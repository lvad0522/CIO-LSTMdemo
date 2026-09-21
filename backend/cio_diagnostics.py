"""独立 CIO 投影与诊断 API。

该路由只负责 CIO 输入、真实模态资产和投影诊断，不加载 LSTM 模型。当前真实可运行
的原始场分支只有 U850；SST 与 SST+U850 联合投影保留稳定契约并明确返回不可用。
"""

from __future__ import annotations

import io
import json
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path, PurePosixPath

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

import cioproj
import live_prediction as prediction


router = APIRouter(prefix="/api/cio", tags=["cio-diagnostics"])

# 可选环境变量覆盖（与 chain.py 的 CHAIN_JOB_ROOT 同形状）：不设 env 时
# 默认路径**逐字节不变**，旧路由不受影响。
JOB_ROOT = Path(os.environ.get(
    "CIO_JOB_ROOT",
    prediction.BACKEND_DIR / "uploads" / "cio_jobs",
))
SUPPORTED_CONFIDENCE = (0.90, 0.95, 0.99)
SPECTRUM_BLOCK_SIZE = 112
SPECTRUM_BOOTSTRAP_SAMPLES = 1000
SPECTRUM_RANDOM_SEED = 20260920

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cio-diagnostic")


def _sst_unavailable() -> HTTPException:
    return HTTPException(
        501,
        detail={
            "code": "SST_NOT_AVAILABLE",
            "message": (
                "SST 投影尚不可用：仓库中没有 SST 原始场及其预处理依据，"
                "不会用 U850-only 结果近似联合 CIO。"
            ),
        },
    )


def _normalize_confidence(value: float) -> float:
    for level in SUPPORTED_CONFIDENCE:
        if abs(value - level) < 1e-9:
            return level
    raise HTTPException(400, "confidence 只支持 0.90、0.95 或 0.99")


def _asset() -> tuple[Path, dict, dict]:
    """返回已通过真实性校验的 MAT 路径、元数据和内容。"""
    meta = prediction._mat_meta()
    path = prediction._mat_path()
    try:
        import scipy.io as sio

        mat = sio.loadmat(str(path))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"读取 CIO 模态资产失败: {exc}") from exc
    return path, meta, mat


def _public_job(job: dict) -> dict:
    fields = (
        "jobId", "status", "stage", "progress", "filename", "inputKind",
        "projectionMode", "calendarKnown", "createdAt", "startedAt",
        "finishedAt", "durationSeconds", "normalization", "projection",
        "warning", "result", "error",
    )
    return {key: job.get(key) for key in fields if key in job}


def _get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "CIO 任务不存在或后端已重启")
        return dict(job)


def _update_job(job_id: str, **changes) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)


def _series_path(job_dir: Path) -> Path:
    return job_dir / "series.npy"


def _normalized_path(job_dir: Path) -> Path:
    return job_dir / "normalized_window.npy"


def _dates_path(job_dir: Path) -> Path:
    return job_dir / "dates.json"


def _completeness_path(job_dir: Path) -> Path:
    return job_dir / "completeness.json"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _spectrum_blocks(values: np.ndarray, dates: list[str] | None) -> np.ndarray:
    """按真实年份或无日期的连续 112 点窗口组织谱估计块。"""
    flat = np.ravel(np.asarray(values, dtype=np.float64))
    if dates is None:
        count = flat.size // SPECTRUM_BLOCK_SIZE
        if count == 0:
            return np.empty((0, SPECTRUM_BLOCK_SIZE), dtype=np.float64)
        return flat[:count * SPECTRUM_BLOCK_SIZE].reshape(count, SPECTRUM_BLOCK_SIZE)

    if len(dates) != flat.size:
        raise ValueError("CIO 序列与日期索引长度不一致")
    by_year: dict[str, list[float]] = {}
    for item, value in zip(dates, flat, strict=True):
        by_year.setdefault(str(item)[:4], []).append(float(value))
    complete = [row for row in by_year.values() if len(row) == SPECTRUM_BLOCK_SIZE]
    if not complete:
        return np.empty((0, SPECTRUM_BLOCK_SIZE), dtype=np.float64)
    return np.asarray(complete, dtype=np.float64)


def _block_periodograms(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算各 112 天块的 Hann 窗周期图，不跨块填补时间缺口。"""
    centered = blocks - np.mean(blocks, axis=1, keepdims=True)
    taper = np.hanning(SPECTRUM_BLOCK_SIZE)
    spectrum = np.fft.rfft(centered * taper, axis=1)
    powers = np.abs(spectrum) ** 2 / np.sum(taper ** 2)
    if SPECTRUM_BLOCK_SIZE % 2 == 0:
        powers[:, 1:-1] *= 2
    else:
        powers[:, 1:] *= 2
    frequencies = np.fft.rfftfreq(SPECTRUM_BLOCK_SIZE, d=1.0)
    valid = frequencies > 0
    periods = 1.0 / frequencies[valid]
    powers = powers[:, valid]
    display = periods >= 5.0
    return periods[display][::-1], powers[:, display][:, ::-1]


def _bootstrap_upper(block_powers: np.ndarray, confidence: float) -> np.ndarray:
    if block_powers.shape[0] == 1:
        return block_powers[0].copy()
    rng = np.random.default_rng(SPECTRUM_RANDOM_SEED)
    indices = rng.integers(
        0,
        block_powers.shape[0],
        size=(SPECTRUM_BOOTSTRAP_SAMPLES, block_powers.shape[0]),
    )
    samples = np.mean(block_powers[indices], axis=1)
    return np.quantile(samples, confidence, axis=0)


def _spectrum_payload(
    values: np.ndarray,
    dates: list[str] | None,
    confidence: float,
    source: str,
    actual_values: np.ndarray | None = None,
) -> dict:
    blocks = _spectrum_blocks(values, dates)
    if blocks.shape[0] == 0:
        return {
            "available": False,
            "confidence": confidence,
            "source": source,
            "blockSize": SPECTRUM_BLOCK_SIZE,
            "blockCount": 0,
            "reason": "功率谱至少需要一个完整的 112 点投影窗口。",
        }

    periods, block_powers = _block_periodograms(blocks)
    mean_power = np.mean(block_powers, axis=0)
    upper = _bootstrap_upper(block_powers, confidence)
    actual_power = None
    actual_upper = None
    if actual_values is not None:
        actual_blocks = _spectrum_blocks(actual_values, dates)
        if actual_blocks.shape[0] == blocks.shape[0]:
            _actual_periods, actual_block_powers = _block_periodograms(actual_blocks)
            actual_power = np.mean(actual_block_powers, axis=0)
            actual_upper = _bootstrap_upper(actual_block_powers, confidence)

    finite = np.isfinite(periods) & np.isfinite(mean_power) & np.isfinite(upper)
    if actual_power is not None and actual_upper is not None:
        finite &= np.isfinite(actual_power) & np.isfinite(actual_upper)
    periods, mean_power, upper = periods[finite], mean_power[finite], upper[finite]
    if actual_power is not None and actual_upper is not None:
        actual_power = actual_power[finite].tolist()
        actual_upper = actual_upper[finite].tolist()
    return {
        "available": True,
        "source": source,
        "method": "112-day block-mean Hann periodogram",
        "confidenceMethod": "block-bootstrap-upper-quantile",
        "confidence": confidence,
        "periodDays": periods.tolist(),
        "projectedPower": mean_power.tolist(),
        "projectedConfidence": upper.tolist(),
        "actualPower": actual_power,
        "actualConfidence": actual_upper,
        "blockSize": SPECTRUM_BLOCK_SIZE,
        "blockCount": int(blocks.shape[0]),
        "periodRangeDays": [5.0, float(SPECTRUM_BLOCK_SIZE)],
        "note": (
            "U850-only 分块诊断谱；各 112 天窗口独立估计后平均，"
            "不等同于论文 SST+U850 全年连续序列。"
        ),
    }


def _reference_for_dates(dates: list[str]) -> np.ndarray:
    _path, _meta, mat = _asset()
    values = np.ravel(np.asarray(mat["pc"], dtype=np.float64)[0])
    start = date(int(np.ravel(mat["yrbeg"])[0]), 1, 1)
    result = []
    for item in dates:
        offset = (date.fromisoformat(item) - start).days
        if offset < 0 or offset >= values.size:
            raise ValueError(f"日期 {item} 超出参考 CIO PC 范围")
        result.append(values[offset])
    return np.asarray(result, dtype=np.float64)


def _npy_completeness(length: int) -> dict:
    return {
        "calendarKnown": False,
        "daysPerYear": None,
        "missingMonths": None,
        "expectedDaysPerYear": 112,
        "blockSize": 112,
        "completeBlocks": length // 112,
        "remainder": length % 112,
        "note": "上传的 .npy 不含日期元数据，不能据此推断年份或缺失月份。",
    }


def _run_npy_job(job_id: str, input_path: Path, job_dir: Path) -> None:
    started = time.time()
    _update_job(
        job_id, status="running", stage="正在读取 CIO 序列", progress=10,
        startedAt=started,
    )
    try:
        raw = np.load(input_path, allow_pickle=False)
        flat = np.asarray(raw)
        if flat.ndim > 1:
            flat = np.squeeze(flat)
        normalized, normalization = prediction._prepare_cio(flat)
        flat = flat.astype(np.float64, copy=False)

        np.save(_series_path(job_dir), flat, allow_pickle=False)
        np.save(_normalized_path(job_dir), normalized, allow_pickle=False)
        completeness = _npy_completeness(int(flat.size))
        _write_json(_completeness_path(job_dir), completeness)

        finished = time.time()
        _update_job(
            job_id,
            status="completed",
            stage="CIO 链路检查完成",
            progress=100,
            calendarKnown=False,
            normalization=normalization,
            warning=normalization.get("warning"),
            result={
                "seriesLength": int(flat.size),
                "normalizedWindowLength": int(normalized.size),
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
            },
            finishedAt=finished,
            durationSeconds=round(finished - started, 3),
        )
    except Exception as exc:  # noqa: BLE001
        finished = time.time()
        _update_job(
            job_id, status="failed", stage="CIO 链路检查失败",
            error=f"{type(exc).__name__}: {exc}", finishedAt=finished,
            durationSeconds=round(finished - started, 3),
        )


def _zip_completeness(meta: dict) -> dict:
    days = meta.get("days_per_year") or {}
    years = meta.get("years") or []
    rows = [
        {
            "year": int(year),
            "days": int(days.get(year, days.get(str(year), 0))),
            "expected": 112,
        }
        for year in years
    ]
    return {
        "calendarKnown": True,
        "daysPerYear": rows,
        "missingMonths": [list(item) for item in meta.get("missing_months") or []],
        "expectedDaysPerYear": 112,
        "blockSize": 112,
        "completeBlocks": sum(1 for row in rows if row["days"] == 112),
        "remainder": None,
        "window": meta.get("window"),
    }


def _run_zip_job(
    job_id: str, zip_path: Path, job_dir: Path, years: list[int], lead: int,
) -> None:
    started = time.time()
    _update_job(
        job_id, status="running", stage="正在解压 U850 原始场", progress=5,
        startedAt=started,
    )
    try:
        raw_dir = job_dir / "raw"
        extracted = prediction._safe_extract_nc(zip_path, raw_dir)
        nc_dir = prediction._locate_nc_dir(raw_dir)
        _update_job(job_id, stage="正在投影 U850-only CIO", progress=25)
        series, meta = cioproj.compute_cioproj(
            which="u850",
            nc_dir=str(nc_dir),
            years=years,
            lead=lead,
            mat_path=str(prediction._mat_path()),
            name="u850",
        )
        dates = meta.pop("dates", None)
        if dates is None:
            raise RuntimeError("U850 投影结果缺少日期索引")

        _update_job(job_id, stage="正在按训练口径归一化", progress=85)
        normalized, normalization = prediction._prepare_cio(series)
        series = np.asarray(series, dtype=np.float64)
        np.save(_series_path(job_dir), series, allow_pickle=False)
        np.save(_normalized_path(job_dir), normalized, allow_pickle=False)
        _write_json(_dates_path(job_dir), dates)
        completeness = _zip_completeness(meta)
        _write_json(_completeness_path(job_dir), completeness)

        asset_meta = prediction._mat_meta()
        projection = {
            "kind": "u850",
            "lead": lead,
            "years": years,
            "window": meta.get("window"),
            "seriesLength": int(series.size),
            "daysPerYear": meta.get("days_per_year"),
            "missingMonths": meta.get("missing_months"),
            "sourceFiles": extracted,
            **asset_meta,
        }
        warnings = [w for w in (normalization.get("warning"), meta.get("warning")) if w]
        finished = time.time()
        _update_job(
            job_id,
            status="completed",
            stage="U850-only CIO 投影完成",
            progress=100,
            calendarKnown=True,
            normalization=normalization,
            projection=projection,
            warning=" ".join(warnings) if warnings else None,
            result={
                "seriesLength": int(series.size),
                "normalizedWindowLength": int(normalized.size),
                "min": float(np.min(series)),
                "max": float(np.max(series)),
            },
            finishedAt=finished,
            durationSeconds=round(finished - started, 3),
        )
    except Exception as exc:  # noqa: BLE001
        finished = time.time()
        _update_job(
            job_id, status="failed", stage="U850-only CIO 投影失败",
            error=f"{type(exc).__name__}: {exc}", finishedAt=finished,
            durationSeconds=round(finished - started, 3),
        )


@router.get("/capabilities")
def cio_capabilities():
    try:
        _path, meta, _mat = _asset()
        u850 = {"available": True, "assetSource": meta["matSource"]}
    except (ValueError, HTTPException) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        u850 = {"available": False, "reason": detail}
    sst_reason = "缺少 SST 原始场与可复现的预处理链路。"
    return {
        "channels": {
            "u850": u850,
            "sst": {"available": False, "reason": sst_reason},
        },
        "jointProjection": {
            "available": False,
            "reason": "论文口径需要 SST 与 U850 同时投影；当前仅有 U850。",
        },
        "spectrum": {
            "available": True,
            "scope": "completed-cio-job",
            "method": "112-day block-mean Hann periodogram",
            "jointAvailable": False,
            "jointReason": "缺少 SST，不能计算论文口径的联合 CIO 功率谱。",
            "blockSize": SPECTRUM_BLOCK_SIZE,
            "supportedConfidence": list(SUPPORTED_CONFIDENCE),
        },
    }


@router.get("/reference")
def cio_reference():
    _path, meta, mat = _asset()
    pc = np.asarray(mat["pc"], dtype=np.float64)
    values = np.ravel(pc[0])
    yrbeg = int(np.ravel(mat["yrbeg"])[0])
    yrend = int(np.ravel(mat["yrend"])[0])
    start = date(yrbeg, 1, 1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(values.size)]
    expected_end = date(yrend, 12, 31).isoformat()
    if not dates or dates[-1] != expected_end:
        raise HTTPException(
            500,
            f"参考 PC 日期长度与 {yrbeg}-{yrend} 不一致（末日 {dates[-1] if dates else 'none'}）",
        )
    return {
        "name": "actual-cio-eof-pc1",
        "startDate": dates[0],
        "endDate": dates[-1],
        "dates": dates,
        "values": values.tolist(),
        "assetSource": meta["matSource"],
        "assetName": prediction._mat_path().name,
    }


@router.get("/mode/u850")
def cio_u850_mode():
    _path, meta, mat = _asset()
    lon = np.ravel(mat["Lon"]).astype(float)
    lat = np.ravel(mat["Lat"]).astype(float)
    raw = np.asarray(mat["mode_U"][0], dtype=np.float64)
    if raw.shape != (lon.size, lat.size):
        raise HTTPException(500, f"mode_U[0] 形状 {raw.shape} 与坐标不一致")
    data = raw.T
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        data = data[:, ::-1]
    expvar = float(np.ravel(mat["expvar"])[0])
    return {
        "field": "u850",
        "mode": 1,
        "rows": int(lat.size),
        "cols": int(lon.size),
        "lat": lat.tolist(),
        "lon": lon.tolist(),
        "data": data.tolist(),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "explainedVariancePercent": expvar,
        "assetSource": meta["matSource"],
    }


@router.get("/spectrum")
def cio_spectrum(confidence: float = Query(0.95)):
    level = _normalize_confidence(confidence)
    return {
        "available": False,
        "confidence": level,
        "confidenceMethod": None,
        "period": None,
        "actualPower": None,
        "projectedPower": None,
        "actualConfidence": None,
        "projectedConfidence": None,
        "reason": (
            "当前缺少 SST+U850 联合投影和连续逐日序列；"
            "不会把 Pearson 临界相关系数用作功率谱置信线。"
        ),
    }


@router.post("/jobs", status_code=202)
async def create_cio_job(
    file: UploadFile = File(...),
    lead: int = Query(1, ge=1, le=20),
    which: str = Query(""),
):
    requested = (which or "").strip().lower()
    if requested in ("sst", "both"):
        raise _sst_unavailable()
    if requested not in ("", "auto", "u850", "cio"):
        raise HTTPException(400, "which 只支持 cio / u850 / sst / both / auto")

    name = file.filename or ""
    suffix = PurePosixPath(name.replace("\\", "/")).suffix.lower()
    if suffix not in (".npy", ".zip"):
        raise HTTPException(400, "只接受 .npy（CIO 序列）或 .zip（U850 原始场）")
    content = await file.read()
    if not content:
        raise HTTPException(400, "上传文件为空")

    if suffix == ".zip" and requested not in ("u850",):
        try:
            kind, _source = prediction._resolve_kind(requested, name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if kind in ("sst", "both"):
            raise _sst_unavailable()
        requested = kind
    if suffix == ".npy" and requested == "u850":
        raise HTTPException(400, ".npy 入口应是已投影的一维 CIO，请使用 which=cio 或留空")

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    created = time.time()

    try:
        if suffix == ".npy":
            try:
                raw = np.load(io.BytesIO(content), allow_pickle=False)
                _normalized, normalization = prediction._prepare_cio(raw)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"CIO 文件校验失败: {exc}") from exc
            input_path = job_dir / "cio_input.npy"
            np.save(input_path, raw, allow_pickle=False)
            job = {
                "jobId": job_id,
                "status": "queued",
                "stage": "等待 CIO 链路检查",
                "progress": 0,
                "filename": name,
                "inputKind": "cio-npy",
                "projectionMode": "uploaded_cio",
                "calendarKnown": False,
                "normalization": normalization,
                "createdAt": created,
                "jobDir": str(job_dir),
            }
            runner = (_run_npy_job, job_id, input_path, job_dir)
        else:
            if not zipfile.is_zipfile(io.BytesIO(content)):
                raise ValueError("上传的 .zip 无法解析（不是有效的 zip 文件）")
            zip_path = job_dir / "u850_fields.zip"
            zip_path.write_bytes(content)
            years = prediction._zip_years(zip_path)
            prediction._mat_meta()
            job = {
                "jobId": job_id,
                "status": "queued",
                "stage": "等待 U850-only CIO 投影",
                "progress": 0,
                "filename": name,
                "inputKind": "raw-zip",
                "projectionMode": "u850_only",
                "calendarKnown": True,
                "projection": {"kind": "u850", "lead": lead, "years": years},
                "createdAt": created,
                "jobDir": str(job_dir),
            }
            runner = (_run_zip_job, job_id, zip_path, job_dir, years, lead)
    except (ValueError, NotImplementedError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc

    with _jobs_lock:
        _jobs[job_id] = job
    _executor.submit(*runner)
    return _public_job(job)


@router.get("/jobs/{job_id}")
def cio_job_status(job_id: str):
    return _public_job(_get_job(job_id))


def _completed_job(job_id: str) -> tuple[dict, Path]:
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "CIO 任务尚未完成")
    return job, Path(job["jobDir"])


@router.get("/jobs/{job_id}/series")
def cio_job_series(job_id: str):
    job, job_dir = _completed_job(job_id)
    raw_path = _series_path(job_dir)
    norm_path = _normalized_path(job_dir)
    if not raw_path.is_file() or not norm_path.is_file():
        raise HTTPException(404, "CIO 序列产物不存在")
    raw = np.load(raw_path, allow_pickle=False)
    normalized = np.load(norm_path, allow_pickle=False)
    dates = _read_json(_dates_path(job_dir)) if _dates_path(job_dir).is_file() else None
    return {
        "projectionMode": job["projectionMode"],
        "calendarKnown": bool(job.get("calendarKnown")),
        "dates": dates,
        "sampleIndices": list(range(int(raw.size))) if dates is None else None,
        "raw": np.ravel(raw).tolist(),
        "normalizedWindow": np.ravel(normalized).tolist(),
    }


@router.get("/jobs/{job_id}/completeness")
def cio_job_completeness(job_id: str):
    _job, job_dir = _completed_job(job_id)
    path = _completeness_path(job_dir)
    if not path.is_file():
        raise HTTPException(404, "CIO 完整性产物不存在")
    return _read_json(path)


@router.get("/jobs/{job_id}/spectrum")
def cio_job_spectrum(job_id: str, confidence: float = Query(0.95)):
    level = _normalize_confidence(confidence)
    job, job_dir = _completed_job(job_id)
    path = _series_path(job_dir)
    if not path.is_file():
        raise HTTPException(404, "CIO 序列产物不存在")
    values = np.load(path, allow_pickle=False)
    dates = _read_json(_dates_path(job_dir)) if _dates_path(job_dir).is_file() else None
    source = "u850-only" if job.get("projectionMode") == "u850_only" else "uploaded-cio"
    actual = _reference_for_dates(dates) if dates is not None else None
    return _spectrum_payload(values, dates, level, source, actual_values=actual)
