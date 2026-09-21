"""一条链：① 投影 CIO → ② 归一化 → ③ LSTM 降水推理。

与 `/api/cio`（投影+诊断）、`/api/predict`（投影+归一化+推理）并列的第三个
router，前缀 `/api/chain`。它**不重新实现任何口径**，只做编排：

  `.zip`（5–9 月逐日原始场）→ ① `cioproj.compute_cioproj`
                             → ② `live_prediction._prepare_cio`
                             → ③ `live_prediction._run_archive_inference`
  `.npy`（已投影一维 CIO）   → 跳过 ①，直接 ② → ③

**段间只走落盘文件**：① 写 `series.npy`（float64）+ `dates.json` +
`completeness.json`，② `np.load(series.npy)` 写 `normalized_window.npy`
（float32），③ `np.load(normalized_window.npy)` 并**直接**喂进
`_run_archive_inference`。③ 不可能拿到别的张量，因此界面上看到的就是喂给
模型的那一份，位级一致由"同一份实现 + 同一个文件"继承。

⚠ 内部契约 ⚠
  下面用到的 `live_prediction` / `cioproj` 的下划线函数
  （`_mat_meta` / `_mat_path` / `_resolve_kind` / `_zip_years` /
  `_safe_extract_nc` / `_locate_nc_dir` / `_prepare_cio` /
  `_run_archive_inference` / `_verify_against_truth`）与 `cio_diagnostics`
  的 `_spectrum_payload` / `_reference_for_dates` / `_normalize_confidence` /
  `_read_json` / `_write_json` 是**跨模块内部契约**：改动它们必须同步本文件。
  之所以直接 import 而不是抽公共模块，是为了让旧路由 `cioproj.py` /
  `live_prediction.py` / `cio_diagnostics.py` 一行不改（零回归），并且让
  「数不许变」由"调用同一份代码"保证，而不是"写两套等价代码再祈祷"。
"""

from __future__ import annotations

import io
import os
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

import cioproj
import cio_diagnostics as diagnostics
import live_prediction as prediction
import verification


router = APIRouter(prefix="/api/chain", tags=["prediction-chain"])

# 可选环境变量覆盖（C-5）：默认路径字节不变，旧路由不受影响。
JOB_ROOT = Path(os.environ.get(
    "CHAIN_JOB_ROOT",
    prediction.BACKEND_DIR / "uploads" / "chain_jobs",
))

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chain")

# 受理校验的唯一规则表：全链与只跑投影不分工，都是 year=2000 & lead=pre1
FIXED_YEAR = 2000
FIXED_LEAD = "pre1"

# 进度条分工：①② 共占 3–33%，③ 占 33–97%，落盘与检验到 100%
STAGE1_SPAN = (3, 30)
STAGE2_SPAN = (30, 33)
INFERENCE_SPAN = (33, 97)

# 未就绪文案按段给出（design D8），不静默返回空数组
STAGE_NOT_READY = {
    1: "投影尚未完成",
    2: "归一化尚未完成",
    3: "降水推理尚未完成",
}

# 可公开给前端的字段白名单：**不含磁盘路径**（服务器绝对路径不外泄；
# 旧 `/api/predict` 与 `/api/cio` 同样不公开，见 C-2）。内部路径一律由
# `_job_dir()` 从 `JOB_ROOT + jobId` 现拼，不进任务状态、不进响应。
_PUBLIC_FIELDS = (
    "jobId", "status", "stage", "progress", "stageIndex", "filename",
    "inputKind", "projectionMode", "calendarKnown", "normalization",
    "projection", "inputShape", "warning", "createdAt", "startedAt",
    "finishedAt", "durationSeconds", "result", "error",
)


# --------------------------------------------------------------- 任务状态

def _public_job(job: dict) -> dict:
    """仅返回可安全公开给前端的任务字段（含阶段门控用的 `stageIndex`）。"""
    return {key: job.get(key) for key in _PUBLIC_FIELDS if key in job}


def _get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "链路任务不存在或后端已重启")
        return dict(job)


def _update_job(job_id: str, **changes) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)


def _advance(job_id: str, stage_index: int, **changes) -> None:
    """阶段完成时把该段字段增量写入任务状态（design D6）。"""
    _update_job(job_id, stageIndex=stage_index, **changes)


def _report(job_id: str, span: tuple[int, int]):
    """把某段内部的 0–100 进度映射到全局进度条区间。"""
    lo, hi = span

    def report(progress: int, stage: str) -> None:
        _update_job(
            job_id,
            progress=lo + int(max(0, min(100, progress)) / 100 * (hi - lo)),
            stage=stage,
        )

    return report


# --------------------------------------------------------- 产物路径与读取

def _series_path(job_dir: Path) -> Path:
    return job_dir / "series.npy"


def _normalized_path(job_dir: Path) -> Path:
    return job_dir / "normalized_window.npy"


def _dates_path(job_dir: Path) -> Path:
    return job_dir / "dates.json"


def _completeness_path(job_dir: Path) -> Path:
    return job_dir / "completeness.json"


def _prediction_path(job_dir: Path) -> Path:
    return job_dir / "prediction.npy"


def _pearson_path(job_dir: Path) -> Path:
    return job_dir / "pearson_r.npy"


def _job_dir(job: dict) -> Path:
    """任务目录 = `JOB_ROOT / jobId` 现拼。

    不在任务状态里存绝对路径：那份状态会经 `_public_job()` 出去，存了就早晚
    跟着漏（C-2）。拼法只有这一处，改 `JOB_ROOT`（env 或猴补丁）也随之生效。
    """
    return JOB_ROOT / job["jobId"]


def _require(job_id: str, stage: int) -> tuple[dict, Path]:
    """取任务与目录，并断言该段已跑完；未就绪 → 409 + 段级文案。"""
    job = _get_job(job_id)
    if int(job.get("stageIndex") or 0) < stage:
        raise HTTPException(409, STAGE_NOT_READY[stage])
    return job, _job_dir(job)


def _require_series(job_id: str) -> tuple[dict, np.ndarray]:
    job, job_dir = _require(job_id, 1)
    path = _series_path(job_dir)
    if not path.is_file():
        raise HTTPException(404, "投影CIO 序列产物不存在")
    return job, np.load(path, allow_pickle=False)


def _require_normalized(job_id: str) -> tuple[dict, np.ndarray]:
    job, job_dir = _require(job_id, 2)
    path = _normalized_path(job_dir)
    if not path.is_file():
        raise HTTPException(404, "归一化窗口产物不存在")
    return job, np.load(path, allow_pickle=False)


def _require_prediction(job_id: str) -> tuple[dict, Path]:
    job, job_dir = _require(job_id, 3)
    path = _prediction_path(job_dir)
    if not path.is_file():
        raise HTTPException(404, "降水预测产物不存在")
    return job, path


def _truth_path(job: dict) -> Path:
    # C-2/B-3：实况路径读**任务顶层** `truthFile`（进程内字段，不在 `_PUBLIC_FIELDS`
    # 里，所以进不了响应）。别再回 `result.verification.truthPath` 读 —— 那个键
    # 已按 PM 口径在调用点摘除，读它等于两条 truth/preview 一起 409。
    meta = (job.get("result") or {}).get("verification") or {}
    truth_path = job.get("truthFile")
    if not truth_path or not Path(truth_path).is_file():
        raise HTTPException(409, meta.get("reason") or "该任务没有可用的实况场")
    return Path(truth_path)


# ------------------------------------------------------------- 受理校验

def _parse_lead(lead: str) -> int:
    """`pre{n}` → n；格式不对统一按权重未覆盖处理（与旧路由文案一致）。"""
    match = prediction.LEAD_RE.match((lead or "").strip())
    if match is None:
        raise HTTPException(400, "当前模型验证仅支持 year=2000、lead=pre1")
    return int(match.group(1))


def _mat_identity(required: bool) -> dict:
    """标准CIO 身份。`.zip` 必需（失败即 400）；`.npy` 尽力而为不阻断。"""
    try:
        meta = prediction._mat_meta()
    except ValueError as exc:
        if required:
            raise HTTPException(400, str(exc)) from exc
        return {"matSource": None, "matPath": None, "matUnavailable": str(exc)}
    return {
        "matSource": meta["matSource"],
        "matPath": Path(meta["matPath"]).name,
        "orthoDeviation": float(meta["orthoDeviation"]),
        # 模态矩阵形状（标准CIO 身份行要用）。列数与 cioproj 的切片口径同源，
        # 不在这里另写 2707/3321 —— 那就是"两套口径"的开始。
        "nModes": int(prediction.MAT_SHAPE[0]),
        "sstColumns": int(cioproj.N_SST),
        "u850Columns": int(cioproj.N_U850),
    }


def _resolve_zip_kind(which: str, filename: str) -> tuple[str, str]:
    """`.zip`：显式参数 > 文件名关键字；判不出就 400（绝不默认 U850）。"""
    try:
        kind, kind_source = prediction._resolve_kind(which, filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if kind in ("sst", "both"):
        try:
            cioproj.compute_cioproj(kind, name=filename)  # 抛 NotImplementedError
        except NotImplementedError as exc:
            raise HTTPException(400, str(exc)) from exc
    return kind, kind_source


def _resolve_npy_which(which: str) -> str:
    """`.npy` 入口应是已投影的一维 CIO：只认 空 / auto / cio。"""
    requested = (which or "").strip().lower()
    if requested in ("", "auto", "cio"):
        return requested or "auto"
    if requested == "u850":
        raise HTTPException(
            400, ".npy 入口应是已投影的一维 CIO，请使用 which=cio 或留空"
        )
    if requested in ("sst", "both"):
        raise HTTPException(400, "SST 分支尚未实现：缺 SST 原始场及其预处理源码")
    raise HTTPException(400, "which 只能是 u850 / sst / both / auto")


def _intake_zip(content: bytes, job_dir: Path, which: str, filename: str,
                year: int, lead: int) -> dict:
    """`.zip` 受理：全程 O(1) 级校验（解压与投影留在后台）。"""
    zip_path = job_dir / "raw_fields.zip"
    zip_path.write_bytes(content)          # 只写这一份，不再另存 bytes（design R12）

    kind, kind_source = _resolve_zip_kind(which, filename)
    identity = _mat_identity(required=True)

    try:
        years = prediction._zip_years(zip_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if year not in years:
        raise HTTPException(
            400,
            f"zip 里没有 {year} 年的 nc（现有年份 {years[0]}–{years[-1]}）；"
            f"模型 pre1/{year} 需要该年作为预测目标年。",
        )

    return {
        "inputKind": "raw-zip",
        "kind": kind,
        "kindSource": kind_source,
        "requestedWhich": (which or "").strip().lower() or "auto",
        "lead": lead,
        "year": year,
        "years": years,
        "window": cioproj.window(lead),
        **identity,
    }


def _intake_npy(content: bytes, job_dir: Path, which: str, year: int,
                lead: int) -> tuple[dict, dict]:
    """`.npy` 受理：预校验（`_prepare_cio`）并立刻落盘 series.npy。"""
    requested = _resolve_npy_which(which)
    try:
        raw = np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"CIO 文件校验失败: {exc}") from exc
    flat = np.asarray(raw)
    if flat.ndim > 1:
        flat = np.squeeze(flat)
    try:
        _selected, normalization = prediction._prepare_cio(flat, pre_year=0)
    except ValueError as exc:
        raise HTTPException(400, f"CIO 文件校验失败: {exc}") from exc

    np.save(job_dir / "cio_input.npy", np.asarray(raw), allow_pickle=False)
    # `.npy` 跳过①，它的落盘序列就是"① 的语义载体"：统一成 float64（design D9）
    series = np.asarray(flat, dtype=np.float64)
    np.save(_series_path(job_dir), series, allow_pickle=False)
    # 阶段性产物一次写全：① 区一解锁就要能画完整性表，这里缺了会 404 撕掉整张卡
    diagnostics._write_json(
        _completeness_path(job_dir), diagnostics._npy_completeness(int(series.size))
    )

    return {
        "inputKind": "cio-npy",
        "kind": "cio",
        "kindSource": "upload",
        "requestedWhich": requested,
        "lead": lead,
        "year": year,
        "seriesLength": int(series.size),
        # 阶段① 没跑，资产身份仍尽力给一份：取得到就显示，取不到就不显示该行
        **_mat_identity(required=False),
    }, normalization


# ------------------------------------------------------------- 三段执行

def _run_zip_stage1(job_id: str, job_dir: Path, projection: dict, lead: int,
                    filename: str) -> dict:
    """① 投影 `.zip` → series.npy / dates.json / completeness.json。"""
    zip_path = job_dir / "raw_fields.zip"
    years = projection["years"]
    kind = projection["kind"]
    report = _report(job_id, STAGE1_SPAN)

    report(5, "正在解压原始气象场")
    raw_dir = job_dir / "raw"
    extracted = prediction._safe_extract_nc(zip_path, raw_dir)
    nc_dir = prediction._locate_nc_dir(raw_dir)

    report(20, f"正在投影 CIO（{len(years)} 年 × 5 月，{extracted} 个 nc）")
    series, meta = cioproj.compute_cioproj(
        which=kind,
        nc_dir=str(nc_dir),
        years=years,
        lead=lead,
        mat_path=str(prediction._mat_path()),
        name=filename,
    )
    dates = meta.get("dates")
    if dates is None:                       # 不吞日期索引：谱与参考线都靠它
        raise RuntimeError("投影结果缺少日期索引（dates）")

    series = np.asarray(series, dtype=np.float64)
    np.save(_series_path(job_dir), series, allow_pickle=False)
    diagnostics._write_json(_dates_path(job_dir), dates)
    diagnostics._write_json(
        _completeness_path(job_dir), diagnostics._zip_completeness(meta)
    )

    fields = {
        "seriesLength": int(series.size),
        "window": meta.get("window"),
        "years": meta.get("years", years),
        "daysPerYear": meta.get("days_per_year"),
        "missingMonths": meta.get("missing_months"),
    }
    _advance(
        job_id, 1,
        calendarKnown=True,
        projection={**projection, **fields},
        result={
            "seriesLength": int(series.size),
            "min": float(np.min(series)),
            "max": float(np.max(series)),
        },
        warning=meta.get("warning"),
    )
    return series


def _run_stage2(job_id: str, job_dir: Path, projection: dict) -> np.ndarray:
    """② 归一化：读落盘的 series.npy → normalized_window.npy（float32）。"""
    report = _report(job_id, STAGE2_SPAN)
    report(10, "正在按训练口径归一化")
    path = _series_path(job_dir)
    if not path.is_file():
        raise RuntimeError("归一化前找不到落盘的 series.npy")
    series = np.load(path, allow_pickle=False)
    selected, normalization = prediction._prepare_cio(series, pre_year=0)
    np.save(_normalized_path(job_dir), selected, allow_pickle=False)

    # 阶段① 在 `_advance` 里已写过自己的 warning（缺月提示，来自 cioproj
    # 的 meta["warning"]）；阶段① 的入参 projection **没有** warning 键，
    # 所以这里必须"有才并、没有不动"，不能无条件写：
    #   `warning=" ".join(...) if warnings else None` 会把阶段① 的 warning
    #   覆盖成 None（旧路由不丢：live_prediction.py 把 proj['warning'] 并进
    #   projection 后再合并）。触发条件是"zip 缺月份但目标年在"，生产可达。
    warnings = [
        w for w in (normalization.get("warning"), projection.get("warning")) if w
    ]
    changes = {
        "normalization": normalization,
        "inputShape": [normalization["inputLength"]],
    }
    if warnings:
        changes["warning"] = " ".join(warnings)
    _advance(
        job_id, 2,
        **changes,
        result={
            "seriesLength": int(series.size),
            "normalizedWindowLength": int(selected.size),
            "min": float(np.min(series)),
            "max": float(np.max(series)),
        },
    )
    return selected


def _run_stage3(job_id: str, job_dir: Path, lead: str, year: int) -> None:
    """③ 推理：读落盘的 normalized_window.npy，直接喂进既有推理函数。"""
    path = _normalized_path(job_dir)
    if not path.is_file():
        raise RuntimeError("推理前找不到落盘的 normalized_window.npy")
    cio = np.load(path, allow_pickle=False)     # 唯一来源就是② 的落盘件

    # ⚠ 这里**不能**再套 `_report(job_id, INFERENCE_SPAN)`：`_run_archive_inference`
    # 的回调已经按**绝对区间**给值（live_prediction.py 里 `lo + processed/N*(hi-lo)`
    # 且 `lo, hi = span`，本处 span=INFERENCE_SPAN）→ 33…97。再映射一次会把
    # 33…97 当 0…100 压缩成 54…95：③ 段少用一半区间，且从 ② 段终点 30 直接跳到 54
    # （C-7）。进度条的全局换算由被调方负责，本段只做透传。
    def report(progress: int, stage: str) -> None:
        _update_job(job_id, progress=progress, stage=stage)

    prediction_array = prediction._run_archive_inference(
        cio, report, span=INFERENCE_SPAN
    )
    report(97, "正在保存 prediction.npy")
    np.save(_prediction_path(job_dir), prediction_array, allow_pickle=False)

    report(98, "正在对比实况场计算相关系数")
    verification = prediction._verify_against_truth(
        prediction_array, job_dir, job_id, {"lead": lead, "year": year}
    )
    # C-2/B-3：实况文件的绝对路径**只留在进程内**（下面的 `truthFile`），摘出来
    # 再进 `result.verification` —— 那份 result 会经 `_public_job()` 出去。
    truth_file = verification.pop("truthPath", None)
    # 检验与预测是同一份实况对照，下载路径由本文件统一指向链路前缀（design D7）
    if verification.get("available"):
        verification["downloadUrl"] = (
            f"/api/chain/jobs/{job_id}/download/pearson"
        )

    _advance(
        job_id, 3,
        result={
            "shape": list(prediction_array.shape),
            "dtype": str(prediction_array.dtype),
            "min": float(np.min(prediction_array)),
            "max": float(np.max(prediction_array)),
            "mean": float(np.mean(prediction_array)),
            "normalizedWindowLength": int(cio.size),
            "downloadUrl": f"/api/chain/jobs/{job_id}/download/prediction",
            "verification": verification,
        },
        truthFile=truth_file,
    )


def _run_chain_job(job_id: str, projection: dict, only_projection: bool,
                   filename: str) -> None:
    """后台执行整条链；异常一律落到 job 的 `error`，与两个旧 router 同写法。"""
    started = time.time()
    lead_index = projection["lead"]
    lead = f"pre{lead_index}"
    year = projection["year"]
    job_dir = JOB_ROOT / job_id

    _update_job(
        job_id, status="running", stage="正在校验输入", progress=1,
        startedAt=started,
    )
    try:
        if projection["inputKind"] == "raw-zip":
            _run_zip_stage1(job_id, job_dir, projection, lead_index, filename)
        else:
            _advance(job_id, 1, calendarKnown=False, projection={
                **projection, "stage1Skipped": True,
            })

        _run_stage2(job_id, job_dir, projection)

        if only_projection:
            # 只跑投影：绝不启动③，也不加载任何 .pt
            finished = time.time()
            _update_job(
                job_id,
                status="completed",
                stage="投影与归一化完成（未跑降水推理）",
                progress=100,
                finishedAt=finished,
                durationSeconds=round(finished - started, 2),
            )
            return

        _run_stage3(job_id, job_dir, lead, year)

        finished = time.time()
        _update_job(
            job_id,
            status="completed",
            stage="链路完成",
            progress=100,
            finishedAt=finished,
            durationSeconds=round(finished - started, 2),
        )
    except Exception as exc:  # 后台线程必须把错误保存给前端
        finished = time.time()
        # 总闸：同旧路由，`error` 的文本会进响应并被前端上屏。净化器与旧路由
        # **共用同一个** `prediction._safe_exc`，不另写一份（口径分叉的老路）。
        # 覆盖范围只限执行路径；受理路径未纳入（见 `_safe_exc` docstring）。
        _update_job(
            job_id,
            status="failed",
            stage="链路失败",
            finishedAt=finished,
            durationSeconds=round(finished - started, 2),
            error=prediction._safe_exc(exc),
        )


# ------------------------------------------------------------------ 路由

@router.post("/jobs", status_code=202)
async def create_chain_job(
    file: UploadFile = File(...),
    year: int = Query(FIXED_YEAR),
    lead: str = Query(FIXED_LEAD),
    which: str = Query(""),
    onlyProjection: bool = Query(False),
):
    """一次提交承载整条链：`.zip` 从① 开始，`.npy` 跳过①、从② 开始。

    参数说明：`year` / `lead` 固定为 2000 / pre1（唯一规则表）；`which` 在
    `.zip` 下交给 `_resolve_kind`，在 `.npy` 下只认 空 / auto / cio；
    `onlyProjection=true` 只跑①②（不加载任何 `.pt`）。
    """
    # 规则表唯一：全链与只跑投影同一套，文案与旧 `/api/predict/jobs` 逐字一致
    if year != FIXED_YEAR or lead != FIXED_LEAD:
        raise HTTPException(400, "当前模型验证仅支持 year=2000、lead=pre1")
    lead_index = _parse_lead(lead)

    name = file.filename or ""
    suffix = PurePosixPath(name.replace("\\", "/")).suffix.lower()
    if suffix not in (".npy", ".zip"):
        raise HTTPException(400, "只接受 .npy（CIO 序列）或 .zip（原始气象场）")
    content = await file.read()
    if not content:
        raise HTTPException(400, "上传文件为空")

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    try:
        if suffix == ".npy":
            projection, normalization = _intake_npy(
                content, job_dir, which, year, lead_index
            )
            projection_mode = "uploaded_cio"
        else:
            if not zipfile.is_zipfile(io.BytesIO(content)):
                raise ValueError("上传的 .zip 无法解析（不是有效的 zip 文件）")
            projection = _intake_zip(
                content, job_dir, which, name, year, lead_index
            )
            normalization = None
            projection_mode = "u850_only"
    except (ValueError, NotImplementedError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)   # 不留半张图
        raise HTTPException(400, str(exc)) from exc
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    job = {
        "jobId": job_id,
        "status": "queued",
        "stage": "等待链路执行",
        "progress": 0,
        "stageIndex": 0,
        "filename": name,
        "inputKind": projection["inputKind"],
        "projectionMode": projection_mode,
        "projection": projection,
        "normalization": normalization,
        "createdAt": time.time(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _executor.submit(
        _run_chain_job, job_id, projection, bool(onlyProjection), name
    )
    return _public_job(job)


@router.get("/jobs/{job_id}")
def chain_job_status(job_id: str):
    return _public_job(_get_job(job_id))


@router.get("/jobs/{job_id}/series")
def chain_job_series(job_id: str):
    """投影序列与归一化窗口。字段名与 `/api/cio/jobs/{id}/series` 逐字一致。"""
    job, raw = _require_series(job_id)
    job_dir = _job_dir(job)
    dates_path = _dates_path(job_dir)
    dates = diagnostics._read_json(dates_path) if dates_path.is_file() else None
    norm_path = _normalized_path(job_dir)
    # ② 未跑完时给空数组而不是 409：① 区一解锁就要能画图（design D6）
    normalized = (
        np.load(norm_path, allow_pickle=False) if norm_path.is_file()
        else np.empty(0, dtype=np.float32)
    )
    return {
        "projectionMode": job["projectionMode"],
        "calendarKnown": bool(job.get("calendarKnown")),
        "dates": dates,
        "sampleIndices": list(range(int(raw.size))) if dates is None else None,
        "raw": np.ravel(raw).tolist(),
        "normalizedWindow": np.ravel(normalized).tolist(),
    }


@router.get("/jobs/{job_id}/completeness")
def chain_job_completeness(job_id: str):
    job, _raw = _require_series(job_id)
    path = _completeness_path(_job_dir(job))
    if not path.is_file():
        raise HTTPException(404, "完整性产物不存在")
    return diagnostics._read_json(path)


def _reference_overlay(dates: list[str]):
    """实际 CIO 参考线，**尽力而为**。

    参考 PC 只覆盖 1982–2017（`_reference_for_dates` 出厂行为），而 `.zip`
    入口的投影窗口是 2000–2020 —— 后半段取不到参考值。与旧 `/api/cio` 不同，
    这里不让整张谱 500：取不到就退回 None，谱本身（projectedPower）照给，
    前端只是少一条蓝线。`_spectrum_payload` 只在块数一致时才启用参考线，
    所以宁可整段不给，也不拼半截序列。
    """
    try:
        return diagnostics._reference_for_dates(dates)
    except ValueError:
        return None


@router.get("/jobs/{job_id}/spectrum")
def chain_job_spectrum(job_id: str, confidence: float = Query(0.95)):
    level = _confidence(confidence)
    job, values = _require_series(job_id)
    job_dir = _job_dir(job)
    dates_path = _dates_path(job_dir)
    dates = diagnostics._read_json(dates_path) if dates_path.is_file() else None
    source = "u850-only" if job["projectionMode"] == "u850_only" else "uploaded-cio"
    actual = _reference_overlay(dates) if dates is not None else None
    return diagnostics._spectrum_payload(
        values, dates, level, source, actual_values=actual
    )


def _confidence(value: float) -> float:
    """confidence 非法 → 400（复用 `/api/cio` 的文案与档位表）。"""
    try:
        return diagnostics._normalize_confidence(value)
    except HTTPException as exc:
        raise HTTPException(400, exc.detail) from exc


@router.get("/jobs/{job_id}/preview")
def chain_preview(
    job_id: str,
    time_index: int = Query(0, ge=0, lt=prediction.N_STEPS),
):
    _job, path = _require_prediction(job_id)
    prediction_array = np.load(path, mmap_mode="r", allow_pickle=False)
    frame = np.asarray(prediction_array[:, :, time_index])
    return {
        "timeIndex": time_index,
        "rows": prediction.GRID_ROWS,
        "cols": prediction.GRID_COLS,
        "min": float(np.min(frame)),
        "max": float(np.max(frame)),
        "data": frame.tolist(),
    }


@router.get("/jobs/{job_id}/truth/preview")
def chain_truth_preview(
    job_id: str,
    time_index: int = Query(0, ge=0, lt=prediction.N_STEPS),
):
    job, _path = _require_prediction(job_id)
    truth_path = _truth_path(job)
    truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
    frame = np.asarray(truth[:, :, time_index], dtype=np.float64)
    return {
        "timeIndex": time_index,
        "rows": int(frame.shape[0]),
        "cols": int(frame.shape[1]),
        "min": float(np.min(frame)),
        "max": float(np.max(frame)),
        "data": frame.tolist(),
    }


@router.get("/jobs/{job_id}/grid")
def chain_grid_series(
    job_id: str,
    i: int = Query(40, ge=0, lt=prediction.GRID_ROWS),
    j: int = Query(50, ge=0, lt=prediction.GRID_COLS),
):
    job, path = _require_prediction(job_id)
    prediction_array = np.load(path, mmap_mode="r", allow_pickle=False)
    response = {
        "i": i,
        "j": j,
        "pred": np.asarray(prediction_array[i, j, :]).tolist(),
        "truth": None,
        "truthName": None,
        "r": None,
    }

    meta = (job.get("result") or {}).get("verification") or {}
    truth_path = job.get("truthFile")
    if meta.get("available") and truth_path and Path(truth_path).is_file():
        truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
        if truth.shape != prediction_array.shape:
            raise HTTPException(
                409,
                f"实况场形状 {truth.shape} 与预测场 {prediction_array.shape} 不一致",
            )
        truth_series = np.asarray(truth[i, j, :], dtype=np.float64)
        response["truth"] = [
            None if not np.isfinite(value) else float(value)
            for value in truth_series
        ]
        response["truthName"] = meta.get("truthName")

        r_path = _pearson_path(path.parent)
        if r_path.is_file():
            r_value = float(np.load(r_path, mmap_mode="r", allow_pickle=False)[i, j])
            response["r"] = r_value if np.isfinite(r_value) else None

    return response


@router.get("/jobs/{job_id}/pearson")
def chain_pearson(job_id: str, confidence: float = Query(0.95)):
    """逐格点 r 图 + 该置信水平下的技巧摘要（点无定义处序列化为 null）。"""
    level = _confidence(confidence)
    job, path = _require_prediction(job_id)
    meta = (job.get("result") or {}).get("verification") or {}
    if not meta.get("available"):
        raise HTTPException(
            409, meta.get("reason") or "该任务没有可用的实况场对比结果"
        )
    r_path = _pearson_path(path.parent)
    if not r_path.is_file():
        raise HTTPException(404, "相关系数图产物不存在")
    r_map = np.load(r_path, allow_pickle=False)
    summary = verification.summarize(r_map, level)
    summary.update({
        "truthName": meta.get("truthName"),
        "downloadUrl": meta.get("downloadUrl"),
    })
    return {
        "rows": int(r_map.shape[0]),
        "cols": int(r_map.shape[1]),
        "min": float(np.nanmin(r_map)),
        "max": float(np.nanmax(r_map)),
        **summary,
        # NaN 不是合法 JSON（Starlette 用 allow_nan=False，直接传会 500），
        # 且前端要区分"无定义"与"r=0"，所以转成 null
        "data": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in r_map
        ],
    }


@router.get("/jobs/{job_id}/download/cio")
def download_cio(job_id: str):
    """投影CIO，`(1, T)` `.npy` —— `.zip` 入口 T=2352，可直接与官方件对拍。"""
    job, series = _require_series(job_id)
    payload = io.BytesIO()
    np.save(payload, np.reshape(np.asarray(series, dtype=np.float64), (1, -1)),
            allow_pickle=False)
    payload.seek(0)
    return _npy_response(payload, f"projected_cio_{FIXED_LEAD}_{FIXED_YEAR}.npy")


@router.get("/jobs/{job_id}/download/normalized")
def download_normalized(job_id: str):
    """归一化窗口 `(112,) float32`，与阶段③ 喂入模型的张量逐位一致。"""
    _job, selected = _require_normalized(job_id)
    payload = io.BytesIO()
    np.save(payload, np.asarray(selected, dtype=np.float32), allow_pickle=False)
    payload.seek(0)
    return _npy_response(
        payload, f"normalized_window_{FIXED_LEAD}_{FIXED_YEAR}.npy"
    )


def _npy_response(payload: io.BytesIO, filename: str) -> StreamingResponse:
    return StreamingResponse(
        payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/jobs/{job_id}/download/prediction")
def download_prediction(job_id: str):
    _job, path = _require_prediction(job_id)
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"prediction_{FIXED_LEAD}_{FIXED_YEAR}.npy",
    )


@router.get("/jobs/{job_id}/download/pearson")
def download_pearson(job_id: str):
    job, path = _require_prediction(job_id)
    r_path = _pearson_path(path.parent)
    if not r_path.is_file():
        raise HTTPException(404, "相关系数图产物不存在")
    return FileResponse(
        r_path,
        media_type="application/octet-stream",
        filename=f"pearson_r_{FIXED_LEAD}_{FIXED_YEAR}.npy",
    )
