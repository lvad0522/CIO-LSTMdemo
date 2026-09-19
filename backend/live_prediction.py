"""逐格点 LSTM checkpoint 在线推理。

两条输入路径，共用同一套后台推理与结果接口：

  ① `.npy`  —— 训练同口径的一维 projected CIO 序列（112 / 2240 / 官方 2352）
  ② `.zip`  —— 原始气象场（5–9 月逐日 nc，21 年 × 5 月），走 `cioproj` 现场投影

两条路径最终都落到 `_prepare_cio`，再顺序读取 pre1_2000.tar.gz 中的 8181
个格点模型，生成 (81, 101, 112) 的预测降水数组。

归一化口径（对 `src/src/prediction.py:54-88` 与 `main_India_new.py:17` 复现）：
  先把序列截/取到训练长度 2240，对整段做 min-max，再按年切 [112k : 112(k+1)]。
  checkpoint 不含 CIO 的 min/max，所以尺度一致性取决于输入序列本身与训练同源。

当前仅开放 year=2000、lead=pre1（已接入的权重只有这一组）；其余组合等权重到位
后自然放开。SST 分支缺原始场与预处理源码，按 `cioproj` 的约定直接报错。
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tarfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Callable

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

import cioproj
import mock_sandbox
import verification


GRID_ROWS = 81
GRID_COLS = 101
N_STEPS = 112

# 训练侧口径：src/src/main_India_new.py:17 把序列截到 2240 = 20 年 × 112 点
TRAIN_SEQ_LEN = N_STEPS * 20
EXPECTED_MODELS = GRID_ROWS * GRID_COLS

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_MODEL_ARCHIVE = (
    PROJECT_ROOT / "新端口-模型训练" / "pre1_2000.tar.gz"
)
MODEL_ARCHIVE = Path(os.environ.get("MODEL_ARCHIVE", DEFAULT_MODEL_ARCHIVE))
JOB_ROOT = BACKEND_DIR / "uploads" / "prediction_jobs"

# CIO 模态固定资产：投影 ∝ e 的后 3321 列。真件在服务器
# /mnt/mydisk2/zxy/data/rain/CIOmode_1982_2017.mat，需手工放入 assets/。
ASSETS_DIR = BACKEND_DIR / "assets"
DEFAULT_MAT = ASSETS_DIR / "CIOmode_1982_2017.mat"

# 服务器真件里 e 的形状（100 个模态 × [SST 2707 | U850 3321]）
MAT_SHAPE = (100, cioproj.N_SST + cioproj.N_U850)
# 真件是正交模态矩阵（e·eᵀ ≈ I，实测偏差 4.2e-15）；夹具是随机数（偏差 6.3e+03）
MAT_ORTHO_TOL = 1e-6

# zip 原始场入口的限额：夹具 105 个 nc 约 31 MB，真实 CESM 场留足余量
MAX_ZIP_FILES = 600
MAX_NC_TOTAL_BYTES = 4 * 1024 ** 3

# 官方命名 cesm.u850.anom.daily.YYYYMM02-28days.nc
NC_YEAR_RE = re.compile(r"\.(\d{4})(\d{2})\d{2}-28days")

LEAD_RE = re.compile(r"^pre(\d{1,2})$")

MODEL_NAME_RE = re.compile(
    r"(?:^|/)pre1/2000/(\d{2})_(\d{2,3})\.pt$"
)

router = APIRouter(prefix="/api/predict", tags=["live-prediction"])

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lstm-predict")


def _public_job(job: dict) -> dict:
    """仅返回可安全公开给前端的任务字段。"""
    fields = (
        "jobId", "status", "stage", "progress", "filename", "inputShape",
        "inputKind", "normalization", "projection", "warning",
        "createdAt", "startedAt", "finishedAt", "durationSeconds", "result",
        "error",
    )
    return {key: job.get(key) for key in fields if key in job}


def _get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "预测任务不存在或后端已重启")
        return dict(job)


def _update_job(job_id: str, **changes) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)


def _prepare_cio(
    raw: np.ndarray, pre_year: int = 0
) -> tuple[np.ndarray, dict]:
    """按训练侧口径准备第 `pre_year` 年的 112 天 CIO。

    复现 src/src/main_India_new.py:17 的 `cio[:, :2240]` 与
    src/src/prediction.py 的 `normalize` + `input[112k : 112(k+1)]`：
    先把序列截到训练长度 2240，对**整段**做 min-max，再按年切块。

    长度策略：
      T ≥ 2240  截前 2240，mode="training-equivalent"（官方 2352 点件走这条）
      112 ≤ T < 2240  整段自归一，mode="partial" + warning（缺月份的真实序列）
      T < 112   报错
    """
    arr = np.asarray(raw)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError("CIO 必须是一维数组，或 squeeze 后为一维数组")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("CIO 数组必须为数值类型")
    if not 0 <= pre_year < 20:
        raise ValueError("pre_year 必须在 0..19")
    if arr.size < N_STEPS:
        raise ValueError(
            f"CIO 序列只有 {arr.size} 点，不足一年（{N_STEPS} 点），无法预测"
        )

    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("CIO 数组包含 NaN 或无穷值")

    if arr.size >= TRAIN_SEQ_LEN:
        base, mode = arr[:TRAIN_SEQ_LEN], "training-equivalent"
    else:
        base, mode = arr, "partial"

    cio_min = float(np.min(base))
    cio_max = float(np.max(base))
    if abs(cio_max - cio_min) < 1e-12:
        raise ValueError("CIO 数组为常数，无法执行 min-max 归一化")

    normalized = (base - cio_min) / (cio_max - cio_min)
    start = N_STEPS * pre_year
    selected = normalized[start:start + N_STEPS].astype(np.float32, copy=False)

    meta = {
        "mode": mode,
        "inputLength": int(arr.size),
        "usedLength": int(base.size),
        "preYear": pre_year,
        "selectedRange": [start, start + N_STEPS],
        "cioMin": cio_min,
        "cioMax": cio_max,
        "warning": None,
    }
    if mode == "partial":
        meta["warning"] = (
            f"序列只有 {arr.size} 点（< 训练长度 {TRAIN_SEQ_LEN}，通常因为缺月份），"
            "按现有长度整段 min-max；尺度与训练不完全一致。"
        )
    return selected, meta


# ------------------------------------------------------- CIOmode .mat 资产

def _mat_path() -> Path:
    """每次调用时解析，便于测试用 CIOPROJ_MAT 指向沙盒里的夹具件。"""
    override = os.environ.get("CIOPROJ_MAT")
    return Path(override) if override else DEFAULT_MAT


def _mat_meta() -> dict:
    """校验 CIO 模态资产可用，返回来源描述。**位置 + 内容**双判据。

    位置：路径是否落在被 `.MOCK_SANDBOX` 标记的模拟数据沙盒里（见 mock_sandbox）。
      - 在沙盒里 → 必须是显式放行（`CIOPROJ_ALLOW_MOCK=1`）才可用，标 mock。
      - 不在沙盒里 → 一律按"必须是真的"处理，**开开关也不放行**。

    内容（沙盒外的件）：真件是正交模态矩阵，`e·eᵀ` 精确等于单位阵
      （2026-09-15 对服务器真件实测偏差 4.2e-15）；夹具件 e 是随机数，
      行范数 ~6e3 且互不正交（偏差 6.3e+03）。阈值 1e-6 隔了 9 个数量级。
      注意**不要**再除以 e.shape[1]——那会把真件的 I 变成 I/6028，反而误判。

    两条都必须"判不出来就报错"，绝不静默出结果。
    """
    path = _mat_path()
    if not path.is_file():
        raise ValueError(
            f"缺少 CIO 模态资产 {path} —— 原始场投影必需。请把服务器上的 "
            "/mnt/mydisk2/zxy/data/rain/CIOmode_1982_2017.mat 放到 "
            "backend/assets/ 下，或用环境变量 CIOPROJ_MAT 指定路径。"
        )
    try:
        import scipy.io as sio

        mat = sio.loadmat(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"读不了 CIO 模态资产 {path}: {exc!r}") from exc

    if "e" not in mat:
        raise ValueError(f"CIO 模态资产 {path} 里没有变量 e")
    e = np.asarray(mat["e"], dtype=np.float64)
    if e.ndim != 2 or e.shape != MAT_SHAPE:
        raise ValueError(f"CIO 模态 e 形状 {e.shape}，期望 {MAT_SHAPE}")

    gram = e @ e.T
    deviation = float(np.max(np.abs(gram - np.eye(e.shape[0]))))

    sandbox = mock_sandbox.find_root(path)
    if sandbox is not None:
        if not mock_sandbox.allow_flag_set():
            raise ValueError(
                f"CIO 模态资产 {path} 在模拟数据沙盒 {sandbox} 里 —— 那是 "
                "make_mock.py 造的假数据，不对应任何真实观测。"
                "正式跑请用服务器真件（backend/assets/ 或 摸库交付_2026-09-15/数据/）；"
                f"确实要用它做联调，显式设 {mock_sandbox.ALLOW_ENV}=1。"
            )
        return {
            "matSource": "mock",
            "matPath": str(path),
            "sandboxRoot": str(sandbox),
            "orthoDeviation": deviation,
        }

    # 沙盒外：必须是真的。ALLOW_MOCK 在这里不构成放行。
    if deviation > MAT_ORTHO_TOL:
        raise ValueError(
            f"CIO 模态资产 {path} 不是服务器真件（e·eᵀ 偏离单位阵 {deviation:.3g} "
            f"> {MAT_ORTHO_TOL:g}）。它也不在模拟数据沙盒里，所以 "
            f"{mock_sandbox.ALLOW_ENV} 对它无效 —— 沙盒外的资产必须是真的。"
            "假数据请放进带 .MOCK_SANDBOX 标记的目录。"
        )
    return {
        "matSource": "real",
        "matPath": str(path),
        "orthoDeviation": deviation,
    }


# ------------------------------------------------------- 原始场 zip 入口

def _resolve_kind(which: str, filename: str) -> tuple[str, str]:
    """交接文档 §9.2：显式参数 > 文件名关键字；判不出就报错，绝不默认 U850。"""
    which = (which or "").strip().lower()
    if which in ("", "auto"):
        return cioproj.detect_kind(filename), "filename"
    if which not in ("u850", "sst", "both"):
        raise ValueError(f"which 只能是 u850 / sst / both / auto，收到 {which!r}")
    try:
        guessed = cioproj.detect_kind(filename)
    except ValueError:
        return which, "explicit-param"
    if guessed != which:
        raise ValueError(
            f"显式指定 which={which}，但文件名 {filename!r} 按关键字判为 {guessed}，"
            "二者冲突；请改用 which=auto，或改文件名以消除歧义。"
        )
    return which, "explicit-param"


def _zip_years(zip_path: Path) -> list[int]:
    """从 zip 成员名推年份（不解压），用于同步校验与投影年份列表。"""
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            PurePosixPath(m.filename.replace("\\", "/")).name
            for m in zf.infolist()
            if not m.is_dir()
        ]
    years = sorted(
        {int(m.group(1)) for n in names if (m := NC_YEAR_RE.search(n))}
    )
    if not years:
        raise ValueError(
            "zip 里的 nc 文件名认不出年份（期望 cesm.*.YYYYMM02-28days.nc）"
        )
    return years


def _safe_extract_nc(zip_path: Path, dest: Path) -> int:
    """把 zip 内的 .nc 扁平解压到 dest；拒绝穿越路径，限制文件数与总体积。"""
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.infolist()
            if not m.is_dir() and m.filename.lower().endswith(".nc")
        ]
        if not members:
            raise ValueError("zip 里没有 .nc 文件（应打包 5–9 月的逐日原始场）")
        if len(members) > MAX_ZIP_FILES:
            raise ValueError(
                f"zip 内 .nc 文件 {len(members)} 个，超过上限 {MAX_ZIP_FILES}"
            )
        for member in members:
            raw_name = member.filename.replace("\\", "/")
            posix = PurePosixPath(raw_name)
            if (
                posix.is_absolute()
                or ".." in posix.parts
                or re.match(r"^[A-Za-z]:", raw_name)
            ):
                raise ValueError(f"zip 内含不安全路径：{member.filename!r}")

            total += member.file_size
            if total > MAX_NC_TOTAL_BYTES:
                raise ValueError(
                    "zip 解压后体积超过 %.1f GB 上限"
                    % (MAX_NC_TOTAL_BYTES / 1024 ** 3)
                )

            target = dest / posix.name
            if target.exists():
                raise ValueError(f"zip 内有重名文件：{posix.name!r}")
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
            count += 1
    return count


def _locate_nc_dir(work_dir: Path) -> Path:
    """find_nc_files 用非递归 glob，所以 nc 要么在顶层，要么在唯一子目录里。"""
    if any(work_dir.glob("*.nc")):
        return work_dir
    subs = [
        d for d in sorted(work_dir.iterdir())
        if d.is_dir() and any(d.glob("*.nc"))
    ]
    if len(subs) == 1:
        return subs[0]
    if not subs:
        raise ValueError("解压后找不到 .nc 文件（zip 里应打包 5–9 月逐日原始场）")
    raise ValueError(
        "解压后多个子目录都含 .nc，无法确定用哪个：%s" % [d.name for d in subs]
    )


def _zip_job_input(
    content: bytes,
    job_dir: Path,
    lead: int,
    which: str,
    filename: str,
    year: int,
) -> tuple[dict, Callable[[Callable[[int, str], None]], tuple]]:
    """同步校验 zip 入口（失败 → 调用方转 HTTP 400），返回 (投影元信息, prepare)。"""
    zip_path = job_dir / "raw_fields.zip"
    zip_path.write_bytes(content)
    work_dir = job_dir / "raw"

    kind, kind_source = _resolve_kind(which, filename)
    if kind in ("sst", "both"):
        cioproj.compute_cioproj(kind, name=filename)  # 直接抛 NotImplementedError
    mat_meta = _mat_meta()

    years = _zip_years(zip_path)
    if year not in years:
        raise ValueError(
            f"zip 里没有 {year} 年的 nc（现有年份 {years[0]}–{years[-1]}）；"
            f"模型 pre1/{year} 需要该年作为预测目标年。"
        )

    def prepare(report):
        report(3, "正在解压原始气象场")
        extracted = _safe_extract_nc(zip_path, work_dir)
        nc_root = _locate_nc_dir(work_dir)
        report(
            8,
            f"正在投影 CIO（{len(years)} 年 × 5 月，{extracted} 个 nc）",
        )
        series, proj = cioproj.compute_cioproj(
            which=kind,
            nc_dir=str(nc_root),
            years=years,
            lead=lead,
            mat_path=str(_mat_path()),
            name=filename,
        )
        report(26, "正在按训练口径归一化")
        cio, normalization = _prepare_cio(series, pre_year=0)

        projection = dict(mat_meta)
        projection.update({
            "inputKind": "raw-zip",
            "requestedWhich": which or "auto",
            "kind": proj.get("which", kind),
            "kindSource": kind_source,
            "lead": lead,
            "window": proj.get("window"),
            "years": proj.get("years", years),
            "seriesLength": proj.get("length"),
            "daysPerYear": proj.get("days_per_year"),
            "missingMonths": proj.get("missing_months"),
            "warning": proj.get("warning"),
        })
        return cio, normalization, projection

    return {
        "inputKind": "raw-zip",
        "requestedWhich": which or "auto",
        "kind": kind,
        "kindSource": kind_source,
        "lead": lead,
        "window": cioproj.window(lead),
        "years": years,
        "matSource": mat_meta["matSource"],
    }, prepare


def _npy_job_input(content: bytes, job_dir: Path) -> tuple[dict, Callable]:
    """同步校验 .npy 入口，返回 (归一化摘要, prepare)。"""
    try:
        raw = np.load(io.BytesIO(content), allow_pickle=False)
        _cio, normalization = _prepare_cio(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"CIO 文件校验失败: {exc}") from exc

    input_path = job_dir / "cio_input.npy"
    np.save(input_path, np.asarray(raw), allow_pickle=False)

    def prepare(report):
        report(5, "正在读取并归一化 CIO 序列")
        raw_now = np.load(input_path, allow_pickle=False)
        cio, norm = _prepare_cio(raw_now, pre_year=normalization["preYear"])
        return cio, norm, {"inputKind": "cio-npy"}

    return normalization, prepare


def _build_model(torch, checkpoint: dict):
    """依据 checkpoint 元数据构造与 src/src/model.py 一致的网络。"""
    nn = torch.nn
    input_size = int(checkpoint.get("input_size", 1))
    hidden_size = int(checkpoint.get("hidden_size", 60))
    num_layers = int(checkpoint.get("num_layers", 2))
    dropout = float(checkpoint.get("dropout", 0.2))

    class RNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                num_layers=num_layers,
            )
            self.linear1 = nn.Linear(hidden_size, 30)
            self.relu1 = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(30, 15)
            self.relu2 = nn.ReLU(inplace=True)
            self.linear3 = nn.Linear(15, 1)

        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.relu1(self.linear1(x))
            x = self.relu2(self.linear2(x))
            return self.linear3(x)

    return RNN()


def _run_archive_inference(
    cio: np.ndarray,
    progress_callback=None,
    span: tuple[int, int] = (3, 95),
) -> np.ndarray:
    """流式读取 gzip tar，逐格点执行前向推理。

    `span` 是进度条区间：投影阶段占掉前面一段，推理从 span[0] 推到 span[1]。
    """
    if not MODEL_ARCHIVE.is_file():
        raise FileNotFoundError(f"模型压缩包不存在: {MODEL_ARCHIVE}")

    # 当前 Windows/Anaconda 环境的 NumPy 与 PyTorch 可能各自携带 OpenMP。
    # 这是本地验证兼容设置；后续部署应统一运行时依赖后移除。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("后端未安装 PyTorch，请执行 pip install torch") from exc

    try:
        from numpy._core.multiarray import scalar as numpy_scalar
    except ImportError:  # NumPy < 2
        from numpy.core.multiarray import scalar as numpy_scalar

    safe_types = [
        numpy_scalar,
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("float32")),
    ]

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    input_tensor = torch.from_numpy(cio).reshape(1, -1, 1)
    prediction = np.full(
        (GRID_ROWS, GRID_COLS, cio.size), np.nan, dtype=np.float32
    )
    seen: set[tuple[int, int]] = set()
    processed = 0

    if progress_callback:
        progress_callback(span[0], "正在打开模型压缩包")

    with torch.serialization.safe_globals(safe_types):
        with tarfile.open(MODEL_ARCHIVE, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".pt"):
                    continue
                match = MODEL_NAME_RE.search(member.name.replace("\\", "/"))
                if not match:
                    continue

                i, j = int(match.group(1)), int(match.group(2))
                if not (0 <= i < GRID_ROWS and 0 <= j < GRID_COLS):
                    raise RuntimeError(f"模型坐标越界: {member.name}")
                if (i, j) in seen:
                    raise RuntimeError(f"模型坐标重复: ({i}, {j})")

                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"无法读取模型: {member.name}")
                checkpoint = torch.load(
                    io.BytesIO(stream.read()),
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                    raise RuntimeError(f"checkpoint 结构不正确: {member.name}")

                model = _build_model(torch, checkpoint)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval()
                with torch.inference_mode():
                    normalized_pred = model(input_tensor).reshape(-1).numpy()

                # 与旧 prediction.py 一致：NaN/负的归一化输出先置 0，再反归一化。
                normalized_pred = np.nan_to_num(
                    normalized_pred, nan=0.0, posinf=0.0, neginf=0.0
                )
                normalized_pred[normalized_pred < 0] = 0
                precip_min = float(checkpoint["precip_min"])
                precip_max = float(checkpoint["precip_max"])
                prediction[i, j, :] = (
                    normalized_pred * (precip_max - precip_min) + precip_min
                ).astype(np.float32, copy=False)

                seen.add((i, j))
                processed += 1
                if progress_callback and (
                    processed == 1 or processed % 25 == 0 or processed == EXPECTED_MODELS
                ):
                    lo, hi = span
                    progress = lo + int(processed / EXPECTED_MODELS * (hi - lo))
                    progress_callback(
                        progress,
                        f"正在推理格点 {processed}/{EXPECTED_MODELS}",
                    )

                del model, checkpoint, normalized_pred

    if processed != EXPECTED_MODELS:
        missing = EXPECTED_MODELS - processed
        raise RuntimeError(
            f"模型数量不完整：读取 {processed}/{EXPECTED_MODELS}，缺少 {missing} 个格点"
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("预测结果包含未填充格点或非有限值")
    return prediction


def _pearson_path(job_dir: Path) -> Path:
    return job_dir / "pearson_r.npy"


def _verify_against_truth(
    prediction: np.ndarray, job_dir: Path, job_id: str, context: dict
) -> dict:
    """对实况场算逐格点 r 图并落盘。**尽力而为：检验失败不让整个预测任务失败。**

    返回的 dict 直接进 `result.verification`。成功时含摘要与下载路径；
    无实况或失败时含 `available: False` + `reason` —— 前端据此显式说明，
    而不是悄悄少一块。预测本身才是主产物，检验是附加信息。
    """
    lead = context.get("lead")
    year = context.get("year")
    try:
        truth = verification.find_truth(lead, year)
    except ValueError as exc:  # 多份实况内容不一致，有歧义
        return {"available": False, "reason": str(exc)}
    if truth is None:
        return {
            "available": False,
            "reason": (
                f"没有 {lead}/{year} 的实况场（Dataset/{lead}/{year}/ 下无 "
                "*_real2.npy），本次只输出预测场，未做技巧检验。"
            ),
        }

    try:
        real = np.asarray(np.load(truth, allow_pickle=False), dtype=np.float64)
        r_map = verification.pearson_map(prediction, real)
        summary = verification.summarize(r_map, 0.95)  # 默认 95%，前端可改
    except (ValueError, OSError) as exc:
        return {"available": False, "reason": f"实况场对比失败: {exc}"}

    np.save(_pearson_path(job_dir), r_map.astype(np.float32), allow_pickle=False)
    summary.update({
        "available": True,
        "truthPath": str(truth),
        "truthName": truth.name,
        "downloadUrl": f"/api/predict/jobs/{job_id}/download/pearson",
    })
    return summary


def _run_job(
    job_id: str, prepare: Callable, result_path: Path, context: dict
) -> None:
    """后台执行：`prepare` 负责（解压 +）投影 + 归一化，随后做全网格推理。

    `context` 携带 `lead` / `year`，推理完成后用它去定位实况场做技巧检验。
    """
    started = time.time()
    _update_job(
        job_id,
        status="running",
        stage="正在校验输入",
        progress=1,
        startedAt=started,
    )
    try:
        def report(progress: int, stage: str) -> None:
            _update_job(job_id, progress=progress, stage=stage)

        cio, normalization, projection = prepare(report)
        projection = projection or {}
        warnings = [
            w for w in (normalization.get("warning"), projection.get("warning")) if w
        ]
        _update_job(
            job_id,
            normalization=normalization,
            projection=projection,
            warning=" ".join(warnings) if warnings else None,
            inputShape=[normalization["inputLength"]],
        )

        # 进度条：投影/归一化占 1–30%，推理占 30–97%，落盘到 100%
        prediction = _run_archive_inference(cio, report, span=(30, 97))
        report(97, "正在保存 prediction.npy")
        np.save(result_path, prediction, allow_pickle=False)

        report(98, "正在对比实况场计算相关系数")
        verification_meta = _verify_against_truth(
            prediction, result_path.parent, job_id, context
        )

        finished = time.time()
        result = {
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
            "min": float(np.min(prediction)),
            "max": float(np.max(prediction)),
            "mean": float(np.mean(prediction)),
            "downloadUrl": f"/api/predict/jobs/{job_id}/download",
            "verification": verification_meta,
        }
        _update_job(
            job_id,
            status="completed",
            stage="预测完成",
            progress=100,
            finishedAt=finished,
            durationSeconds=round(finished - started, 2),
            result=result,
        )
    except Exception as exc:  # 后台线程必须把错误保存给前端
        finished = time.time()
        _update_job(
            job_id,
            status="failed",
            stage="预测失败",
            finishedAt=finished,
            durationSeconds=round(finished - started, 2),
            error=f"{type(exc).__name__}: {exc}",
        )


@router.post("/jobs", status_code=202)
async def create_prediction_job(
    file: UploadFile = File(...),
    year: int = Query(2000),
    lead: str = Query("pre1"),
    which: str = Query(""),
):
    """上传 CIO `.npy` 或原始气象场 `.zip`，创建后台预测任务。

    `.npy`：训练同口径的一维 projected CIO 序列。
    `.zip`：5–9 月逐日原始场，走 `cioproj` 现场投影。`which` 为空时按文件名判类，
            判不出即报错（绝不默认按 U850 算）。
    """
    if year != 2000 or lead != "pre1":
        raise HTTPException(400, "当前模型验证仅支持 year=2000、lead=pre1")
    lead_index = int(LEAD_RE.match(lead).group(1))

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
    result_path = job_dir / "prediction.npy"

    normalization = None
    try:
        if suffix == ".npy":
            normalization, prepare = _npy_job_input(content, job_dir)
            projection = {"inputKind": "cio-npy"}
        else:
            if not zipfile.is_zipfile(io.BytesIO(content)):
                raise ValueError("上传的 .zip 无法解析（不是有效的 zip 文件）")
            projection, prepare = _zip_job_input(
                content, job_dir, lead_index, which, name, year
            )
    except (ValueError, NotImplementedError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc

    job = {
        "jobId": job_id,
        "status": "queued",
        "stage": "等待后台推理",
        "progress": 0,
        "filename": name,
        "inputKind": projection["inputKind"],
        "normalization": normalization,
        "projection": projection,
        "createdAt": time.time(),
        "resultPath": str(result_path),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _executor.submit(
        _run_job, job_id, prepare, result_path, {"lead": lead, "year": year}
    )
    return _public_job(job)


@router.get("/jobs/{job_id}")
def prediction_job_status(job_id: str):
    return _public_job(_get_job(job_id))


@router.get("/jobs/{job_id}/preview")
def prediction_preview(
    job_id: str,
    time_index: int = Query(0, ge=0, lt=N_STEPS),
):
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "预测尚未完成")
    result_path = Path(job["resultPath"])
    if not result_path.is_file():
        raise HTTPException(404, "预测结果文件不存在")
    prediction = np.load(result_path, mmap_mode="r", allow_pickle=False)
    frame = np.asarray(prediction[:, :, time_index])
    return {
        "timeIndex": time_index,
        "rows": GRID_ROWS,
        "cols": GRID_COLS,
        "min": float(np.min(frame)),
        "max": float(np.max(frame)),
        "data": frame.tolist(),
    }


@router.get("/jobs/{job_id}/grid")
def prediction_grid_series(
    job_id: str,
    i: int = Query(40, ge=0, lt=GRID_ROWS),
    j: int = Query(50, ge=0, lt=GRID_COLS),
):
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "预测尚未完成")
    result_path = Path(job["resultPath"])
    if not result_path.is_file():
        raise HTTPException(404, "预测结果文件不存在")
    prediction = np.load(result_path, mmap_mode="r", allow_pickle=False)
    return {
        "i": i,
        "j": j,
        "pred": np.asarray(prediction[i, j, :]).tolist(),
    }


@router.get("/jobs/{job_id}/pearson")
def prediction_pearson(
    job_id: str,
    confidence: float = Query(0.95),
):
    """逐格点 Pearson r 图 + 该置信水平下的技巧摘要。

    `confidence` 直接对应前端「CIO参数设置 → 显著性置信水平」，只影响**阈值**
    （哪些格点算显著），不改变 r 本身 —— r 与置信水平无关，重算阈值是纯后处理。
    """
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "预测尚未完成")
    meta = (job.get("result") or {}).get("verification") or {}
    if not meta.get("available"):
        raise HTTPException(
            409, meta.get("reason") or "该任务没有可用的实况场对比结果"
        )

    try:
        level = verification.normalize_confidence(confidence)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    r_path = _pearson_path(Path(job["resultPath"]).parent)
    if not r_path.is_file():
        raise HTTPException(404, "相关系数图文件不存在")
    r_map = np.load(r_path, allow_pickle=False)

    summary = verification.summarize(r_map, level)
    summary.update({
        "truthName": meta.get("truthName"),
        "downloadUrl": meta.get("downloadUrl"),
    })
    rows = r_map.shape[0]
    cols = r_map.shape[1]
    return {
        "rows": rows,
        "cols": cols,
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


@router.get("/jobs/{job_id}/truth/preview")
def prediction_truth_preview(
    job_id: str,
    time_index: int = Query(0, ge=0, lt=N_STEPS),
):
    """实况场的某一时间步，用来和预测场并排看。"""
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "预测尚未完成")
    meta = (job.get("result") or {}).get("verification") or {}
    truth_path = meta.get("truthPath")
    if not truth_path or not Path(truth_path).is_file():
        raise HTTPException(409, meta.get("reason") or "该任务没有可用的实况场")

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


@router.get("/jobs/{job_id}/download/pearson")
def download_pearson(job_id: str):
    """下载逐格点 r 图（float32, (81, 101)）；无定义格点为 NaN。"""
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "预测尚未完成")
    r_path = _pearson_path(Path(job["resultPath"]).parent)
    if not r_path.is_file():
        raise HTTPException(404, "相关系数图文件不存在")
    return FileResponse(
        r_path,
        media_type="application/octet-stream",
        filename="pearson_r_pre1_2000.npy",
    )


@router.get("/jobs/{job_id}/download")
def download_prediction(job_id: str):
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "预测尚未完成")
    result_path = Path(job["resultPath"])
    if not result_path.is_file():
        raise HTTPException(404, "预测结果文件不存在")
    return FileResponse(
        result_path,
        media_type="application/octet-stream",
        filename="prediction_pre1_2000.npy",
    )
