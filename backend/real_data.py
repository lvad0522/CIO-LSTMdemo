"""真实数据加载模块 —— 从 Dataset/ 加载 CIO+LSTM 降水预测的真实输出。

数据源约定
==========
- 目录结构：`{DATA_ROOT}/pre{lead}/{year}/`，文件命名：
  `pre_{lead}_{year}({fold})_20_40_100_125_0.25_{type}2.npy`
  - lead: 提前期 1-20（pre1~pre20）
  - year: 2000-2019
  - fold: 1-6（LSTM 重复实验）
  - type: pearson2 / predict2 / real2
- 区域：东亚 20-40°N, 100-125°E @0.25°，网格 81×101（GRID_ROWS×GRID_COLS）
- 时间：112 天（6-9 月每月 2-29 号）
- 数据根目录：默认 `../Dataset`（相对 backend 目录），可用环境变量 `DATA_ROOT` 覆盖

文件语义（源自 main_India_new.py，见 gen_guide.py:711-810）
- pearson2: 形状 (81,101,3) = [行索引, 列索引, Pearson r]（第 3 通道才是 r）
- predict2: 形状 (81,101,112) float32，LSTM 预测的带通滤波后降水（可含负值）
- real2:   形状 (81,101,112) float64，真实滤波后降水（带通滤波，58% 负值）
- pearson 中的 NaN 已被生成脚本替换为 0（0 可能是"无有效相关"而非真实 r=0）

已知数据情况（2026-08-25 实测）
- pre19/2002 的 6 个 real2 原缺失，已用 pre1/2002 同 fold 复制填补
  （2002 年 real2 跨 pre1~pre20 逐字节一致——真实值不依赖 lead；pearson2 随 lead 变化，勿同法处理）
- pre3/pre5/pre20 存在 300 个畸形冗余 predict2 文件（缺左括号，如 pre_3_20001)_...），
  加载一律使用精确文件名，不参与任何加载逻辑

source 标记约定
- dict 型字段（skillMap/timeSeries）及 list 元素（table/s2s）带 `"source"`：
  `"dataset"` = 真实数据，`"mock"` = 降级数据
- CIO 模块（cioTS/cioCorr）：mock 已于 2026-08-25 关停（伪造数据会误导用户），
  /api/run 返回 null；待上传 SST+Uwind 的真实计算接入后再恢复

统计口径说明
- skillMap 的 r 为 6 个 fold 逐格点取最大值（最乐观 fold，2026-08-28 按需求调整，
  原为平均）；timeSeries 的 "r" 字段与热力图同口径（同取 fold 最大值），保证同格点一致
- timeSeries 的 pred/real 曲线仍为 6 个 fold 逐时间点平均（降水序列展示，非技能评分）
- table 的 r/RMSE/MAE 为全网格全序列池化计算（predict2 vs real2），
  与论文"逐格点 r 区域平均"口径不同，仅作稳定性展示
"""

import os
import re
import threading
from collections import OrderedDict

import numpy as np

# ── 路径与常量 ────────────────────────────────────────────
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Dataset"),
)

FILE_PATTERN = re.compile(
    r"^pre_(\d+)_(\d{4})\((\d)\)_20_40_100_125_0\.25_(pearson2|predict2|real2)\.npy$"
)

GRID_ROWS = 81
GRID_COLS = 101
N_DAYS = 112
N_FOLDS = 6
LEADS = list(range(1, 21))
YEARS = list(range(2000, 2020))

# 数据完整性：2026-08-25 起 7200 个文件全部齐备
# （pre19/2002 的 6 个 real2 曾缺失，已用 pre1/2002 同 fold 复制填补，见模块 docstring）


class ValidationError(ValueError):
    """参数校验错误，main.py 中捕获后转为 HTTP 400。"""


class DataNotFoundError(FileNotFoundError):
    """数据集文件缺失，main.py 中捕获后转为 HTTP 404。"""


# ── 有界 LRU 缓存（线程安全）──────────────────────────────
class _LRUCache:
    """容量受限的 LRU 缓存，线程安全。"""

    def __init__(self, capacity):
        self._capacity = capacity
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            return None

    def put(self, key, value):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)


# 容量按 (lead, year, fold) 组 × 3 类型计算：16 组 × 3 ≈ 48 个文件条目 ≈ 171MB
# （每组 pearson 0.19MB + predict 3.5MB + real 7.2MB ≈ 10.7MB）
# 一次 /api/run 需要 18 个文件条目，48 容量可容纳 2-3 组完整组合，避免重复请求抖动重读
_cache = _LRUCache(capacity=48)


# ── 加载原语 ──────────────────────────────────────────────
def _file_path(lead, year, fold, kind):
    """精确拼接文件路径（禁止 glob 通配，避免命中畸形文件）。"""
    return os.path.join(
        DATA_ROOT, f"pre{lead}", str(year),
        f"pre_{lead}_{year}({fold})_20_40_100_125_0.25_{kind}.npy",
    )


def _load_npy(lead, year, fold, kind):
    """加载单个 npy（精确文件名 + 缓存 + 缺失/损坏防御）。"""
    key = (lead, year, fold, kind)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    path = _file_path(lead, year, fold, kind)
    if not os.path.exists(path):
        raise DataNotFoundError(f"数据集文件缺失: {path}")
    try:
        arr = np.load(path)
    except Exception as exc:
        raise DataNotFoundError(f"数据集文件损坏: {path} ({exc})") from exc

    _cache.put(key, arr)
    return arr


def _check_lead_year(lead, year):
    """校验 lead/year 范围，非法抛 ValidationError。"""
    if lead not in LEADS:
        raise ValidationError(f"无效提前期: pre{lead}，可选 pre1~pre20")
    if year not in YEARS:
        raise ValidationError(f"无效年份: {year}，可选 2000-2019")


def _check_grid(i, j):
    """校验格点索引范围。"""
    if not (0 <= i < GRID_ROWS):
        raise ValidationError(f"格点行索引 i={i} 超出范围 (0-{GRID_ROWS - 1})")
    if not (0 <= j < GRID_COLS):
        raise ValidationError(f"格点列索引 j={j} 超出范围 (0-{GRID_COLS - 1})")


# ── 数据集完整性校验 ──────────────────────────────────────
def verify_dataset():
    """按 7200 个精确文件名逐一检查，返回缺失清单与意外文件统计。

    返回: {"missing": list[str], "malformed": int, "total_expected": int}
    """
    missing = []
    for lead in LEADS:
        for year in YEARS:
            for fold in range(1, N_FOLDS + 1):
                for kind in ("pearson2", "predict2", "real2"):
                    if not os.path.exists(_file_path(lead, year, fold, kind)):
                        missing.append(_file_path(lead, year, fold, kind))

    # 意外文件统计：目录下存在但正则不匹配的 npy（畸形冗余等）
    malformed = 0
    if os.path.isdir(DATA_ROOT):
        for root, _dirs, names in os.walk(DATA_ROOT):
            for name in names:
                if name.endswith(".npy") and not FILE_PATTERN.match(name):
                    malformed += 1
    return {
        "missing": missing,
        "malformed": malformed,
        "total_expected": len(LEADS) * len(YEARS) * N_FOLDS * 3,
    }


# ── 真实数据 gen_* ────────────────────────────────────────
def gen_skill_map(year=2019, lead=6):
    """技能地图：6 个 fold 的 pearson r 逐格点取最大值（最乐观 fold）。"""
    _check_lead_year(lead, year)
    r_maps = [_load_npy(lead, year, f, "pearson2")[..., 2] for f in range(1, N_FOLDS + 1)]
    r_max = np.max(np.stack(r_maps, axis=0), axis=0)
    return {
        "data": r_max.tolist(),
        "rows": GRID_ROWS,
        "cols": GRID_COLS,
        "source": "dataset",
    }


def gen_time_series(year=2019, lead=6, grid_i=40, grid_j=50):
    """格点时序：6 个 fold 的 predict/real 逐时间点平均。

    "r" 字段 = 该格点 6 个 fold 的 pearson2 相关最大值（与 gen_skill_map 同口径，
    最乐观 fold），前端时序图直接展示此值，保证与热力图同格点颜色一致。
    注意：max(r_fold) ≠ r(mean_pred, mean_real)（非线性算子，平均抑制噪声），
    因此前端禁止自行重算 r（见 prediction.py 截断/归一化语义）。
    """
    _check_lead_year(lead, year)
    _check_grid(grid_i, grid_j)
    preds = [_load_npy(lead, year, f, "predict2") for f in range(1, N_FOLDS + 1)]
    reals = [_load_npy(lead, year, f, "real2") for f in range(1, N_FOLDS + 1)]
    pred_mean = np.mean(np.stack([p[grid_i, grid_j, :] for p in preds], axis=0), axis=0)
    real_mean = np.mean(np.stack([r[grid_i, grid_j, :] for r in reals], axis=0), axis=0)
    fold_rs = [_load_npy(lead, year, f, "pearson2")[grid_i, grid_j, 2]
               for f in range(1, N_FOLDS + 1)]
    return {
        "real": real_mean.tolist(),
        "pred": pred_mean.tolist(),
        "r": round(float(np.max(fold_rs)), 4),
        "source": "dataset",
    }


def gen_results_table(year=2019, lead=6):
    """实验统计表：6 个 fold 各自的池化 r / RMSE / MAE。"""
    _check_lead_year(lead, year)
    table = []
    for fold in range(1, N_FOLDS + 1):
        pred = _load_npy(lead, year, fold, "predict2").flatten()
        real = _load_npy(lead, year, fold, "real2").flatten()
        table.append({
            "experiment": fold,
            "pearsonR": round(_pearson_r(real, pred), 3),
            "rmse": round(float(np.sqrt(np.mean((pred - real) ** 2))), 3),
            "mae": round(float(np.mean(np.abs(pred - real))), 3),
            "source": "dataset",
        })
    return table


def _pearson_r(a, b):
    """向量化的 Pearson 相关系数（常数序列返回 0）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ── S2S 模式对比（真实数据：corr-2003.nc + corr-avg-2003.nc + LSTM 重算）──
S2S_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "新绘图资料"
)
S2S_MODEL_ORDER = [
    "BOM", "CMA", "CNRM", "ECCC", "ECMWF",
    "HMCR", "ISAC", "JMA", "KMA", "NCEP", "UKMO",
]
# LSTM 蓝框口径（与 新绘图资料/lead_corr_2003.py 一致）：
# 112-121E, 23-27N @ 0.25° → 数组切片 行 12:29, 列 48:85
ROI_ROW_SLICE = slice(12, 29)
ROI_COL_SLICE = slice(48, 85)
S2S_LEADS = list(range(1, 21))
S2S_YEAR = 2003


def _load_s2s_curves():
    """读 corr-2003.nc(11,20) 与 corr-avg-2003.nc(20)，返回 (curves, s2s_mean)。"""
    import xarray as xr

    corr_file = os.path.join(S2S_DATA_DIR, "corr-2003.nc")
    avg_file = os.path.join(S2S_DATA_DIR, "corr-avg-2003.nc")
    missing = [f for f in (corr_file, avg_file) if not os.path.exists(f)]
    if missing:
        raise DataNotFoundError(f"S2S 对比数据缺失: {[os.path.basename(m) for m in missing]}")
    with xr.open_dataset(corr_file) as ds:
        corr_np = ds["corr"].values  # (11, 20)
    with xr.open_dataset(avg_file) as ds:
        s2s_mean = ds["corr"].values  # (20,)
    if corr_np.shape != (11, 20) or s2s_mean.shape != (20,):
        raise DataNotFoundError(f"S2S 对比数据维度异常: {corr_np.shape} / {s2s_mean.shape}")
    return {name: corr_np[i] for i, name in enumerate(S2S_MODEL_ORDER)}, s2s_mean


def _recompute_lstm_curve():
    """按锁定口径重算 LSTM 20 个 lead 的蓝框区域平均 r（缺失 lead 返回 None 断线）。"""
    curve = []
    for lead in S2S_LEADS:
        fold_rs = []
        for fold in range(1, N_FOLDS + 1):
            try:
                fold_rs.append(_load_npy(lead, S2S_YEAR, fold, "pearson2")[..., 2])
            except DataNotFoundError:
                continue
        if not fold_rs:
            curve.append(None)
            continue
        max_field = np.fmax.reduce(fold_rs)
        curve.append(float(max_field[ROI_ROW_SLICE, ROI_COL_SLICE].mean()))
    return curve


from mock_data import (  # noqa: E402
    gen_s2s_comparison as _mock_s2s,
)


def gen_s2s_comparison(lead="pre6", region="eastasia"):
    """S2S 模式对比 —— 真实数据（2003 年，lead 1-20）；读取失败降级 mock。

    返回 20 个行对象：{"lead": 1-20, "<11 模式名>": r, "S2S_Mean": r,
    "LSTM": r, "source": "dataset"}。
    """
    try:
        curves, s2s_mean = _load_s2s_curves()
        lstm_curve = _recompute_lstm_curve()
    except (DataNotFoundError, OSError, ImportError) as exc:
        print(f"[WARN] S2S 对比数据读取失败，降级 mock: {exc}")
        models = _mock_s2s(lead, region)
        for m in models:
            m["source"] = "mock"
        return models

    rows = []
    for i, lead_val in enumerate(S2S_LEADS):
        row = {"lead": lead_val}
        for name in S2S_MODEL_ORDER:
            row[name] = round(float(curves[name][i]), 4)
        row["S2S_Mean"] = round(float(s2s_mean[i]), 4)
        lstm_v = lstm_curve[i]
        row["LSTM"] = round(lstm_v, 4) if lstm_v is not None else None
        row["source"] = "dataset"
        rows.append(row)
    return rows
