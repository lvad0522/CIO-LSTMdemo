"""真实数据加载模块 —— 加载 CIO+LSTM 降水预测的真实输出（Dataset/ + 重训产出的 model/）。

数据源约定
==========
- 目录结构：`{DATA_ROOT}/pre{lead}/{year}/`，文件命名：
  `pre_{lead}_{year}({fold})_20_40_100_125_0.25_{type}2.npy`
  - lead: 提前期 1-20（pre1~pre20）
  - year: 2000-2019
  - fold: 1-6（LSTM 重复实验）；**pre7 重训后只有 1**，折数一律用 `_folds()` 实查。
    展示用哪几折由 `DISPLAY_MODE` 决定（foldmax=各折取最大 / fold1=只折 1）
  - type: pearson2 / predict2 / real2
- 区域：东亚 20-40°N, 100-125°E @0.25°，网格 81×101（GRID_ROWS×GRID_COLS）
- 时间：112 天（6-9 月每月 2-29 号）
- 数据根目录：默认 `../Dataset`（相对 backend 目录），可用环境变量 `DATA_ROOT` 覆盖
- 重训 lead 改指 `{MODEL_ROOT}/`（默认 `../model`，环境变量 `MODEL_ROOT` 覆盖），
  逐 lead 登记在 `LEAD_DATA_DIRS`（含是否平铺），见 `_year_dir()`

文件语义（源自 main_India_new.py，见 gen_guide.py:711-810）
- pearson2: 形状 (81,101,3) = [行索引, 列索引, Pearson r]（第 3 通道才是 r）
- predict2: 形状 (81,101,112) float32，LSTM 预测的带通滤波后降水（可含负值）
- real2:   形状 (81,101,112) float64，真实滤波后降水（带通滤波，58% 负值）
- pearson 中的 NaN 已被生成脚本替换为 0（0 可能是"无有效相关"而非真实 r=0）

已知数据情况（2026-08-25 实测；命名口径 2026-09-17 修订；pre7 2026-09-19 接新件）
- **旧 pre7 是坏模型**（输入是常数垃圾场，输出年不变、r 虚高）：论文口径跨 20 年平均
  0.1534，其余 19 个 lead 只有 0.1240~0.1287。2026-09-19 起改用重训批次——干净输入、
  20 年平均 +0.1274，落在健康区间。**新件在 `model/pre7_单折/`，且只有折 1**，由
  `_year_dir()` 接手；`Dataset/pre7/` 那 360 个旧件原封未动（不删不改，留作对照）。
  显示口径 2026-09-19 已拍板为**全站只用折 1**（见下"统计口径说明"）——单折的 pre7
  与 6 折的邻居只有在这个口径下才可比
- pre19/2002 的 6 个 real2 原缺失，已用 pre1/2002 同 fold 复制填补
  （2002 年 real2 跨 pre1~pre20 逐字节一致——真实值不依赖 lead；pearson2 随 lead 变化，勿同法处理）
  [2026-09-19 复核：填补件与 pre1/2002 同 fold 逐字节一致，且 mtime 秒级继承自 pre1/2002
   （copy2 保留时间戳），确认系本地填补操作，非原始产出]
- pre3/pre5 全部年份 + pre20 的 2010-2019 共 50 个目录里混了**两批** predict2：正常名的是
  "第一遍"，缺左括号的（如 pre_3_20001)_...，共 300 个）是"第二遍"。目录里的 pearson2
  **全部由第二遍算出**（恒等式复核 50/50 成立），故 predict2 优先取第二遍那份，见 _file_path。
  磁盘上一律不动——改名会永久覆盖第一遍，无法回退

source 标记约定
- dict 型字段（skillMap/timeSeries）及 list 元素（table/s2s）带 `"source"`：
  `"dataset"` = 真实数据，`"mock"` = 降级数据
- CIO 模块（cioTS/cioCorr）：mock 已于 2026-08-25 关停（伪造数据会误导用户），
  /api/run 返回 null；待上传 SST+Uwind 的真实计算接入后再恢复

统计口径说明（**两条分支的唯一差别：`DISPLAY_MODE` 常量**）
- 两个展示口径，两条分支的代码除该常量外完全一致（见 `_display_folds()`）：
  - `"foldmax"`（**main**，2026-08-28 起）：逐格点取各折最大值（最乐观折）
  - `"fold1"`（**feature/display-single-fold**）：只取折 1（一次训练）
- **为什么要有 fold1**：pre7 重训件只有折 1，"折最大"对它等于它自己，与邻居 6 折取最大
  不可比（2003 年 S2S 蓝框：pre7 0.3410 而邻居 ~0.50，看着像 pre7 最差）；统一折 1 后
  pre7 0.3410 落在邻居 0.32~0.38 区间内。代价：单折单格点噪声大、可为负
  （2000 年 pre6 @(40,50) = 0.0102），这是折 1 的真实水平
- **为什么 main 仍留 foldmax**：foldmax 不是任何一次训练的能力，实测把全网格平均放大
  约 2.2~3.7 倍（2000 年 0.15 → 0.36）。保留它只为对照，对外口径以 fold1 分支为准
- 两个口径下 **skillMap 与 timeSeries 的 "r" 都同源同口径**（同走 `_display_pearson()`），
  保证同格点颜色与数字一致
- foldmax 的 pred/real 曲线是**各折逐时间点平均**（"最大"对时间序列无意义）；
  fold1 是折 1 的原始序列。两者都走 `_display_series()`
- table 的 r/RMSE/MAE 为全网格全序列池化计算（predict2 vs real2），
  与论文"逐格点 r 区域平均"口径不同，仅作稳定性展示；foldmax 6 行 / fold1 1 行
- `_folds()` 保留：`verify_dataset` 仍按**实际存在的折**做完整性校验（数据层面的事，
  与展示口径无关）
"""

import os
import re
import threading
from collections import OrderedDict

import numpy as np

# ── 路径与常量 ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(_HERE, "..", "Dataset"))
# 模型侧产出根目录（2026-09-19 规矩：用 .pt 跑出来的东西放 model/，不往 Dataset/ 塞）
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(_HERE, "..", "model"))

# 数据目录逐 lead 解析：默认 Dataset/pre{lead}/{year}/；重训过的 lead 在此登记，
# 值为 (根目录, 是否平铺)。Dataset/ 按年份分子目录，model/ 下的重训产出平铺一层
# （年份已编码在文件名里）。加新条目即可，其余代码不用动。见 model/README.md 与
# 摸库交付_2026-09-15/04_文档/pre7单折与显示口径_给后端agent.md
LEAD_DATA_DIRS = {
    7: (os.path.join(MODEL_ROOT, "pre7_单折"), True),   # 09-16 用干净输入重训，仅折 1
}


def _year_dir(lead, year):
    """该 (lead, year) 的文件所在目录（Dataset/ 是 pre{lead}/{year}/，重训产出是平铺）。"""
    spec = LEAD_DATA_DIRS.get(lead)
    if spec is None:
        return os.path.join(DATA_ROOT, f"pre{lead}", str(year))
    root, flat = spec
    return root if flat else os.path.join(root, str(year))


FILE_PATTERN = re.compile(
    r"^pre_(\d+)_(\d{4})\((\d)\)_20_40_100_125_0\.25_(pearson2|predict2|real2)\.npy$"
)

GRID_ROWS = 81
GRID_COLS = 101
N_DAYS = 112
# 重复次数（"折"）不写死：多数 lead 是 6，pre7 重训后只有 1。一律用 _folds() 实查
LEADS = list(range(1, 21))
YEARS = list(range(2000, 2020))

# 数据完整性：2026-09-19 起期望 6900 个文件（= 19 个 lead × 20 年 × 6 折 × 3 + pre7 的
# 20 年 × 1 折 × 3）。pre19/2002 的 6 个 real2 曾缺失，已用 pre1/2002 同 fold 复制填补；
# pre7 换成 model/ 下的单折重训件，见模块 docstring

# ── 展示口径（**两条分支只差这一个常量**）────────────────
# "foldmax"：逐格点取各折最大值（最乐观折，2026-08-28 起的口径）—— main 分支
# "fold1"  ：只取折 1（一次训练）—— feature/display-single-fold 分支
# pre7 重训件只有折 1，只有 fold1 口径下它与邻居才可比。详见模块 docstring。
DISPLAY_MODE = "foldmax"


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
    """精确拼接文件路径（禁止 glob 通配，避免命中畸形文件）。

    predict2 特例：pre3/pre5 全部年份 + pre20 的 2010-2019 共 50 个目录，里面混有
    "第二遍"跑出的文件名少左括号的 predict2（如 `pre_3_20001)_...`）；该目录的
    pearson2 正是由这份算出的（恒等式复核 50/50 成立，见 01_探针/verify_identity_all400.py）。
    为保证热力图 r 与曲线同源，predict2 优先取这份，没有则回落正常名。
    pearson2 / real2 的名字是正常的，不受影响。
    """
    d = _year_dir(lead, year)
    if kind == "predict2":
        m = os.path.join(d, f"pre_{lead}_{year}{fold})_20_40_100_125_0.25_predict2.npy")
        if os.path.exists(m):
            return m
    return os.path.join(d, f"pre_{lead}_{year}({fold})_20_40_100_125_0.25_{kind}.npy")


def _folds(lead, year, kind="pearson2"):
    """该 (lead, year) 目录下**实际存在**的折号（升序）。pre7 重训后只有 [1]。

    只认正常名（带括号）：截断名 `pre_3_20001)_...` 不匹配 FILE_PATTERN，自然被排除，
    所以不需要额外正则。用 os.listdir + 正则而非 glob 通配。
    """
    d = _year_dir(lead, year)
    out = set()
    if os.path.isdir(d):
        for name in os.listdir(d):
            m = FILE_PATTERN.match(name)
            if m and int(m.group(1)) == lead and int(m.group(2)) == year \
                    and m.group(4) == kind:
                out.add(int(m.group(3)))
    return sorted(out)


def _display_folds(lead, year):
    """当前展示口径下要参与聚合的折号（`DISPLAY_MODE` 的**唯一**落点）。

    foldmax → 该目录实有的全部折（通常 6）
    fold1   → [1]
    缺折 1 时直接抛错，**不静默换折**——悄悄退到别的折就是悄悄换了口径，
    数字会变而没人知道（同 Dataset 口径修复单里"不许静默回退"的规矩）。
    """
    folds = _folds(lead, year, "pearson2")
    if not folds:
        raise DataNotFoundError(f"pre{lead}/{year} 无可用折")
    if DISPLAY_MODE == "fold1":
        if 1 not in folds:
            raise DataNotFoundError(f"pre{lead}/{year} 缺折 1，无法按 fold1 口径取数")
        return [1]
    return folds


def _display_pearson(lead, year):
    """按展示口径返回该 (lead, year) 的 pearson r 场，(81, 101)。

    热力图与曲线上的 "r" 都从这里取，保证同格点一致。
    fold1 时只有一折，`np.fmax.reduce` 退化为直接取它自己。
    """
    maps = [_load_npy(lead, year, f, "pearson2")[..., 2]
            for f in _display_folds(lead, year)]
    return np.fmax.reduce(maps)


def _display_series(lead, year, grid_i, grid_j):
    """按展示口径返回该格点的 (pred, real) 两条 112 天序列。

    fold1 时只有一折，`np.mean` 退化为直接取它自己（各元素完全相等，无精度损失）。
    """
    folds = _display_folds(lead, year)
    preds = [_load_npy(lead, year, f, "predict2")[grid_i, grid_j, :] for f in folds]
    reals = [_load_npy(lead, year, f, "real2")[grid_i, grid_j, :] for f in folds]
    return (np.mean(np.stack(preds, axis=0), axis=0),
            np.mean(np.stack(reals, axis=0), axis=0))


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
    """按**实际存在**的折逐一检查，返回缺失清单与意外文件统计。

    折号不写死：以该目录 pearson2 的折集为准（pre7 重训后只有 [1]），再查 predict2 /
    real2 是否按同一批折配套齐全。整目录查不到 pearson2 也记一笔，避免"没有折 ⇒ 无需
    检查"的空转（若只按实有折循环，missing 会恒为空，校验就废了）。

    返回: {"missing": list[str], "malformed": int, "total_expected": int}
    """
    missing = []
    total = 0
    for lead in LEADS:
        for year in YEARS:
            folds = _folds(lead, year, "pearson2")
            if not folds:
                missing.append(_year_dir(lead, year))
                continue
            for fold in folds:
                for kind in ("pearson2", "predict2", "real2"):
                    total += 1
                    if not os.path.exists(_file_path(lead, year, fold, kind)):
                        missing.append(_file_path(lead, year, fold, kind))

    # 意外文件统计：目录下存在但正则不匹配的 npy（畸形冗余等）。
    # 只扫 DATA_ROOT——model/ 是模型产出目录，不属于"数据集"范畴
    malformed = 0
    if os.path.isdir(DATA_ROOT):
        for root, _dirs, names in os.walk(DATA_ROOT):
            for name in names:
                if name.endswith(".npy") and not FILE_PATTERN.match(name):
                    malformed += 1
    return {
        "missing": missing,
        "malformed": malformed,
        "total_expected": total,
    }


# ── 真实数据 gen_* ────────────────────────────────────────
def gen_skill_map(year=2019, lead=6):
    """技能地图：该 lead 的 pearson r 逐格点图（口径见 `DISPLAY_MODE`）。

    foldmax：逐格点取各折最大（放大 2.2~3.7 倍，不是任何一次训练的能力）
    fold1  ：折 1 那一次的原始 r 图
    """
    _check_lead_year(lead, year)
    return {
        "data": _display_pearson(lead, year).tolist(),
        "rows": GRID_ROWS,
        "cols": GRID_COLS,
        "source": "dataset",
    }


def gen_time_series(year=2019, lead=6, grid_i=40, grid_j=50):
    """格点时序：该 lead 的 predict/real 序列 + 同格点 r（口径见 `DISPLAY_MODE`）。

    "r" 与 gen_skill_map 同一次取数，保证与热力图同格点颜色一致。
    注意：任何口径下前端都**禁止**自行对曲线重算 r（见 prediction.py 的
    截断/归一化语义），也**禁止**自行改口径。
    """
    _check_lead_year(lead, year)
    _check_grid(grid_i, grid_j)
    pred, real = _display_series(lead, year, grid_i, grid_j)
    r = _display_pearson(lead, year)[grid_i, grid_j]
    return {
        "real": real.tolist(),
        "pred": pred.tolist(),
        "r": round(float(r), 4),
        "source": "dataset",
    }


def gen_results_table(year=2019, lead=6):
    """实验统计表：各展示折的池化 r / RMSE / MAE（foldmax 6 行 / fold1 1 行）。"""
    _check_lead_year(lead, year)
    table = []
    for fold in _display_folds(lead, year):
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
    """重算 LSTM 20 个 lead 的蓝框区域平均 r（缺折的 lead 返回 None 断线）。

    口径与热力图一致（同走 `_display_pearson`）。foldmax 口径下邻居 ~0.50 而
    pre7（单折，取最大等于它自己）0.3410 显得最低；fold1 口径下 pre7 落在
    邻居 0.32~0.38 内 —— 这就是引入 fold1 的直接原因。
    """
    curve = []
    for lead in S2S_LEADS:
        try:
            r_map = _display_pearson(lead, S2S_YEAR)
        except DataNotFoundError:
            curve.append(None)
            continue
        curve.append(float(r_map[ROI_ROW_SLICE, ROI_COL_SLICE].mean()))
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
