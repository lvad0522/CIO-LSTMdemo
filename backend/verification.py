# -*- coding: utf-8 -*-
"""预测技巧检验：在线推理结果对比 **TRMM 实况场**，算逐格点 Pearson r。

## 为什么是逐格点 r，而不是一个全局数

实况 `real2` 与预测同为 `(81, 101, 112)` = (纬度, 经度, 时间步)。降水距平场
在空间上高度不均匀，把它整个展平算一个相关系数会把"哪里报得准、哪里报不准"
抹平，还会被大值区主导。本仓库与实验室口径一致，都取**逐格点沿时间维**的 r：

    r[i, j] = corr( pred[i, j, :], real[i, j, :] )      # 每个格点一条 112 点序列

n = 112 天，逐格点一张 (81, 101) 的 r 图。实验室的 `*_pearson2.npy` 就是这个
东西（第 3 通道），所以本模块的输出可以直接和它对。

## 实况场的歧义必须报错，不能随便挑一个

`Dataset/pre1/2000/` 下有 **6 个** `real2`（每折一份）。实测 6 份逐字节相同，
所以真值无歧义；但**这个前提要每次现验**，不能写死：
万一将来某折的实况被换成不同的口径，静默挑第一个会得到一个说不清来历的 r。

因此 `find_truth()` 读全部候选，逐字节比对，不一致就报错。判不出来就报错，
绝不静默出结果 —— 与 `cioproj.detect_kind` / `live_prediction._mat_meta` 同一条规矩。

## 恒定点：0/0 是真无定义，如实记 NaN

个别格点的预测在这 112 天里**输出完全恒定**（实测 pre1/2000 有 9 个），
时间维方差为 0 ⇒ r = 0/0。这不是数值噪声，是模型在该格点退化。
本模块如实记为 NaN，并在 `summary` 里单独计数（`undefinedPoints`），
**不计入**均值/占比分母 —— 把 NaN 当成 0 会把技巧整体拉低，当成 1 会拉高。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

N_STEPS = 112
GRID_ROWS = 81
GRID_COLS = 101

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)


def _truth_roots() -> list[Path]:
    """实况场搜索根。`TRUTH_DIR` 可整体覆盖（测试用）。"""
    override = os.environ.get("TRUTH_DIR")
    if override:
        return [Path(override)]
    return [
        PROJECT_ROOT / "Dataset",
        PROJECT_ROOT.parent / "Dataset",
    ]


def truth_candidates(lead: str, year: int) -> list[Path]:
    """列出 `{root}/{lead}/{year}/*_real2.npy`。"""
    found: list[Path] = []
    for root in _truth_roots():
        day = root / lead / str(year)
        if day.is_dir():
            found.extend(sorted(day.glob("*_real2.npy")))
    return found


def find_truth(lead: str, year: int) -> Path | None:
    """定位实况场，**并验证候选之间逐字节一致**。

    没有任何候选 → 返回 None（调用方按"无实况、跳过检验"处理）。
    有多个但内容不一致 → 抛 ValueError（口径有歧义，不敢挑）。
    """
    paths = truth_candidates(lead, year)
    if not paths:
        return None

    first = paths[0]
    reference = np.load(first, allow_pickle=False)
    for other in paths[1:]:
        candidate = np.load(other, allow_pickle=False)
        if candidate.shape != reference.shape or not np.array_equal(
            candidate, reference
        ):
            raise ValueError(
                f"{lead}/{year} 有 {len(paths)} 份实况场，但内容不一致"
                f"（{first.name} vs {other.name}）。实况口径有歧义，"
                "请先确认哪一份是对的，再重跑检验。"
            )
    return first


def pearson_map(pred: np.ndarray, real: np.ndarray) -> np.ndarray:
    """逐格点沿时间维的 Pearson r，返回 (81, 101) float64，无定义处为 NaN。

    无定义 = 任一序列在时间维上方差为 0（0/0），或含非有限值。
    """
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if pred.shape != real.shape:
        raise ValueError(f"预测场形状 {pred.shape} 与实况场 {real.shape} 不一致")
    if pred.ndim != 3:
        raise ValueError(f"预测场应为 (纬度, 经度, 时间) 三维，收到 {pred.shape}")

    finite = np.isfinite(pred).all(axis=2) & np.isfinite(real).all(axis=2)
    a = pred - pred.mean(axis=2, keepdims=True)
    b = real - real.mean(axis=2, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=2) * (b * b).sum(axis=2))

    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(axis=2) / denominator
    r[~finite | ~(denominator > 0)] = np.nan
    return np.clip(r, -1.0, 1.0)


def critical_r(n: int, confidence: float) -> float:
    """双尾 t 检验下 |r| 的显著性阈值（自由度 n-2）。

    n=112 时：0.90 → 0.1562，0.95 → 0.1857，0.99 → 0.2425。
    """
    if n < 3:
        raise ValueError(f"样本量 {n} 太小，无法做相关性检验")
    from scipy import stats

    t = float(stats.t.ppf(0.5 + confidence / 2.0, n - 2))
    return t / np.sqrt(n - 2 + t * t)


def normalize_confidence(value) -> float:
    """把前端的 '0.95' / 0.95 归一化成 float，并限制在支持的档位。"""
    try:
        level = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"置信水平 {value!r} 不是数字") from exc
    if not 0.5 < level < 1.0:
        raise ValueError(f"置信水平 {level} 必须在 (0.5, 1) 之间")
    return level


def summarize(r_map: np.ndarray, confidence: float) -> dict:
    """r 图的摘要统计。只统计有定义的格点。

    显著性按**双尾** t 检验判 |r| ≥ 阈值，但拆成正/负两侧分别报：
    降水技巧图上「显著正相关」才意味着该格点有技巧，「显著负相关」是反相关
    （比气候态还差）。只报一个合计数会把这两件事混成一件。
    """
    defined = np.isfinite(r_map)
    values = r_map[defined]
    total = int(r_map.size)
    if values.size == 0:
        raise ValueError("r 图所有格点都无定义，无法给出技巧摘要")

    threshold = critical_r(N_STEPS, confidence)
    significant = np.abs(values) >= threshold
    denominator = values.size
    return {
        "n": N_STEPS,
        "confidence": confidence,
        "criticalR": round(threshold, 4),
        "meanR": round(float(values.mean()), 4),
        "medianR": round(float(np.median(values)), 4),
        "minR": round(float(values.min()), 4),
        "maxR": round(float(values.max()), 4),
        "positiveFraction": round(float((values > 0).mean()), 4),
        "significantPositiveFraction": round(
            float((significant & (values > 0)).sum() / denominator), 4
        ),
        "significantNegativeFraction": round(
            float((significant & (values < 0)).sum() / denominator), 4
        ),
        "significantFraction": round(float(significant.mean()), 4),
        "definedPoints": int(defined.sum()),
        "undefinedPoints": total - int(defined.sum()),
        "totalPoints": total,
    }
