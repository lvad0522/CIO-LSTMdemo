# -*- coding: utf-8 -*-
"""技巧检验自检：逐格点 Pearson r、显著性阈值、实况场歧义判据。

分两块：

**A. 合成本例**（永远跑）—— 用能手工验算的构造数据钉住算法本身：
完美正/负相关、手算 r、时间维恒定点、NaN、形状不符、多份实况不一致。

**B. 真实数据**（有 `Dataset/` 才跑）—— 在线推理产出对官方实况场，
断言与实验室 `pearson2` 口径一致：n=112、均值 ~0.153、无定义点 9 个。

跑法:  python test_verification.py
退出码: 0 = 通过 | 1 = 有断言失败 | 2 = **没有真数据，B 块没验证**（不是通过！）
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import verification as V                            # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "!!", label,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        FAILS.append(label)
    return cond


print("test_verification: 技巧检验自检")
print("=" * 78)

# ================================================ A. 合成本例

print("A. 逐格点 r 的算法（合成本例，可手工验算）")

rng = np.random.default_rng(2026)
ROWS, COLS, T = 4, 5, 112

real = rng.standard_normal((ROWS, COLS, T))

# A1 完美正相关
r = V.pearson_map(real, real)
check(np.allclose(r, 1.0), "pred == real 时逐点 r 全为 1", r.ravel()[:4])

# A2 完美负相关
r = V.pearson_map(-real, real)
check(np.allclose(r, -1.0), "pred == -real 时逐点 r 全为 -1", r.ravel()[:4])

# A3 手工验算：某个格点用 np.corrcoef 对齐
pred = rng.standard_normal((ROWS, COLS, T))
r = V.pearson_map(pred, real)
hand = np.corrcoef(pred[2, 3, :], real[2, 3, :])[0, 1]
check(abs(r[2, 3] - hand) < 1e-12,
      "格点 (2,3) 与 np.corrcoef 逐位一致", (r[2, 3], hand))

# A4 时间维恒定的格点 -> 0/0，必须是 NaN 而不是 0
flat = pred.copy()
flat[0, 0, :] = 3.14
r = V.pearson_map(flat, real)
check(np.isnan(r[0, 0]), "预测恒定点 -> NaN（不是 0）", r[0, 0])
check(np.isfinite(r[1, 1]), "同图其余格点不受影响", r[1, 1])

# A5 任一侧含 NaN -> 该格点 NaN，不影响别处
dirty = real.copy()
dirty[1, 2, 7] = np.nan
r = V.pearson_map(pred, dirty)
check(np.isnan(r[1, 2]) and np.isfinite(r[1, 3]),
      "实况含 NaN 的格点 -> NaN，邻点正常", (r[1, 2], r[1, 3]))

# A6 形状不符必须报错
try:
    V.pearson_map(pred[:, :, :10], real)
    check(False, "形状不符时抛 ValueError", "居然没抛")
except ValueError as exc:
    check(True, "形状不符时抛 ValueError -> %s" % str(exc)[:40])

# A7 临界 r：手册值 + 随置信水平单调
manual = {0.90: 0.1562, 0.95: 0.1857, 0.99: 0.2425}
for level, want in manual.items():
    got = V.critical_r(112, level)
    check(abs(got - want) < 5e-5, "临界 r @%.2f = %.4f（手册 %.4f）"
          % (level, got, want), got)
levels = [V.critical_r(112, c) for c in (0.90, 0.95, 0.99)]
check(levels[0] < levels[1] < levels[2], "临界 r 随置信水平单调递增", levels)

# A8 摘要：无定义点不计入分母
r_map = np.full((2, 2), 0.5)
r_map[0, 0] = np.nan
s = V.summarize(r_map, 0.95)
check(s["definedPoints"] == 3 and s["undefinedPoints"] == 1
      and s["totalPoints"] == 4, "摘要正确区分有定义/无定义点", s)
check(s["meanR"] == 0.5, "均值只用有定义点算", s["meanR"])

# A9 显著正/负必须分开报
mixed = np.array([[0.9, -0.9], [0.05, 0.05]])
s = V.summarize(mixed, 0.95)
check(s["significantPositiveFraction"] == 0.25
      and s["significantNegativeFraction"] == 0.25
      and s["significantFraction"] == 0.5,
      "显著正/负分开计数（合计数会把反相关算成技巧）", s)

# A10 置信水平输入归一化
check(V.normalize_confidence("0.95") == 0.95
      and V.normalize_confidence(0.9) == 0.9, "置信水平字符串/数字都能收")
for bad in ("abc", 0.2, 1.5):
    try:
        V.normalize_confidence(bad)
        check(False, "非法置信水平 %r 应报错" % (bad,), "居然没抛")
    except ValueError:
        check(True, "非法置信水平 %r 报错" % (bad,))

# A11 find_truth：缺失 / 一致 / 不一致
print("B. 实况场歧义判据（临时目录构造）")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    os.environ["TRUTH_DIR"] = str(root)
    try:
        check(V.find_truth("pre1", 2000) is None, "没有任何实况 -> None")

        day = root / "pre1" / "2000"
        day.mkdir(parents=True)
        arr = rng.standard_normal((ROWS, COLS, T))
        for k in (1, 2, 3):
            np.save(day / ("pre_1_2000(%d)_real2.npy" % k), arr)
        found = V.find_truth("pre1", 2000)
        check(found is not None and found.name.endswith("(1)_real2.npy"),
              "3 份内容一致 -> 取第一份", found)

        np.save(day / "pre_1_2000(3)_real2.npy", arr + 1.0)
        try:
            V.find_truth("pre1", 2000)
            check(False, "3 份内容不一致 -> 必须报错", "居然挑了一份返回")
        except ValueError as exc:
            check("不一致" in str(exc),
                  "3 份内容不一致 -> 报错（不静默挑）", str(exc)[:60])
    finally:
        os.environ.pop("TRUTH_DIR", None)

# ================================================ C. 真实数据
print("C. 真实数据（在线推理产出 vs 官方实况）")
os.environ.pop("TRUTH_DIR", None)
live = HERE.parent / "在线推理产出" / "live_pre1_2000.npy"
truth = V.find_truth("pre1", 2000)

if truth is None or not live.is_file():
    print("!" * 78)
    print("!! 没找到实况场或缺 live 产出 —— C 块【没验证】，退出码 2（不是通过）")
    print("!!   实况: Dataset/pre1/2000/*_real2.npy")
    print("!!   产出: 在线推理产出/live_pre1_2000.npy")
    print("!" * 78)
    print("A/B 块结果:", "全部通过" if not FAILS else "有失败: %s" % FAILS)
    sys.exit(1 if FAILS else 2)

pred = np.load(live)
real = np.load(truth)
r_map = V.pearson_map(pred, real)
ref = np.load(str(truth).replace("real2", "pearson2"))[:, :, 2]

check(pred.shape == real.shape == (81, 101, 112),
      "预测与实况同形 (81,101,112)", (pred.shape, real.shape))

s = V.summarize(r_map, 0.95)
check(s["n"] == 112, "样本量 n=112", s["n"])
check(s["undefinedPoints"] == 9,
      "无定义格点 9 个（模型在这些点输出时间维恒定）", s["undefinedPoints"])
check(abs(s["meanR"] - 0.1531) < 5e-4,
      "live 全图平均 r ≈ 0.1531", s["meanR"])
check(abs(s["meanR"] - float(np.mean(ref))) < 5e-4,
      "与实验室 pearson2 的均值同水位（%.4f vs %.4f）"
      % (s["meanR"], float(np.mean(ref))), s["meanR"])

finite = np.isfinite(r_map)
spatial = float(np.corrcoef(r_map[finite], ref[finite])[0, 1])
check(spatial > 0.6, "live 的 r 图与 pearson2 的空间相关 = %.4f" % spatial,
      spatial)

print("  实况: %s" % truth.name)
for key in ("confidence", "criticalR", "meanR", "medianR", "minR", "maxR",
            "positiveFraction", "significantPositiveFraction",
            "significantNegativeFraction", "definedPoints", "undefinedPoints"):
    print("    %-28s %s" % (key, s[key]))

print("=" * 78)
print("判定:", "全部通过" if not FAILS else "有失败: %s" % FAILS)
sys.exit(0 if not FAILS else 1)
