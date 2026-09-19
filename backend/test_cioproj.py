# -*- coding: utf-8 -*-
"""投影模块自检：从原始 nc 一路算到 CIO 序列，对金标准逐点比。

## 两种模式（自动选，差别很大）

**真数据模式（首选）** —— 找到 `摸库交付_2026-09-15/数据/` 时启用。
真 U850 + 真 `.mat` 对**服务器官方** `CIOproj_*.npy`。这是**真正的正确性验证**：
金标准是独立产出的，不是本仓库造的。

**夹具模式（退而求其次）** —— 只在没有 `数据/` 时启用，用 `03_夹具/`。
⚠️ 这是**循环论证**：夹具的 `.mat` 和夹具金标准是 `make_mock.py` 用同一套约定
造的，必然自洽。它只能当**回归测试**用（查"代码有没有被改坏"），
**不能当正确性证据**。跑起来会打醒目横幅提醒。

判据两边一致：除刻意做坏的 pre7 外都应逐点吻合（残差只来自 nc 存的是 float32）。
**pre7 是负面用例** —— 服务器上那个文件本身就是爆的（实测最大差 5.87e+37），
它必须对不上，否则说明测试根本没在比。

跑法:  python test_cioproj.py
退出码: 0 = 通过 | 1 = 有不一致 | 2 = **没找到任何数据，什么都没验证**（不是通过！）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cioproj                                     # noqa: E402

BAD_LEAD = 7                    # 服务器上爆掉的那一年
CORR_MIN = 0.999999             # 逐点一致要求的相关
REL_MAX = 1e-5                  # 最大差 / 金标准 std
LEADS = list(range(1, 21))

HERE = os.path.dirname(os.path.abspath(__file__))


def _first_dir(cands, probe):
    for rel in cands:
        cand = os.path.abspath(os.path.join(HERE, rel))
        if os.path.isdir(os.path.join(cand, probe)):
            return cand
    return None


def find_real():
    """服务器真数据包；含 U850/ + CIOmode .mat + 官方金标准。"""
    env = os.environ.get("CIOPROJ_REAL")
    if env:
        return env
    return _first_dir(
        [os.path.join("..", "摸库交付_2026-09-15", "数据"),
         os.path.join("..", "..", "摸库交付_2026-09-15", "数据"),
         "数据", os.path.join("..", "数据")],
        "20_40_100_125")


def find_fixture():
    """模拟数据沙盒（见其 .MOCK_SANDBOX 说明）。"""
    env = os.environ.get("CIOPROJ_FIXTURE")
    if env:
        return env
    return _first_dir(
        [os.path.join("..", "摸库交付_2026-09-15", "03_夹具"),
         "03_夹具",
         os.path.join("..", "03_夹具"),
         os.path.join("..", "..", "摸库交付_2026-09-15", "03_夹具"),
         os.path.join("..", "_mock"),
         os.path.join("..", "..", "_mock"),
         "D:/大三上/科研/lab/_mock"],
        "U850")


print("test_cioproj: 原始 nc -> CIO 序列，对金标准逐点比")
print("=" * 78)

REAL = find_real()
if REAL:
    ROOT = REAL
    MODE = "real"
    print("模式 = 真数据（真正正确性验证：对服务器官方金标准）")
else:
    ROOT = find_fixture()
    MODE = "mock"
    if ROOT:
        print("!" * 78)
        print("!! 夹具模式 —— 这【不是】正确性验证，是循环论证的回归测试")
        print("!! 夹具 .mat 与夹具金标准同出于 make_mock.py 的同一套约定，必然自洽。")
        print("!! 它只能查【代码有没有被改坏】。要真验证请把 摸库交付_2026-09-15/数据/ 放到位。")
        print("!" * 78)
    else:
        print("⚠  没找到真数据（数据/）也没找到夹具 —— 什么都没验证。")
        print("   真数据: 摸库交付_2026-09-15/数据/   可用 CIOPROJ_REAL 指定")
        print("   夹具  : 摸库交付_2026-09-15/03_夹具/ 可用 CIOPROJ_FIXTURE 指定")
        print("=" * 78)
        print("判定: 跳过（退出码 2 —— 这不是通过）")
        sys.exit(2)

print("数据根 =", ROOT)
NC = os.path.join(ROOT, "U850")
MAT = os.path.join(ROOT, "CIOmode_1982_2017.mat")
if not os.path.isfile(MAT):                      # 夹具的 .mat 在 rain/ 下
    MAT = os.path.join(ROOT, "rain", "CIOmode_1982_2017.mat")
D20 = os.path.join(ROOT, "20_40_100_125")
if not os.path.isdir(D20):                       # 夹具的金标准在 rain/ 下
    D20 = os.path.join(ROOT, "rain", "20_40_100_125")
print(".mat  =", MAT)
print("金标准 =", D20)

YEARS = list(range(2000, 2021))
ok = True
for lead in LEADS:
    series, meta = cioproj.compute_cioproj(
        which="u850", nc_dir=NC, years=YEARS, lead=lead, mat_path=MAT)
    gold = np.load(os.path.join(
        D20, "CIOproj_-20_20_40_120_1degree_pre%d.npy" % lead)).squeeze()
    n = min(series.size, gold.size)
    diff = np.abs(series[:n] - gold[:n])
    corr = float(np.corrcoef(series[:n], gold[:n])[0, 1])
    rel = float(diff.max() / gold.std())
    matched = series.size == gold.size and corr >= CORR_MIN and rel <= REL_MAX

    if lead == BAD_LEAD:
        # 负面用例：爆掉的年份必须对不上
        verdict = "爆值年份，已确认对不上" if not matched else "本该对不上，却对上了！"
        if matched:
            ok = False
    else:
        verdict = "一致" if matched else "对不上！"
        if not matched:
            ok = False

    print("  pre%-2d %4d/%-4d 天/年=%-3d 窗口=%s  corr=%.8f 最大差=%.3e 相对=%.2e  [%s]"
          % (lead, series.size, gold.size, meta["days_per_year"][YEARS[0]],
             meta["window"], corr, diff.max(), rel, verdict))
    if meta.get("missing_months"):
        print("        缺月份:", meta["missing_months"])

print("=" * 78)
print("判定:", "全部符合预期" if ok else "有不一致，看上面")
if MODE == "mock":
    print("!! 本次是夹具模式，只证明【没被改坏】，不证明【算得对】。")

# 顺带验分派：判不出来必须报错，不能默认成 U850
try:
    cioproj.detect_kind("some_random_file.npy")
    print("分派自检: 失败（本该报错）")
    ok = False
except ValueError as exc:
    print("分派自检: 判不出来时报错 ->", exc)
print("分派自检: 文件名含 uwnd ->", cioproj.detect_kind("cesm.uwnd.2010.nc"))
print("分派自检: sst 分支 ->", end=" ")
try:
    cioproj.compute_cioproj("sst", mat_path=MAT)
    print("失败（本该报错）")
    ok = False
except NotImplementedError as exc:
    print("按预期拒绝：", str(exc)[:40], "...")

print("退出码 =", 0 if ok else 1)
sys.exit(0 if ok else 1)
