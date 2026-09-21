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
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ==================================================== 任务根隔离（C-4 同形状 / R5-T2）
# 本套件只算投影、本身不建任务目录；但两个任务根 env 一律**在 import 之前**设好指向
# 本进程专属临时区 —— 免得将来有人在这里加一条建任务的路径就又开始往真
# `backend/uploads/` 里倒桩产物。
REAL_UPLOADS = Path(HERE) / "uploads"
TMP_BASE = Path(tempfile.mkdtemp(prefix="r5t2_cioproj_%d_" % os.getpid()))
JOB_ROOT = TMP_BASE / "prediction_jobs"
CIO_JOB_ROOT = TMP_BASE / "cio_jobs"
TRUTH_ROOT = TMP_BASE / "truth"
for _d in (JOB_ROOT, CIO_JOB_ROOT, TRUTH_ROOT):
    _d.mkdir(parents=True, exist_ok=True)
os.environ["PREDICTION_JOB_ROOT"] = str(JOB_ROOT)
os.environ["CIO_JOB_ROOT"] = str(CIO_JOB_ROOT)
os.environ["TRUTH_DIR"] = str(TRUTH_ROOT)
atexit.register(shutil.rmtree, TMP_BASE, ignore_errors=True)   # 异常/提前退出也清理

import cioproj                                     # noqa: E402

# ------------------------------------------------ 真工作区零写入的自检装置
REAL_JOB_DIRS = (REAL_UPLOADS / "chain_jobs",
                 REAL_UPLOADS / "prediction_jobs",
                 REAL_UPLOADS / "cio_jobs")


def _inside(path, base):
    """`path` 是否（严格地）落在 `base` 之下。"""
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


# 守卫：两个任务根 env 必须都落在临时区，否则不许继续（env 没生效就会命中）。
_LEAKED = [("%s=%s" % (name, root))
           for name, root in (("PREDICTION_JOB_ROOT", JOB_ROOT),
                              ("CIO_JOB_ROOT", CIO_JOB_ROOT))
           if any(_inside(root, real) for real in REAL_JOB_DIRS)]
if _LEAKED:
    print("⚠  任务根没隔离到临时区，自检会往真工作区里写文件：")
    for _d in _LEAKED:
        print("     %s" % _d)
    print("   拒绝继续（C-4）—— 这不是通过。")
    sys.exit(2)


def _snapshot_tree(base):
    """目录名全集 + 各条目 mtime + **根目录 mtime**（见兄弟套件同名函数说明）。"""
    plain = str(base)
    if not base.is_dir():
        return {"key": plain, "exists": False, "root_mtime": None, "entries": {}}
    entries = {}
    for path in sorted(base.rglob("*")):
        try:
            entries[str(path)] = path.stat().st_mtime
        except OSError:
            continue
    try:
        root_mtime = base.stat().st_mtime
    except OSError:
        root_mtime = None
    return {"key": plain, "exists": True, "root_mtime": root_mtime,
            "entries": entries}


def _no_write_diff(before, after):
    """比对跑前/跑后快照，返回 (是否零差异, 差异描述)。"""
    if before.get("exists") != after.get("exists"):
        return False, "存在性变了：%r -> %r" % (before["exists"], after["exists"])
    if not before.get("exists"):
        return True, "目录本就不存在（无既有件，亦未新建）"
    b, a = before["entries"], after["entries"]
    parts = []
    stale = sorted(set(b) - set(a))
    fresh = sorted(set(a) - set(b))
    touched = [p for p in sorted(set(b) & set(a)) if b[p] != a[p]]
    if stale:
        parts.append("%d 个条目被删：%s" % (len(stale), stale[:3]))
    if fresh:
        parts.append("%d 个条目新增：%s" % (len(fresh), fresh[:3]))
    if touched:
        parts.append("%d 个条目 mtime 被改：%s" % (len(touched), touched[:3]))
    if before.get("root_mtime") != after.get("root_mtime"):
        parts.append("根目录 mtime 变了（建了又删？）")
    return (not parts), ("；".join(parts) if parts else "无差异")


# 跑前快照：这之后才是全部用例体
REAL_SNAPSHOT_BEFORE = {str(d): _snapshot_tree(d) for d in REAL_JOB_DIRS}


def _final_isolation_selfcheck():
    """末置自检（C-4 同形状）：真工作区 uploads/ 本轮零写入，**进退出判定**。

    两条出口（"没数据 → 2" 与正常收尾）都要调用；失败时返回 False，由调用方把
    退出码落到 1（真失败）而不是 2（没验证）或 0（通过）。
    """
    print("  隔离自检（C-4）：任务根在临时区，不写 backend/uploads/")
    print("    PREDICTION_JOB_ROOT -> %s" % os.environ.get("PREDICTION_JOB_ROOT"))
    print("    CIO_JOB_ROOT        -> %s" % os.environ.get("CIO_JOB_ROOT"))
    print("    TRUTH_DIR           -> %s" % os.environ.get("TRUTH_DIR"))
    print("    真工作区（仓库内）    -> %s" % REAL_UPLOADS)
    good = True
    if not (_inside(JOB_ROOT, TMP_BASE) and _inside(CIO_JOB_ROOT, TMP_BASE)
            and _inside(TRUTH_ROOT, TMP_BASE)):
        print("  [!!] 任务根/实况根没落在本进程临时区")
        good = False
    for real in REAL_JOB_DIRS:
        okk, why = _no_write_diff(REAL_SNAPSHOT_BEFORE[str(real)],
                                  _snapshot_tree(real))
        print("  [%s] 真工作区 %s 本轮一个字节都没被写%s"
              % ("OK" if okk else "!!", real.name, "" if okk else "   <- " + why))
        good = good and okk
    return good

BAD_LEAD = 7                    # 服务器上爆掉的那一年
CORR_MIN = 0.999999             # 逐点一致要求的相关
REL_MAX = 1e-5                  # 最大差 / 金标准 std
LEADS = list(range(1, 21))


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
        _iso_ok = _final_isolation_selfcheck()
        print("判定: 跳过（退出码 2 —— 这不是通过）")
        sys.exit(2 if _iso_ok else 1)

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

if not _final_isolation_selfcheck():
    ok = False
print("退出码 =", 0 if ok else 1)
sys.exit(0 if ok else 1)
