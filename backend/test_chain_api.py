# -*- coding: utf-8 -*-
"""链路自检：`/api/chain` 的受理校验、三段编排、产物端点与**逐位对拍**。

分块：
  A. 受理与 API 契约（打桩 `_run_archive_inference`，秒级）：`.zip` 全链、
     `.npy` 入口、`onlyProjection=true`、全部降级 400 分支
  B. 数值一致性：同一份 zip 走新链与旧 `/api/predict/jobs`，比**喂进模型的
     张量**与**产出的 prediction.npy**（验收 1）；`_prepare_cio` 只被调一次
  C. 官方件对拍：新链 `/series` 的 `raw` 对 `摸库交付_2026-09-15/数据/
     20_40_100_125/CIOproj_-20_20_40_120_1degree_pre1.npy`，最大差 < 1e-6
  D. 未就绪语义（409/404）与下载格式
  E. **离线对拍**（真跑 8181 个 checkpoint，分钟级，用 CHAIN_E2E=1 才跑）：
     拿 `backend/uploads/prediction_jobs/<job>/raw_fields.zip` 跑完整链路，
     与同目录 `prediction.npy` 逐位比 —— 这是验收 1 的直接证据。

跑法:  python test_chain_api.py          # 只跑 A–D（快）
       set CHAIN_E2E=1 && python test_chain_api.py   # 加上 E（真推理）
"""
import io
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

try:                                    # Windows 控制台默认 GBK，中文断言读不清
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

FAILS = []

# 已确认的**陈旧基线**：`prediction.npy` 是旧版代码的产物（test 阶段实测：
# 拿它们对拍会报出 75.26 那样的伪差异）。显式列出，让"挑错基线"变成硬错误，
# 而不是静默拿它当验收 1 的对照（C-3）。
#
# ⚠ 键是 **8 字符前缀**，两条来源路径给过来的却是**全名**（`cand.name` /
# `env_job` 都是 32 位 jobId）—— 所以匹配必须走 `startswith`，不能 `in`。
# 上一版写成 `cand.name in STALE_BASELINES` 导致本守卫**恒为 False**、
# 从未生效（B-1：和 C-1 同一类"保证不保证"）。新增陈旧件只要加前缀即可。
STALE_BASELINES = {
    "16655a7a": "prediction.npy 是旧版产物（5d8e66c1 一代），对拍报伪差异",
    "847e8f0f": "prediction.npy 是旧版产物（5d8e66c1 一代），对拍报伪差异",
}


def is_stale_baseline(name: str) -> bool:
    """任务目录名/ jobId 是否命中陈旧黑名单（前缀匹配，见 STALE_BASELINES）。"""
    return any(str(name).startswith(key) for key in STALE_BASELINES)


# C-2/B-3：服务器绝对路径**绝不该**出现在任何响应里。扫描是**递归**的（当初
# 的真实泄漏在 `result.verification.truthPath`，第 3 层），标记取自本机实际
# 路径而不是猜的驱动盘/仓库名 —— 部署环境路径不含仓库名时，硬编码仓库名一个
# 都查不出。
LEAK_MARKERS = (
    str(HERE.resolve()),           # backend 绝对路径（串里的真泄漏就是这种）
    str(HERE.parent.resolve()),    # 仓库根绝对路径
    "backend\\", "backend/",       # 相对形态的目录标记
    "uploads\\", "uploads/",
)
_DRIVE_RE = re.compile(r"[A-Za-z]:[\\/]")

# **allowlist 现在故意留空**：B-3（`truthPath` 泄漏）已按 PM 口径修掉 ——
# 实况绝对路径改为只存**任务顶层 `truthFile`**，而该键不在任何 `_public_job()`
# 的字段白名单里（live_prediction.py 与 chain.py 各有一份），结构上不可能进响应；
# 两个 `truth/preview` 消费者改读它。因此这条递归扫描从此是 B-3 的**真断言**：
# 只要绝对路径再出现在任一响应里（新路由或旧路由），本套件立刻判红。
#
# 保留这个空元组而不是删掉机制，是为了让"重新豁免"必须是一次**显式登记**。
# 配套的自检在下面 `check(not LEAK_ALLOWLIST, ...)`：谁把它填回去，谁就先红。
LEAK_ALLOWLIST = ()


# B-3 修复的"其余一切不变"侧：`result.verification` 的**键集**。改前含
# `truthPath`（18 键），改后只少这一个（17 键）。用集合等值断言而不是"逐项检查
# 字段还在" —— 后者挡不住"顺手多塞一个键"这种口径漂移，也挡不住少一个。
# 14 个统计键来自 `verification.summarize()`，3 个附加键来自
# `_verify_against_truth()`（其中 downloadUrl 由各 router 覆写成自己的前缀）。
VERIFICATION_KEYS = {
    "n", "confidence", "criticalR", "meanR", "medianR", "minR", "maxR",
    "positiveFraction", "significantPositiveFraction",
    "significantNegativeFraction", "significantFraction", "definedPoints",
    "undefinedPoints", "totalPoints",
    "available", "truthName", "downloadUrl",
}


def _is_allowlisted(path: tuple) -> bool:
    return any(path == allowed for allowed in LEAK_ALLOWLIST)


def _leak_hits(node, path=()):
    """递归找出响应体里带服务器绝对路径的字段，返回 (JSON 路径, 值) 列表。

    容器（dict/list）继续下钻，字符串对 LEAK_MARKERS 逐个 find。
    命中 LEAK_ALLOWLIST 的字段**仍会打印**出来，但不算作未授权泄漏 —— 这正是
    "不假装它不存在"。B-3 修复后该 allowlist 为空，所以打印分支不该再触发。
    """
    hits = []
    if isinstance(node, dict):
        for key, value in node.items():
            hits.extend(_leak_hits(value, path + (str(key),)))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            hits.extend(_leak_hits(value, path + ("[%d]" % index,)))
    elif isinstance(node, str):
        merged = node.replace("\\", "/")
        if any(m.replace("\\", "/") in merged for m in LEAK_MARKERS) \
                or _DRIVE_RE.search(node):
            hits.append((".".join(path), node))
    return hits


def check_no_leak(doc, label):
    """断言语义：响应体里没有任何**未授权**的服务器绝对路径（递归扫描）。"""
    hits = _leak_hits(doc)
    allowed = [h for h in hits if _is_allowlisted(tuple(h[0].split(".")))]
    unallowed = [h for h in hits if h not in allowed]
    for where, value in allowed:
        print("      [i] 登记在 LEAK_ALLOWLIST 的豁免命中（B-3 已修，不该再有）："
              "%s = %s" % (where, value))
    return check(not unallowed, label,
                 ["%s = %s" % (w, v) for w, v in unallowed][:3])


def check(cond, label, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "!!", label,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        FAILS.append(label)
    return cond


def find_real():
    """真数据包：U850/ + CIOmode .mat + 官方金标准（首选，能验真伪）。"""
    env = os.environ.get("CIOPROJ_REAL")
    if env:
        return Path(env)
    for rel in ("../摸库交付_2026-09-15/数据", "../../摸库交付_2026-09-15/数据"):
        cand = (HERE / rel).resolve()
        if (cand / "20_40_100_125").is_dir():
            return cand
    return None


def find_fixture():
    env = os.environ.get("CIOPROJ_FIXTURE")
    if env:
        return Path(env)
    for rel in ("../摸库交付_2026-09-15/03_夹具", "03_夹具", "../03_夹具",
                "../../摸库交付_2026-09-15/03_夹具", "../_mock", "../../_mock"):
        cand = (HERE / rel).resolve()
        if (cand / "U850").is_dir():
            return cand
    return None


def locate_data():
    """返回 (数据根, nc 目录, .mat, 官方金标准目录, 是否真件)。"""
    real = find_real()
    if real is not None:
        return real, real / "U850", real / "CIOmode_1982_2017.mat", real / "20_40_100_125", True
    fix = find_fixture()
    if fix is None:
        return None
    return (fix, fix / "U850", fix / "rain" / "CIOmode_1982_2017.mat",
            fix / "rain" / "20_40_100_125", False)


print("test_chain_api: /api/chain 链路自检")
print("=" * 78)

DATA = locate_data()
if DATA is None:
    print("⚠  既没找到真数据也没找到夹具 —— 什么都没验证。")
    print("   真数据: 摸库交付_2026-09-15/数据/   夹具: 摸库交付_2026-09-15/03_夹具/")
    print("=" * 78)
    print("判定: 跳过（退出码 2 —— 这不是通过）")
    sys.exit(2)

ROOT, NC, MAT, GOLD, IS_REAL = DATA
print("数据根 = %s（%s）" % (ROOT, "真件" if IS_REAL else "夹具件"))
print(".mat  = %s" % MAT)

if IS_REAL:
    os.environ["CIOPROJ_MAT"] = str(MAT)          # 真件：沙盒外，必须真
    os.environ.pop("CIOPROJ_ALLOW_MOCK", None)
else:
    os.environ["CIOPROJ_MAT"] = str(MAT)
    os.environ["CIOPROJ_ALLOW_MOCK"] = "1"        # 夹具件只在联调里放行

# 实况场指向受控临时目录：技巧检验的断言不能取决于本机有没有 Dataset/
# 任务目录也隔离到临时区：**两个 router 都要**。
#   ① 新链 `chain.JOB_ROOT` 读 `CHAIN_JOB_ROOT`（本轮补的 env 覆盖，C-5）；
#   ② 旧 `/api/predict` 的 `live_prediction.JOB_ROOT` 读 `PREDICTION_JOB_ROOT`。
# 两个 env 都必须在 import 之前设好：两个模块都在 import 期求值 JOB_ROOT，
# 只在运行期改模块属性是**补不住 import 期派生值**的。
#
# 临时根**按 pid 唯一化**（R-2）：原来的固定名 + 启动时无条件 rmtree，会让同一
# checkout 里并行跑的两个实例互清任务目录（表现为 FileNotFoundError 或目录计数
# 对不上），把并发干扰伪装成真实回归。加 pid 后两实例各写各的，互不相干。
TMP_BASE = HERE / ("_tmp_chain_%d" % os.getpid())
shutil.rmtree(TMP_BASE, ignore_errors=True)
JOB_ROOT = TMP_BASE / "chain_jobs"                # 新链任务目录
LEGACY_JOB_ROOT = TMP_BASE / "prediction_jobs"    # 旧 /api/predict 任务目录
TRUTH_ROOT = TMP_BASE / "truth"                   # 实况场目录
for _d in (TRUTH_ROOT, JOB_ROOT, LEGACY_JOB_ROOT):
    _d.mkdir(parents=True, exist_ok=True)
os.environ["TRUTH_DIR"] = str(TRUTH_ROOT)
os.environ["CHAIN_JOB_ROOT"] = str(JOB_ROOT)
os.environ["PREDICTION_JOB_ROOT"] = str(LEGACY_JOB_ROOT)

import chain as ch                                   # noqa: E402
import cio_diagnostics as cd                         # noqa: E402  （第5轮：/api/cio 终态扫描）
import live_prediction as lp                         # noqa: E402
import verification                                  # noqa: E402  （D3 打桩 find_truth）

ch.JOB_ROOT = JOB_ROOT                               # 与 env 同值，双保险
lp.JOB_ROOT = LEGACY_JOB_ROOT
# 第 5 轮：`cio_diagnostics.JOB_ROOT` 是**硬编码**的
# `prediction.BACKEND_DIR / "uploads" / "cio_jobs"`，**不读 env**（cd:29）——
# 不改它在运行期兜底，`/api/cio/jobs` 会直接往真工作区里写任务目录。
# 这里猴补丁是安全的：`cd` 只在函数体内用 `JOB_ROOT`（cd:538），没有 import 期
# 派生值，而且补丁在任何请求之前就位。下面 REAL_JOB_DIRS 守卫一并收紧。
CD_JOB_ROOT = TMP_BASE / "cio_jobs"
CD_JOB_ROOT.mkdir(parents=True, exist_ok=True)
cd.JOB_ROOT = CD_JOB_ROOT


def _inside(path, base):
    """`path` 是否（严格地）落在 `base` 之下。"""
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


# 真工作区的两个任务库：本轮自检**一个字节都不许往里写**（C-4）。
REAL_JOB_DIRS = (HERE / "uploads" / "chain_jobs",
                 HERE / "uploads" / "prediction_jobs",
                 HERE / "uploads" / "cio_jobs")     # 第5轮：/api/cio 也要证明没写
LEAKED_ROOTS = [d for d in (ch.JOB_ROOT, lp.JOB_ROOT, cd.JOB_ROOT)
                if any(_inside(d, real) for real in REAL_JOB_DIRS)]
if LEAKED_ROOTS:
    print("⚠  任务目录没隔离到临时区，自检会往真工作区里写文件：")
    for _d in LEAKED_ROOTS:
        print("     %s" % _d)
    print("   拒绝继续（C-4）—— 这不是通过。")
    sys.exit(2)
if os.environ.get("CHAIN_REQUIRE_ISOLATED"):
    _in_repo = [d for d in (ch.JOB_ROOT, lp.JOB_ROOT)
                if _inside(d, HERE.parent)]
    if _in_repo:
        print("⚠  CHAIN_REQUIRE_ISOLATED=1 但任务根不在仓库外：%s" % _in_repo)
        sys.exit(2)

from fastapi import FastAPI                          # noqa: E402
from fastapi.testclient import TestClient            # noqa: E402

app = FastAPI()
app.include_router(ch.router)
client = TestClient(app)
# 旧路由的对照客户端（B 段用它跑 `/api/predict/jobs`）
legacy = FastAPI()
legacy.include_router(lp.router)
legacy_client = TestClient(legacy)
# 第 5 轮新增：CIO 链路检查路由（此前本套件**从未调用过**，所以它的终态
# `projection` 同样带着 `matPath` 却没人扫 —— 见下面 A6 段的说明）。
cio_app = FastAPI()
cio_app.include_router(cd.router)
cio_client = TestClient(cio_app)
CHAIN_PREFIX = "/api/chain/jobs"

real_inference = lp._run_archive_inference
CAPTURED = {}


def _snapshot_tree(base: Path) -> dict:
    """目录名全集 + 各目录/文件 mtime + **根目录 mtime**（B-1/B-4）。

    根 mtime 是关键：没有它，"建一个目录又把它删掉"在前后快照里看不出任何
    差别（名字集合与 mtime 都恢复原状），而真工作区恰恰会被这种手法悄悄碰过。
    """
    plain = str(base)
    posix = str(base).replace("\\", "/")
    if not base.is_dir():
        return {"key": plain, "exists": False, "root_mtime": None,
                "entries": {}}
    entries = {}
    for path in sorted(base.rglob("*")):
        try:
            entries[str(path)] = path.stat().st_mtime
        except OSError:                        # 竞态下目录可能刚被移走
            continue
    try:
        root_mtime = base.stat().st_mtime
    except OSError:
        root_mtime = None
    return {"key": plain, "posix": posix, "exists": True,
            "root_mtime": root_mtime, "entries": entries}


def verify_no_write(before: dict, after: dict, label: str) -> bool:
    """比对跑前/跑后快照，证明**这一个字节都没被写**（不是"打印一句"）。"""
    if before.get("exists") != after.get("exists"):
        return check(False, "%s：存在性变了" % label, after["key"])
    if not before.get("exists"):
        print("  [i] %s：目录本就不存在（无既有件，亦未新建）" % label)
        return True
    b_entries, a_entries = before["entries"], after["entries"]
    stale = sorted(set(b_entries) - set(a_entries))
    fresh = sorted(set(a_entries) - set(b_entries))
    touched = [(p, b_entries[p], a_entries[p]) for p in
               sorted(set(b_entries) & set(a_entries))
               if b_entries[p] != a_entries[p]]
    parts = []
    if stale:
        parts.append("(%d 个条目被删)" % len(stale))
        parts.extend("  - %s" % p for p in stale[:3])
    if fresh:
        parts.append("(%d 个条目新增)" % len(fresh))
        parts.extend("  + %s" % p for p in fresh[:3])
    if touched:
        parts.append("(%d 个条目 mtime 被改)" % len(touched))
        parts.extend("  ~ %s" % p for p, _b, _a in touched[:3])
    if before.get("root_mtime") != after.get("root_mtime"):
        parts.append("根目录 mtime 变了（建了又删？）：%r -> %r"
                     % (before.get("root_mtime"), after.get("root_mtime")))
    return check(not parts, label, "；".join(parts) if parts else "无差异")


# 跑前快照：这段之后才是 A–E 全部用例体。跑后快照在最后的 finally 里 ——
# 中间**所有**路径（含 monkey patch、临时目录创建失败等异常路径）都被覆盖。
REAL_SNAPSHOT_BEFORE = {str(d): _snapshot_tree(d) for d in REAL_JOB_DIRS}
REAL_SNAPSHOT_AFTER = None      # finally 里赋值；异常提前退出时下面的报错要用
write_checks_done = False       # 零写入比对是否真的跑过（不许静默跳过）


def fake_inference(cio, progress_callback=None, span=(3, 95), **kw):
    """桩：记下真正喂进模型的张量，回一个由它铺出来的场。

    `**kw` 吸收 `_run_archive_inference` 新增的 keyword-only `lead`/`year`
    （import-all-models §6.1：只同步签名，**不改任何断言口径**）。
    """
    CAPTURED["cio"] = np.array(cio, copy=True)
    if progress_callback:
        progress_callback(span[0], "桩推理开始")
        progress_callback(span[1], "桩推理结束")
    return np.tile(cio.astype(np.float32), (lp.GRID_ROWS, lp.GRID_COLS, 1))


def build_zip(nc_paths, inner="uwnd_5-9", corrupt=False, name_of=None):
    """打包 nc 成 zip。

    `corrupt=True`  —— 成员名/目录结构照旧，只把 nc **内容**换成垃圾字节：
                       T4-4 的载体（受理与解压只看名字，必然走到读 nc 才炸）。
    `name_of`       —— 自定义成员名（收 (path, index)），用来造"月份一个都匹配
                       不上"的第二种载体。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for index, path in enumerate(nc_paths):
            base = name_of(path, index) if name_of else path.name
            name = "%s/%s" % (inner, base)
            if corrupt:
                zf.writestr(name, b"NOT-A-NETCDF-" + b"x" * 512)
            else:
                zf.write(path, name)
    return buf.getvalue()


def npy_bytes(array):
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


POLLS = []            # wait() 的轮询留痕（B-3/C-7 的进度证据）
INFERENCE_LO = ch.INFERENCE_SPAN[0]   # ③ 段绝对进度区间的下界（=33），不写死
INFERENCE_HI = ch.INFERENCE_SPAN[1]   # 上界（=97）：二次映射会让终值停在 95


def wait(job_id, timeout=300, prefix=CHAIN_PREFIX):
    if prefix.startswith("/api/predict"):
        http = legacy_client
    elif prefix.startswith("/api/cio"):
        http = cio_client
    else:
        http = client
    end = time.time() + timeout
    while time.time() < end:
        data = http.get("%s/%s" % (prefix, job_id)).json()
        # 每次轮询留痕：B-3/C-7 要的"进度单调 + ③ 段起止"只能从这些**历史**
        # 采样点重建 —— 作业跑完后 GET 到的进度恒为 100，事后补测不出来。
        POLLS.append({"prefix": prefix, "jobId": job_id,
                      "stage": data.get("stage"), "progress": data.get("progress"),
                      "stageIndex": data.get("stageIndex")})
        if data.get("status") in ("completed", "failed"):
            return data
        if os.environ.get("CHAIN_DEBUG"):
            print("      [dbg] %s %s %s %s" % (prefix, data.get("status"),
                  data.get("stage"), data.get("progress")))
        time.sleep(0.1)
    return {"status": "timeout"}


def monotone_polls(job_id, prefix=CHAIN_PREFIX):
    """把某个作业的轮询采样折成**单调递增**序列（每次只记新高，返回 (stage, progress)）。

    C-7 首轮说的"进度倒退"经评审探针实测**证伪**，所以这条断言只钉住
    "不倒退"这一半；真正的判别项是 ③ 段的**绝对区间**（下面的 span 断言）。
    """
    series = []
    for rec in POLLS:
        if rec["jobId"] != job_id or rec["prefix"] != prefix:
            continue
        progress = rec.get("progress")
        if progress is None:
            continue
        if not series or progress > series[-1][1]:
            series.append((rec.get("stage") or "", int(progress)))
    return series


NCS = sorted(NC.glob("*.nc"))
ZIP_BYTES = build_zip(NCS)
GOLD_ARR = np.load(GOLD / "CIOproj_-20_20_40_120_1degree_pre1.npy").squeeze()

# CHAIN_E2E=1 时 B 段也走**真推理**（分钟级），验收 1 就有了真实端到端的依据；
# 否则打桩，秒级跑完 A–D。
E2E = bool(os.environ.get("CHAIN_E2E"))
B_STUB = not E2E
lp._run_archive_inference = fake_inference if B_STUB else real_inference
try:
    # B-3 自检：豁免表必须为空。这条断言的作用不是"扫得多严"，而是让**重新豁免**
    # 这件事必须先把这条断言改红 —— 否则后人给新泄漏加一条 allowlist 就能静默糊过去。
    check(not LEAK_ALLOWLIST,
          "泄漏扫描没有登记任何豁免（B-3 已修，豁免表保持空）",
          LEAK_ALLOWLIST)

    # =============================================== A. 受理与 API 契约
    print("A. 受理与 API 契约（打桩推理）")

    r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                    files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES,
                                    "application/zip")})
    check(r.status_code == 202, "zip 建任务 -> 202", r.text[:200])
    created = r.json()
    check(created["stageIndex"] == 0
          and created["status"] in ("queued", "running")
          and created["inputKind"] == "raw-zip",
          "受理返回 stageIndex=0 / 未完成 / raw-zip",
          {k: created.get(k) for k in ("stageIndex", "status", "inputKind")})
    check(created["projectionMode"] == "u850_only",
          ".zip 入口 projectionMode=u850_only", created.get("projectionMode"))
    check(created["projection"]["matSource"] == ("real" if IS_REAL else "mock")
          and created["projection"]["matPath"] == MAT.name,
          "受理即透出标准CIO 身份（matSource/matPath）", created.get("projection"))
    # C-2：服务器绝对路径不外泄（两个旧 router 同样不公开）
    check("jobDir" not in created,
          "受理响应不含 jobDir（服务器路径不外泄）", sorted(created))
    check_no_leak(created,
                  "受理响应里没有任何字段带服务器绝对路径（递归扫描）")
    job_dir = JOB_ROOT / created["jobId"]
    check((job_dir / "raw_fields.zip").is_file(),
          "受理阶段 raw_fields.zip 已落盘", str(job_dir))

    job = wait(created["jobId"])
    check(job["status"] == "completed", "zip 全链跑完 -> completed", job.get("error"))
    if job["status"] == "completed":
        check(job["stageIndex"] == 3, "全链 stageIndex=3", job.get("stageIndex"))
        check(job["result"]["shape"] == [81, 101, 112], "结果形状 (81,101,112)",
              job["result"]["shape"])
        check(job["result"]["downloadUrl"]
              == "/api/chain/jobs/%s/download/prediction" % job["jobId"],
              "result.downloadUrl 指向链路前缀", job["result"]["downloadUrl"])
        proj = job["projection"]
        check(proj["kind"] == "u850" and proj["kindSource"] == "explicit-param",
              "投影类别留痕 kind/kindSource", proj)
        check(proj["seriesLength"] == 2352 and not proj["missingMonths"],
              "投影 2352 点、无缺月", (proj.get("seriesLength"), proj.get("missingMonths")))
        check(job["normalization"]["mode"] == "training-equivalent",
              "归一化走 training-equivalent", job["normalization"]["mode"])
        check(job["result"]["verification"]["available"] is False,
              "无实况场 -> verification.available=False",
              job["result"]["verification"])
        check((job_dir / "series.npy").is_file()
              and (job_dir / "normalized_window.npy").is_file()
              and (job_dir / "dates.json").is_file()
              and (job_dir / "completeness.json").is_file(),
              "四份落盘产物齐备（series/normalized/dates/completeness）")
        series_dtype = np.load(job_dir / "series.npy", allow_pickle=False).dtype
        norm_dtype = np.load(job_dir / "normalized_window.npy", allow_pickle=False).dtype
        check(series_dtype == np.float64 and norm_dtype == np.float32,
              "series.npy=float64 / normalized_window.npy=float32",
              (series_dtype, norm_dtype))

    # --- A2: `.npy` 入口跳过①，受理即落盘 series.npy
    r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=cio",
                    files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
    check(r.status_code == 202, "npy 建任务 -> 202", r.text[:200])
    created = r.json()
    check(created["stageIndex"] == 0 and created["inputKind"] == "cio-npy"
          and created["projectionMode"] == "uploaded_cio",
          "npy 受理 stageIndex=0 / cio-npy / uploaded_cio",
          {k: created.get(k) for k in ("stageIndex", "inputKind", "projectionMode")})
    npy_dir = JOB_ROOT / created["jobId"]
    check((npy_dir / "cio_input.npy").is_file() and (npy_dir / "series.npy").is_file(),
          "npy 受理即落盘 cio_input.npy + series.npy")
    check(created["projection"].get("matSource") == ("real" if IS_REAL else "mock")
          and created["projection"].get("matPath") == MAT.name,
          "npy 受理也带标准CIO 身份（尽力而为，不阻断）",
          (created["projection"].get("matSource"), created["projection"].get("matUnavailable")))
    job = wait(created["jobId"])
    check(job["status"] == "completed" and job["stageIndex"] == 3,
          "npy 全链跑完 stageIndex=3", job.get("error") or job.get("stageIndex"))

    # --- A3: onlyProjection=true 只跑①②
    CAPTURED.clear()
    r = client.post(
        "/api/chain/jobs?year=2000&lead=pre1&which=u850&onlyProjection=true",
        files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES, "application/zip")})
    check(r.status_code == 202, "onlyProjection 建任务 -> 202", r.text[:200])
    partial = wait(r.json()["jobId"])
    check(partial["status"] == "completed" and partial["stageIndex"] == 2,
          "只跑投影：completed 且 stageIndex=2",
          (partial.get("status"), partial.get("stageIndex")))
    check("未跑" in (partial.get("stage") or ""),
          "只跑投影的 stage 文案点明未跑推理", partial.get("stage"))
    check(not CAPTURED, "只跑投影：从未调用推理（未加载任何 .pt）", list(CAPTURED))
    r = client.get("%s/%s/preview" % (CHAIN_PREFIX, partial["jobId"]))
    check(r.status_code == 409 and "降水推理尚未完成" in r.text,
          "只跑投影后 /preview -> 409 降水推理尚未完成",
          "%s %s" % (r.status_code, r.text[:120]))

    # --- A4: 阶段① 的 warning 不被阶段② 覆盖（B-2/C-6）
    # 缺陷机制：`_intake_zip` 建的 projection **没有 warning 键**，而 `_run_stage2`
    # 用 `projection.get("warning")` 参与合并、再 `warning=... if warnings else None`
    # 写回 → 把阶段① 刚写的缺月提示覆盖成 None（旧路由不会丢：live_prediction 把
    # proj['warning'] 并进 projection）。触发条件"缺月"，生产可达但低频。
    #
    # 为什么这里用**故障注入**：本验收的 year=2000/lead=pre1 窗口恰好是交付包的
    # 全部 5–9 月，原样 zip 的 missing 恒为空、永远触发不了这条路径（实测
    # missing_months=[] / warning=None）。所以把"阶段① 给出 warning"注入进去，
    # 验的仍是 `_run_stage2` 真实的合并逻辑 —— 没有放宽任何产品行为。
    _real_cioproj_stage1 = ch.cioproj.compute_cioproj
    _injected = {"n": 0}

    def _warn_injecting_cioproj(*args, **kwargs):
        series, meta = _real_cioproj_stage1(*args, **kwargs)
        meta["warning"] = "缺这些月份的源文件 [(2001, 5)] —— 本用例注入"
        _injected["n"] += 1
        return series, meta

    ch.cioproj.compute_cioproj = _warn_injecting_cioproj
    try:
        r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                        files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES,
                                        "application/zip")})
        check(r.status_code == 202, "注入 warning 的 zip 仍受理 -> 202", r.text[:200])
        warned_job = wait(r.json()["jobId"])
    finally:
        ch.cioproj.compute_cioproj = _real_cioproj_stage1
    check(_injected["n"] == 1, "阶段① 走的是被注入的 compute_cioproj（用例前提成立）",
          _injected["n"])
    check(warned_job["status"] == "completed", "注入 warning 的链路跑完",
          warned_job.get("error"))
    warn = warned_job.get("warning")
    check(bool(warn) and "缺这些月份" in warn,
          "阶段② 之后，阶段① 的 warning 仍在（没被覆盖成 None）", warn)

    # --- A5: 降级路径全部在受理阶段 400
    short = np.arange(111, dtype=np.float64)
    nan_series = np.array([1.0, 2.0, np.nan] * 40)
    const_series = np.full(2352, 3.0)
    only2001 = build_zip([p for p in NCS if ".2001" in p.name])
    bad_cases = [
        (dict(files={"file": ("readme.txt", b"hello")}, params={}),
         "只接受 .npy（CIO 序列）或 .zip（原始气象场）", "非 npy/zip"),
        (dict(files={"file": ("原始场.zip", ZIP_BYTES)}, params={}),
         "判不出输入类别", "zip 名判不出类别"),
        (dict(files={"file": ("cesm.sst.2000.zip", ZIP_BYTES)},
              params={"which": "sst"}), "SST 分支尚未实现", "which=sst"),
        (dict(files={"file": ("cesm.sst.2000.zip", ZIP_BYTES)},
              params={"which": "u850"}), "二者冲突", "which 与文件名冲突"),
        (dict(files={"file": ("cesm.uwnd.zip", only2001)}, params={"which": "u850"}),
         "zip 里没有 2000 年的 nc", "zip 缺目标年"),
        (dict(files={"file": ("CIO.npy", npy_bytes(short))}, params={"which": "cio"}),
         "不足一年", ".npy 短于一年"),
        (dict(files={"file": ("CIO.npy", npy_bytes(nan_series))}, params={"which": "cio"}),
         "包含 NaN 或无穷值", ".npy 含 NaN"),
        (dict(files={"file": ("CIO.npy", npy_bytes(const_series))}, params={"which": "cio"}),
         "为常数，无法执行 min-max 归一化", ".npy 为常数"),
        (dict(files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))}, params={"which": "u850"}),
         ".npy 入口应是已投影的一维 CIO", ".npy 却声明 u850"),
        (dict(files={"file": ("CIO.npy", b"")}, params={"which": "cio"}),
         "上传文件为空", "空文件"),
    ]
    for kwargs, want_msg, label in bad_cases:
        params = dict(kwargs.pop("params", {}))
        params.setdefault("year", 2000)
        params.setdefault("lead", "pre1")
        r = client.post("/api/chain/jobs", params=params, **kwargs)
        check(r.status_code == 400 and want_msg in r.text,
              "400：%s" % label, "%s %s" % (r.status_code, r.text[:160]))

    r = client.post("/api/chain/jobs",
                    files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES,
                                    "application/zip")})
    check(r.status_code == 202, "不带 year/lead（默认 2000/pre1）也受理", r.text[:160])
    default_params_job = wait(r.json()["jobId"])          # 必须等完，否则占着单 worker
    check(default_params_job["status"] == "completed",
          "默认参数的链路任务跑完", default_params_job.get("error"))

    for params, want, label in (
            ({"year": 2001, "lead": "pre1"}, "pre1/2001", "year=2001"),
            ({"year": 2000, "lead": "pre2"}, "pre2/2000", "lead=pre2"),
            ({"year": 2001, "lead": "pre1", "onlyProjection": "true"},
             "pre1/2001", "year=2001 + onlyProjection（不放宽）"),
            ({"year": 2000, "lead": "pre2", "onlyProjection": "true"},
             "pre2/2000", "lead=pre2 + onlyProjection（不放宽）")):
        r = client.post("/api/chain/jobs", params=params,
                        files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
        # 口径改了（import-all-models §4.1/§4.2）：不再是"唯一规则表"那句，
        # 而是查可用性 —— 文案含组合标识与原因，且与"格式非法"分开。
        check(r.status_code == 400 and want in r.text
              and "格式" not in r.text,
              "400：%s（组合不可用文案）" % label,
              "%s %s" % (r.status_code, r.text[:160]))

    # 被拒的任务不留目录：A1 zip 全链 + A2 npy 全链 + A3 只跑投影 + 缺月 zip
    # + 无参数 = 5
    leftovers = sorted(p.name for p in JOB_ROOT.iterdir())
    check(len(leftovers) == 5, "五次成功受理 = 5 个目录，被拒的都不留店",
          leftovers)

    # ================== A7. 权重组清单与放开受理（import-all-models，AC1/AC3/AC4）
    #
    # 判据来源：`.harness/spec/changes/import-all-models/` 的 tasks.md §1/§2/§3/§4。
    # **组数是动态的**（权重仍在下载）：下面一律实测，不写死任何组数。
    print("A7. 权重组清单与放开受理（import-all-models）")
    _a7_path_markers = ("C:", "D:", ":\\", "\\\\")

    # --- A7.1 清单端点：形状 / 不变量 / 无路径
    r = client.get("/api/chain/models")
    check(r.status_code == 200, "GET /api/chain/models -> 200", r.text[:200])
    catalog = r.json()
    check_no_leak(catalog, "清单响应里没有任何磁盘路径（递归扫描）")
    _leads = {item["lead"]: item for item in catalog.get("leads", [])}
    check(sorted(_leads) == ["pre1", "pre10", "pre15", "pre3", "pre5"],
          "清单覆盖 4 个 lead + 冻结归档 pre1", sorted(_leads))
    _GRID_YEARS = [2000, 2002, 2003, 2005, 2007, 2008, 2009, 2010, 2012]
    for _lead in ("pre3", "pre5", "pre10", "pre15"):
        check([y["year"] for y in _leads[_lead]["years"]] == _GRID_YEARS,
              "%s 列出 9 个候选年" % _lead,
              [y["year"] for y in _leads[_lead]["years"]])
    _p1 = _leads["pre1"]["years"]
    check([y["year"] for y in _p1] == [2000] and _p1[0]["source"] == "archive",
          "pre1 只有冻结归档 2000（source=archive）", _p1)

    _mismatch, _missing_reason, _n_avail = [], [], 0
    for _item in catalog["leads"]:
        for _y in _item["years"]:
            _live = lp.resolve_model_group(_item["lead"], _y["year"])
            if bool(_live["available"]) != bool(_y["available"]):
                _mismatch.append((_item["lead"], _y["year"],
                                  _y["available"], _live["available"]))
            if _y["available"]:
                _n_avail += 1
            elif not _y.get("reason"):
                _missing_reason.append((_item["lead"], _y["year"]))
    check(not _mismatch,
          "清单 available ⇔ resolve_model_group 逐组相等（唯一谓词）", _mismatch)
    check(not _missing_reason,
          "不可用项一律带 reason（前端禁选项的提示文案）", _missing_reason)
    check(_n_avail > 0, "至少有一组可用（数量动态，不写死）", _n_avail)
    _avail_pair = next(
        ((i["lead"], y["year"])
         for i in catalog["leads"] for y in i["years"] if y["available"]), None)
    # 原写法 `_avail_pair != ("pre1", 2000) or True` 恒真（不论取到什么都是 True），
    # 是"断言群恒真"那类无牙检查，改为对生成器耗尽这一真实失败模式取值。
    check(_avail_pair is not None,
          "取到一组可用组合（数量动态，不写死）", _avail_pair)

    # --- A7.2 组合不可用 => 400（文案含组合标识 + 原因，且不含路径）
    _UNAVAIL = ("pre3", 2011)                # 该年不在候选网格里，恒不可用
    r = client.post("/api/chain/jobs?year=%d&lead=%s&which=cio" % (_UNAVAIL[1], _UNAVAIL[0]),
                    files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
    check(r.status_code == 400
          and ("%s/%d" % _UNAVAIL) in r.text
          and not any(m in r.text for m in _a7_path_markers),
          "400：不可用组合带组合标识与原因（无绝对路径）",
          "%s %s" % (r.status_code, r.text[:200]))
    # 4.2：格式非法与"组合不可用"必须是两句不同的话
    r = client.post("/api/chain/jobs?year=2000&lead=prex&which=cio",
                    files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
    check(r.status_code == 400 and "格式" in r.text,
          "400：lead 格式非法走另一句文案", "%s %s" % (r.status_code, r.text[:160]))
    r = client.post("/api/chain/jobs?year=2000&lead=pre2&which=cio",
                    files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
    check(r.status_code == 400 and "pre2/2000" in r.text and "格式" not in r.text,
          "400：lead 合法但组合不可用 -> 组合文案（不是格式文案）",
          "%s %s" % (r.status_code, r.text[:160]))

    # --- A7.3 可用组合：受理通过 + 落盘 + 下载文件名跟随实际 (lead, year)
    #
    # ⚠ 这一单同时是 **AC4「技巧检验对每组都取到 Dataset/{lead}/{year} 真值」**
    # 的锚点（C012），所以发单前先把 `TRUTH_DIR` **临时摘掉**：
    # 套件全局把它指向临时夹具根（:218），而夹具只有 pre1/2000 一份（:1283）——
    # 照原样发这条 job，它**永远查不到实况**（available=False），AC4 在套件里
    # 结构上无从钉起。摘掉后回落到 `verification._truth_roots()` 的真实搜索根，
    # 这一单才真的跟 `Dataset/pre3/2012/` 的真件对拍（跑完立刻在 finally 里还回去）。
    _truth_expect = verification.PROJECT_ROOT / "Dataset" / "pre3" / "2012"
    _truth_on_disk = (sorted(p.name for p in _truth_expect.glob("*_real2.npy"))
                      if _truth_expect.is_dir() else [])
    check(bool(_truth_on_disk),
          "A7.3 前提：Dataset/pre3/2012/ 下有实况件（AC4 的对照盘面就位）",
          _truth_expect)
    _saved_truth_dir = os.environ.pop("TRUTH_DIR", None)
    try:
        r = client.post("/api/chain/jobs?year=2012&lead=pre3&which=cio",
                        files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
        check(r.status_code == 202, "A7：可用组合 (pre3,2012) 受理 -> 202", r.text[:160])
        a7_job = wait(r.json()["jobId"])
    finally:
        if _saved_truth_dir is not None:
            os.environ["TRUTH_DIR"] = _saved_truth_dir
    check(a7_job.get("status") == "completed", "A7：(pre3,2012) 链路跑完",
          a7_job.get("error"))

    # AC4（C012）：技巧检验取到的那份真值必须**归属本任务请求的那一组**，
    # 不是写死/串到别的组（pre1/2000 是最危险的那个"默认值"——pre1/2000 的
    # 既有用例 :1289 逐字不动，它对本条要抓的错**结构上无区分力**：请求的恰好
    # 就是 pre1/2000）。判据全部是**结构性**的：目录归属 + 盘上文件名集合；
    # `truthFile` 取自进程内任务态（绝对路径不出公开响应，那里只有 basename）。
    _ver7 = (a7_job.get("result") or {}).get("verification") or {}
    _truth_file = str(ch._get_job(a7_job["jobId"]).get("truthFile") or "")
    _truth_dir7 = Path(_truth_file).resolve().parent if _truth_file else None
    print("      [i] A7.3b (pre3,2012) 实况来源 = %s" % (_truth_file or "<无>"))
    check(_ver7.get("available") is True,
          "A7.3b：(pre3,2012) 真取到了实况并算了 r（不是静默跳过/静默降级）",
          _ver7)
    check(_truth_dir7 == _truth_expect.resolve(),
          "A7.3b：实况件归属 Dataset/pre3/2012/（串到别组即红：不是 pre1/2000）",
          _truth_file)
    check(_ver7.get("truthName") == Path(_truth_file).name
          and _ver7.get("truthName") in _truth_on_disk,
          "A7.3b：truthName 是 Dataset/pre3/2012/ 盘上真在的那一个",
          (_ver7.get("truthName"), _truth_on_disk[:2]))
    # 内容级：上屏的实况序列必须逐点等于**该组盘上真件**本身。对照物取自
    # `Dataset/pre3/2012/`（**不是** job 自己报的那个 `truthFile`），所以这条
    # 同样是跨组判据：串到别组就会拿另一组的场来比，逐点对不上。
    _grid7, _expect7 = {}, None
    _r7g = client.get("%s/%s/grid?i=40&j=50" % (CHAIN_PREFIX, a7_job["jobId"]))
    _disk7 = (_truth_expect / _truth_on_disk[0]) if _truth_on_disk else None
    if _r7g.status_code == 200 and _disk7 is not None and _disk7.is_file():
        _grid7 = _r7g.json()
        _real7 = np.load(_disk7, mmap_mode="r", allow_pickle=False)
        _expect7 = [None if not np.isfinite(v) else float(v)
                    for v in np.asarray(_real7[40, 50, :], dtype=np.float64)]
    check(_expect7 is not None and _grid7.get("truth") == _expect7
          and _grid7.get("truthName") == _ver7.get("truthName"),
          "A7.3b：/grid 上屏的实况逐点等于 Dataset/pre3/2012/ 盘上真件"
          "（对照物独立于 job 自报的 truthFile）",
          (_r7g.status_code, _grid7.get("truthName"), _truth_file))

    _p = a7_job.get("projection") or {}
    check(_p.get("lead") == 3 and _p.get("year") == 2012,
          "A7：projection 落库实际 (lead, year)", (_p.get("lead"), _p.get("year")))
    _norm = a7_job.get("normalization") or {}
    # 21 年（2000–2020）各 112 天 ⇒ 2012 的窗口起点 = 12 × 112
    check(_norm.get("windowStart") == 12 * 112 and _norm.get("targetYear") == 2012,
          "A7：窗口按目标年定位（windowStart=1344 / targetYear=2012）",
          {k: _norm.get(k) for k in ("windowStart", "targetYear", "selectedRange")})
    check(_norm.get("mode") == "training-equivalent" and not _norm.get("warning"),
          "A7：目标年仍落训练基准内 -> mode 不变、无 warning",
          (_norm.get("mode"), _norm.get("warning")))
    for _tail, _want in (("prediction", "prediction_pre3_2012.npy"),
                         ("normalized", "normalized_window_pre3_2012.npy"),
                         ("cio", "projected_cio_pre3_2012.npy")):
        _r7 = client.get("%s/%s/download/%s" % (CHAIN_PREFIX, a7_job["jobId"], _tail))
        _cd = _r7.headers.get("content-disposition", "")
        check(_r7.status_code == 200 and _want in _cd,
              "A7：download/%s 文件名跟随实际组合" % _tail,
              "%s %s" % (_r7.status_code, _cd))
    # 4.5：pre1/2000 上逐字不变
    _r7 = client.get("%s/%s/download/prediction"
                     % (CHAIN_PREFIX, default_params_job["jobId"]))
    check("prediction_pre1_2000.npy" in _r7.headers.get("content-disposition", ""),
          "A7：pre1/2000 下载文件名逐字不变",
          _r7.headers.get("content-disposition"))

    # --- A7.4 接线（P0）：推理函数必须收到**请求的** (lead, year)，不是默认值
    _seen_infer = {}

    def _capturing_inference(cio, progress_callback=None, span=(3, 95), **kw):
        _seen_infer.update(kw)
        return fake_inference(cio, progress_callback, span)

    lp._run_archive_inference = _capturing_inference
    try:
        r = client.post("/api/chain/jobs?year=2012&lead=pre3&which=cio",
                        files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
        _cap_job = wait(r.json()["jobId"])
    finally:
        lp._run_archive_inference = fake_inference
    check(_cap_job.get("status") == "completed", "A7：接线用例跑完",
          _cap_job.get("error"))
    check(_seen_infer.get("lead") == "pre3" and _seen_infer.get("year") == 2012,
          "A7：推理入口收到 (lead=pre3, year=2012) —— 漏传即静默回退 pre1/2000",
          _seen_infer)

    # --- A7.5 .npy 入口：均匀年表 + 越界年被拒（受理期，不留到后台）
    for _bad_year in (1990, 2021):
        _d7 = TMP_BASE / ("a7_npy_%d" % _bad_year)
        _d7.mkdir(parents=True, exist_ok=True)
        try:
            ch._intake_npy(npy_bytes(GOLD_ARR), _d7, "cio", _bad_year, 3)
            _ok7, _msg7 = False, "没有抛错"
        except Exception as _exc7:          # noqa: BLE001
            _ok7 = getattr(_exc7, "status_code", None) == 400
            _msg7 = str(getattr(_exc7, "detail", _exc7))
        check(_ok7, ".npy 受理：year=%d 被 400（不在序列年表里）" % _bad_year, _msg7)

    # --- A7.6 按年份定位（3.1/3.2/3.5）：2020 必须能取到，AC2 口径不变
    _dpy = ((default_params_job.get("projection") or {}).get("daysPerYear"))
    _src_dir = JOB_ROOT / default_params_job["jobId"]
    _series21 = np.load(_src_dir / "series.npy", allow_pickle=False)
    _dates21 = cd._read_json(_src_dir / "dates.json")
    check(bool(_dpy) and isinstance(_series21.size, int) and _series21.size == 2352,
          "A7.6 前提：21 年真件序列 2352 点 + 投影年表在手",
          (_series21.size, None if not _dpy else len(_dpy)))
    check(lp.locate_year_window({2000 + k: 112 for k in range(21)}, 2020, 2352) == 2240,
          "A7.6：逐年累加定位 2020 -> 2240",
          lp.locate_year_window({2000 + k: 112 for k in range(21)}, 2020, 2352))
    _sel20, _meta20 = lp._prepare_cio(
        _series21, calendar={"daysPerYear": _dpy, "year": 2020})
    check(_meta20["windowStart"] == 2240 and _meta20["targetYear"] == 2020,
          "A7.6：2020 的窗口起点 = 2240（首个落在基准之外的年份）",
          {k: _meta20.get(k) for k in ("windowStart", "targetYear", "mode")})
    check(_meta20["mode"] == "extended-base" and bool(_meta20.get("warning"))
          and "2240" in _meta20["warning"] and "2020" in _meta20["warning"],
          "A7.6：扩基准必须留痕（mode=extended-base + warning 写明 2240/2020）",
          (_meta20.get("mode"), _meta20.get("warning")))
    check(_sel20.size == 112 and _sel20.dtype == np.float32,
          "A7.6：摘出的仍是 112 天 float32", (_sel20.size, _sel20.dtype))
    check(_dates21[2240] == "2020-05-29" and _dates21[2351] == "2020-09-28",
          "A7.6：2020 窗口 = 2020-05-29 … 2020-09-28（pre1 口径）",
          (_dates21[2240], _dates21[2351]))
    # 前序年缺月：只平移 + warning，不阻断（3.4）
    _shift = dict(_dpy)
    _shift[2005] = 56
    check(lp.locate_year_window(_shift, 2020, 2352) == 2240 - 56,
          "A7.6：前序年缺月只让起点平移（2020 -> 2184）",
          lp.locate_year_window(_shift, 2020, 2352))
    _sel_s, _meta_s = lp._prepare_cio(
        _series21, calendar={"daysPerYear": _shift, "year": 2020})
    check("2005" in (_meta_s.get("warning") or "")
          and _meta_s["windowStart"] == 2240 - 56,
          "A7.6：前序年缺月列出该年、不阻断（warning 非空）",
          (_meta_s.get("windowStart"), _meta_s.get("warning")))
    # 目标年自身不足 112 天：硬拒（3.1）
    _bad = dict(_dpy)
    _bad[2020] = 84
    try:
        lp.locate_year_window(_bad, 2020, 2352)
        _ok_bad, _msg_bad = False, "没有抛错"
    except ValueError as _exc_bad:
        _ok_bad, _msg_bad = "2020" in str(_exc_bad), str(_exc_bad)
    check(_ok_bad, "A7.6：目标年只有 84 天 -> ValueError（不补齐、不降级）", _msg_bad)

    # --- A7.7 目录分支：缺件错误指向**该组目录**（三参形式，不泄绝对路径）
    # ⚠ 这里必须用 `real_inference`（本文件开头留的**原始**函数对象），不能用
    # `lp._run_archive_inference` —— A 段前面几节把后者换成了打桩推理，
    # 打桩对谁都不抛错，断言会变成恒真的"没有抛错"。
    _real_roots = lp.MODEL_ROOTS
    lp.MODEL_ROOTS = [TMP_BASE / "a7_no_such_root"]
    try:
        try:
            real_inference(np.zeros(lp.N_STEPS, np.float32),
                           lead="pre10", year=2011)
            _ok7b, _msg7b = False, "没有抛错"
        except FileNotFoundError as _exc7b:
            _safe = lp._safe_exc(_exc7b)
            _ok7b = ("2011" in _safe
                     and not any(m in _safe for m in _a7_path_markers))
            _msg7b = _safe
        except Exception as _exc7b:          # noqa: BLE001
            _ok7b, _msg7b = False, repr(_exc7b)
    finally:
        lp.MODEL_ROOTS = _real_roots
    check(_ok7b, "A7.7：目录源缺件 -> 三参 FileNotFoundError（文案无绝对路径）",
          _msg7b)
    # 外部根掉线（OSError）不得冒泡成 5xx（D4/R6）
    _missing_root = TMP_BASE / "a7_missing_root"
    lp.MODEL_ROOTS = [_missing_root]
    try:
        _live7 = lp.resolve_model_group("pre3", 2000)
    finally:
        lp.MODEL_ROOTS = _real_roots
    check(_live7["available"] is False and bool(_live7.get("reason")),
          "A7.7：根不存在 -> 优雅不可用（available=False + reason），不抛 OSError",
          _live7)

    # --- A7.8 目录分支共用的内层循环 + 尾部完整性检查（2.2/2.4）+ 组标识交叉断言（2.6）
    import tarfile as _tar7                              # noqa: PLC0415
    _grp = TMP_BASE / "a7_model_root" / "pre10" / "2011"
    _grp.mkdir(parents=True, exist_ok=True)
    _picked = 0
    with _tar7.open(lp.MODEL_ARCHIVE, mode="r|gz") as _tf7:
        for _m7 in _tf7:
            if not _m7.isfile() or not _m7.name.endswith(".pt"):
                continue
            _leaf7 = _m7.name.replace("\\", "/").rsplit("/", 1)[-1]
            if not lp.GROUP_MEMBER_RE.match(_leaf7):
                continue
            (_grp / _leaf7).write_bytes(_tf7.extractfile(_m7).read())
            _picked += 1
            if _picked >= 2:
                break
    check(_picked == 2, "A7.8 前提：从归档里取出 2 个真 checkpoint 当目录源", _picked)
    _real_resolver = lp.resolve_model_group
    _descriptor = {"lead": "pre10", "year": 2011, "available": True,
                   "reason": None, "source": "dir", "files": 2, "bytes": 0,
                   "directory": _grp}
    lp.resolve_model_group = lambda lead, year: dict(_descriptor)
    try:
        try:
            real_inference(np.zeros(lp.N_STEPS, np.float32),
                           lead="pre10", year=2011)
            _ok8, _msg8 = False, "没有抛错"
        except RuntimeError as _exc8:
            _ok8, _msg8 = "不完整" in str(_exc8), str(_exc8)
        except Exception as _exc8:          # noqa: BLE001
            _ok8, _msg8 = False, repr(_exc8)
        check(_ok8, "A7.8：目录分支下尾部完整性检查仍生效（2/8181 -> RuntimeError）",
              _msg8)
        # 2.6：实际加载的组标识 == 请求的 (lead, year)，不等即 RuntimeError
        lp.resolve_model_group = lambda lead, year: dict(
            _descriptor, lead="pre3", year=2000)
        try:
            real_inference(np.zeros(lp.N_STEPS, np.float32),
                           lead="pre10", year=2011)
            _ok8b, _msg8b = False, "没有抛错"
        except RuntimeError as _exc8b:
            _txt8b = str(_exc8b)
            _ok8b = ("pre10" in _txt8b and "11" in _txt8b
                     and "pre3" in _txt8b and "2000" in _txt8b)
            _msg8b = _txt8b
        except Exception as _exc8b:          # noqa: BLE001
            _ok8b, _msg8b = False, repr(_exc8b)
        check(_ok8b, "A7.8：组标识不一致 -> RuntimeError（文案含两边组合）", _msg8b)
    finally:
        lp.resolve_model_group = _real_resolver

    # ============================================ B. 与旧路由逐位对拍
    print("B. 数值一致性：新链 vs /api/predict/jobs（打桩边界）")
    legacy_job_dir = None

    CAPTURED.clear()
    r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                    files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES, "application/zip")})
    chain_job = wait(r.json()["jobId"], timeout=3600 if E2E else 300)
    chain_cio = CAPTURED.get("cio")
    chain_pred = np.load(JOB_ROOT / chain_job["jobId"] / "prediction.npy",
                         allow_pickle=False)
    check(chain_job["status"] == "completed",
          "新链跑完", chain_job.get("error"))
    if B_STUB:
        check(chain_cio is not None and chain_cio.shape == (112,)
              and chain_cio.dtype == np.float32,
              "桩模式下抓到喂给模型的 (112,) float32 张量",
              None if chain_cio is None else (chain_cio.shape, chain_cio.dtype))
    else:
        check(chain_cio is None, "真推理模式下没有桩捕获（走的是真模型）", chain_cio)

    CAPTURED.clear()
    r = legacy_client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                           files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES,
                                           "application/zip")})
    check(r.status_code == 202, "旧路由建任务 -> 202",
          "%s %s" % (r.status_code, r.text[:200]))
    legacy_job = wait(r.json()["jobId"], prefix="/api/predict/jobs",
                      timeout=3600 if E2E else 300)
    check(legacy_job.get("status") == "completed", "旧路由任务跑完",
          "%s %s" % (legacy_job.get("status"), legacy_job.get("error")))

    # =========================================== A6. 终态响应无绝对路径（A/B 组判别用例）
    # 第 5 轮返工。**为什么必须扫终态、而不是扫 202**：
    #   202 的 `projection` 是受理期瘦身版（`.zip` 入口只回 matSource 一族），
    #   `matPath` / `sandboxRoot` 是 `_run_job` 把 `prepare()` 返回的**完整**
    #   projection 写回 job 之后才出现的（live_prediction.py:717-723），而
    #   `_public_job` 把 `projection` 当**整体放行的容器键**（lp:97 / cd:80）。
    #   → 泄漏**只**在终态响应上。扫 202 会写出一条永远绿的假断言。
    # **为什么以前一直漏掉**（不是"扫得不够深"，是"扫的对象不对"）：
    #   本套件此前扫过的旧路由任务都是 **.npy** 入口（`projection` 只有
    #   `{"inputKind": "cio-npy"}`，结构上就没有 matPath）；而 `/api/cio/jobs`
    #   在本套件里**从未被调用过**。这个 `.zip` 旧路由任务一直没进扫描名单。
    def _terminal_no_abs_path(job, label, want_mat_name):
        """终态响应：既不能漏绝对路径，也不能把资产身份行改没了。"""
        check_no_leak(job, "%s：终态响应递归扫不到任何绝对路径" % label)
        proj = job.get("projection") or {}
        check(proj.get("matSource") == ("real" if IS_REAL else "mock")
              and proj.get("matPath") == want_mat_name,
              "%s：资产身份行照常显示（matSource + 纯文件名）" % label,
              {k: proj.get(k) for k in ("matSource", "matPath")})
        check("sandboxRoot" not in proj,
              "%s：sandboxRoot 键已删（全仓零消费方，无需保留）" % label,
              sorted(proj))

    _terminal_no_abs_path(legacy_job, "旧路由 /api/predict/jobs（.zip 入口）", MAT.name)

    r = cio_client.post("/api/cio/jobs?lead=1&which=u850",
                        files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES,
                                        "application/zip")})
    check(r.status_code == 202, "/api/cio/jobs 建任务 -> 202", r.text[:200])
    cio_job = wait(r.json()["jobId"], prefix="/api/cio/jobs")
    check(cio_job.get("status") == "completed", "/api/cio 任务跑完",
          "%s %s" % (cio_job.get("status"), cio_job.get("error")))
    _terminal_no_abs_path(cio_job, "/api/cio/jobs（.zip 入口）", MAT.name)
    if cio_job.get("status") == "completed":
        shutil.rmtree(cd.JOB_ROOT / cio_job["jobId"], ignore_errors=True)

    legacy_cio = CAPTURED.get("cio")
    legacy_job_dir = lp.JOB_ROOT / legacy_job["jobId"]
    legacy_pred = np.load(legacy_job_dir / "prediction.npy", allow_pickle=False)

    if chain_cio is not None and legacy_cio is not None:
        check(chain_cio.shape == legacy_cio.shape
              and chain_cio.dtype == legacy_cio.dtype == np.float32
              and np.array_equal(chain_cio, legacy_cio, equal_nan=False),
              "喂进模型的 (112,) float32 张量逐位一致",
              (chain_cio.shape, chain_cio.dtype, legacy_cio.dtype))
    check(chain_pred.shape == legacy_pred.shape
          and chain_pred.dtype == legacy_pred.dtype
          and np.array_equal(chain_pred, legacy_pred, equal_nan=False),
          "prediction.npy 逐位一致（验收 1：%s）" % ("真推理" if E2E else "桩"),
          (chain_pred.shape, chain_pred.dtype, legacy_pred.dtype))
    if E2E:
        print("    （真推理：新链 vs 旧路由各跑一次，逐位比）")
    shutil.rmtree(legacy_job_dir, ignore_errors=True)
    shutil.rmtree(JOB_ROOT / chain_job["jobId"], ignore_errors=True)

    # C/D 段固定打桩：这两段验的是编排、调用次数与端点契约，不是模型输出；
    # 打桩才能秒级跑完。真推理只在 B（CHAIN_E2E=1）与 E（CHAIN_E2E=1）里发生。
    print("C. 官方件对拍与「全链只归一化一次」（打桩推理）")
    CAPTURED.clear()
    calls = {"n": 0}
    real_prepare = lp._prepare_cio
    # ⚠ 快照取自 `_prepare_cio` 返回处，而 `np.save(normalized_window.npy)` 在
    # **其后**才执行（chain.py：先 `_prepare_cio` 再 `np.save`）—— 所以下面那条
    # "③ 开跑前一次都没被读过"的断言**是恒真的**，它证明不了②→③ 的交接
    # （R-1，已如实标注）。本段真正钉住行为的是 `read_counter` 的**读取计数**，
    # 它也是 C-1 的唯一判别器（删了变异体就会畅通）。
    read_after_stage2 = {"n": None}
    read_counter = {"n": 0}          # 在计数桩里前向引用，此处先建

    def counting_prepare(raw, pre_year=0, *, calendar=None, **kw):
        # `calendar` 是 import-all-models 新增的 keyword-only 形参（年表定位窗口）；
        # 计数桩必须原样**转发**，否则会悄悄按 `preYear` 老口径归一化，
        # 本段后面「与阶段③ 喂入的张量逐位一致」的断言就名不副实了。
        calls["n"] += 1
        out = real_prepare(raw, pre_year, calendar=calendar, **kw)
        read_after_stage2["n"] = read_counter["n"]
        return out

    lp._prepare_cio = counting_prepare
    try:
        stage3_saw_file = {"value": None}
        stub_loaded = {"value": None}
        # C-7 的观测点：`state` = 真正写进任务状态的 progress（判别项）；
        # `callback` = 回调给的原始值（留作对照，单独看它区分不出二次映射）。
        stage3_progress = {"state": [], "callback": []}
        # 判别性探针（C-1）：光比"值"区分不了"读落盘件"与"内存传值"——
        # 内存值与落盘件本就逐位相同。这里数**读取行为**：本任务目录下
        # `normalized_window.npy` 被 `np.load` 读了几次。
        #   正确实现（③ np.load 落盘件）：1
        #   变异体（③ 收内存值、落盘照常）：0 —— 值比对对它恒真，只有计数能钉住
        # 作用域限定**本任务目录**，不用全局 rglob：别的任务的同名文件不算数。
        read_scope = {"file": None}
        np_load_real = np.load

        def counting_load(path, *args, **kwargs):
            target = read_scope["file"]
            if target is not None:
                try:
                    if Path(str(path)).resolve() == target:
                        read_counter["n"] += 1
                except (OSError, ValueError):
                    pass
            return np_load_real(path, *args, **kwargs)

        real_infer_for_probe = lp._run_archive_inference

        def probing_inference(cio, progress_callback=None, span=(3, 95), **kw):
            # 桩自己读本任务的落盘件：③ 的执行边界内该读什么就读什么
            path = JOB_ROOT / read_scope["jobId"] / "normalized_window.npy"
            on_disk = (np_load_real(path, allow_pickle=False)
                       if path.is_file() else None)
            stub_loaded["value"] = None if on_disk is None else on_disk.size
            # 兼容变异体签名：cio 为 None 时形状比对无意义，判 False 即可
            try:
                got = np.asarray(cio, dtype=np.float32)
            except (TypeError, ValueError):
                got = None
            stage3_saw_file["value"] = (
                on_disk is not None and got is not None
                and np.array_equal(got, on_disk)
            )
            return fake_inference(cio, progress_callback, span)

        def stage3_reporter(cio, progress_callback=None, span=(3, 95), **kw):
            def spy(progress, stage):
                # ⚠ 这里抓到的是**回调给的原始值**（span 两端 33/97），不是
                # `_run_stage3` 最终写进任务状态的值 —— 拿它断言会漏掉二次映射
                # （实测：只回退 C-7 一行的 mut3 曾在本断言下"全部通过"）。
                # 所以真实判别项是下面 wrap 任务状态写入的 `spy_update_job`。
                stage3_progress["callback"].append((progress, stage))
                if progress_callback:
                    progress_callback(progress, stage)

            return probing_inference(cio, spy if progress_callback else None, span)

        # 观测点放在**任务状态的写入处**：不管 `_run_stage3` 内部怎么映射，
        # `_update_job(progress=...)` 收到的才是前端**真正看到**的值。
        # 连 stage 文案一起记，才能把 ③ 段的写入从全链里切出来（否则末尾的
        # 98/100 会被算进来，max 永远到不了 97）。
        # C-7 的判别就靠它：修好的实现写 33…97；二次映射写 54…95。
        real_update_job = ch._update_job

        def spy_update_job(job_id, **changes):
            if "progress" in changes:
                stage3_progress["state"].append(
                    (int(changes["progress"]), changes.get("stage")))
            return real_update_job(job_id, **changes)

        ch._update_job = spy_update_job
        lp._run_archive_inference = stage3_reporter
        np.load = counting_load
        try:
            r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                            files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES,
                                            "application/zip")})
            jid = r.json()["jobId"]
            read_scope["jobId"] = jid
            read_scope["file"] = (JOB_ROOT / jid
                                  / "normalized_window.npy").resolve()
            job = wait(jid)          # 轮询留痕进 POLLS（C-7 的进度证据）
        finally:
            np.load = np_load_real
        lp._run_archive_inference = fake_inference
        check(job["status"] == "completed", "计数任务跑完", job.get("error"))
        check(calls["n"] == 1, "全链中 _prepare_cio 恰好被调用 1 次", calls["n"])
        check(read_counter["n"] == 1,
              "阶段③ 确实读落盘件：本任务目录 normalized_window.npy 被读 1 次",
              read_counter["n"])
        # ⚠ 恒真的伴随断言（R-1）：快照点在 `_prepare_cio` 返回处，那时
        # `np.save` 还没跑，文件压根不存在 —— 这一条与实现无关，永远为 0。
        # 保留只为把"它恒真"写在套件里（真实判别项是上面那条计数）。
        check(read_after_stage2["n"] == 0,
              "（恒真，仅存证）② 快照点早于 np.save，该文件此刻必未创建",
              read_after_stage2["n"])
        check(stub_loaded["value"] == 112,
              "桩推理自己按路径读到落盘件（112 点）",
              stub_loaded["value"])
        check(stage3_saw_file["value"] is True,
              "阶段③ 收到的张量与落盘件逐位一致（③ 的输入来自② 的落盘件）")

        # --- B-3/C-7：进度条的**绝对区间**（不是"倒退"）。
        # ③ 段区间是 INFERENCE_SPAN=(33,97)，而 `_run_archive_inference` 的回调
        # 给的已经就是这个绝对值。`_run_stage3` 若再套一层 `_report(...)` 二次
        # 映射，实际只覆盖 33+int(33%*64)=54 … 33+int(97%*64)=95 → 终值**只有
        # 95**，且从② 段终点 30 直接跳到 54。
        prog = monotone_polls(jid)
        check(prog == sorted(prog, key=lambda x: x[1]),
              "进度条全链不倒退（折成单调递增后逐点相等）", prog[:12])
        # 判别项 = 写进任务状态的值。按**绝对区间**把 ③ 段的写入切出来：
        # 本段只允许写 [33, 97]，而本段之后的收尾写的是 98 / 100（各是一笔
        # 独立写入），因此不会混进来。两条断言互为姊妹：
        #   min == 33 → 起点没错位；max == 97 → 终值没被压在 95。
        # 修好的实现写 33…97；二次映射写 54…95（两条同时红）。
        # 不按 stage 文案切：打桩给的文案是"桩推理开始/结束"，真实推理的文案是
        # "正在推理格点 i/N" —— 按文案切等于把断言绑死在某一侧的实现上。
        stage3_state = [p for p, _s in stage3_progress["state"]
                        if INFERENCE_LO <= p <= INFERENCE_HI]
        check(bool(stage3_state) and min(stage3_state) == INFERENCE_LO,
              "阶段③ 写进状态的起点正好是绝对区间下界 %d（二次映射会变 54）"
              % INFERENCE_LO, stage3_progress["state"])
        check(bool(stage3_state) and max(stage3_state) == INFERENCE_HI,
              "阶段③ 写进状态的终值正好是绝对区间上界 %d（二次映射会停在 95）"
              % INFERENCE_HI, stage3_progress["state"])

        r = client.get("%s/%s/series" % (CHAIN_PREFIX, jid))
        check(r.status_code == 200, "/series -> 200", r.text[:120])
        series = r.json()
        check(set(series) == {"projectionMode", "calendarKnown", "dates",
                              "sampleIndices", "raw", "normalizedWindow"},
              "/series 字段名与旧端点逐字一致", sorted(series))
        raw = np.asarray(series["raw"], dtype=np.float64)
        diff = float(np.max(np.abs(raw - GOLD_ARR)))
        check(raw.size == GOLD_ARR.size and diff < 1e-6,
              "投影序列与官方 pre1 件最大差 < 1e-6（验收 2）", diff)
        check(len(series["normalizedWindow"]) == 112,
              "/series 的 normalizedWindow 为 112 点",
              len(series["normalizedWindow"]))

        # 未就绪语义 + 下载格式
        print("D. 未就绪语义与下载格式")
        r = client.get("/api/chain/jobs/nope/preview")
        check(r.status_code == 404 and "链路任务不存在或后端已重启" in r.text,
              "不存在的 jobId -> 404", "%s %s" % (r.status_code, r.text[:120]))

        r = client.get("%s/%s/download/cio" % (CHAIN_PREFIX, jid))
        saved = np.load(io.BytesIO(r.content))
        check(saved.shape == (1, 2352) and saved.dtype == np.float64,
              "download/cio（.zip 入口）为 (1,2352) float64",
              (saved.shape, saved.dtype))
        check(np.array_equal(saved.ravel(), raw),
              "download/cio 与 /series 的 raw 逐位一致")

        r = client.get("%s/%s/download/normalized" % (CHAIN_PREFIX, jid))
        norm_saved = np.load(io.BytesIO(r.content))
        check(norm_saved.shape == (112,) and norm_saved.dtype == np.float32,
              "download/normalized 为 (112,) float32",
              (norm_saved.shape, norm_saved.dtype))
        check(np.array_equal(norm_saved, lp._prepare_cio(raw)[0]),
              "download/normalized 与阶段③ 喂入的张量逐位一致（验收 2）")

        r = client.get("%s/%s/spectrum?confidence=0.8" % (CHAIN_PREFIX, jid))
        check(r.status_code == 400 and "confidence 只支持 0.90、0.95 或 0.99" in r.text,
              "非法 confidence -> 400", "%s %s" % (r.status_code, r.text[:120]))
        r = client.get("%s/%s/spectrum?confidence=0.95" % (CHAIN_PREFIX, jid))
        spec = r.json()
        check(r.status_code == 200 and spec.get("available") is True,
              "/spectrum 正常返回（验收 2）", r.text[:160])
        check(spec["blockCount"] == 21 and len(spec["periodDays"]) == len(spec["projectedPower"]),
              "/spectrum 为 21 个 112 天块",
              (spec.get("blockCount"), len(spec.get("periodDays", []))))
        check(spec["actualPower"] is None and spec["actualConfidence"] is None,
              "参考 PC 只到 2017，越界时整段不给参考线（不 500）",
              spec.get("actualPower") is not None)
        r = client.get("%s/%s/completeness" % (CHAIN_PREFIX, jid))
        comp = r.json()
        check(r.status_code == 200 and comp["calendarKnown"] is True
              and comp["completeBlocks"] == 21,
              "/completeness 为 21 个完整年", r.text[:160])
        r = client.get("%s/%s/series" % (CHAIN_PREFIX, jid))
        check(r.status_code == 200 and r.json()["projectionMode"] == "u850_only",
              "/series projectionMode=u850_only", r.status_code)
        # C-2：任务状态端点也不外泄服务器绝对路径。
        # 这里是 `jid`（C 段计数任务）的状态响应：它跑在实况件创建**之前**，
        # `verification.available=False`，**这份响应里没有实况路径可泄漏** ——
        # 所以它证明不了"有实况时也不泄漏"，那件事由 D 段带实况件的那份响应
        # （下面的 check_no_leak ③）负责（B-2 的第二重失效就在于此）。
        status_doc = client.get("%s/%s" % (CHAIN_PREFIX, jid)).json()
        check("jobDir" not in status_doc,
              "GET /jobs/{id} 不含 jobDir（服务器路径不外泄）",
              sorted(status_doc))
        check_no_leak(status_doc,
                      "GET /jobs/{id} 里没有任何字段带服务器绝对路径（递归扫描）")
        check(Path(status_doc["jobId"]).name == status_doc["jobId"],
              "jobId 是纯 hex，可直接拼进临时目录名", status_doc.get("jobId"))

        # --- D2: 受理态（stageIndex=0）下所有产物端点 -> 409 + 段级文案
        r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                        files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES,
                                        "application/zip")})
        fresh = r.json()["jobId"]
        for path, stage, want in (
                ("series", "①", "投影尚未完成"),
                ("completeness", "①", "投影尚未完成"),
                ("spectrum", "①", "投影尚未完成"),
                ("download/cio", "①", "投影尚未完成"),
                ("download/normalized", "②", "归一化尚未完成"),
                ("preview", "③", "降水推理尚未完成"),
                ("truth/preview", "③", "降水推理尚未完成"),
                ("grid", "③", "降水推理尚未完成"),
                ("pearson", "③", "降水推理尚未完成"),
                ("download/prediction", "③", "降水推理尚未完成")):
            resp = client.get("%s/%s/%s" % (CHAIN_PREFIX, fresh, path))
            check(resp.status_code == 409 and want in resp.text,
                  "未就绪 %s（阶段%s）-> 409 %s" % (path, stage, want),
                  "%s %s" % (resp.status_code, resp.text[:120]))

        r = client.get("%s/%s" % (CHAIN_PREFIX, fresh))
        check(int(r.json().get("stageIndex") or 0) == 0,
              "受理后 stageIndex=0", r.json().get("stageIndex"))

        pending = wait(fresh)
        check(pending["status"] == "completed" and pending["stageIndex"] == 3,
              "等待中的任务最终跑完", pending.get("error"))

        r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=cio",
                        files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
        npy_job = wait(r.json()["jobId"])
        r = client.get("%s/%s/download/cio" % (CHAIN_PREFIX, npy_job["jobId"]))
        npy_cio = np.load(io.BytesIO(r.content))
        check(npy_cio.shape == (1, GOLD_ARR.size) and npy_cio.dtype == np.float64,
              "download/cio（.npy 入口）为 (1, T) float64", npy_cio.shape)
        r = client.get("%s/%s/series" % (CHAIN_PREFIX, npy_job["jobId"]))
        check(r.json()["projectionMode"] == "uploaded_cio"
              and r.json()["calendarKnown"] is False,
              ".npy 入口 /series 的 projectionMode/calendarKnown 正确", r.json().get("projectionMode"))
        # 阶段① 卡片把 completeness 和 series 一起 Promise.all —— 这里 404 会撕掉整张卡
        r = client.get("%s/%s/completeness" % (CHAIN_PREFIX, npy_job["jobId"]))
        npy_comp = r.json() if r.status_code == 200 else {}
        check(r.status_code == 200 and npy_comp.get("calendarKnown") is False
              and npy_comp.get("completeBlocks") == GOLD_ARR.size // 112,
              ".npy 入口 /completeness 可用（受理即落盘）",
              "%s %s" % (r.status_code, r.text[:120]))

        # 路由表不含三个资产端点（它们仍在 /api/cio）
        paths = {p for route in app.routes if (p := getattr(route, "path", None))}
        check(not any("capabilities" in p or "reference" in p or "mode/u850" in p
                      for p in paths),
              "链路模块不重复实现 capabilities/reference/mode/u850",
              sorted(p for p in paths if p.startswith("/api/chain")))

        # --- 实况对照的链路前缀（有实况时才成立）
        # ⚠ 真值件的内容必须与被比较的任务**同源**，否则下面 `pred == truth` 恒假：
        #   `jid`（zip 入口）的序列是**现投影**，`.npy 入口` 的序列是官方金标准
        #   `GOLD_ARR`，两者实测 max|Δ| = 1.56e-07（同套件 C 段也只敢按 1e-6 判等）。
        #   归一化后这点差落在 float32 的最后一个 ulp（实测窗口 max|Δ| = 5.96e-08），
        #   `==` 必然为假 —— 与 import-all-models 无关（同一序列上，带不带 calendar
        #   的窗口逐位相同；归因探针见 .harness/tmp/import-all-models/_d2_probe.py）。
        #   改用 `npy_job`（与后面两个任务同为 GOLD_ARR 口径）造真值件，断言仍是
        #   严格 `==`，且钉住的语义不变：端点吐出的实况就是**写进文件的那份值**。
        day = TRUTH_ROOT / "pre1" / "2000"
        day.mkdir(parents=True, exist_ok=True)
        truth = np.tile(
            np.load(JOB_ROOT / npy_job["jobId"]
                    / "normalized_window.npy").astype(np.float64),
            (81, 101, 1))
        truth_file = "pre_1_2000(1)_20_40_100_125_0.25_real2.npy"
        np.save(day / truth_file, truth)
        r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=cio",
                        files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
        job = wait(r.json()["jobId"])
        ver = (job.get("result") or {}).get("verification") or {}
        check(ver.get("available") is True
              and ver.get("downloadUrl")
              == "/api/chain/jobs/%s/download/pearson" % job["jobId"],
              "有实况 -> verification.downloadUrl 指向链路前缀", ver)
        # C-2/B-3：**这份**才是带实况件的响应（verification.available=True）。
        # B-3 修好后这里**一个字面量都不豁免**。修法是"改存私有键"而不是
        # "出口处擦除"：实况绝对路径只存**任务顶层 `truthFile`**，该键不在
        # `_public_job()` 白名单里 → 结构上进不了响应。所以下面三条要一起成立：
        #   ① 响应里扫不到任何绝对路径（真断言，不再是 allowlist 安慰剂）；
        #   ② 进程内确实有 truthFile 且指向真文件 —— 证明①不是靠"把实况路径
        #      整个丢掉/换成 None"换来的（那会让两个 truth/preview 一起 409）；
        #   ③ 对外可见的 truthName/downloadUrl 原样保留，键集与改前逐键一致。
        # 缺②，一条"删掉整块 verification"的实现也能让①变绿。
        check_no_leak(job,
                      "有实况件的 GET /api/chain/jobs/{id} 里没有任何服务器绝对路径"
                      "（递归扫描，无豁免）")
        inner = ch._get_job(job["jobId"])
        check(isinstance(inner.get("truthFile"), str)
              and Path(inner["truthFile"]).is_file(),
              "链路：实况绝对路径改存进程内顶层 truthFile（消费者仍读得到）",
              inner.get("truthFile"))
        check("truthFile" not in job and "jobDir" not in job
              and "resultPath" not in job,
              "链路：truthFile/jobDir/resultPath 都不在公开响应里（白名单默认不公开）",
              sorted(job))
        check(set(ver) == VERIFICATION_KEYS,
              "链路 verification 键集与改前逐键一致（只少了 truthPath）",
              sorted(set(ver) ^ VERIFICATION_KEYS))
        check(ver.get("truthName") == truth_file and ver.get("downloadUrl"),
              "链路 verification.truthName/downloadUrl 原样保留（前端零改动）", ver)
        r = client.get("%s/%s/pearson?confidence=0.95" % (CHAIN_PREFIX, job["jobId"]))
        pe = r.json()
        check(r.status_code == 200 and pe["rows"] == 81 and pe["cols"] == 101
              and pe["data"][0][0] is not None,
              "/pearson 可用且 data 可 JSON 解析", r.status_code)
        r = client.get("%s/%s/truth/preview?time_index=0" % (CHAIN_PREFIX, job["jobId"]))
        check(r.status_code == 200 and r.json()["rows"] == 81,
              "链路 /truth/preview 仍然 200（B-3 最容易改崩的点）",
              "%s %s" % (r.status_code, r.text[:120]))
        r = client.get("%s/%s/grid?i=40&j=50" % (CHAIN_PREFIX, job["jobId"]))
        grid = r.json()
        check(r.status_code == 200 and len(grid["pred"]) == 112
              and len(grid["truth"]) == 112,
              "链路格点端点同时返回 112 步预测与 real2 实况", r.status_code)
        check(grid["pred"] == grid["truth"] and grid["truthName"] == truth_file,
              "链路格点实况来自本任务已校验的 real2 文件", grid.get("truthName"))
        check(grid["r"] is not None and abs(grid["r"] - 1.0) < 1e-6,
              "链路格点端点返回同口径 Pearson r", grid["r"])
        r = client.get("%s/%s/download/pearson" % (CHAIN_PREFIX, job["jobId"]))
        check(r.status_code == 200, "download/pearson 可下载", r.status_code)

        # --- 旧路由同口径：B-3 的源头在共享函数 `_verify_against_truth`，两个
        # 路由都会被它波及，所以两侧必须一起验。旧路由这一份是**改动的另一半**
        # 证据：只修链路会让两条路由口径分叉（B-3 明令禁止的那种修法）。
        r = legacy_client.post("/api/predict/jobs?year=2000&lead=pre1&which=cio",
                               files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
        check(r.status_code == 202, "旧路由建实况对照任务 -> 202", r.text[:160])
        legacy_truth_job = wait(r.json()["jobId"], prefix="/api/predict/jobs")
        lver = (legacy_truth_job.get("result") or {}).get("verification") or {}
        check(lver.get("available") is True and lver.get("truthName") == truth_file,
              "旧路由有实况 -> available/truthName 原样保留", lver)
        check(set(lver) == VERIFICATION_KEYS,
              "旧路由 verification 键集与改前逐键一致（只少了 truthPath）",
              sorted(set(lver) ^ VERIFICATION_KEYS))
        check_no_leak(legacy_truth_job,
                      "旧路由 GET /api/predict/jobs/{id} 同样扫不到任何绝对路径"
                      "（递归扫描，无豁免）")
        legacy_inner = lp._get_job(legacy_truth_job["jobId"])
        check(isinstance(legacy_inner.get("truthFile"), str)
              and Path(legacy_inner["truthFile"]).is_file(),
              "旧路由：实况绝对路径也改存进程内顶层 truthFile",
              legacy_inner.get("truthFile"))
        r = legacy_client.get("/api/predict/jobs/%s/truth/preview?time_index=0"
                              % legacy_truth_job["jobId"])
        check(r.status_code == 200 and r.json()["rows"] == 81,
              "旧路由 /truth/preview 仍然 200（改崩这里就是 409）",
              "%s %s" % (r.status_code, r.text[:120]))
        r = legacy_client.get("/api/predict/jobs/%s/grid?i=40&j=50"
                              % legacy_truth_job["jobId"])
        legacy_grid = r.json()
        check(r.status_code == 200 and len(legacy_grid["pred"]) == 112
              and len(legacy_grid["truth"]) == 112,
              "旧路由格点端点同时返回 112 步预测与 real2 实况", r.status_code)
        check(legacy_grid["pred"] == legacy_grid["truth"]
              and legacy_grid["truthName"] == truth_file,
              "旧路由格点实况来自本任务已校验的 real2 文件",
              legacy_grid.get("truthName"))
        check(legacy_grid["r"] is not None and abs(legacy_grid["r"] - 1.0) < 1e-6,
              "旧路由格点端点返回同口径 Pearson r", legacy_grid["r"])
        shutil.rmtree(lp.JOB_ROOT / legacy_truth_job["jobId"], ignore_errors=True)
        (day / truth_file).unlink(missing_ok=True)

        # ============================================ D3. 异常文本的出口净化
        # 追加要求（PM 批准）：`reason` / `error` 会把**异常文本**直接送进响应
        # （`error` 还会被前端上屏，见 R3-4）。`OSError.filename` 是服务器绝对
        # 路径，所以执行路径上的两处**总闸**与实况检验的 reason 都要过
        # `_safe_exc()`。
        #
        # ⚠ 这段一律打桩，**不指向 Dataset/ 下任何真实文件**：三个用例都是
        # "实况读失败/读坏"，拿真件去触发等于靠改真件——而且真件被改了别处会红。
        print("D3. 异常文本出口净化（打桩，不碰真实况）")
        real_find_truth = verification.find_truth
        real_archive_infer = lp._run_archive_inference
        no_path_markers = ("C:", "D:", ":\\", "\\\\")   # 硬字符串，不用正则

        def _verification_case(label, truth_path, want_shape_text=None):
            """把 find_truth 钉到指定件上跑一条链，返回 (job, reason)。"""
            verification.find_truth = lambda lead, year: truth_path
            try:
                r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=cio",
                                files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
                check(r.status_code == 202, "%s：受理 -> 202" % label,
                      "%s %s" % (r.status_code, r.text[:160]))
                job = wait(r.json()["jobId"])
            finally:
                verification.find_truth = real_find_truth
            ver = (job.get("result") or {}).get("verification") or {}
            reason = ver.get("reason") or ""
            check(job.get("status") == "completed" and ver.get("available") is False,
                  "%s：检验失败**不拖垮任务**（completed + available=False）" % label,
                  (job.get("status"), ver, job.get("error")))
            check(not any(m in reason for m in no_path_markers),
                  "%s：reason 里没有绝对路径标记（硬字符串检测）" % label, reason)
            check_no_leak(job, "%s：整份响应里没有绝对路径（递归扫描）" % label)
            if want_shape_text:
                check(want_shape_text in reason,
                      "%s：诊断信息没被误删（reason 含 %s）"
                      % (label, want_shape_text), reason)
            return job, reason

        # 这段只验**文本**，不验数值：强制打桩，免得真推理把 E 段拖长几分钟。
        lp._run_archive_inference = fake_inference
        try:
            # ① OSError 分支：真实存在的 OSError 才会带 `.filename`（绝对路径）
            missing = TMP_BASE / "no_such_dir" / "pre_1_2000_missing_real2.npy"
            _job, reason = _verification_case("实况文件不存在", missing)
            check(missing.name in reason and "FileNotFoundError" in reason,
                  "实况文件不存在：reason 保留了**文件名**（可诊断）", reason)

            # ② 真 0 字节实况件 -> EOFError。**必须是真 0 字节**：截断件
            #    （\x93NUMPY... 或缺 magic）抛的是 ValueError，早就被接住了，
            #    拿截断件测这一条会假绿（实测确认）。
            zero = TMP_BASE / "zero_real2.npy"
            zero.write_bytes(b"")
            _job, reason = _verification_case("0 字节实况件", zero)
            check("EOFError" in reason, "0 字节实况件：reason 点出 EOFError", reason)

            # ③ 形状不一致：ValueError，`str(exc)` 必须**原样保留** ——
            #    "(81, 101, 112) 与 (5, 5, 5) 不一致" 是最常用的诊断信息，
            #    无条件套 `[Errno None] None` 会把它毁掉。
            odd = TMP_BASE / "odd_real2.npy"
            np.save(odd, np.zeros((5, 5, 5), dtype=np.float64))
            _job, reason = _verification_case("实况件形状不一致", odd, "(5, 5, 5)")
            check("不一致" in reason, "形状不一致：ValueError 原文保留", reason)

            # ④ 两处**总闸**：任务执行期抛出的任意异常也走 `_safe_exc`。
            #    路径取 TMP_BASE 下的不存在文件，basename 唯一。
            boom_path = TMP_BASE / "missing_model_archive.tar.gz"

            def boom(*_a, **_kw):
                raise FileNotFoundError(2, "No such file or directory", str(boom_path))

            lp._run_archive_inference = boom
            for label, http, prefix in (
                    ("链路总闸", client, CHAIN_PREFIX),
                    ("旧路由总闸", legacy_client, "/api/predict/jobs")):
                r = http.post("%s?year=2000&lead=pre1&which=cio" % prefix,
                              files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
                job = wait(r.json()["jobId"], prefix=prefix)
                err = job.get("error") or ""
                check(job.get("status") == "failed", "%s：任务确实 failed" % label,
                      job.get("status"))
                check(boom_path.name in err, "%s：error 里保留了文件名" % label, err)
                check(not any(m in err for m in no_path_markers),
                      "%s：error 里没有绝对路径标记（硬字符串检测）" % label, err)
                check_no_leak(job, "%s：整份响应里没有绝对路径（递归扫描）" % label)

            # ⑤ **我们自己 `raise` 的异常**：单参 `FileNotFoundError(f"...{PATH}")`
            #    的 `.filename` 是 None，会绕过总闸的 `.filename` 分支走 else，
            #    把模型包的绝对路径原样吐进 `error`（该串前端上屏，R3-4）。
            #    这里让**真的** `_run_archive_inference` 跑：把 `MODEL_ARCHIVE`
            #    指到临时区的不存在文件，它在 `is_file()` 处立刻抛，不会真开包。
            #    ⚠ 打桩只碰临时区，不指向真实模型包。
            #    ⚠ 这里必须用 `real_inference`（:269 在**任何打桩之前**抓的原函数），
            #    不能用 D3 开头的 `real_archive_infer` —— 那个抓到的是当时的
            #    `fake_inference`，拿它"跑真函数"会让本条恒真（实测踩过）。
            bogus_archive = TMP_BASE / "no_such_model_archive.tar.gz"
            lp._run_archive_inference = real_inference
            real_model_archive = lp.MODEL_ARCHIVE
            lp.MODEL_ARCHIVE = bogus_archive
            try:
                for label, http, prefix in (
                        ("模型包缺失·链路", client, CHAIN_PREFIX),
                        ("模型包缺失·旧路由", legacy_client, "/api/predict/jobs")):
                    r = http.post("%s?year=2000&lead=pre1&which=cio" % prefix,
                                  files={"file": ("CIO.npy", npy_bytes(GOLD_ARR))})
                    job = wait(r.json()["jobId"], prefix=prefix)
                    err = job.get("error") or ""
                    check(job.get("status") == "failed",
                          "%s：任务 failed" % label, job.get("status"))
                    check(bogus_archive.name in err,
                          "%s：error 保留了模型包**文件名**（还能诊断）" % label, err)
                    check(not any(m in err for m in no_path_markers),
                          "%s：error 里没有绝对路径标记（硬字符串检测）"
                          % label, err)
                    check_no_leak(job, "%s：整份响应里没有绝对路径" % label)
            finally:
                lp.MODEL_ARCHIVE = real_model_archive
                lp._run_archive_inference = fake_inference

            # ⑥ **执行路径**上"路径只在 message 里、`.filename` 为空"的异常（T4-4）。
            #    `.filename` 分支天然接不住这一类：`cioproj.read_nc()` 聚合四个 nc
            #    后端后抛的是 `RuntimeError`（`.filename` 恒无），nc 的绝对路径写在
            #    message 里 → 走 else 原样透出 → `error` 前端上屏（R3-4 同一条链）。
            #    载体用**真**坏件：成员名/结构照旧，只把 nc 内容换成垃圾字节 ——
            #    受理与解压都只看名字，所以必然一路走到读 nc 才炸。
            def _bad_input_case(label, payload):
                """投一份必然在**执行路径**上炸的 zip，返回终态 job。"""
                resp = client.post(
                    "%s?year=2000&lead=pre1&which=u850" % CHAIN_PREFIX,
                    files={"file": ("cesm.uwnd.5-9.zip", payload,
                                    "application/zip")})
                check(resp.status_code == 202,
                      "%s：受理仍 202（O(1) 校验只看名字）" % label,
                      "%s %s" % (resp.status_code, resp.text[:160]))
                return wait(resp.json()["jobId"])

            bad_job = _bad_input_case("nc 内容坏", build_zip(NCS, corrupt=True))
            bad_err = bad_job.get("error") or ""
            check(bad_job.get("status") == "failed",
                  "nc 内容坏：任务终态 failed", bad_job.get("status"))
            check(bool(bad_err), "nc 内容坏：error 文案非空（不是闷失败）",
                  bad_err[:160])
            check(_DRIVE_RE.search(bad_err) is None,
                  "执行路径 error 不含 <盘符>:<分隔符> 片段（message 带路径、"
                  ".filename 为空）", bad_err[:200])
            check_no_leak(bad_job,
                          "执行路径失败的终态响应递归扫不到绝对路径（T4-4）")
            shutil.rmtree(JOB_ROOT / bad_job["jobId"], ignore_errors=True)

            # ⑦ 同族的**第二个可达点**：路径不是第三方载荷，而是模块自己用 `%s`
            #    拼进 message 的（`cioproj.load_field` 的 `nc_dir=%s`）。成员名保留
            #    年份（`_zip_years` 要认得出 2000），但月份给 12 —— 窗口只取 5–9 月，
            #    于是一天都没读到 → 抛带绝对目录的 `RuntimeError`。这条只有**通用
            #    剥离**挡得住，钉住"不是只修一处特例"。
            #    ⚠ 2026-09-21 契约变更（import-all-models）：受理阶段新增"目标年
            #    5–9 月必须齐"的**必要**条件预判（frontend-contract §2），链路入口
            #    对这份载荷**同步 400** —— 该可达点在链路侧按契约失效（守卫没丢，
            #    是被提前拦下了）。旧路由 `/api/predict` 的受理口本轮**不动**
            #    （红线：不碰 prediction.py），同一份载荷仍能走到 `load_field`，
            #    所以"模块自抛的绝对**目录**也要被剥掉"这条覆盖**改由旧路由承载**，
            #    断言一条不减。
            odd_payload = build_zip(
                NCS, name_of=lambda _p, i:
                "cesm.u850.anom.daily.20001202-28days_%02d.nc" % i)
            resp = client.post(
                "%s?year=2000&lead=pre1&which=u850" % CHAIN_PREFIX,
                files={"file": ("cesm.uwnd.5-9.zip", odd_payload,
                                "application/zip")})
            check(resp.status_code == 400 and "[5, 6, 7, 8, 9]" in resp.text,
                  "nc 名不匹配：链路受理同步 400（目标年 5–9 月缺失，"
                  "不再是跑完才炸）",
                  "%s %s" % (resp.status_code, resp.text[:160]))
            check(_DRIVE_RE.search(resp.text) is None,
                  "nc 名不匹配：链路 400 文案里没有绝对路径", resp.text[:200])
            legacy_odd = legacy_client.post(
                "/api/predict/jobs?year=2000&lead=pre1&which=u850",
                files={"file": ("cesm.uwnd.5-9.zip", odd_payload,
                                "application/zip")})
            check(legacy_odd.status_code == 202,
                  "nc 名不匹配：旧路由受理仍 202（受理只看名字）",
                  "%s %s" % (legacy_odd.status_code, legacy_odd.text[:160]))
            odd_job = wait(legacy_odd.json()["jobId"], prefix="/api/predict/jobs")
            odd_err = odd_job.get("error") or ""
            check(odd_job.get("status") == "failed",
                  "nc 名不匹配：任务终态 failed", odd_job.get("status"))
            check(_DRIVE_RE.search(odd_err) is None,
                  "执行路径 error 里模块自抛的绝对目录也被剥掉", odd_err[:200])
            check_no_leak(odd_job, "nc 名不匹配：整份响应递归扫不到绝对路径")
            shutil.rmtree(lp.JOB_ROOT / odd_job["jobId"], ignore_errors=True)

        finally:
            lp._run_archive_inference = real_archive_infer
    finally:
        lp._prepare_cio = real_prepare
        lp._run_archive_inference = fake_inference
        ch._update_job = real_update_job

    # ====================================================== E. 离线对拍
    print("E. 离线逐位对拍（真跑 8181 个 checkpoint）")
    baseline = None
    ready = False                      # 离线对拍将真跑 = 本轮验收 1 有直接证据

    def pick_baseline():
        """选离线对拍基线：**显式来源优先，绝不静默挑"排序第一个"**（C-1 的姊妹问题 C-3）。

        ① `CHAIN_BASELINE_DIR`   —— 直接指定任务目录
        ② `CHAIN_BASELINE_JOB`   —— 指定 backend/uploads/prediction_jobs/<jobId>
        ③ 都没有 → 跳过并显著标注（返回 None / label 说明原因）
        **不自动回退**：uploads 里的目录是可变状态，且已知有旧版产物会让对拍
        报出 75.26 那样的**伪差异**。宁可 skip，也不给假保证。
        """
        env_dir = os.environ.get("CHAIN_BASELINE_DIR")
        if env_dir:
            cand = Path(env_dir)
            if not (cand / "raw_fields.zip").is_file():
                return None, "CHAIN_BASELINE_DIR 所指目录没有 raw_fields.zip", False
            if not (cand / "prediction.npy").is_file():
                return None, "CHAIN_BASELINE_DIR 所指目录没有 prediction.npy", False
            if is_stale_baseline(cand.name):
                return None, "CHAIN_BASELINE_DIR 指向的目录是**已确认的陈旧件**", True
            return cand, "CHAIN_BASELINE_DIR", False
        env_job = os.environ.get("CHAIN_BASELINE_JOB")
        if env_job:
            cand = HERE / "uploads" / "prediction_jobs" / Path(env_job).name
            if not (cand / "raw_fields.zip").is_file():
                return None, "CHAIN_BASELINE_JOB=%s 没有 raw_fields.zip" % env_job, False
            if not (cand / "prediction.npy").is_file():
                return None, "CHAIN_BASELINE_JOB=%s 没有 prediction.npy" % env_job, False
            if is_stale_baseline(env_job):
                return None, "CHAIN_BASELINE_JOB 指定的是**已确认的陈旧件**", True
            return cand, "CHAIN_BASELINE_JOB", False
        return None, ("未指定基线：离线对拍没有可复现的输入。设 "
                      "CHAIN_BASELINE_JOB=<jobId>（或 CHAIN_BASELINE_DIR=<目录>）"
                      "后再跑——不自动回退，uploads 里已知有陈旧件"), False

    baseline, baseline_src, baseline_stale = pick_baseline()
    # B-1 的**自检**（同 C-1 判别性实验的思路：拿"应该命中的输入"证明守卫是活的）。
    # 评审实测：这两条曾恒为 False（键是 8 字符前缀，比较用 32 位全名），
    # 于是陈旧件被当成可用基线、报出 maxAbsDiff 75.2594 的伪回归而无人察觉。
    # 这里用全名喂进去 —— 只要匹配退回等值判断，本断言立刻变红。
    check(is_stale_baseline("16655a7a764c491889da30ee1d9ca6e7")
          and is_stale_baseline("847e8f0f" + "0" * 24),
          "陈旧基线守卫是活的：32 位全名（走 startswith）也能命中黑名单",
          sorted(STALE_BASELINES))
    check(not is_stale_baseline("41e4d53d" + "0" * 24),
          "陈旧基线守卫不误伤：正常基线不会被黑名单命中",
          "41e4d53d…")
    # 陈旧件不是"跳过"，是**判据错**：它会报出像 75.26 那样的伪差异（或反向
    # 掩盖真实偏差），跑赢了只会误导。显式红 + 非 0 退出。
    ready = bool(baseline and os.environ.get("CHAIN_E2E")
                 and not baseline_stale)
    if baseline_stale:
        print("  ⚠ 基线不可用：%s —— 请改指其它任务目录" % baseline_src)
    elif baseline is None:
        print("  跳过：%s" % baseline_src)
    elif not os.environ.get("CHAIN_E2E"):
        print("  跳过：设 CHAIN_E2E=1 才真跑（分钟级）；基线来源 = %s" % baseline_src)
    else:
        archived = np.load(baseline / "prediction.npy", allow_pickle=False)
        lp._run_archive_inference = real_inference
        zip_bytes = (baseline / "raw_fields.zip").read_bytes()
        r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                        files={"file": ("raw_fields.zip", zip_bytes,
                                        "application/zip")})
        check(r.status_code == 202, "离线对拍建任务 -> 202", r.text[:200])
        job = wait(r.json()["jobId"], timeout=3600)
        check(job["status"] == "completed", "离线对拍全链跑完", job.get("error"))
        if job["status"] == "completed":
            got = np.load(JOB_ROOT / job["jobId"] / "prediction.npy",
                          allow_pickle=False)
            check(got.shape == archived.shape and got.dtype == archived.dtype,
                  "形状/dtype 与历史产物一致",
                  (got.shape, got.dtype, archived.shape, archived.dtype))
            check(np.array_equal(got, archived, equal_nan=False),
                  "prediction.npy 与历史 /api/predict 产物逐位一致（验收 1）",
                  "最大差 %.3e" % float(np.max(np.abs(
                      got.astype(np.float64) - archived.astype(np.float64)))))
            print("    基线目录 = %s（来源：%s）" % (baseline, baseline_src))

    # C-4：夹具件/真件是**退出判定的一部分**，不是只印一行
    if os.environ.get("CHAIN_REQUIRE_REAL") and not IS_REAL:
        check(False, "CHAIN_REQUIRE_REAL=1 但只找到夹具件（本轮验的不是真件）",
              str(ROOT))
    if not IS_REAL and ready:
        check(False, "离线对拍在**夹具件**上跑：校验力弱于真件，不能当验收 1 证据",
              str(ROOT))
    print("  隔离自检（C-4）：两个任务根都在临时区，不写 backend/uploads/")
    print("    chain     -> %s" % ch.JOB_ROOT)
    print("    旧 /predict -> %s" % lp.JOB_ROOT)
    # B-1/B-4：跑前/跑后快照在此比对（**不是**无条件打印"未写入"）。
    # 快照从 import 完成后取到本行为止，覆盖了 A–E 全部用例体。
    for _real in REAL_JOB_DIRS:
        _key = str(_real)
        verify_no_write(
            REAL_SNAPSHOT_BEFORE[_key], _snapshot_tree(_real),
            "真工作区 %s 本轮一个字节都没被写" % _real.name)
    write_checks_done = True

finally:
    lp._run_archive_inference = real_inference
    os.environ.pop("TRUTH_DIR", None)
    shutil.rmtree(TMP_BASE, ignore_errors=True)     # 本实例自己的临时根（pid 唯一）
    # 跑后快照必须取在这里：把它放在第 5 步的 finally 里，连"临时目录创建失败"
    # 这类中途异常路径也照样删掉中途产物 —— 快照取在用例体末尾反而会漏掉它们。
    REAL_SNAPSHOT_AFTER = {str(d): _snapshot_tree(d) for d in REAL_JOB_DIRS}

print("=" * 78)
# C-4：验的是"夹具件"还是"真件"是**退出判定的一部分**，不只是一行打印。
# 离线对拍在夹具件上跑出来的"逐位一致"校验力弱于真件，不能当验收 1 证据。
E2E_ON = bool(os.environ.get("CHAIN_E2E"))
if E2E_ON and not ready:
    check(False,
          "CHAIN_E2E=1 但离线对拍未跑（基线缺失/不可用）——本轮验收 1 无证据",
          baseline_src)
print("本轮数据来源 = %s（%s）" % (ROOT, "真件" if IS_REAL else "夹具件"))
# B-1/B-4 的证据闭环：上面那条"真工作区零写入"的 check 只在用例体正常跑完时
# 才会执行。若中途抛异常（含临时目录创建失败），下面的报错是**诚实**的，不能
# 让本次运行在"未证明零写入"的情况下悄悄判绿。
if not write_checks_done:
    check(False,
          "异常退出：真工作区零写入**未**完成比对（本轮不能声称不写 uploads）",
          str(REAL_JOB_DIRS))
if FAILS:
    print("失败 %d 项：" % len(FAILS))
    for item in FAILS:
        print("  -", item)
    sys.exit(1)
print("判定: 全部通过")
