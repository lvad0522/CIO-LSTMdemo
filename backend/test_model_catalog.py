# -*- coding: utf-8 -*-
"""权重组清单（400 组网格）与扫描缓存/预热的不变量自检。

变更 `expand-model-catalog`。与 `test_chain_api.py` **分开**成新文件的理由（design D9）：
后者正被并行会话改动（`/series` 键集），是最高撞车点；能少动就少动。

覆盖 **INV-1 … INV-11**（连续编号；tasks 第 4 组同表）：

| # | 不变量 | 落点 |
|---|---|---|
| INV-1 | 清单恰 20 lead × 20 year = 400 项，每 lead 的 `years` 非空且恒 20 项 | `A.` |
| INV-2 | `years` 绝不为空数组（前端 `isValidModelCatalog` 是 all-or-nothing） | `A.` |
| INV-3 | 清单 `available` ⇔ 受理口 `chain._require_available` 逐组一致 | `H.` |
| INV-4 | 根顺序 `D→E→F`：路径 (i) 前根**无**该目录 / (ii) 前根**半截** / 仲裁 / 归档短路 | `B.` |
| INV-5 | 掉盘与**权限类**错误 ⇒ 清单仍 200、相关组 `available:false`（含正向对照） | `C.` |
| INV-6 | 缓存：同一组两次逐字段相同；`mtime` 变则重扫；`is_dir` 不缓存 | `D.` |
| INV-7 | 结构性断言（AC6）：热态清单请求枚举次数 == 0 **+ 正向对照** | `E.` |
| INV-8 | manifest：指纹不匹配丢弃；单条 `mtime` 不一致丢弃；不同根列表互不覆盖 | `F.` |
| INV-9 | TTL **不上**请求路径 + 周期轮次**真的在跑**且受预算 K 约束 | `G.` |
| INV-10 | 预热：导入无副作用、不阻断启动、可关闭、单飞 | `I.` |
| INV-11 | 响应递归脱敏（无 `C:`/`D:`/`:\\`/`\\\\`） | 每个端到端响应处 + `J.` |

## 两条**硬约束**（违反则本文件失去意义）

1. **不依赖真实外置盘**（design C004）：所有根顺序/半截/掉盘断言一律用**临时合成夹具**
   （猴补 `MODEL_ROOTS`）。`E:/` `F:/` 只出现在 `J.` 段的**带 skip 诊断项**里 ——
   盘不在就打印跳过原因，**不计入失败**。
2. **不写真实工作区**：`MODEL_SCAN_MANIFEST_DIR` 与任务根都猴补到 `TMP_BASE`
   （`backend/_tmp_catalog_<pid>/`），跑完由 `J.` 段断言真实 `backend/uploads/` 下
   没有本套件写的 manifest。

跑法：`python test_model_catalog.py`
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ── 必须在 import 后端模块之前（tasks 4.5 / design D7）─────────────────────
# 预热**默认开启**，且冷盘上的全网格扫描可达数分钟。若测试直接 import 触发预热，
# 整个套件会被外置盘拖死。这里用 setdefault：显式设 1 的人自己承担后果。
os.environ.setdefault("MODEL_PREWARM", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:                                    # Windows 控制台默认 GBK，中文断言读不清
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FAILS = []
SKIPS = []

# 临时区**按 pid 唯一化**（与 test_chain_api 同一条理由 R-2：并行实例各写各的）
TMP_BASE = HERE / ("_tmp_catalog_%d" % os.getpid())
shutil.rmtree(TMP_BASE, ignore_errors=True)
MANIFEST_DIR = TMP_BASE / "manifests"
JOB_ROOT = TMP_BASE / "chain_jobs"
TRUTH_ROOT = TMP_BASE / "truth"
for _d in (MANIFEST_DIR, JOB_ROOT, TRUTH_ROOT):
    _d.mkdir(parents=True, exist_ok=True)
os.environ["CHAIN_JOB_ROOT"] = str(JOB_ROOT)
os.environ["TRUTH_DIR"] = str(TRUTH_ROOT)

import live_prediction as lp                    # noqa: E402
import chain as ch                              # noqa: E402
from fastapi import FastAPI, HTTPException      # noqa: E402
from fastapi.testclient import TestClient       # noqa: E402

# 隔离：manifest 落盘目录猴补到临时区（不猴补就会读写真实的 backend/uploads/）
lp.MODEL_SCAN_MANIFEST_DIR = MANIFEST_DIR

# 真实 `backend/uploads/` 的 manifest **现状签名**（名字 → (大小, mtime_ns, sha256)）：
# 本套件只许读它、不许写它（C-4）。⚠ 不能断言"真实目录下一个 manifest 都没有" ——
# 生产/实测正常跑过一次预热就会留下**合法产物**，那种断言会红在别人的合法产物上。
# 要证的是"**本套件**没写它" ⇒ 跑前跑后签名必须相等。
def _manifest_sig(root) -> dict:
    """签名的第三个分量是 **sha256**，不是装饰：见 `_live_backend()` 上方的归因规则。"""
    sig = {}
    for p in sorted(Path(root).glob("model_scan_manifest_*.json")):
        try:
            raw = p.read_bytes()
        except OSError:                      # 读不到就退回时间戳判据（宁严不宽）
            sig[p.name] = (p.stat().st_size, p.stat().st_mtime_ns, None)
            continue
        sig[p.name] = (len(raw), p.stat().st_mtime_ns,
                       hashlib.sha256(raw).hexdigest())
    return sig


_REAL_MANIFEST_SIG = _manifest_sig(HERE / "uploads")

app = FastAPI()
app.include_router(ch.router)
client = TestClient(app)


# ---------------------------------------------------------------- 断言与脱敏
def check(cond, label, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "!!", label,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        FAILS.append(label)
    return cond


def skip(label, why):
    print("  [skip] %s：%s" % (label, why))
    SKIPS.append(label)


# 与 `test_chain_api.check_no_leak` **同口径**（递归 + 盘符正则）。此处独立实现
# 是为了让本文件能单跑（import test_chain_api 会把那个脚本整个执行一遍）。
_DRIVE_RE = __import__("re").compile(r"[A-Za-z]:[\\/]")


def _leak_hits(node, path=()):
    hits = []
    if isinstance(node, dict):
        for key, value in node.items():
            hits.extend(_leak_hits(value, path + (str(key),)))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            hits.extend(_leak_hits(value, path + ("[%d]" % index,)))
    elif isinstance(node, str):
        if _DRIVE_RE.search(node) or "\\\\" in node:
            hits.append((".".join(path), node))
    return hits


def check_no_leak(doc, label):
    hits = _leak_hits(doc)
    return check(not hits, label,
                 ["%s = %s" % (w, v) for w, v in hits][:3])


# ───────────────────────────────────────────────────────────── 归因规则（J 段用）
# 真实 `backend/uploads/` 里的 manifest **不是本套件独有**：只要有一个后端在跑，
# 它的 `_background_scan_loop` 就每 `MODEL_SCAN_TTL`(默认 300 s) 调一次
# `write_model_scan_manifest()`（C006 的周期兜底，**PM 确认过必须有**），而且写的
# 是**同一份文件**、内容逐字节相同，只是 **mtime 前进**。
#
# 所以 (size, mtime) 相等这条判据**只在没有并发后端时才可判定**：
#   · 无并发后端 ⇒ "文件变了"只能是本套件写的 ⇒ 判红（判据照旧，牙口不变）；
#   · 有并发后端 ⇒ "文件变了"既可能是它刷新、也可能是本套件写的 ⇒ **不可归因**，
#     只能跳过 —— 硬判就是拿一个无法归因的量去定罪。
#
# 实证（2026-09-28）：20:10:57 → 20:15:57 **精确 300.0 s** 一次，正是 8000 启动
# 时刻 17:30:51 + 9600 s(=32×300 s) 那一拍 + 6 s 扫描耗时；观察期间未跑任何测试。
# 套件时长 ~90 s ⇒ 后端在跑时约 **30% 概率假红**（运行手册描述的"两站点常驻"正是
# 这个前提）。假红的代价不只是噪声：失败文案写的是"本套件一个字节都没写它"，
# 后来者会据此去改**本来正确**的猴补。
def _live_backend() -> str:
    """探测"正在按周期重写真实清单"的后端（运行手册的三个端口），返回 `host:port` 或 ""。

    判据必须能**认出是本项目后端**，不能只看端口通不通 —— 别的程序占着 8000 也会
    让探测为真，把一个本该判红的运行误跳过（假跳过 = 白丢覆盖）。这里要求
    `/openapi.json` 的 `paths` 里含 `/api/chain/models`（本项目自有路由）。
    顺带 `ProxyHandler({})` 关掉代理：本机回环不该走桌面代理。
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in (8000, 8001, 8002):
        try:
            with opener.open("http://127.0.0.1:%d/openapi.json" % port,
                             timeout=1.0) as resp:
                if resp.status != 200:
                    continue
                doc = json.loads(resp.read().decode("utf-8"))
        except Exception:                    # noqa: BLE001 连不上/非 JSON 一律算"没有"
            continue
        if "/api/chain/models" in (doc.get("paths") or {}):
            return "127.0.0.1:%d" % port
    return ""


# ---------------------------------------------------------------- 合成夹具
# 「完整组」的三项守卫判据值：8181 个成员、字节和 1558039617、坐标覆盖 81×101。
# 用**稀疏文件**（`truncate`，不落数据块）造出与真件**逐字节相同的形状**：
#   8100 × 190446 + 81 × 190457 == 1558039617   （真件的两种 `.pt` 尺寸）
# 单组建 5.1 s；建一次模板后用**硬链接**分发，单组 1.9 s。
_N_SMALL, _N_LARGE = 190446, 190457
_TEMPLATE = {"dir": None, "names": None}


def _member_plan():
    """(成员名, 目标字节数) 全量清单 —— 列 100 的 81 个用大尺寸，凑出精确字节和。"""
    plan = []
    for i in range(lp.GRID_ROWS):
        for j in range(lp.GRID_COLS):
            plan.append(("%02d_%03d.pt" % (i, j),
                         _N_LARGE if j == lp.GRID_COLS - 1 else _N_SMALL))
    return plan


def _fill(directory: Path, plan):
    directory.mkdir(parents=True, exist_ok=True)
    for name, size in plan:
        with open(directory / name, "wb") as fh:
            fh.truncate(size)


def complete_group(directory: Path) -> Path:
    """造一组**通过三项守卫**的合成组目录（形状与真件逐字节相同的判据值）。"""
    if _TEMPLATE["dir"] is None:
        tpl = TMP_BASE / "_tpl_complete"
        plan = _member_plan()
        _fill(tpl, plan)
        _TEMPLATE["dir"], _TEMPLATE["names"] = tpl, plan
    tpl = _TEMPLATE["dir"]
    directory.mkdir(parents=True, exist_ok=True)
    for name, size in _TEMPLATE["names"]:
        try:
            os.link(tpl / name, directory / name)
        except OSError:                     # 跨卷/无权限退化成真建
            with open(directory / name, "wb") as fh:
                fh.truncate(size)
    return directory


def partial_group(directory: Path, count: int = 247) -> Path:
    """造一组**存在但不完整**的合成组目录（默认 247 个成员，与半截副本同量级）。"""
    directory.mkdir(parents=True, exist_ok=True)
    made = 0
    for i in range(lp.GRID_ROWS):
        for j in range(lp.GRID_COLS):
            if made >= count:
                return directory
            with open(directory / ("%02d_%03d.pt" % (i, j)), "wb") as fh:
                fh.write(b"\0" * 16)
            made += 1
    return directory


def group_dir(root: Path, lead: str, year: int) -> Path:
    return root / lead / str(year)


# ---------------------------------------------------------------- 打桩工具
class scandir_counter:
    """统计 `os.scandir` 调用次数与路径（AC6 的结构性判据载体）。

    **必须给正向对照**：只在"热态应 0 次"上断言的话，计数器一旦失效（例如
    被别处换成别的实现）该断言会**恒真**。所以每次 0 次断言都配一条"同一计数器
    在冷态能数到 >0"的对照。
    """

    def __init__(self, deny_prefix=None):
        self.paths = []
        self._deny = deny_prefix
        self._real = os.scandir

    def __enter__(self):
        counter = self

        def counting(path=".", *args, **kwargs):
            counter.paths.append(str(path))
            if counter._deny is not None and str(path).startswith(counter._deny):
                raise PermissionError(13, "Access is denied", str(path))
            return counter._real(path, *args, **kwargs)

        os.scandir = counting
        return self

    def __exit__(self, *_exc):
        os.scandir = self._real
        return False

    @property
    def count(self):
        return len(self.paths)


class deny_fs:
    """让 `Path.is_dir` / `Path.stat` 对某前缀下的路径抛 `PermissionError`（D11 正向对照）。

    **为什么必须有这条**：盘符不存在时 `pathlib` 的 `is_dir()` 返回 `False` **而不抛**
    （只吞 ENOENT/ENOTDIR/EBADF/ELOOP），所以"MODEL_ROOTS 指向不存在路径"的夹具
    **永远到不了** `except OSError`。只测它等于没测"权限不足"那条硬要求。
    """

    def __init__(self, prefix):
        self.prefix = str(prefix)
        self._is_dir = Path.is_dir
        self._stat = Path.stat

    def __enter__(self):
        prefix = self.prefix
        real_is_dir, real_stat = self._is_dir, self._stat

        def is_dir(self, *args, **kwargs):
            if str(self).startswith(prefix):
                raise PermissionError(13, "Access is denied", str(self))
            return real_is_dir(self, *args, **kwargs)

        def stat(self, *args, **kwargs):
            if str(self).startswith(prefix):
                raise PermissionError(13, "Access is denied", str(self))
            return real_stat(self, *args, **kwargs)

        Path.is_dir, Path.stat = is_dir, stat
        return self

    def __exit__(self, *_exc):
        Path.is_dir, Path.stat = self._is_dir, self._stat
        return False


class roots:
    """临时替换 `lp.MODEL_ROOTS`（`chain.prediction` 是同一个模块对象）。"""

    def __init__(self, value):
        self.value = list(value)

    def __enter__(self):
        self.saved = lp.MODEL_ROOTS
        lp.MODEL_ROOTS = list(self.value)
        return self

    def __exit__(self, *_exc):
        lp.MODEL_ROOTS = self.saved
        return False


def catalog_items(doc):
    for lead_item in doc.get("leads", []):
        for item in lead_item.get("years", []):
            yield lead_item["lead"], item


def http_available(lead, year):
    """受理口是否放行（**直接调谓词**，不提交真任务 —— 那会起真实推理负载）。"""
    try:
        ch._require_available(lead, year)
        return True, None
    except HTTPException as exc:
        return False, str(exc.detail)


def wait_thread_idle(timeout=5.0):
    for _ in range(int(timeout / 0.05) + 1):
        thread = lp.active_background_scan_thread()
        if thread is None:
            return True
        time.sleep(0.05)
    return False


def poll_until(pred, timeout=15.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


# ==================================================================== 主流程
print("test_model_catalog: 权重组清单（400 组）与扫描缓存/预热不变量自检")
print("=" * 78)
print("TMP_BASE = %s" % TMP_BASE)
print("MODEL_PREWARM(env) = %s / lp.MODEL_PREWARM = %s"
      % (os.environ.get("MODEL_PREWARM"), lp.MODEL_PREWARM))

# 本套件专用的三套合成根（全部在 TMP_BASE 下，**不碰真实盘**）：
#   ROOT_A —— 根顺序里靠前的根；(pre1,2014) 完整
#   ROOT_B —— 靠后的根；(pre3,2014) 完整、(pre1,2014) 完整（仲裁用）
#   ROOT_C —— 只有半截的 (pre3,2014)
ROOT_A = TMP_BASE / "root_a"
ROOT_B = TMP_BASE / "root_b"
ROOT_C = TMP_BASE / "root_c"
print("构造合成夹具（稀疏文件 + 硬链接）…")
_t0 = time.time()
complete_group(group_dir(ROOT_A, "pre1", 2014))
complete_group(group_dir(ROOT_B, "pre3", 2014))
complete_group(group_dir(ROOT_B, "pre1", 2014))
complete_group(group_dir(ROOT_B, "pre7", 2005))
partial_group(group_dir(ROOT_C, "pre3", 2014))
partial_group(group_dir(ROOT_C, "pre1", 2000), count=101)   # 归档短路用
print("夹具就绪（%.1f s）" % (time.time() - _t0))

# ------------------------------------------------------------ A. INV-1 / INV-2
print("\nA. 清单形状：20 lead × 20 year = 400（INV-1 / INV-2）")
lp._scan_cache_clear()
with roots([ROOT_A, ROOT_B, ROOT_C]):
    r = client.get("/api/chain/models")
    check(r.status_code == 200, "GET /api/chain/models -> 200", r.text[:200])
    doc = r.json()
    check_no_leak(doc, "清单响应递归脱敏（无盘符/UNC 路径）")

_leads = [item["lead"] for item in doc["leads"]]
check(_leads == list(lp.GROUP_LEADS),
      "lead 分组 == 声明式网格 GROUP_LEADS（顺序也一致）", _leads[:5])
check(len(_leads) == 20 and len(set(_leads)) == 20,
      "恰 20 个 lead 分组（不写死 4 个）", len(_leads))

# 独立路径交叉验证：不拿 GROUP_LEADS 自证，而是按**规格文本的区间** pre1..pre20 重算
check(set(_leads) == {"pre%d" % n for n in range(1, 21)},
      "lead 集合 == 规格区间 pre1..pre20（独立于常量重算）",
      sorted(set(_leads) - {"pre%d" % n for n in range(1, 21)}))

_years_ref = list(range(2000, 2020))
_n_items = 0
_empty, _wrong_years, _wrong_len = [], [], []
for _item in doc["leads"]:
    _ys = [y["year"] for y in _item["years"]]
    _n_items += len(_ys)
    if not _ys:
        _empty.append(_item["lead"])
    if len(_ys) != 20:
        _wrong_len.append((_item["lead"], len(_ys)))
    if _ys != _years_ref:
        _wrong_years.append((_item["lead"], _ys))
check(_n_items == 400, "条目总数 == 20 × 20 = 400", _n_items)
check(not _empty, "每个 lead 的 years 都非空（前端 all-or-nothing）", _empty)
check(not _wrong_len, "每个 lead 的 years 恒 20 项", _wrong_len)
check(not _wrong_years, "每个 lead 的 years == 2000..2019 且**递增有序**", _wrong_years[:2])

_srcs = {item["source"] for _l, item in catalog_items(doc)}
check(_srcs <= {"archive", "dir"}, "source 只取 archive / dir 两个枚举值", _srcs)
_reason_missing = [(lead, it["year"]) for lead, it in catalog_items(doc)
                   if not it["available"] and not it.get("reason")]
check(not _reason_missing, "不可用条目一律带 reason", _reason_missing[:3])

# ------------------------------------------------------------ B. INV-4 路径
print("\nB. 根顺序与命中判据（INV-4）")

# --- B.1 路径 (i)：靠前的根上**没有**该目录 ⇒ 落到下一根（合成夹具，非真实盘）
lp._scan_cache_clear()
_root_empty = TMP_BASE / "root_empty"
_root_empty.mkdir(parents=True, exist_ok=True)
with roots([_root_empty, ROOT_B]):
    g = lp.resolve_model_group("pre3", 2014)
check(g["available"] is True and g["source"] == "dir"
      and Path(g["directory"]) == group_dir(ROOT_B, "pre3", 2014),
      "路径(i) 前根无该目录 ⇒ 落到后根（available/dir/落点）",
      (g["available"], g["source"], g["directory"]))

# --- B.2 路径 (ii)：靠前的根上**存在但半截** ⇒ 落到下一根（本次新增行为）
lp._scan_cache_clear()
with roots([ROOT_C, ROOT_B]):
    g = lp.resolve_model_group("pre3", 2014)
check(g["available"] is True and g["source"] == "dir"
      and Path(g["directory"]) == group_dir(ROOT_B, "pre3", 2014),
      "路径(ii) 前根半截(247) ⇒ 落到后根完整组（SHALL NOT 报'不完整'）",
      (g["available"], g["source"], g["directory"]))
check(g.get("reason") is None
      and g["files"] == lp.GROUP_FILE_COUNT
      and g["bytes"] == lp.GROUP_TOTAL_BYTES,
      "路径(ii) 命中后：reason 为 None，且 files/bytes 取自**胜出的完整根**"
      "（不是被跳过的半截根的 247/47040184）",
      (g.get("reason"), g["files"], g["bytes"], lp.GROUP_FILE_COUNT))
# 正向对照：把后根摘掉，同一前根**必须**报"不完整（247/8181）" —— 证明上面那条
# 不是因为前根压根没被看（半截夹具真的被扫到了）
with roots([ROOT_C]):
    g_partial = lp.resolve_model_group("pre3", 2014)
check(g_partial["available"] is False
      and "247" in (g_partial.get("reason") or "")
      and "8181" in (g_partial.get("reason") or ""),
      "正向对照：只给半截根 ⇒ available:false 且 reason 含实测 247/8181",
      (g_partial["available"], g_partial.get("reason")))

# --- B.3 全部根都半截 ⇒ 报**最具体**的守卫原因（D8，不得被'未找到'覆盖）
lp._scan_cache_clear()
with roots([ROOT_C, TMP_BASE / "root_absent"]):
    g_all_partial = lp.resolve_model_group("pre3", 2014)
check(g_all_partial["available"] is False
      and "247" in (g_all_partial.get("reason") or ""),
      "全部根都半截 ⇒ reason 取第一个发生者（守卫原因，含实测值）",
      g_all_partial.get("reason"))
check("未找到" not in (g_all_partial.get("reason") or ""),
      "全部根都半截 ⇒ SHALL NOT 报'未找到该组权重目录'",
      g_all_partial.get("reason"))

# --- B.4 多根同名 ⇒ 由根顺序唯一裁决（合成夹具：两临时根各一份完整组）
lp._scan_cache_clear()
with roots([ROOT_A, ROOT_B]):
    g1 = lp.resolve_model_group("pre1", 2014)
    g2 = lp.resolve_model_group("pre1", 2014)
    g_flip = None
with roots([ROOT_B, ROOT_A]):
    g_flip = lp.resolve_model_group("pre1", 2014)
check(Path(g1["directory"]) == group_dir(ROOT_A, "pre1", 2014),
      "(pre1,2014) 两副本 ⇒ 命中**靠前**的根（可用性也随根顺序裁决）",
      g1["directory"])
check(g1 == g2, "同一盘面重复解析结果确定（可复算）", (g1, g2))
check(Path(g_flip["directory"]) == group_dir(ROOT_B, "pre1", 2014),
      "把根顺序对调 ⇒ 裁决随之对调（证明'顺序即优先级'真的在起作用）",
      g_flip["directory"])

# --- B.5 pre1/2000 归档短路：无条件高于任何目录根，且**不读任何目录**（D4）
lp._scan_cache_clear()
with scandir_counter() as cnt:
    with roots([ROOT_C, ROOT_A, ROOT_B]):       # ROOT_C 里有一份 pre1/2000 目录
        g_arch = lp.resolve_model_group("pre1", 2000)
check(g_arch["available"] is True and g_arch["source"] == "archive"
      and g_arch["directory"] is None,
      "pre1/2000 ⇒ source=archive、不打目录命中",
      (g_arch["available"], g_arch["source"], g_arch["directory"]))
check(cnt.count == 0,
      "pre1/2000 归档短路**一次目录枚举都没做**（同名目录存在也不读）",
      cnt.paths)
with roots([ROOT_C, ROOT_A, ROOT_B]):
    _arch_item = [it for it in
                  next(x for x in lp.list_model_groups() if x["lead"] == "pre1")["years"]
                  if it["year"] == 2000][0]
check(_arch_item["source"] == "archive" and _arch_item["available"] is True,
      "清单里 pre1/2000 同样是 source=archive / available（网格侧无特例代码）",
      _arch_item)

# ------------------------------------------------------------ C. INV-5 掉盘
print("\nC. 掉盘与不可访问（INV-5 / D11）")
lp._scan_cache_clear()
_absent = TMP_BASE / "root_absent"
with roots([_absent]):
    r = client.get("/api/chain/models")
    doc_absent = r.json()
    _n_ok = sum(1 for _l, it in catalog_items(doc_absent) if it["available"])
    _n_arch = sum(1 for l, it in catalog_items(doc_absent)
                  if it["available"] and it["source"] == "archive")
check(r.status_code == 200, "根指向不存在路径 ⇒ 清单仍 200（无 5xx）", r.status_code)
check_no_leak(doc_absent, "掉盘清单响应递归脱敏")
check(_n_ok == 1 and _n_arch == 1,
      "掉盘时只有 pre1/2000（归档短路）可用，其余 399 组 available:false",
      (_n_ok, _n_arch))
_bad_reason = [(l, it["year"], it.get("reason")) for l, it in catalog_items(doc_absent)
               if it["available"] is False and "不可达" not in (it.get("reason") or "")]
check(not _bad_reason,
      "掉盘的 399 组 reason 都是'模型根不可达'（2026-09-26 PM 指示：根整根不可用"
      "要标注不可达，不能再伪装成'未找到'）", _bad_reason[:2])
# 正向对照（D.3 的"目录消失"用例是另一侧）：根**可用**但没有这一组 ⇒ 仍是"未找到"。
# 两条一起才证明新文案**区分了两种状态**，而不是把旧的"未找到"整类改词。
_confused = [(l, it["year"], it.get("reason")) for l, it in catalog_items(doc_absent)
             if it["available"] is False and "未找到" in (it.get("reason") or "")]
check(not _confused,
      "掉盘的 399 组 SHALL NOT 报'未找到该组权重目录'（与'根可用但无此组'可区分）",
      _confused[:2])

# --- C.1b 三档优先级（规格 :251）：①存在但不完整 ＞ ②根不可达 ＞ ③未找到
lp._scan_cache_clear()
with roots([_absent, ROOT_C]):
    g_prio = lp.resolve_model_group("pre3", 2014)
check(g_prio["available"] is False
      and "247" in (g_prio.get("reason") or "")
      and "8181" in (g_prio.get("reason") or "")
      and "不可达" not in (g_prio.get("reason") or ""),
      "前根不可达 + 后根半截 ⇒ reason 取更具体的'不完整（247/8181）'，"
      "后写的 ② 不得覆盖先前更具体的 ①",
      g_prio.get("reason"))

# --- C.2 **权限类正向对照**（D11 的唯一守门断言）
# 盘符不存在时 is_dir() 返回 False 而不抛 ⇒ 上面那条**到不了** except OSError。
_DENY = TMP_BASE / "deny_root"
(_DENY / "pre3" / "2000").mkdir(parents=True, exist_ok=True)
lp._scan_cache_clear()
with deny_fs(_DENY):
    with roots([_DENY]):
        try:
            g_deny = lp.resolve_model_group("pre3", 2000)
            _raised = None
        except BaseException as exc:            # noqa: BLE001
            g_deny, _raised = None, repr(exc)
        r = client.get("/api/chain/models")
check(_raised is None, "权限类 OSError 不得从 resolve_model_group 冒出", _raised)
check(bool(g_deny) and g_deny["available"] is False
      and "不可访问" in (g_deny.get("reason") or "")
      and "PermissionError" in (g_deny.get("reason") or ""),
      "is_dir/stat 抛 PermissionError ⇒ 转成'模型根不可访问（PermissionError）'",
      None if not g_deny else g_deny.get("reason"))
check(r.status_code == 200, "权限类错误 ⇒ 清单仍 200（SHALL NOT 5xx）", r.status_code)
check_no_leak(r.json(), "权限类错误下响应仍不含盘符")
_deny_reason = [it.get("reason") for _l, it in catalog_items(r.json())
                if not it["available"]]
check(bool(_deny_reason) and all("不可访问" in (x or "") for x in _deny_reason),
      "权限类错误下 399 组 reason 都是'模型根不可访问'",
      _deny_reason[:2])

# ------------------------------------------------------------ D. INV-6 缓存
print("\nD. 缓存行为（INV-6）")
_solo = TMP_BASE / "root_solo"
complete_group(group_dir(_solo, "pre5", 2007))
lp._scan_cache_clear()
with roots([_solo]):
    d1 = lp.resolve_model_group("pre5", 2007)
    with scandir_counter() as c2:
        d2 = lp.resolve_model_group("pre5", 2007)
check(d1 == d2, "同一组连解两次：描述符逐字段相同", (d1, d2))
check(c2.count == 0, "第二次解析命中缓存 ⇒ 零枚举", c2.paths)
# 独立路径交叉验证：缓存结果 == 直接扫盘的结果
_direct = lp._scan_group_directory(group_dir(_solo, "pre5", 2007))
check((d1["files"], d1["bytes"], d1["reason"]) ==
      (_direct["files"], _direct["bytes"], _direct["problem"]),
      "缓存返回值 == 直接 _scan_group_directory 的实测值（逐字段）",
      (d1["files"], d1["bytes"], d1["reason"], _direct))

# --- D.2 目录 mtime 变化 ⇒ 重扫
_gdir = group_dir(_solo, "pre5", 2007)
_cur = _gdir.stat().st_mtime_ns
os.utime(_gdir, ns=(_cur - 10 ** 9, _cur - 10 ** 9))     # 回拨 1 s：mtime 键必变
with roots([_solo]):
    with scandir_counter() as c3:
        d3 = lp.resolve_model_group("pre5", 2007)
check(c3.count == 1, "目录 mtime 变化 ⇒ 该组重扫（枚举 1 次）", c3.paths)
check(d3["available"] is True, "重扫后判定不变（内容没动）", d3["reason"])

# --- D.3 is_dir 不缓存：目录消失/复现都在下一个请求即反映
_live = TMP_BASE / "root_live"
complete_group(group_dir(_live, "pre6", 2008))
lp._scan_cache_clear()
with roots([_live]):
    d_before = lp.resolve_model_group("pre6", 2008)
    _gd = group_dir(_live, "pre6", 2008)
    _moved = _live / "pre6" / "2008_moved"
    _gd.rename(_moved)                       # 目录消失（缓存里有它的条目）
    d_gone = lp.resolve_model_group("pre6", 2008)
    _moved.rename(_gd)                       # 目录复现
    d_back = lp.resolve_model_group("pre6", 2008)
check(d_before["available"] is True, "前提：该组本来可用", d_before["reason"])
check(d_gone["available"] is False and
      d_gone.get("reason") == "未找到该组权重目录",
      "目录消失 ⇒ **下一个调用**即报不可用（存在性探测未被缓存）",
      (d_gone["available"], d_gone.get("reason")))
check(d_back["available"] is True,
      "目录复现 ⇒ 下一个调用即恢复可用（缓存不固化'整组不可用'）",
      (d_back["available"], d_back.get("reason")))

# --- D.4 半截 ⇒ 补全（mtime 变）⇒ 转可用，不跨请求固化
_fix = TMP_BASE / "root_fix"
partial_group(group_dir(_fix, "pre8", 2009), count=100)
lp._scan_cache_clear()
with roots([_fix]):
    f_before = lp.resolve_model_group("pre8", 2009)
    _fd = group_dir(_fix, "pre8", 2009)
    _filler = _member_plan()
    for name, size in _filler:               # 补全到 8181 / 精确字节和
        target = _fd / name
        # 已存在的那 100 个成员是**占位尺寸**（16 B），必须**原地改写**成正确尺寸：
        # 只加新文件的话字节和永远差 100×(190446−16)，"补齐后转可用"就永远红。
        # `os.link` 撞已存在路径抛 FileExistsError（OSError 子类）⇒ 自动落到改写分支。
        try:
            os.link(_TEMPLATE["dir"] / name, target)
        except OSError:
            with open(target, "wb") as fh:
                fh.truncate(size)
    os.utime(_fd, ns=(_fd.stat().st_mtime_ns - 10 ** 9,) * 2)
    f_after = lp.resolve_model_group("pre8", 2009)
check(f_before["available"] is False and "不完整" in (f_before.get("reason") or ""),
      "前提：半截组判不可用", f_before.get("reason"))
check(f_after["available"] is True,
      "补齐后（mtime 变）⇒ 转可用（缓存 SHALL NOT 把'半截'跨请求固化）",
      (f_after["available"], f_after.get("reason")))

# ------------------------------------------------------------ E. INV-7 结构性
print("\nE. 热态请求零枚举 + 正向对照（INV-7 / AC6）")
lp._scan_cache_clear()
with roots([ROOT_A, ROOT_B, ROOT_C]):
    with scandir_counter() as c_cold:
        r = client.get("/api/chain/models")
    _cold_doc = r.json()
    _cold_derived = sum(1 for _l, it in catalog_items(_cold_doc)
                        if it["available"] and it["source"] == "dir")
    with scandir_counter() as c_hot:
        r = client.get("/api/chain/models")
    _hot_doc = r.json()
check(c_cold.count == _cold_derived and _cold_derived > 0,
      "正向对照：缓存冷时枚举次数 == 目录源可用组数（**派生值**，非写死常量）",
      (c_cold.count, _cold_derived, c_cold.paths[:4]))
check(c_hot.count == 0,
      "热态（缓存已填）一次清单请求的枚举次数 == 0（结构性判据）",
      (c_hot.count, c_hot.paths[:4]))
check([d == e for d, e in zip(_cold_doc["leads"], _hot_doc["leads"])].count(False) == 0,
      "冷/热两次清单内容一致（缓存不改变判定）", None)
# pre1/2000 走归档短路 ⇒ 目录源可用组数恒比网格可用组数少 1（**不写死那个数字**）
_with_archive = sum(1 for _l, it in catalog_items(_hot_doc) if it["available"])
check(_with_archive == _cold_derived + 1,
      "归档短路使'可用组数'恰比'目录源可用组数'多 1（pre1/2000，派生对照）",
      (_with_archive, _cold_derived))

# ------------------------------------------------------------ F. INV-8 manifest
print("\nF. 落盘 manifest（INV-8）")
lp._scan_cache_clear()
with roots([ROOT_A, ROOT_B]):
    _n_written = lp.prewarm_model_groups()
    _m_path = lp.model_scan_manifest_path()
    _m_exists = _m_path.is_file()
    _fp = lp.model_roots_fingerprint()
    _roots_here = [str(p) for p in lp.MODEL_ROOTS]   # 必须在 with 内取：外面已还原成真根
check(_m_exists and _m_path.parent == MANIFEST_DIR,
      "预热把扫描结论落盘到猴补后的目录（不碰真实 backend/uploads/）",
      _m_path)
check(_fp[:16] in _m_path.name,
      "根列表指纹落在**文件名**里（两站点各自独立，不会写到一起）", _m_path.name)
_raw = json.loads(_m_path.read_text("utf-8"))
check(_raw.get("fingerprint") == _fp and _roots_here == _raw.get("roots"),
      "manifest 内容带根列表指纹与根序列", _raw.get("roots"))
check(all(isinstance(e, list) and len(e) == 5 for e in _raw.get("entries", []))
      and bool(_raw.get("entries")),
      "条目 schema ==(组目录, mtime_ns, files, bytes, problem) 五元组", _raw.get("entries", [])[:1])

# --- F.2 加载：mtime 一致 ⇒ 采纳；单条不一致 ⇒ 只丢那一条
lp._scan_cache_clear()
with roots([ROOT_A, ROOT_B]):
    _adopted = lp.load_model_scan_manifest()
    with scandir_counter() as c_loaded:
        r = client.get("/api/chain/models")
check(_adopted == len(_raw["entries"]),
      "manifest 条目全部被采纳（mtime 一致）", (_adopted, len(_raw["entries"])))
check(c_loaded.count == 0,
      "冷启动首个清单请求按 manifest 结论给出，**零枚举**（对齐+盘面未变）",
      c_loaded.paths[:4])

_entries = [list(e) for e in _raw["entries"]]
check(len(_entries) >= 2, "前提：manifest 至少 2 条（够做单条丢弃对照）", len(_entries))
_mixed = [list(_entries[0]), list(_entries[1])]
_mixed[1][1] = int(_mixed[1][1]) - 10 ** 9           # 只把第 2 条的 mtime 改坏
lp._scan_cache_clear()
with roots([ROOT_A, ROOT_B]):
    _m_path.write_text(json.dumps(
        {"fingerprint": _fp, "roots": [str(p) for p in lp.MODEL_ROOTS],
         "entries": _mixed}), encoding="utf-8")
    _adopted_mixed = lp.load_model_scan_manifest()
check(_adopted_mixed == 1,
      "单条 mtime 不一致 ⇒ **只丢那一条**（其余仍复用，不判整份无效）",
      _adopted_mixed)

# --- F.3 根列表指纹不匹配 ⇒ 整份丢弃
lp._scan_cache_clear()
with roots([ROOT_A, ROOT_B]):
    _good = lp.model_scan_manifest_path()
    _good.write_text(json.dumps(
        {"fingerprint": "0" * 16, "roots": ["Z:/other"], "entries": _entries}),
        encoding="utf-8")
    _adopted_bad = lp.load_model_scan_manifest()
check(_adopted_bad == 0, "指纹/根序列不匹配 ⇒ 丢弃、不沿用", _adopted_bad)

# --- F.4 不同根列表 ⇒ 不同文件名，互不覆盖
with roots([ROOT_A]):
    _p_a = lp.model_scan_manifest_path()
with roots([ROOT_B]):
    _p_b = lp.model_scan_manifest_path()
check(_p_a != _p_b and _p_a.name != _p_b.name,
      "两套根列表的 manifest 路径不同（8001/8002 两站点互不覆盖）",
      (_p_a.name, _p_b.name))
lp._scan_cache_clear()
with roots([ROOT_A]):
    lp.prewarm_model_groups()
    _p_a_exists = _p_a.is_file()
with roots([ROOT_B]):
    lp.prewarm_model_groups()
check(_p_a_exists and _p_a.is_file() and _p_b.is_file(),
      "各写各的，先写的那份仍在（未被后者覆盖）",
      (_p_a.is_file(), _p_b.is_file()))

# ------------------------------------------------------------ G. INV-9 TTL
print("\nG. TTL 不上请求路径 + 周期轮次（INV-9）")
_saved_ttl, _saved_budget = lp.MODEL_SCAN_TTL, lp.MODEL_SCAN_BUDGET
_saved_prewarm_g = lp.MODEL_PREWARM
try:
    lp._scan_cache_clear()
    with roots([ROOT_A, ROOT_B]):
        lp.prewarm_model_groups()                    # 填缓存（400 个键都会被 touch）
        lp.MODEL_SCAN_TTL = 0.02                     # TTL 调成极小
        time.sleep(0.1)                              # 保证**全部**条目已"过期"
        with scandir_counter() as c_ttl:
            r = client.get("/api/chain/models")
    check(r.status_code == 200 and c_ttl.count == 0,
          "TTL 全部过期后请求清单 ⇒ 枚举次数仍为 0（TTL 不在请求路径上）",
          (c_ttl.count, c_ttl.paths[:4]))

    # --- 周期轮次**真的在跑**（正向对照，否则 TTL 是死参数）
    # 本文件顶部把 env `MODEL_PREWARM` 设成 0（否则 import 即全网格扫外置盘），
    # 而 `MODEL_PREWARM` 默认关闭时 `start_background_model_scan()` 正确地返回 None。
    # 所以这里必须**显式打开模块属性**，否则整段 G 都测不到周期轮次。
    lp.MODEL_PREWARM = True
    lp.MODEL_SCAN_BUDGET = 3
    lp.MODEL_SCAN_TTL = 0.05
    lp._scan_cache_clear()
    with roots([ROOT_A, ROOT_B]):
        lp.prewarm_model_groups()
        stats_before = dict(lp.background_scan_stats())
        thread = lp.start_background_model_scan()
        _ran = poll_until(
            lambda: lp.background_scan_stats()["rounds"] > stats_before.get("rounds", 0),
            timeout=20.0)
        stats_after = dict(lp.background_scan_stats())
    # 先停线程**再**量请求：`scandir_counter` 打的是**进程级**的 `os.scandir`，
    # 后台轮次若正在扫描会把那些枚举一并记进来（TTL=0.05 ⇒ 每 50 ms 一轮，几乎
    # 必然撞上）—— 那样量到的是"轮次在扫盘"，与"请求路径是否扫盘"无关，是假红。
    lp.stop_background_model_scan()
    wait_thread_idle()
    with roots([ROOT_A, ROOT_B]):
        with scandir_counter() as c_round:
            r_round = client.get("/api/chain/models")
    check(thread is not None and _ran,
          "周期轮次**真的发生**（后台线程跑了至少一轮；否则 TTL 是死参数）",
          (thread, stats_before.get("rounds"), stats_after.get("rounds")))
    _sizes = stats_after.get("round_sizes", [])[stats_before.get("rounds", 0):]
    check(bool(_sizes) and all(1 <= n <= lp.MODEL_SCAN_BUDGET for n in _sizes),
          "每轮重扫条数 >= 1 且 <= K（预算约束生效）",
          (_sizes[:6], lp.MODEL_SCAN_BUDGET))
    check(stats_after.get("rescanned", 0) >= 1,
          "周期轮次累计重扫条数 >= 1", stats_after.get("rescanned"))
    check(r_round.status_code == 200 and c_round.count == 0,
          "一轮周期重扫**之后**的清单请求：枚举次数仍为 0（轮次未把扫描带回请求路径）",
          (c_round.count, c_round.paths[:4]))

    # --- K=0 不算满足：只要存在已到期条目，单轮至少重扫 1 条（C009）
    lp.MODEL_SCAN_BUDGET = 0
    lp.MODEL_SCAN_TTL = 0.0          # 刚预热完的条目 seen≈now ⇒ 不清零则**没有**到期项
    lp._scan_cache_clear()
    with roots([ROOT_A, ROOT_B]):
        lp.prewarm_model_groups()
        with scandir_counter() as c_k0:
            _n_k0 = lp.run_scan_round()
    check(_n_k0 >= 1,
          "K=0 时单轮仍至少重扫 1 条（'从不重扫'不算满足）",
          (_n_k0, c_k0.count))
    lp.MODEL_SCAN_BUDGET = _saved_budget
finally:
    lp.MODEL_SCAN_TTL = _saved_ttl
    lp.MODEL_SCAN_BUDGET = _saved_budget
    lp.MODEL_PREWARM = _saved_prewarm_g
    lp.stop_background_model_scan()
wait_thread_idle()

# ------------------------------------------------------------ H. INV-3 一致性
print("\nH. 清单 ⇔ 受理口逐组一致（INV-3 / AC7）")
lp._scan_cache_clear()
with roots([ROOT_A, ROOT_B, ROOT_C]):
    doc_h = None
_mismatch, _reason_bad = [], []
# ⚠ 逐组核对**必须整段在 `with roots(...)` 内**：清单与受理口两次调用要用**同一套
# 根**。放到 with 外面比对的是"合成根下的清单 vs 真实 D:/E:/F: 下的受理口"，
# 会凭空产出 19 条 pre1/2001..2019 的假不一致（真实盘上那些组存在）。
with roots([ROOT_A, ROOT_B, ROOT_C]):
    doc_h = client.get("/api/chain/models").json()
    for _lead, _it in catalog_items(doc_h):
        _ok, _detail = http_available(_lead, _it["year"])
        if _ok != bool(_it["available"]):
            _mismatch.append((_lead, _it["year"], _it["available"], _ok))
        if not _ok:
            _want = "%s/%d" % (_lead, _it["year"])
            if _want not in (_detail or ""):
                _reason_bad.append((_want, _detail))
check(not _mismatch,
      "400 组逐组：清单 available ⇔ chain._require_available 放行",
      _mismatch[:5])
check(not _reason_bad,
      "400 组逐组：被拒的 400 文案含组合标识与原因",
      _reason_bad[:3])
check(any(it["available"] for _l, it in catalog_items(doc_h))
      and any(not it["available"] for _l, it in catalog_items(doc_h)),
      "前提：夹具同时含可用与不可用两组（一致性断言两侧都非空）", None)
# 反向：清单里不可用的组合**真的**会 400（走端点），且不留任务目录
_unavail_lead, _unavail_year = next(((l, it["year"]) for l, it in catalog_items(doc_h)
                                     if not it["available"]), (None, None))
_before_dirs = set(p.name for p in JOB_ROOT.iterdir())
import numpy as _np                              # noqa: E402
_cio = _np.linspace(0.0, 1.0, lp.N_STEPS).astype(_np.float64)
_buf = __import__("io").BytesIO()
_np.save(_buf, _cio, allow_pickle=False)
with roots([ROOT_A, ROOT_B, ROOT_C]):
    r_un = client.post("/api/chain/jobs?year=%d&lead=%s&which=cio"
                       % (_unavail_year, _unavail_lead),
                       files={"file": ("CIO.npy", _buf.getvalue())})
_after_dirs = set(p.name for p in JOB_ROOT.iterdir())
check(r_un.status_code == 400
      and ("%s/%d" % (_unavail_lead, _unavail_year)) in r_un.text
      and "格式" not in r_un.text,
      "清单判不可用的组合 ⇒ 端点 400（含组合标识、不是格式文案）",
      "%s %s" % (r_un.status_code, r_un.text[:160]))
check(_after_dirs == _before_dirs,
      "被拒的请求不留任务目录（集合相等，不数个数）",
      sorted(_after_dirs - _before_dirs))
check_no_leak(r_un.json(), "被拒响应无盘符")

# ------------------------------------------------------------ I. INV-10 预热
print("\nI. 预热：导入无副作用 / 不阻断 / 可关闭 / 单飞（INV-10 / D7）")
import main as _main                              # noqa: E402
_live_named = [t.name for t in threading.enumerate() if t.name == "model-scan-prewarm"]
check(lp.active_background_scan_thread() is None and not _live_named,
      "导入 live_prediction **后**不存在活跃预热线程（导入期零副作用）",
      (lp.active_background_scan_thread(), _live_named))

_saved_prewarm = lp.MODEL_PREWARM
try:
    lp.MODEL_PREWARM = False
    with roots([ROOT_A]):
        _t_off = lp.start_background_model_scan()
    check(_t_off is None and lp.active_background_scan_thread() is None,
          "MODEL_PREWARM=0 ⇒ 不起线程", _t_off)

    lp.MODEL_PREWARM = True
    # 让后台扫描**真的有活干且干得慢**：前根放一个半截组（247 文件），并在
    # scandir 上注入 0.3 s 延迟 —— 于是"启动钩子立即返回"可判定：
    # 若它内联执行，本次调用本身就会花掉 >= 0.3 s。
    _slow = TMP_BASE / "root_slow"
    partial_group(group_dir(_slow, "pre9", 2010), count=247)
    lp._scan_cache_clear()
    with scandir_counter() as _c_slow:
        _real_scandir = os.scandir
        os.scandir = lambda *a, **k: (time.sleep(0.3), _real_scandir(*a, **k))[1]
        try:
            with roots([_slow]):
                _t0 = time.time()
                _thr = lp.start_background_model_scan()
                _elapsed = time.time() - _t0
                _thr2 = lp.start_background_model_scan()
                _named_now = [t.name for t in threading.enumerate()
                              if t.name == "model-scan-prewarm"]
                # C020：**真钩子体**（不只是 AST 形状）也要在延迟窗口内实测 —— 上面测的
                # 是 `start_background_model_scan()`，而 main.py 的钩子体里还可能塞别的
                # 同步动作。`main` 已在延迟注入**之前**导入（导入耗时不得计入本次计时）。
                _t0h = time.time()
                _main.start_model_scan_on_startup()
                _elapsed_hook = time.time() - _t0h
        finally:
            os.scandir = _real_scandir
    check(_elapsed < 0.25 and _thr is not None and _thr.daemon,
          "预热开启时启动钩子**立即返回**（毫秒级，不是数百秒）且线程是 daemon",
          (_elapsed, _thr, None if _thr is None else _thr.daemon))
    check(_elapsed_hook < 0.05,
          "真钩子体 `main.start_model_scan_on_startup()` 在注入 0.3 s 扫盘延迟下仍 < 0.05 s"
          "（C020：直接覆盖钩子体本身，不只断言 AST 形状）",
          _elapsed_hook)
    check(len(_named_now) <= 1,
          "单飞：同一时刻至多一个预热线程在跑",
          _named_now)
    lp.stop_background_model_scan()
    wait_thread_idle()
    check(lp.active_background_scan_thread() is None,
          "停线程后无残留预热线程（跑完确认无遗留）", None)
finally:
    lp.MODEL_PREWARM = _saved_prewarm

# --- I.2 启动钩子的**结构性**断言（C005：不得内联/不得 await）
import ast                                       # noqa: E402
_src = (HERE / "main.py").read_text("utf-8")
_tree = ast.parse(_src)
_hooks = []
for _node in ast.walk(_tree):
    if isinstance(_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for _dec in _node.decorator_list:
            _txt = ast.unparse(_dec)
            if "on_event" in _txt and "startup" in _txt:
                _hooks.append(_node)
_hook_calls = [ast.unparse(n) for h in _hooks for n in ast.walk(h)
               if isinstance(n, ast.Call)]
check(bool(_hooks), "main.py 有 startup 事件钩子", [h.name for h in _hooks])
check(any("start_background_model_scan" in c for c in _hook_calls),
      "startup 钩子调用了 start_background_model_scan（预热接线存在）", _hook_calls)
check(not any("prewarm_model_groups" in c for c in _hook_calls),
      "startup 钩子**没有**内联调用 prewarm_model_groups（C005：会阻塞启动）",
      _hook_calls)
check(not any(isinstance(h, ast.AsyncFunctionDef) for h in _hooks),
      "startup 钩子不是 async（没有 await 执行体的余地）",
      [type(h).__name__ for h in _hooks])

# --- I.3 env 解析保护（C015）：坏 env **不得让 import 失败**，须记警告并回落默认值
# 三个常量都在 **import 期**求值 ⇒ env 写成非数字时，旧实现会让 `import live_prediction`
# 直接抛 ValueError，两个在线站点连**启动**都失败。**必须用子进程测**：本进程里模块早
# 已导入，改不了"导入期"的行为（直接改 `lp.MODEL_SCAN_TTL` 测的是另一条路径）。
_defaults = (lp.MODEL_SCAN_TTL, lp.MODEL_SCAN_BUDGET, lp.MODEL_SCAN_STALL_LIMIT)
check(_defaults == (300.0, 4, 30.0),
      "父进程默认值 == 运行手册 §10.2 记载的 300 s / 4 / 30 s（下面的回落比对以此为基准）",
      _defaults)
_probe_src = ("import live_prediction as lp;"
              "print(lp.MODEL_SCAN_TTL, lp.MODEL_SCAN_BUDGET, lp.MODEL_SCAN_STALL_LIMIT)")


def _probe_env(ttl, budget, stall):
    """子进程里带指定 env 导入 live_prediction ⇒ (rc, 三值, stderr)。"""
    env = dict(os.environ)
    env.update({"MODEL_PREWARM": "0",
                "MODEL_SCAN_TTL": ttl,
                "MODEL_SCAN_BUDGET": budget,
                "MODEL_SCAN_STALL_LIMIT": stall})
    proc = subprocess.run([sys.executable, "-c", _probe_src], cwd=str(HERE), env=env,
                          capture_output=True, timeout=300)
    # 不指定 text：stderr 在 Windows 上多为 cp936，用 errors="replace" 解码才不会炸；
    # 断言只匹配**纯 ASCII** 的变量名，与编码无关。
    out = proc.stdout.decode("utf-8", "replace").strip()
    err = proc.stderr.decode("utf-8", "replace")
    vals = tuple(float(x) for x in out.split()) if proc.returncode == 0 and out else ()
    return proc.returncode, vals, err


for _name, _bad in (("abc / xyz / zzz", ("abc", "xyz", "zzz")),
                    ("0 / 0 / -5", ("0", "0", "-5"))):
    _rc, _vals, _err = _probe_env(*_bad)
    check(_rc == 0 and _vals == _defaults
          and "MODEL_SCAN_TTL" in _err,
          "坏 env（%s）⇒ import **不抛**、三值回落默认、且**记了警告**（只静默回落不算）" % _name,
          (_rc, _vals, [ln for ln in _err.splitlines() if "MODEL_SCAN_TTL" in ln]))

# 正向对照：合法 env 必须**真的被读进去**（否则"回落"可以靠"永远返回默认值"假通过），
# 且**不得**打这条警告（否则警告断言恒真）。
_rc, _vals, _err = _probe_env("120", "7", "2.5")
check(_rc == 0 and _vals == (120.0, 7.0, 2.5) and "MODEL_SCAN_TTL" not in _err,
      "正向对照：合法 env（120 / 7 / 2.5）被真实采纳，且**未**误报解析警告",
      (_rc, _vals, [ln for ln in _err.splitlines() if "MODEL_SCAN_TTL" in ln]))

# ------------------------------------------------------------ J. 收尾与诊断
print("\nJ. 真实工作区未被写入 + 真实盘面诊断（带 skip）")
_real_after = _manifest_sig(HERE / "uploads")
_peer = _live_backend()
if _peer and _real_after != _REAL_MANIFEST_SIG:
    # 有并发后端 + 文件确实变了 ⇒ 无法归因（见 `_live_backend()` 上方的规则）。
    # 注意跳过的是**本项**，不是整段 J：下面的盘面诊断照跑。
    _only_mtime = ({k: v[2] for k, v in _real_after.items()}
                   == {k: v[2] for k, v in _REAL_MANIFEST_SIG.items()})
    skip("真实 backend/uploads/ 的 manifest 未被本套件写入",
         "检测到并发后端 %s：它按 MODEL_SCAN_TTL(%.0f s) 周期重写**同一份文件**"
         "（%s），本轮无法把变更归因给本套件 ⇒ 不可判定。"
         "要拿到这项结论，请停掉全部后端后重跑本套件"
         % (_peer, lp.MODEL_SCAN_TTL,
            "内容 sha256 未变、仅 mtime 前进，形态与周期刷新吻合" if _only_mtime
            else "内容与 mtime 均变"))
else:
    # 无并发后端（或文件根本没变）⇒ 判据与修此假红之前**逐字相同**：牙口不变。
    check(_real_after == _REAL_MANIFEST_SIG,
          "真实 backend/uploads/ 的 manifest 签名**跑前跑后一致**（本套件一个字节都没写它）",
          (sorted(set(_real_after) ^ set(_REAL_MANIFEST_SIG)),
           sorted(_real_after.items())[:2]))

_d_re, _e_re, _f_re = (HERE.parent / "model" / "服务器模型_pt",
                       Path("E:/服务器模型_pt"), Path("F:/服务器模型_pt"))
print("  真实根存在性：D=%s E=%s F=%s"
      % (_d_re.is_dir(), _e_re.is_dir(), _f_re.is_dir()))
print("  真实盘面诊断（**不跑全量 400 组扫盘**，PM 口径；只做定点探测）")
# 守卫必须**自足**（评审 C013）：断言真值依赖"E: 上没有该目录"，所以守卫里也要查 E:。
# 只查 F: 的话，E: 一旦再现同名目录就会**假红**（本盘面 24h 内已变两次，不是假想场景）。
_e_pre3 = (_e_re / "pre3" / "2014").is_dir()
_f_pre3 = (_f_re / "pre3" / "2014").is_dir()
if _e_pre3 and _f_pre3:
    skip("pre3/2014 的真实盘面 oracle",
         "E: 与 F: 同时存在该目录 ⇒ 归属由根顺序裁决，此处**不作归属断言**"
         "（规则由 B 段合成夹具承担）")
elif _f_pre3:
    with roots([_d_re, _e_re, _f_re]):
        _oracle = lp.resolve_model_group("pre3", 2014)
    check(_oracle["available"] is True
          and Path(_oracle["directory"]) == _f_re / "pre3" / "2014",
          "诊断（跳过后置）：E: 无该目录 ⇒ pre3/2014 现行归属为 F: 且可用",
          (_oracle["available"], _oracle["directory"]))
else:
    skip("pre3/2014 ⇒ F: 的真实盘面 oracle",
         "F:/服务器模型_pt/pre3/2014 不在本机（外置盘未挂载）")
# `pre1/2014` 同理。**历史**：2026-09-25 时 E:/F: 均有完整副本 ⇒ 由根顺序裁决命中 E:；
# 2026-09-26 E: 侧副本被删（非本变更所为）⇒ 现在与 pre3/2014 同走路径 (i) 落到 F:。
# 这里断言的是**当前盘面**归属，守卫按同一条自足规则写（E: 无 且 F: 有）。
_e_pre1 = (_e_re / "pre1" / "2014").is_dir()
_f_pre1 = (_f_re / "pre1" / "2014").is_dir()
if _e_pre1 and _f_pre1:
    skip("pre1/2014 的真实盘面 oracle",
         "E: 与 F: 同时存在该目录 ⇒ 归属由根顺序裁决，此处**不作归属断言**"
         "（规则由 B 段合成夹具承担）")
elif _f_pre1:
    with roots([_d_re, _e_re, _f_re]):
        _oracle2 = lp.resolve_model_group("pre1", 2014)
    check(_oracle2["available"] is True
          and Path(_oracle2["directory"]) == _f_re / "pre1" / "2014",
          "诊断（跳过后置）：E: 无该目录 ⇒ pre1/2014 现行归属为 F: 且可用",
          (_oracle2["available"], _oracle2["directory"]))
else:
    skip("pre1/2014 ⇒ F: 的真实盘面 oracle",
         "F:/服务器模型_pt/pre1/2014 不在本机（外置盘未挂载）")

# ------------------------------------------------------------ 收尾
shutil.rmtree(TMP_BASE, ignore_errors=True)

print("=" * 78)
print("用例失败 %d 项，跳过 %d 项" % (len(FAILS), len(SKIPS)))
for _f in FAILS:
    print("  !! %s" % _f)
for _s in SKIPS:
    print("  [skip] %s" % _s)
print("判定: %s" % ("全绿" if not FAILS else "有失败"))
sys.exit(1 if FAILS else 0)
