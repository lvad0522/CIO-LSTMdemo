# -*- coding: utf-8 -*-
"""预测链自检：归一化口径、投影入口、API 契约。

三块：
  A. `_prepare_cio` 的长度边界表（2352 / 2240 / 1932 / 112 / 越界与非法输入）
  B. 与训练侧口径等价：同一序列过 `_prepare_cio` 和 `src/src/prediction.py` 的
     normalize + 切片，逐元素一致（`src/` 不在 git 里，缺了就跳过这块）
  C. API 契约：用 FastAPI TestClient 真跑 `POST /api/predict/jobs`，覆盖
     zip 原始场入口、npy 入口、以及四类应当 400 的输入。
     `_run_archive_inference` 被打桩（真跑 8181 个模型要 1–5 分钟，不进自检）。

跑法:  python test_predict_chain.py
"""
import atexit
import io
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ==================================================== 任务根隔离（C-4 同形状 / R5-T2）
# 跑测试**绝不写**真 `backend/uploads/`：任务根与实况根一律指向本进程专属临时区。
# `live_prediction.JOB_ROOT` 是 **import 期求值**（lp:58-61），env 必须在下面那行
# `import live_prediction` **之前**设好 —— 运行期再改模块属性补不住 import 期派生值。
REAL_UPLOADS = HERE / "uploads"
TMP_BASE = Path(tempfile.mkdtemp(prefix="r5t2_predict_chain_%d_" % os.getpid()))
JOB_ROOT = TMP_BASE / "prediction_jobs"           # 旧 `/api/predict` 的任务目录
TRUTH_ROOT = TMP_BASE / "truth"                   # 实况场目录
for _d in (JOB_ROOT, TRUTH_ROOT):
    _d.mkdir(parents=True, exist_ok=True)
os.environ["PREDICTION_JOB_ROOT"] = str(JOB_ROOT)
os.environ["TRUTH_DIR"] = str(TRUTH_ROOT)
atexit.register(shutil.rmtree, TMP_BASE, ignore_errors=True)   # 异常/跳过路径也清理

import live_prediction as lp                        # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "!!", label,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        FAILS.append(label)
    return cond


# ------------------------------------------------ 真工作区零写入的自检装置
# 三个任务库一个字节都不许写（R5-T2：`cio_jobs` 也是泄漏口，c4_snapshot 旧版只看两个）。
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


# 守卫用的是**模块内实际求出的** JOB_ROOT（不是本文件的局部变量）：env 若没生效，
# 这里就会命中，直接拒绝继续 —— 不许"隔离失败还照样跑"。
_LEAKED = [root for root in (lp.JOB_ROOT,)
           if any(_inside(root, real) for real in REAL_JOB_DIRS)]
if _LEAKED:
    print("⚠  任务根没隔离到临时区，自检会往真工作区里写文件：")
    for _d in _LEAKED:
        print("     %s" % _d)
    print("   拒绝继续（C-4）—— 这不是通过。")
    sys.exit(2)


def _snapshot_tree(base):
    """目录名全集 + 各条目 mtime + **根目录 mtime**。

    根 mtime 是关键：没有它，"建一个目录又把它删掉"在前后快照里看不出差别
    （名字集合与 mtime 都恢复原状），而真工作区恰恰会被这种手法悄悄碰过。
    """
    plain = str(base)
    if not base.is_dir():
        return {"key": plain, "exists": False, "root_mtime": None, "entries": {}}
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


# ======================================================= A. _prepare_cio 边界

print("test_predict_chain: 预测链自检")
print("=" * 78)
print("A. _prepare_cio 长度策略")
rng = np.random.default_rng(2026)

cases = [
    (2352, "training-equivalent", False, "官方件：21 年 × 112，切前 2240 再整段归一"),
    (2240, "training-equivalent", False, "正好训练长度"),
    (1932, "partial", True, "缺月份的短序列：整段自归一 + warning"),
    (112, "partial", True, "只有一年"),
]
for n, want_mode, want_warn, label in cases:
    cio, meta = lp._prepare_cio(rng.standard_normal(n))
    check(cio.shape == (112,) and cio.dtype == np.float32
          and meta["mode"] == want_mode
          and bool(meta["warning"]) == want_warn,
          "T=%-5d -> %-20s warning=%s  (%s)"
          % (n, meta["mode"], bool(meta["warning"]), label),
          (cio.shape, cio.dtype, meta["mode"], meta["warning"]))

for bad, label in ((np.zeros(111), "111 点：不足一年"),
                   (np.zeros(300), "常数：min == max"),
                   (np.array([1.0, 2.0, np.nan] * 40), "含 NaN"),
                   (np.zeros((112, 3)), "二维且 squeeze 不掉")):
    try:
        lp._prepare_cio(bad)
        check(False, label + " 应报错", "没报错")
    except ValueError as exc:
        check(True, "%s -> %s" % (label, str(exc)[:46]))

# 2352 的关键点：min/max 只在前 2240 点上取（后 112 点若含极值不影响结果）
series = rng.standard_normal(2352)
series[2300] = 99.0                       # 把极值放到训练长度之外
cio_2352, meta_2352 = lp._prepare_cio(series)
cio_2240, meta_2240 = lp._prepare_cio(series[:2240])
check(meta_2352["cioMax"] == meta_2240["cioMax"]
      and np.array_equal(cio_2352, cio_2240),
      "2352 与 2240 结果一致（极值落在第 2240 点之后时仍不走样）",
      (meta_2352["cioMax"], meta_2240["cioMax"]))

# ================================================ B. 与训练侧 prediction.py 等价

print("B. 与训练侧口径等价（src/src/prediction.py）")
SRC = (HERE / ".." / "src" / "src").resolve()
if not (SRC / "prediction.py").is_file():
    print("  跳过：没有 %s（src/ 不在 git 里，属正常）" % SRC)
else:
    sys.path.insert(0, str(SRC))
    try:
        import prediction as train_pred            # noqa: E402

        class _Borrow:
            """只为借那两个方法，避开 Predict.__init__ 挑 cuda 设备。"""
            maxmin_norm = train_pred.Predict.maxmin_norm
            normalize = train_pred.Predict.normalize

        for n, label in ((2352, "官方件"), (2240, "训练长度")):
            data = rng.standard_normal(n)
            truncated = data[:2240]                # main_India_new.py:17 的先截断
            want = _Borrow().normalize(truncated)[:112]
            got, _meta = lp._prepare_cio(data)
            # 逐位比：_prepare_cio 末尾按模型输入转 float32，所以比 float32 结果
            same = np.array_equal(got, want.astype(np.float32))
            check(same, "%s：_prepare_cio == prediction.normalize + [0:112]（%s）"
                  % (label, n),
                  "float32 后最大差 %.3e" % np.abs(
                      got.astype(np.float64) - want).max())

        # 反向确认：不截断直接归一化会得到不同结果（说明截断这一步真的在起作用）
        data = rng.standard_normal(2352)
        data[2300] = 99.0
        naive = _Borrow().normalize(data)[:112]
        got, _ = lp._prepare_cio(data)
        check(not np.array_equal(got.astype(np.float64), naive),
              "不截断直接归一化会走样（证明 2240 截断不是摆设）")
    except Exception as exc:                       # noqa: BLE001
        print("  跳过：import prediction 失败 %r" % (exc,))

# ============================================================ C. API 契约

print("C. API 契约（POST /api/predict/jobs）")
FIX = find_fixture()
if FIX is None:
    print("  跳过：没找到夹具（设 CIOPROJ_FIXTURE 指定）")
else:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    MAT = FIX / "rain" / "CIOmode_1982_2017.mat"
    NC = FIX / "U850"
    os.environ["CIOPROJ_MAT"] = str(MAT)
    os.environ["CIOPROJ_ALLOW_MOCK"] = "1"         # 夹具件只在本机联调里放行

    # 实况场指向受控临时目录（模块顶部已建好并出口到 TRUTH_DIR）：技巧检验的断言
    # 不能取决于本机有没有 Dataset/。这里只需把 pre1/2000 建出来（起始为空）。
    (TRUTH_ROOT / "pre1" / "2000").mkdir(parents=True, exist_ok=True)

    app = FastAPI()
    app.include_router(lp.router)
    client = TestClient(app)

    real_inference = lp._run_archive_inference

    def fake_inference(cio, progress_callback=None, span=(3, 95)):
        """桩：不读 8181 个权重，直接回一个由 cio 铺出来的场。"""
        if progress_callback:
            progress_callback(span[0], "桩推理开始")
            progress_callback(span[1], "桩推理结束")
        return np.tile(cio.astype(np.float32), (lp.GRID_ROWS, lp.GRID_COLS, 1))

    lp._run_archive_inference = fake_inference

    def wait(job_id, timeout=180):
        end = time.time() + timeout
        while time.time() < end:
            data = client.get("/api/predict/jobs/%s" % job_id).json()
            if data["status"] in ("completed", "failed"):
                return data
            time.sleep(0.2)
        return {"status": "timeout"}

    try:
        # --- C1: zip 原始场入口（105 个 nc，套一层壳测子目录定位）
        buf = io.BytesIO()
        ncs = sorted(NC.glob("*.nc"))
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in ncs:
                zf.write(p, "uwnd_5-9/" + p.name)
        zip_bytes = buf.getvalue()

        r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                        files={"file": ("cesm.uwnd.5-9.zip", zip_bytes,
                                        "application/zip")})
        check(r.status_code == 202, "zip 建任务 -> 202", r.text[:200])
        job = wait(r.json()["jobId"])
        check(job["status"] == "completed", "zip 任务跑完 -> completed",
              job.get("error"))
        if job["status"] == "completed":
            proj, norm = job["projection"], job["normalization"]
            check(job["result"]["shape"] == [81, 101, 112], "结果形状 (81,101,112)",
                  job["result"]["shape"])
            check(proj["kind"] == "u850" and proj["kindSource"] == "explicit-param",
                  "投影类别留痕 kind/kindSource", proj)
            check(proj["seriesLength"] == 2352 and not proj["missingMonths"],
                  "投影 2352 点、无缺月", (proj["seriesLength"], proj["missingMonths"]))
            check(norm["mode"] == "training-equivalent",
                  "归一化走 training-equivalent", norm["mode"])
            check(proj["matSource"] == "mock", "matSource 如实标为 mock",
                  proj["matSource"])
            check(job["inputKind"] == "raw-zip", "inputKind = raw-zip",
                  job["inputKind"])
            # 实况目录此时是空的：检验必须如实标"没做"，不能悄悄少一块
            ver = job["result"]["verification"]
            check(ver["available"] is False and "没有 pre1/2000 的实况场" in ver["reason"],
                  "无实况场 -> verification.available=False 且给出原因", ver)

        # --- C2: npy 入口（用夹具金标准 2352 点）
        gold = np.load(FIX / "rain" / "20_40_100_125" /
                       "CIOproj_-20_20_40_120_1degree_pre1.npy").squeeze()
        for arr, want_mode, want_warn, label in (
                (gold, "training-equivalent", False, "2352 点官方件"),
                (gold[:1932], "partial", True, "1932 点（缺月份）")):
            b = io.BytesIO()
            np.save(b, arr, allow_pickle=False)
            r = client.post("/api/predict/jobs?year=2000&lead=pre1",
                            files={"file": ("CIO.npy", b.getvalue())})
            check(r.status_code == 202, "npy(%s) 建任务 -> 202" % label, r.text[:200])
            job = wait(r.json()["jobId"])
            check(job["status"] == "completed", "npy(%s) 跑完" % label,
                  job.get("error"))
            if job["status"] == "completed":
                check(job["normalization"]["mode"] == want_mode,
                      "npy(%s) mode=%s" % (label, want_mode),
                      job["normalization"]["mode"])
                check(bool(job.get("warning")) == want_warn,
                      "npy(%s) warning=%s" % (label, want_warn), job.get("warning"))

        # --- C3: 应当 400 的四类
        bad_cases = [
            (dict(files={"file": ("readme.txt", b"hello")}),
             "只接受 .npy（CIO 序列）或 .zip（原始气象场）", "非 npy/zip"),
            (dict(files={"file": ("原始场.zip", zip_bytes)}),
             "判不出输入类别", "zip 名判不出类别且未显式指定 which"),
            (dict(files={"file": ("cesm.sst.2000.zip", zip_bytes)},
                  params={"which": "sst"}),
             "SST 分支尚未实现", "which=sst"),
            (dict(files={"file": ("cesm.sst.2000.zip", zip_bytes)},
                  params={"which": "u850"}),
             "二者冲突", "which 与文件名关键字冲突"),
            (dict(files={"file": ("CIO.npy", b"\x93NUMPY not really")}),
             "CIO 文件校验失败", "假 npy"),
        ]
        for kwargs, want_msg, label in bad_cases:
            params = kwargs.pop("params", {})
            params.setdefault("year", 2000)
            params.setdefault("lead", "pre1")
            r = client.post("/api/predict/jobs", params=params, **kwargs)
            check(r.status_code == 400 and want_msg in r.text,
                  "400：%s" % label, "%s %s" % (r.status_code, r.text[:160]))

        # --- C4: 缺 .mat 资产 -> 400 且文案指向资产
        os.environ["CIOPROJ_MAT"] = str(HERE / "_no_such_mat.mat")
        r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                        files={"file": ("cesm.uwnd.5-9.zip", zip_bytes)})
        check(r.status_code == 400 and "缺少 CIO 模态资产" in r.text,
              "400：缺 .mat 资产", "%s %s" % (r.status_code, r.text[:160]))
        os.environ["CIOPROJ_MAT"] = str(MAT)

        # --- C5: 沙盒里的夹具件，没开 ALLOW_MOCK 时必须被拒
        del os.environ["CIOPROJ_ALLOW_MOCK"]
        r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                        files={"file": ("cesm.uwnd.5-9.zip", zip_bytes)})
        check(r.status_code == 400 and "模拟数据沙盒" in r.text,
              "400：沙盒里的夹具件在未放行时被拒（绝不静默出结果）",
              "%s %s" % (r.status_code, r.text[:160]))
        os.environ["CIOPROJ_ALLOW_MOCK"] = "1"

        # --- C5b: 沙盒【外】的假件，开着 ALLOW_MOCK 也必须被拒
        #     （2026-09-15 沙盒设计的关键保证：环境变量单独不构成放行）
        import mock_sandbox as _ms
        outside = HERE / "_outside_fake.mat"
        shutil.copy(MAT, outside)
        try:
            check(not _ms.is_sandboxed(outside),
                  "沙盒外临时件确实不在沙盒里", str(_ms.find_root(outside)))
            check(_ms.allow_flag_set(), "此时 ALLOW_MOCK 是打开的（反证前提）")
            os.environ["CIOPROJ_MAT"] = str(outside)
            r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                            files={"file": ("cesm.uwnd.5-9.zip", zip_bytes)})
            check(r.status_code == 400 and "沙盒外的资产必须是真的" in r.text,
                  "400：沙盒外的假件即使开着 ALLOW_MOCK 也被拒",
                  "%s %s" % (r.status_code, r.text[:160]))
        finally:
            outside.unlink(missing_ok=True)
            os.environ["CIOPROJ_MAT"] = str(MAT)

        # --- C6: zip 里没有目标年 -> 400
        buf2 = io.BytesIO()
        with zipfile.ZipFile(buf2, "w") as zf:
            for p in ncs:
                if ".2001" in p.name:
                    zf.write(p, p.name)
        r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                        files={"file": ("cesm.uwnd.zip", buf2.getvalue())})
        check(r.status_code == 400 and "没有 2000 年的 nc" in r.text,
              "400：zip 里没有目标年 2000", "%s %s" % (r.status_code, r.text[:160]))

        # --- C7: 路径穿越防护
        evil = HERE / "_tmp_evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../evil.nc", b"x")
        try:
            lp._safe_extract_nc(evil, HERE / "_tmp_extract")
            check(False, "路径穿越应被拒", "没报错")
        except ValueError as exc:
            check("不安全路径" in str(exc), "路径穿越被拒 -> %s" % str(exc)[:50])
        evil.unlink(missing_ok=True)
        # --- C8: 有实况场时的技巧检验端点
        #     实况构造成与桩输出**逐位相同**，于是 r 处处为 1 —— 断言可以写死。
        day = TRUTH_ROOT / "pre1" / "2000"
        cio_used, _ = lp._prepare_cio(gold)
        truth = np.tile(cio_used.astype(np.float64), (81, 101, 1))
        truth_file = "pre_1_2000(1)_20_40_100_125_0.25_real2.npy"
        np.save(day / truth_file, truth)

        b = io.BytesIO()
        np.save(b, gold, allow_pickle=False)
        r = client.post("/api/predict/jobs?year=2000&lead=pre1",
                        files={"file": ("CIO.npy", b.getvalue())})
        job = wait(r.json()["jobId"])
        check(job["status"] == "completed", "有实况时任务跑完", job.get("error"))
        ver = job["result"]["verification"]
        check(ver["available"] is True and ver["truthName"] == truth_file,
              "有实况 -> verification.available=True 且留痕实况文件名", ver)
        jid = job["jobId"]

        r = client.get("/api/predict/jobs/%s/pearson?confidence=0.95" % jid)
        check(r.status_code == 200, "pearson 端点 -> 200", r.text[:160])
        pe = r.json()
        check(pe["rows"] == 81 and pe["cols"] == 101,
              "r 图形状 (81,101)", (pe["rows"], pe["cols"]))
        check(pe["n"] == 112 and abs(pe["criticalR"] - 0.1857) < 5e-5,
              "n=112、95% 临界 r = 0.1857", (pe["n"], pe["criticalR"]))
        check(abs(pe["meanR"] - 1.0) < 1e-6, "实况==预测 -> 平均 r = 1", pe["meanR"])
        check(pe["undefinedPoints"] == 0 and pe["definedPoints"] == 8181,
              "无恒定格点，全部有定义", (pe["definedPoints"], pe["undefinedPoints"]))
        check(pe["significantPositiveFraction"] == 1.0
              and pe["significantNegativeFraction"] == 0.0,
              "全格点显著正相关，无显著负相关",
              (pe["significantPositiveFraction"], pe["significantNegativeFraction"]))
        check(pe["data"][0][0] is not None and abs(pe["data"][0][0] - 1.0) < 1e-6,
              "data 首元素可按 JSON 解析且 = 1", pe["data"][0][0])

        # 置信水平只改阈值，不改 r
        r = client.get("/api/predict/jobs/%s/pearson?confidence=0.99" % jid)
        pe99 = r.json()
        check(abs(pe99["criticalR"] - 0.2425) < 5e-5
              and pe99["meanR"] == pe["meanR"],
              "换置信水平只换阈值、不换 r", (pe99["criticalR"], pe99["meanR"]))

        # 类型错误由 FastAPI 在入参校验阶段挡下（422）；范围错误才是我们自己的 400
        r = client.get("/api/predict/jobs/%s/pearson?confidence=abc" % jid)
        check(r.status_code == 422, "非数字置信水平 -> 422（FastAPI 入参校验）",
              r.status_code)
        for bad in ("0.2", "1.5"):
            r = client.get("/api/predict/jobs/%s/pearson?confidence=%s" % (jid, bad))
            check(r.status_code == 400 and "必须在" in r.text,
                  "越界置信水平 %s -> 400（域校验）" % bad,
                  "%s %s" % (r.status_code, r.text[:120]))

        r = client.get("/api/predict/jobs/%s/download/pearson" % jid)
        check(r.status_code == 200 and len(r.content) > 0,
              "r 图可下载", r.status_code)
        saved = np.load(io.BytesIO(r.content))
        check(saved.shape == (81, 101) and saved.dtype == np.float32,
              "下载的 r 图为 (81,101) float32", (saved.shape, saved.dtype))

        r = client.get("/api/predict/jobs/%s/truth/preview?time_index=3" % jid)
        check(r.status_code == 200 and r.json()["rows"] == 81
              and len(r.json()["data"]) == 81,
              "实况场预览端点可用", r.status_code)

        # --- C9: 实况有歧义（两份内容不同）必须拒绝，不能挑一份
        np.save(day / "pre_1_2000(2)_20_40_100_125_0.25_real2.npy", truth + 1.0)
        b = io.BytesIO()
        np.save(b, gold, allow_pickle=False)
        r = client.post("/api/predict/jobs?year=2000&lead=pre1",
                        files={"file": ("CIO.npy", b.getvalue())})
        job = wait(r.json()["jobId"])
        ver = job["result"]["verification"]
        check(job["status"] == "completed" and ver["available"] is False
              and "不一致" in ver["reason"],
              "实况有两份且不一致 -> 拒绝检验并说明（预测本身仍完成）", ver)

        # 清理歧义件，恢复成单份
        (day / "pre_1_2000(2)_20_40_100_125_0.25_real2.npy").unlink()

    finally:
        lp._run_archive_inference = real_inference
        shutil.rmtree(HERE / "_tmp_extract", ignore_errors=True)
        shutil.rmtree(TMP_BASE, ignore_errors=True)
        os.environ.pop("TRUTH_DIR", None)
        os.environ.pop("PREDICTION_JOB_ROOT", None)

# ---- 末置自检（C-4 同形状）：真工作区 uploads/ 本轮零写入，**进退出判定** ----
print("  隔离自检（C-4）：任务根在临时区，不写 backend/uploads/")
print("    任务根 PREDICTION_JOB_ROOT -> %s" % lp.JOB_ROOT)
print("    实况根 TRUTH_ROOT          -> %s" % TRUTH_ROOT)
print("    真工作区（仓库内）          -> %s" % REAL_UPLOADS)
check(str(lp.JOB_ROOT) == str(JOB_ROOT),
      "PREDICTION_JOB_ROOT 环境覆盖确实生效（任务根落在临时区）",
      (str(lp.JOB_ROOT), str(JOB_ROOT)))
for _real in REAL_JOB_DIRS:
    _ok, _why = _no_write_diff(REAL_SNAPSHOT_BEFORE[str(_real)],
                               _snapshot_tree(_real))
    check(_ok, "真工作区 %s 本轮一个字节都没被写" % _real.name, _why)

print("=" * 78)
if FAILS:
    print("失败 %d 项：" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("判定: 全部通过")
