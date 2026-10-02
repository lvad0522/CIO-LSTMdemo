# -*- coding: utf-8 -*-
"""Tester 独立验证 ①  ——  `/api/chain` 的受理校验、旧路由保留、落盘件对拍。

刻意与 Developer 的 test_chain_api.py **不共享任何代码**（不 import 它），只 import
被测模块本身。分块：

  T1. 受理阶段 6 类降级路径必须 400，且**不留半张图**（任务目录被清干净、任务未登记）
  T2. 旧路由 `/api/cio/*` 与 `/api/predict/*` 的路由表逐条还在（不是只测健康检查）
  T3. 验收 2b：落盘 `series.npy` 对官方件 `CIOproj_-20_20_40_120_1degree_pre1.npy`
      逐点 < 1e-6；形状/dtype 同官方
  T4. 阶段边界语义（409/200）与下载端点 `(1,T)` float64 形状

跑法:  cd backend && python _t_tester/test_chain_tester.py
"""
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent           # backend/_t_tester
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ---- 环境隔离：绝不让本测试写进 backend/uploads/prediction_jobs 或 Dataset ----
PRED_ROOT = HERE / "_pred_jobs"
TRUTH_ROOT = HERE / "_truth"
for p in (PRED_ROOT, TRUTH_ROOT):
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
os.environ["PREDICTION_JOB_ROOT"] = str(PRED_ROOT)
os.environ["TRUTH_DIR"] = str(TRUTH_ROOT)
os.environ["CIOPROJ_MAT"] = str(BACKEND / "assets" / "CIOmode_1982_2017.mat")
os.environ.pop("CIOPROJ_ALLOW_MOCK", None)

import chain as ch                                # noqa: E402
import live_prediction as lp                      # noqa: E402
import main                                       # noqa: E402
from fastapi.testclient import TestClient         # noqa: E402

GOLD = (BACKEND.parent / "摸库交付_2026-09-15" / "数据" / "20_40_100_125"
        / "CIOproj_-20_20_40_120_1degree_pre1.npy")

FAILS = []
PASSES = []


def check(cond, label, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "!!", label,
                           "" if cond else "   <- " + str(detail)))
    (PASSES if cond else FAILS).append(label)
    return bool(cond)


def chain_dirs():
    root = ch.JOB_ROOT
    return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


def make_zip(members: dict, prefix: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in members.items():
            zf.writestr(prefix + name, payload)
    return buf.getvalue()


# 夹具 nc：顶层指向 backend/_mock/（若存在），否则现场从真件里取一个小前缀
def fixture_nc(y: int, m: int, marker: bytes = b"\x00") -> tuple[str, bytes]:
    """返回 (标准 nc 文件名, 内容)。内容用真件 U850 的前 2KB 截断即可——
    受理校验只看文件名，解压在后台；本文件里所有 zip 用例都在受理期就 400 了。"""
    real = BACKEND.parent / "摸库交付_2026-09-15" / "数据" / "U850"
    cand = real / ("cesm.u850.anom.daily.%04d%02d02-28days.nc" % (y, m))
    if cand.is_file():
        return cand.name, cand.read_bytes()[:2048]
    return ("cesm.u850.anom.daily.%04d%02d02-28days.nc" % (y, m)), marker


def zip_years(years, months=(5, 6, 7, 8, 9)) -> bytes:
    members = {}
    for y in years:
        for m in months:
            n, b = fixture_nc(y, m)
            members[n] = b
    return make_zip(members)


print("test_chain_tester: Tester 独立验证（不 import Developer 的测试）")
print("=" * 78)
print("chain.JOB_ROOT        =", ch.JOB_ROOT)
print("live_prediction.JOB_ROOT =", lp.JOB_ROOT)
print("TRUTH_DIR             =", os.environ["TRUTH_DIR"])
assert lp.JOB_ROOT == PRED_ROOT, "PREDICTION_JOB_ROOT 未生效，拒绝在真目录上跑"

client = TestClient(main.app)

# ============================================ T1. 受理阶段 6 类降级路径
print("\nT1. 受理阶段降级路径（全部必须 400，且不留半张图）")

CASES = []
# (a) 文件名判不出类别
CASES.append(("缺类别关键字（which 留空）",
              ("mystery_package.zip", zip_years([2000, 2001])),
              "?year=2000&lead=pre1"))
# (b) zip 里没有 2000 年
CASES.append(("zip 里没有 2000 年 nc",
              ("cesm.uwnd.anom.2001-2002.zip", zip_years([2001, 2002])),
              "?year=2000&lead=pre1&which=u850"))
# (c) .npy 短于 112 点
CASES.append(("CIO .npy 只有 111 点",
              ("cio_short.npy", None),  # 内容在下面按 npy 单独生成
              "?year=2000&lead=pre1"))
# (d) 权重未覆盖组合（lead=pre6）
CASES.append(("权重未覆盖 lead=pre6",
              ("cesm.uwnd.anom.2000.zip", zip_years([2000, 2001])),
              "?year=2000&lead=pre6&which=u850"))
# (e) 非 .zip/.npy 后缀
CASES.append(("上传 .txt（后缀不合法）",
              ("readme.txt", b"hello"), "?year=2000&lead=pre1"))
# (f) 非法 confidence —— 在**查询端点**上（受理端点没有 confidence 参数）
CASES.append(("zip 无法解析（假 zip）",
              ("cesm.uwnd.anom.2000.zip", b"not a zip at all"),
              "?year=2000&lead=pre1&which=u850"))

for label, (fname, payload), query in CASES:
    before = chain_dirs()
    if payload is None:                      # (c) 111 点 CIO npy
        buf = io.BytesIO()
        np.save(buf, np.arange(111, dtype=np.float64))
        payload = buf.getvalue()
    r = client.post("/api/chain/jobs" + query,
                    files={"file": (fname, payload, "application/octet-stream")})
    body = ""
    try:
        body = r.json().get("detail", "")
    except Exception:                        # noqa: BLE001
        body = r.text[:120]
    check(r.status_code == 400, "%s -> 400" % label,
          "status=%s body=%s" % (r.status_code, body[:200]))
    check(bool(body.strip()), "%s -> 有明确文案" % label, repr(body))
    after = chain_dirs()
    check(after == before, "%s -> 任务目录未新增（不留半张图）" % label,
          "before=%s after=%s" % (before[-3:], after[-3:]))

# (f2) 非法 confidence：拿一个真 .npy 建任务（只跑到 ① 就有 series）
buf = io.BytesIO()
np.save(buf, np.sin(np.arange(2400) / 17.0).astype(np.float64))
r = client.post("/api/chain/jobs?year=2000&lead=pre1",
                files={"file": ("cio_ok.npy", buf.getvalue(),
                                "application/octet-stream")})
check(r.status_code == 202, "合法 .npy 受理 -> 202", r.text[:200])
job_id_npy = r.json()["jobId"] if r.status_code == 202 else None
if job_id_npy:
    # .npy 入口 series.npy 是受理期同步落盘的，无需等后台
    r2 = client.get("/api/chain/jobs/%s/spectrum?confidence=0.5" % job_id_npy)
    check(r2.status_code == 400, "非法 confidence=0.5 -> 400", r2.status_code)
    check("confidence" in r2.json().get("detail", ""),
          "非法 confidence 文案含 confidence",
          r2.json().get("detail"))
    r3 = client.get("/api/chain/jobs/%s/spectrum?confidence=0.95" % job_id_npy)
    check(r3.status_code == 200, "合法 confidence=0.95 -> 200", r3.status_code)

# ============================================ T2. 旧路由逐条还在
print("\nT2. 旧路由表逐条保留（/api/cio + /api/predict）")


def route_table():
    """本机 FastAPI 把 include_router 包成 `_IncludedRouter`（path/methods 都是
    None，真 router 挂在 `.original_router` 上），所以必须递归进去，
    否则会把三个 router 的端点全判成"缺失"。"""
    out = set()

    def walk(routes, prefix=""):
        for r in routes:
            sub = getattr(r, "routes", None) or \
                getattr(getattr(r, "original_router", None), "routes", None)
            if sub:
                walk(sub, prefix + (getattr(r, "path", None) or ""))
                continue
            path = getattr(r, "path", None)
            methods = getattr(r, "methods", None)
            if path and methods:
                for m in methods:
                    if m not in ("HEAD", "OPTIONS"):
                        out.add((m, prefix + path))

    walk(main.app.routes)
    return out


TABLE = route_table()
EXPECT_OLD = {
    ("POST", "/api/predict/jobs"),
    ("GET", "/api/predict/jobs/{job_id}"),
    ("GET", "/api/predict/jobs/{job_id}/preview"),
    ("GET", "/api/predict/jobs/{job_id}/grid"),
    ("GET", "/api/predict/jobs/{job_id}/pearson"),
    ("GET", "/api/predict/jobs/{job_id}/truth/preview"),
    ("GET", "/api/predict/jobs/{job_id}/download"),
    ("GET", "/api/predict/jobs/{job_id}/download/pearson"),
    ("POST", "/api/cio/jobs"),
    ("GET", "/api/cio/jobs/{job_id}"),
    ("GET", "/api/cio/jobs/{job_id}/series"),
    ("GET", "/api/cio/jobs/{job_id}/completeness"),
    ("GET", "/api/cio/jobs/{job_id}/spectrum"),
    ("GET", "/api/cio/capabilities"),
    ("GET", "/api/cio/reference"),
    ("GET", "/api/cio/mode/u850"),
}
missing = sorted(EXPECT_OLD - TABLE)
check(not missing, "旧路由 %d 条逐条还在" % len(EXPECT_OLD), "缺失: %s" % missing)

EXPECT_NEW = {
    ("POST", "/api/chain/jobs"),
    ("GET", "/api/chain/jobs/{job_id}"),
    ("GET", "/api/chain/jobs/{job_id}/series"),
    ("GET", "/api/chain/jobs/{job_id}/completeness"),
    ("GET", "/api/chain/jobs/{job_id}/spectrum"),
    ("GET", "/api/chain/jobs/{job_id}/preview"),
    ("GET", "/api/chain/jobs/{job_id}/truth/preview"),
    ("GET", "/api/chain/jobs/{job_id}/grid"),
    ("GET", "/api/chain/jobs/{job_id}/pearson"),
    ("GET", "/api/chain/jobs/{job_id}/download/cio"),
    ("GET", "/api/chain/jobs/{job_id}/download/normalized"),
    ("GET", "/api/chain/jobs/{job_id}/download/prediction"),
    ("GET", "/api/chain/jobs/{job_id}/download/pearson"),
}
missing_new = sorted(EXPECT_NEW - TABLE)
check(not missing_new, "新路由 %d 条齐备" % len(EXPECT_NEW), "缺失: %s" % missing_new)

# 旧路由不是空壳：真打一次 .npy 建任务 + 读状态
buf = io.BytesIO()
np.save(buf, np.sin(np.arange(2400) / 17.0).astype(np.float64))
r = client.post("/api/predict/jobs?year=2000&lead=pre1",
                files={"file": ("cio_ok.npy", buf.getvalue(),
                                "application/octet-stream")})
check(r.status_code == 202, "旧 /api/predict/jobs 受理 .npy -> 202", r.text[:200])
old_id = r.json()["jobId"] if r.status_code == 202 else None
if old_id:
    rj = client.get("/api/predict/jobs/%s" % old_id)
    check(rj.status_code == 200 and rj.json()["jobId"] == old_id,
          "旧 /api/predict/jobs/{id} 可读", rj.status_code)
    check(old_id in sorted(p.name for p in lp.JOB_ROOT.iterdir()),
          "旧路由任务落在 PREDICTION_JOB_ROOT（隔离生效）",
          str(lp.JOB_ROOT))

for path in ("/api/cio/capabilities", "/api/cio/reference", "/api/cio/mode/u850"):
    rr = client.get(path)
    check(rr.status_code == 200, "资产端点 %s -> 200" % path,
          "%s %s" % (rr.status_code, rr.text[:120]))

# ============================================ T3/T4. 真 ZIP 全链（贵，慢）
print("\nT3/T4. 真 ZIP：① 落盘件 vs 官方金标准 + 阶段门控 + 下载形状")
ZIP_SRC = BACKEND / "uploads" / "prediction_jobs" / \
    "30cc03890812494fbf3e101333926ce2" / "raw_fields.zip"
ZIP_SKIP = bool(os.environ.get("SKIP_ZIP"))
if not ZIP_SRC.is_file():
    check(False, "找不到真 zip 夹具 %s" % ZIP_SRC)
    ZIP_SKIP = True

if not ZIP_SKIP:
    import time
    before = chain_dirs()
    zb = ZIP_SRC.read_bytes()
    t0 = time.time()
    r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850&onlyProjection=true",
                    files={"file": ("cesm.uwnd.anom.daily.5-9.zip", zb,
                                    "application/zip")})
    check(r.status_code == 202, "真 zip 受理 -> 202", r.text[:200])
    jid = r.json()["jobId"]
    jdir = ch.JOB_ROOT / jid
    check(str(jdir) in [str(ch.JOB_ROOT / d) for d in chain_dirs()],
          "受理后任务目录已创建（唯一新增）",
          "%s -> %s" % (before[-3:], chain_dirs()[-3:]))

    job = None
    while time.time() - t0 < 900:
        job = client.get("/api/chain/jobs/%s" % jid).json()
        if job.get("status") in ("completed", "failed"):
            break
        time.sleep(2)
    check(job and job["status"] == "completed",
          "onlyProjection=true 只跑①② 到 completed",
          (job or {}).get("error"))
    if job and job["status"] == "completed":
        check(job.get("stageIndex") == 2,
              "只跑投影 -> stageIndex=2（未启动③）", job.get("stageIndex"))
        check(not (jdir / "prediction.npy").exists(),
              "只跑投影 -> 不落 prediction.npy（绝不加载 .pt）")
        check((jdir / "series.npy").is_file()
              and (jdir / "normalized_window.npy").is_file(),
              "①② 产物落盘齐备")

        # ---- 验收 2b：落盘 series.npy 对官方件
        s = np.load(jdir / "series.npy", allow_pickle=False)
        g = np.load(GOLD, allow_pickle=False)
        gf = np.ravel(g).astype(np.float64)
        check(GOLD.is_file(), "官方件存在 %s" % GOLD)
        check(s.shape == (2352,) if s.size == 2352 else False,
              "落盘 series.npy 是 1 维 2352 点（zip 入口口径）", s.shape)
        check(str(s.dtype) == "float64", "落盘 series dtype=float64", s.dtype)
        if s.size == gf.size:
            d = float(np.max(np.abs(s - gf)))
            check(d < 1e-6,
                  "落盘 series.npy vs 官方 pre1 逐点最大差 < 1e-6（实测 %.3e）" % d,
                  d)
        else:
            check(False, "series 长度 %d != 官方 %d" % (s.size, gf.size))

        # ---- 阶段门控：只跑①② 时 /preview 应 409
        r409 = client.get("/api/chain/jobs/%s/preview" % jid)
        check(r409.status_code == 409,
              "③ 未跑时 /preview -> 409（不是空图）", r409.status_code)
        check("降水推理" in r409.json().get("detail", ""),
              "409 文案是段级文案「降水推理尚未完成」", r409.json().get("detail"))

        # ---- 下载端点形状
        rc = client.get("/api/chain/jobs/%s/download/cio" % jid)
        check(rc.status_code == 200, "download/cio -> 200", rc.status_code)
        arr = np.load(io.BytesIO(rc.content), allow_pickle=False)
        check(arr.shape == (1, 2352) and arr.dtype == np.float64,
              "download/cio 是 (1,2352) float64", (arr.shape, arr.dtype))
        check(np.array_equal(np.ravel(arr), s),
              "download/cio 与落盘 series.npy 逐位一致")
        rn = client.get("/api/chain/jobs/%s/download/normalized" % jid)
        check(rn.status_code == 200, "download/normalized -> 200", rn.status_code)
        nn = np.load(io.BytesIO(rn.content), allow_pickle=False)
        on = np.load(jdir / "normalized_window.npy", allow_pickle=False)
        check(nn.shape == (112,) and nn.dtype == np.float32,
              "download/normalized 是 (112,) float32", (nn.shape, nn.dtype))
        check(np.array_equal(nn, on),
              "download/normalized 与落盘 normalized_window.npy 逐位一致")

        # ---- ② 的因果性（轻量版）：把落盘 series 改一个比特，重跑② 必须跟着变
        sp = jdir / "series.npy"
        keep = s.copy()
        s2 = s.copy()
        s2[0] += 0.001
        np.save(sp, s2, allow_pickle=False)
        ch._run_stage2(jid, jdir, {"warning": None})
        n2 = np.load(jdir / "normalized_window.npy", allow_pickle=False)
        np.save(sp, keep, allow_pickle=False)
        check(not np.array_equal(n2, on),
              "改了落盘 series.npy -> ② 产物跟着变（② 读的是落盘件，不是内存变量）")

        # ---- /series 返回的 raw 与落盘件一致
        rs = client.get("/api/chain/jobs/%s/series" % jid)
        check(rs.status_code == 200, "/series -> 200", rs.status_code)
        raw = np.asarray(rs.json()["raw"], dtype=np.float64)
        check(raw.size == 2352 and np.array_equal(raw, keep),
              "/series.raw 与落盘 series.npy 逐位一致", raw.size)
        nw = np.asarray(rs.json()["normalizedWindow"], dtype=np.float32)
        check(nw.size == 112, "/series.normalizedWindow 长度 112", nw.size)

print("\n" + "=" * 78)
print("PASS %d / FAIL %d" % (len(PASSES), len(FAILS)))
if FAILS:
    print("失败项:")
    for f in FAILS:
        print("   !!", f)
    sys.exit(1)
print("判定: 全部通过")
