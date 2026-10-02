# -*- coding: utf-8 -*-
"""C-2 归因：`truthPath` 是本次新引入的，还是旧路由/上游函数本来就有的？

两问：
  ① 旧 `/api/predict/jobs` 的响应体里有没有同一条绝对路径？（有 → 非本次引入）
  ② 生产口径（不设 TRUTH_DIR，走真实 Dataset/）下 `find_truth` 能不能成功？
     （能 → truthPath 分支在生产上是活的，不是只在测试沙盒里才出现）
"""
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
REPO = BACKEND.parent
DATA = REPO / "摸库交付_2026-09-15" / "数据"

print("=" * 74)
print("① 生产口径：TRUTH_DIR 未设，直接问 verification.find_truth('pre1', 2000)")
print("=" * 74)
os.environ.pop("TRUTH_DIR", None)
import verification                                     # noqa: E402
cands = verification.truth_candidates("pre1", 2000)
print("truth_candidates 命中 %d 份:" % len(cands))
for c in cands[:6]:
    print("   ", c)
try:
    t = verification.find_truth("pre1", 2000)
    print("find_truth -> %s" % t)
    print("=> 生产上 available=True 可达，truthPath=%s 会进响应" % t)
except ValueError as exc:
    print("find_truth 抛 ValueError: %s" % exc)
    print("=> 生产上 available=False，truthPath 不进响应（该泄漏在生产口径下不可达）")

print()
print("=" * 74)
print("② 旧路由 /api/predict/jobs 是否同样给出 truthPath")
print("=" * 74)
import tempfile                                           # noqa: E402
TMP = Path(tempfile.gettempdir()) / "chain_c2_probe_old"
for sub in ("chain_jobs", "predict_jobs", "truth/pre1/2000"):
    (TMP / sub).mkdir(parents=True, exist_ok=True)
os.environ["CHAIN_JOB_ROOT"] = str(TMP / "chain_jobs")
os.environ["PREDICTION_JOB_ROOT"] = str(TMP / "predict_jobs")
os.environ["TRUTH_DIR"] = str(TMP / "truth")
os.environ["CIOPROJ_MAT"] = str(DATA / "CIOmode_1982_2017.mat")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np                                         # noqa: E402
import live_prediction as lp                               # noqa: E402
from fastapi import FastAPI                                # noqa: E402
from fastapi.testclient import TestClient                  # noqa: E402

assert BACKEND / "uploads" not in lp.JOB_ROOT.parents, lp.JOB_ROOT
print("隔离: lp.JOB_ROOT=%s" % lp.JOB_ROOT)

app = FastAPI()
app.include_router(lp.router)
client = TestClient(app)

real_inference = lp._run_archive_inference
lp._run_archive_inference = lambda cio, progress_callback=None, span=(30, 97): np.tile(
    np.asarray(cio, dtype=np.float32), (lp.GRID_ROWS, lp.GRID_COLS, 1))

# 先造一份实况场（形状对齐）
truth_dir = TMP / "truth" / "pre1" / "2000"
lp.JOB_ROOT.mkdir(parents=True, exist_ok=True)


def build_zip(nc_dir, inner="uwnd_5-9"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(nc_dir.glob("*.nc")):
            zf.write(p, "%s/%s" % (inner, p.name))
    return buf.getvalue()


ZIP = build_zip(DATA / "U850")


def wait(job_id, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        d = client.get("/api/predict/jobs/%s" % job_id).json()
        if d.get("status") in ("completed", "failed"):
            return d
        time.sleep(0.1)
    return {"status": "timeout"}


# 第一趟拿 normalized（用于造实况），第二趟才有 verification
r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                files={"file": ("cesm.uwnd.5-9.zip", ZIP, "application/zip")})
print("旧路由 POST -> %s" % r.status_code)
j1 = wait(r.json()["jobId"])
print("旧路由 status=%s err=%s" % (j1.get("status"), j1.get("error")))
probe_path = None
for sub in lp.JOB_ROOT.glob("*"):
    p = sub / "prediction.npy"       # 旧 router 不落 normalized_window.npy，用预测场造实况
    if p.is_file():
        probe_path = p
        break
if probe_path is None:
    print("!! 没找到 prediction.npy，无法造实况")
    sys.exit(3)
norm = np.load(probe_path, allow_pickle=False)
assert norm.ndim == 3 and norm.shape[:2] == (lp.GRID_ROWS, lp.GRID_COLS), norm.shape
np.save(truth_dir / "pre_1_2000(1)_20_40_100_125_0.25_real2.npy",
        norm.astype(np.float64), allow_pickle=False)

r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                files={"file": ("cesm.uwnd.5-9.zip", ZIP, "application/zip")})
jid = r.json()["jobId"]
job = wait(jid)
ver = (job.get("result") or {}).get("verification") or {}
print("旧路由 GET 响应顶层键: %s" % sorted(job))
print("旧路由 verification 键: %s" % sorted(ver))
print("旧路由 result.verification.truthPath = %r" % ver.get("truthPath"))
print()
print("=> 旧 /api/predict 是否同样外泄服务器绝对路径: %s"
      % ("是（说明这是**上游函数/旧路由既有**行为，非 /api/chain 新引入）"
         if ver.get("truthPath") else "否"))
lp._run_archive_inference = real_inference
