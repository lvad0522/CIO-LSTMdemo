# -*- coding: utf-8 -*-
"""Tester 独立验证 ② ——  同一份 zip 走**旧** /api/predict/jobs 与**新** /api/chain/jobs
的三方对拍（旧路由 vs 新链 vs 仓库里存的历史 prediction.npy）。

用法: python old_vs_new.py <prediction_jobs 下的 jobId>
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

JOB_ID = sys.argv[1]
REAL = BACKEND / "uploads" / "prediction_jobs" / JOB_ID
TMP = HERE / ("pred_" + JOB_ID)
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True, exist_ok=True)
os.environ["PREDICTION_JOB_ROOT"] = str(TMP)
os.environ["CIOPROJ_MAT"] = str(BACKEND / "assets" / "CIOmode_1982_2017.mat")
os.environ.pop("CIOPROJ_ALLOW_MOCK", None)
TRUTH = HERE / ("truth_pred_" + JOB_ID)
shutil.rmtree(TRUTH, ignore_errors=True)
TRUTH.mkdir(parents=True, exist_ok=True)
os.environ["TRUTH_DIR"] = str(TRUTH)

from fastapi.testclient import TestClient       # noqa: E402
import chain as ch                              # noqa: E402
import live_prediction as lp                    # noqa: E402
import main                                     # noqa: E402

hist = np.load(REAL / "prediction.npy", allow_pickle=False)
zip_bytes = (REAL / "raw_fields.zip").read_bytes()

client = TestClient(main.app)
out = {"jobId": JOB_ID, "histMd5": hashlib.md5(hist.tobytes()).hexdigest()}

# ---------- 旧路由 ----------
t0 = time.time()
r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                files={"file": ("cesm.uwnd.anom.daily.5-9.zip", zip_bytes,
                                "application/zip")})
assert r.status_code == 202, r.text[:300]
oid = r.json()["jobId"]
ojob = None
while time.time() - t0 < 900:
    ojob = client.get("/api/predict/jobs/%s" % oid).json()
    if ojob.get("status") in ("completed", "failed"):
        break
    time.sleep(2)
# `_public_job` 不公开 resultPath，但 PREDICTION_JOB_ROOT 已被我指到临时区，
# 直接按 jobId 在临时区里找目录即可（绝不在真目录里翻）。
old_path = TMP / oid / "prediction.npy"
assert old_path.is_file(), "旧路由未落 prediction.npy：%s" % old_path
old = np.load(old_path, allow_pickle=False)
out["oldStatus"] = ojob["status"]
out["oldErr"] = ojob.get("error")
out["oldMd5"] = hashlib.md5(old.tobytes()).hexdigest()
out["oldSum"] = float(old.sum())
out["oldSeconds"] = round(time.time() - t0, 1)
out["oldVsHist"] = bool(np.array_equal(old, hist))
out["oldVsHistMaxAbsDiff"] = float(np.max(np.abs(old.astype(np.float64)
                                                 - hist.astype(np.float64))))

print("[ovn] 旧路由完成 %s md5=%s sum=%.4f" % (out["oldStatus"], out["oldMd5"], out["oldSum"]))

# ---------- 新链（同一进程、同一份 zip） ----------
t1 = time.time()
r2 = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                 files={"file": ("cesm.uwnd.anom.daily.5-9.zip", zip_bytes,
                                 "application/zip")})
assert r2.status_code == 202, r2.text[:300]
nid = r2.json()["jobId"]
njob = None
while time.time() - t1 < 900:
    njob = client.get("/api/chain/jobs/%s" % nid).json()
    if njob.get("status") in ("completed", "failed"):
        break
    time.sleep(2)
npath = ch.JOB_ROOT / nid / "prediction.npy"
new = np.load(npath, allow_pickle=False)
out["newJobId"] = nid
out["newStatus"] = njob["status"]
out["newErr"] = njob.get("error")
out["newMd5"] = hashlib.md5(new.tobytes()).hexdigest()
out["newSum"] = float(new.sum())
out["newSeconds"] = round(time.time() - t1, 1)
out["newVsOldBitwise"] = bool(np.array_equal(new, old))
out["newVsOldMaxAbsDiff"] = float(np.max(np.abs(new.astype(np.float64)
                                                - old.astype(np.float64))))
out["newVsHist"] = bool(np.array_equal(new, hist))
out["newVsHistMaxAbsDiff"] = float(np.max(np.abs(new.astype(np.float64)
                                                 - hist.astype(np.float64))))

# 新链的落盘① 序列 vs 官方件（同一份 zip 再核一遍验收 2）
gold = (BACKEND.parent / "摸库交付_2026-09-15" / "数据" / "20_40_100_125"
        / "CIOproj_-20_20_40_120_1degree_pre1.npy")
if gold.is_file():
    g = np.ravel(np.load(gold, allow_pickle=False)).astype(np.float64)
    s = np.load(ch.JOB_ROOT / nid / "series.npy", allow_pickle=False)
    out["seriesMaxAbsDiffVsGold"] = (float(np.max(np.abs(s - g)))
                                     if s.size == g.size else None)

print("[ovn] 新链完成 %s md5=%s sum=%.4f | 新vs旧逐位=%s"
      % (out["newStatus"], out["newMd5"], out["newSum"], out["newVsOldBitwise"]))

print("[ovn] RESULT " + json.dumps(out, ensure_ascii=False))
(HERE / ("old_vs_new_%s.json" % JOB_ID)).write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
