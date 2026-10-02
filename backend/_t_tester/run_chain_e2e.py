# -*- coding: utf-8 -*-
"""Tester 独立 E2E：换一份基线目录跑完整 /api/chain 链，与仓库里存的 prediction.npy 逐位对拍。

用法: python run_chain_e2e.py <prediction_jobs 下的 jobId>
不写任何 backend/uploads/ 下的真目录（两个 JOB_ROOT 都指到临时区）。
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # backend/_t_tester
BACKEND = HERE.parent                           # backend
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

JOB_ID = sys.argv[1]
REAL_JOBS = BACKEND / "uploads" / "prediction_jobs"
SRC = REAL_JOBS / JOB_ID
assert (SRC / "raw_fields.zip").is_file(), SRC
assert (SRC / "prediction.npy").is_file(), SRC

TMP = HERE / ("jobs_" + JOB_ID)
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True, exist_ok=True)

# 隔离（B-5）：chain.JOB_ROOT / live_prediction.JOB_ROOT 都跟着各自的 env 走。
#   ⚠ 两个 env 都必须设：原先只设了 PREDICTION_JOB_ROOT，而 chain 那一侧在
#   C-5 之后是读 CHAIN_JOB_ROOT 的 → 漏设就会回落到真实
#   backend/uploads/chain_jobs，与本文件 docstring 的"不写真目录"直接矛盾。
os.environ["PREDICTION_JOB_ROOT"] = str(TMP)
os.environ["CHAIN_JOB_ROOT"] = str(TMP)
os.environ["CIOPROJ_MAT"] = str(BACKEND / "assets" / "CIOmode_1982_2017.mat")
os.environ.pop("CIOPROJ_ALLOW_MOCK", None)
TRUTH = HERE / ("truth_" + JOB_ID)
shutil.rmtree(TRUTH, ignore_errors=True)
TRUTH.mkdir(parents=True, exist_ok=True)
os.environ["TRUTH_DIR"] = str(TRUTH)

from fastapi.testclient import TestClient       # noqa: E402
import chain as ch                              # noqa: E402
import live_prediction as lp                    # noqa: E402
import main                                     # noqa: E402

print("[e2e] jobId =", JOB_ID)
print("[e2e] chain.JOB_ROOT =", ch.JOB_ROOT)
print("[e2e] live_prediction.JOB_ROOT =", lp.JOB_ROOT)
# 断言的是**隔离真成立**（而不是"我知道它会往哪写"）：两个 JOB_ROOT 都必须
# 落在 TMP 里。任一回落真实 uploads，这里立刻红 —— 隔离成功反而断言失败的那种
# 自相矛盾不会再出现。
assert ch.JOB_ROOT == TMP, ch.JOB_ROOT
assert lp.JOB_ROOT == TMP, lp.JOB_ROOT

zip_bytes = (SRC / "raw_fields.zip").read_bytes()
ref = np.load(SRC / "prediction.npy", allow_pickle=False)
print("[e2e] 参照 prediction.npy shape=%s dtype=%s md5=%s"
      % (ref.shape, ref.dtype, hashlib.md5(ref.tobytes()).hexdigest()))

t0 = time.time()
with TestClient(main.app) as client:
    r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                    files={"file": ("cesm.uwnd.anom.daily.5-9.zip", zip_bytes,
                                    "application/zip")})
    print("[e2e] POST /api/chain/jobs ->", r.status_code)
    assert r.status_code == 202, r.text[:400]
    jid = r.json()["jobId"]
    job_dir = ch.JOB_ROOT / jid
    deadline = time.time() + 1800
    job = None
    while time.time() < deadline:
        job = client.get("/api/chain/jobs/%s" % jid).json()
        if job.get("status") in ("completed", "failed"):
            break
        time.sleep(3)
    print("[e2e] status=%s stage=%s err=%s 用时=%.1fs"
          % (job.get("status"), job.get("stage"), job.get("error"), time.time() - t0))

    out = {
        "jobId": JOB_ID,
        "outJobId": jid,
        "status": job.get("status"),
        "error": job.get("error"),
        "stageIndex": job.get("stageIndex"),
        "seriesLength": (job.get("projection") or {}).get("seriesLength"),
        "usedSeconds": round(time.time() - t0, 1),
    }

    if job.get("status") == "completed":
        newp = np.load(job_dir / "prediction.npy", allow_pickle=False)
        out["newShape"] = list(newp.shape)
        out["newDtype"] = str(newp.dtype)
        out["md5New"] = hashlib.md5(newp.tobytes()).hexdigest()
        out["md5Ref"] = hashlib.md5(ref.tobytes()).hexdigest()
        out["bitwiseEqual"] = bool(newp.shape == ref.shape
                                   and newp.dtype == ref.dtype
                                   and np.array_equal(newp, ref))
        out["maxAbsDiff"] = (float(np.max(np.abs(newp.astype(np.float64)
                                                  - ref.astype(np.float64))))
                             if newp.shape == ref.shape else None)
        # ③ 喂给模型的张量（落盘件）也要留证据
        nw = np.load(job_dir / "normalized_window.npy", allow_pickle=False)
        cio, norm = lp._prepare_cio(np.load(job_dir / "series.npy"), pre_year=0)
        out["stage2MatchesPrep"] = bool(np.array_equal(nw, cio)
                                        and nw.dtype == cio.dtype
                                        and np.array_equal(nw, cio.astype(np.float32)))
        out["normalizedShape"] = list(nw.shape)
        out["normalizedDtype"] = str(nw.dtype)
        # 落盘 sequence 与官方件对拍（如果拿得到）
        gold = (BACKEND.parent / "摸库交付_2026-09-15" / "数据" / "20_40_100_125"
                / "CIOproj_-20_20_40_120_1degree_pre1.npy")
        if gold.is_file():
            g = np.load(gold, allow_pickle=False)
            s = np.load(job_dir / "series.npy", allow_pickle=False)
            gf = np.ravel(g).astype(np.float64)
            out["goldShape"] = list(g.shape)
            out["goldDtype"] = str(g.dtype)
            out["seriesShape"] = list(s.shape)
            out["seriesDtype"] = str(s.dtype)
            if gf.size == s.size:
                out["seriesMaxAbsDiffVsGold"] = float(np.max(np.abs(s - gf)))
            else:
                out["seriesMaxAbsDiffVsGold"] = None
                out["goldLenMismatch"] = [int(gf.size), int(s.size)]

    (HERE / ("result_%s.json" % JOB_ID)).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[e2e] RESULT " + json.dumps(out, ensure_ascii=False))
