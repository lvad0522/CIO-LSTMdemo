# -*- coding: utf-8 -*-
"""R5 Tester T21：复现 E 段判红（最大差 2.107e+01）。

E 段原样：基线目录 = r5_baseline（raw_fields.zip sha 76b11f98），真推理，
          POST 文件名 = "raw_fields.zip"。
本探针在**沙盒后端**里跑（与 E 段同一套 code/data），把结果存到持久位置以便分析差异。
"""
import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SANDBOX = Path(r"C:\Users\Dong_\AppData\Local\Temp\r5sandbox")
BACKEND = SANDBOX / "backend"
DUMP = Path(r"C:\Users\Dong_\AppData\Local\Temp\r5_t21_artifacts")
DUMP.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND))

TMP = Path(tempfile.mkdtemp(prefix="t21_e_"))
CH = TMP / "chain_jobs"
TR = TMP / "truth"
CH.mkdir(parents=True, exist_ok=True)
TR.mkdir(parents=True, exist_ok=True)
os.environ["CHAIN_JOB_ROOT"] = str(CH)
os.environ["TRUTH_DIR"] = str(TR)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import chain as ch  # noqa: E402
import live_prediction as lp  # noqa: E402

import numpy as np  # noqa: E402

print("chain.__file__ =", ch.__file__)
print("  sha256 =", hashlib.sha256(Path(ch.__file__).read_bytes()).hexdigest())
print("lp.__file__    =", lp.__file__)
print("  sha256 =", hashlib.sha256(Path(lp.__file__).read_bytes()).hexdigest())
print("lp.MODEL_ARCHIVE =", lp.MODEL_ARCHIVE, "| is_file =", lp.MODEL_ARCHIVE.is_file())

BASELINE = Path(r"C:\Users\Dong_\AppData\Local\Temp\r5_baseline")
archived = np.load(BASELINE / "prediction.npy", allow_pickle=False)
zb = (BASELINE / "raw_fields.zip").read_bytes()
print("基线 zip sha =", hashlib.sha256(zb).hexdigest())
print("基线 pred sha =", hashlib.sha256((BASELINE / 'prediction.npy').read_bytes()).hexdigest())

app = FastAPI()
app.include_router(ch.router)
client = TestClient(app)

print("\n真推理（不设任何桩；`_run_archive_inference` 保持 import 时的原函数）…")
print("  当前 lp._run_archive_inference =", lp._run_archive_inference)

t0 = time.time()
r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                files={"file": ("raw_fields.zip", zb, "application/zip")})
print("POST ->", r.status_code, "| jobId =", r.json().get("jobId"))
jid = r.json()["jobId"]
d = None
for _ in range(3600):
    d = client.get("/api/chain/jobs/%s" % jid).json()
    if d["status"] in ("completed", "failed"):
        break
    time.sleep(0.3)
print("终态 =", d["status"], "| error =", d.get("error"), "| 用时 %.0fs" % (time.time() - t0))

pn = CH / jid / "prediction.npy"
got = np.load(pn, allow_pickle=False)
gsha = hashlib.sha256(pn.read_bytes()).hexdigest()
print("\n本次 sha256 =", gsha)
print("归档 sha256 = 63084c26ceb6a9effcb470deffac4ee48490e47b6b3a9cf34bd51a21153138bd")

diff = np.abs(got.astype(np.float64) - archived.astype(np.float64))
print("\n最大差 = %.6e  <-- E 段报的是 2.107e+01" % diff.max())
print("均值差 = %.6e" % diff.mean())
nz = int((diff > 0).sum())
print("不等的格点数 = %d / %d (%.4f%%)" % (nz, diff.size, 100.0 * nz / diff.size))
if nz:
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print("最大差位置 (i,j,t) =", idx, " got =", got[idx], " archived =", archived[idx])
    # 逐时间层统计
    per_t = diff.max(axis=(0, 1))
    print("逐时间层最大差 前 5:", np.round(np.sort(per_t)[-5:], 4))
    print("逐时间层最大差 后 5:", np.round(np.sort(per_t)[:5], 4))

# 落盘留存
shutil.copy(pn, DUMP / "t21_prediction.npy")
np.save(DUMP / "t21_diff.npy", diff.astype(np.float32))
import json
json.dump({"sha": gsha, "maxdiff": float(diff.max()),
           "n_diff": nz, "size": int(diff.size),
           "chain_sha": hashlib.sha256(Path(ch.__file__).read_bytes()).hexdigest(),
           "lp_sha": hashlib.sha256(Path(lp.__file__).read_bytes()).hexdigest()},
          open(DUMP / "t21_summary.json", "w"), indent=2)
print("\n留存 ->", DUMP)
shutil.rmtree(TMP, ignore_errors=True)
