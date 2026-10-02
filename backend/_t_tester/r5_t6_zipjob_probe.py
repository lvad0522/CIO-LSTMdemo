# -*- coding: utf-8 -*-
"""R5 Tester T6：单跑 zip 任务，看 raw/ 与 prediction.npy 的真实关系。

精确复刻 test_predict_chain.py C1 段（含它的 fake_inference 桩）。
"""
import hashlib
import io
import os
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIX = (HERE / "../摸库交付_2026-09-15/03_夹具").resolve()
NC = FIX / "U850"
print("FIX =", FIX, "| U850 nc =", len(list(NC.glob("*.nc"))))

os.environ["CIOPROJ_MAT"] = str(FIX / "rain" / "CIOmode_1982_2017.mat")
os.environ["CIOPROJ_ALLOW_MOCK"] = "1"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import live_prediction as lp  # noqa: E402

app = FastAPI()
app.include_router(lp.router)
client = TestClient(app)

buf = io.BytesIO()
ncs = sorted(NC.glob("*.nc"))
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in ncs:
        zf.write(p, "uwnd_5-9/" + p.name)
zb = buf.getvalue()
print("zip sha =", hashlib.sha256(zb).hexdigest())


def fake_inference(cio, progress_callback=None, span=(3, 95)):
    if progress_callback:
        progress_callback(span[0], "桩推理开始")
        progress_callback(span[1], "桩推理结束")
    return __import__("numpy").tile(cio.astype("float32"),
                                    (lp.GRID_ROWS, lp.GRID_COLS, 1))


lp._run_archive_inference = fake_inference

r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                files={"file": ("cesm.uwnd.5-9.zip", zb, "application/zip")})
print("POST ->", r.status_code)
jid = r.json()["jobId"]
for _ in range(900):
    d = client.get("/api/predict/jobs/%s" % jid).json()
    if d["status"] in ("completed", "failed"):
        break
    time.sleep(0.2)

print("status =", d["status"], "| error =", d.get("error"))
jd = lp.JOB_ROOT / jid
print("jobdir =", jd)
print("contents =", sorted(p.name for p in jd.iterdir()))
raw = jd / "raw"
print("raw/ exists =", raw.exists(),
      "| raw/ 文件数 =", len(list(raw.iterdir())) if raw.exists() else 0)
pn = jd / "prediction.npy"
if pn.exists():
    print("prediction.npy sha =",
          hashlib.sha256(pn.read_bytes()).hexdigest())
print("期望桩输出 sha      = 1125875a185aedd48bc8487957ef43064bd16f0d2205748f41783ca130307e93")
