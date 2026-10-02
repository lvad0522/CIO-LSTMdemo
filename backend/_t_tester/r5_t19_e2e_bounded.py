# -*- coding: utf-8 -*-
"""R5 Tester T19：E 段离线逐位对拍的**有界独立复跑**（我自己跑，不依赖挂住的整轮套件）。

E 段本体 = 「拿一份**真推理**归档 prediction.npy，再用它的 raw_fields.zip 建新链任务，
          真跑一遍，断言逐位一致」。这里就干这件事，一次，有界。

基线 = `r5_baseline/prediction.npy`（sha 63084c26…，我在 T15 真推理相位亲手产出）。
       ⚠ 该目录的 raw_fields.zip 是沙盒副本重建的（76b11f98，mtime 进了 zip 条目头），
         与磁盘真件 9caed34e 字节不同但**内容同一**（105 个 nc 逐位相同）。
         本探针**不**用它，改用真夹具目录重建（→ 9caed34e），把变量钉死。
"""
import hashlib
import io
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(r"D:\大三上\科研\降水预测-新任务")
BACKEND = REPO / "backend"
FIX = REPO / "摸库交付_2026-09-15" / "03_夹具"
MAT = FIX / "rain" / "CIOmode_1982_2017.mat"
NC = FIX / "U850"
BASELINE = Path(r"C:\Users\Dong_\AppData\Local\Temp\r5_baseline")
BASE_SHA = "63084c26ceb6a9effcb470deffac4ee48490e47b6b3a9cf34bd51a21153138bd"

sys.path.insert(0, str(BACKEND))

TMP = Path(tempfile.mkdtemp(prefix="t19_e2e_"))
CH = TMP / "chain_jobs"
LP = TMP / "prediction_jobs"
TR = TMP / "truth"
for d in (CH, LP, TR):
    d.mkdir(parents=True, exist_ok=True)

os.environ["CHAIN_JOB_ROOT"] = str(CH)
os.environ["PREDICTION_JOB_ROOT"] = str(LP)
os.environ["TRUTH_DIR"] = str(TR)
os.environ["CIOPROJ_MAT"] = str(MAT)
os.environ["CIOPROJ_ALLOW_MOCK"] = "1"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import chain as ch  # noqa: E402

print("chain.__file__   =", ch.__file__)
print("chain sha256     =", hashlib.sha256(Path(ch.__file__).read_bytes()).hexdigest())
assert ch.JOB_ROOT == CH.resolve(), "隔离失败"

import numpy as np  # noqa: E402

archived = np.load(BASELINE / "prediction.npy", allow_pickle=False)
print("归档 prediction.npy sha =", hashlib.sha256(
    (BASELINE / "prediction.npy").read_bytes()).hexdigest())
assert hashlib.sha256((BASELINE / "prediction.npy").read_bytes()).hexdigest() == BASE_SHA

ncs = sorted(NC.glob("*.nc"))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in ncs:
        zf.write(p, "uwnd_5-9/" + p.name)
zb = buf.getvalue()
zsha = hashlib.sha256(zb).hexdigest()
print("提交 zip sha256   =", zsha, "| nc =", len(ncs))

app = FastAPI()
app.include_router(ch.router)
client = TestClient(app)

print("\n提交新链任务（真推理，无任何打桩）…")
t0 = time.time()
r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                files={"file": ("raw_fields.zip", zb, "application/zip")})
print("POST ->", r.status_code)
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

print("\n" + "=" * 74)
print("  E 段断言（逐字同口径）")
print("=" * 74)
print("  归档 sha256 =", BASE_SHA)
print("  新链 sha256 =", gsha)
print("  形状/dtype  = %s %s | %s %s" % (got.shape, got.dtype,
                                        archived.shape, archived.dtype))
ok_shape = got.shape == archived.shape and got.dtype == archived.dtype
print("  [%s] 形状/dtype 与历史产物一致" % ("OK" if ok_shape else "!!"))
same = bool(ok_shape and np.array_equal(got, archived, equal_nan=False))
if ok_shape:
    dx = float(np.max(np.abs(got.astype(np.float64) - archived.astype(np.float64))))
else:
    dx = float("nan")
print("  [%s] prediction.npy 与历史 /api/predict 产物逐位一致（验收 1） 最大差 %.3e"
      % ("OK" if same else "!!", dx))

# 隔离自检
print("\n  隔离自检：chain 任务根 =", ch.JOB_ROOT)
print("  真 uploads/ 未被写 =",
      not (BACKEND / "uploads" / "chain_jobs" / jid).exists())

ok = same and (d["status"] == "completed")
print("\nE_SECTION_EQUIV =", ok)
sys.exit(0 if ok else 1)
