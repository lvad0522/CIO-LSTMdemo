# -*- coding: utf-8 -*-
"""R5 Tester T18：验收 2 —— 新链阶段① 投影序列 vs 官方 CIOproj pre1 逐点对拍。

只读真 `backend/`（不改一字节），三根隔离到进程私有 tmp。
`chain.__file__` 打印自证；隔离断言不过就拒绝继续。

口径：官方 `CIOproj_-20_20_40_120_1degree_pre1.npy` 是 2352 点（21 年 x 112），
       其 `[:112]` = 2000 年块（与 `01_探针/qa_pre1_2000.py` 的 `cio_n[:112]` 同口径）。
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
GOLD = FIX / "rain" / "20_40_100_125" / "CIOproj_-20_20_40_120_1degree_pre1.npy"
NC = FIX / "U850"

sys.path.insert(0, str(BACKEND))

TMP = Path(tempfile.mkdtemp(prefix="t18_cioproj_"))
CH = TMP / "chain_jobs"
TR = TMP / "truth"
for d in (CH, TR):
    d.mkdir(parents=True, exist_ok=True)

os.environ["CHAIN_JOB_ROOT"] = str(CH)
os.environ["TRUTH_DIR"] = str(TR)
os.environ["CIOPROJ_MAT"] = str(MAT)
os.environ["CIOPROJ_ALLOW_MOCK"] = "1"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import chain as ch  # noqa: E402

print("chain.__file__  =", ch.__file__)
print("chain sha256    =", hashlib.sha256(Path(ch.__file__).read_bytes()).hexdigest())
print("ch.JOB_ROOT     =", ch.JOB_ROOT)
assert ch.JOB_ROOT == CH.resolve(), "隔离失败：拒绝继续"
assert Path(ch.__file__).resolve() == (BACKEND / "chain.py").resolve(), "不是真 backend 的 chain.py"

app = FastAPI()
app.include_router(ch.router)
client = TestClient(app)

ncs = sorted(NC.glob("*.nc"))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in ncs:
        zf.write(p, "uwnd_5-9/" + p.name)
zb = buf.getvalue()
print("zip nc 数 =", len(ncs), "| zip sha256 =", hashlib.sha256(zb).hexdigest())

print("\n[1] 提交 onlyProjection 任务（不加载任何 .pt）")
t0 = time.time()
r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850&onlyProjection=true",
                files={"file": ("cesm.uwnd.5-9.zip", zb, "application/zip")})
print("  POST ->", r.status_code, r.text[:200])
jid = r.json()["jobId"]
d = None
for _ in range(1800):
    d = client.get("/api/chain/jobs/%s" % jid).json()
    if d["status"] in ("completed", "failed"):
        break
    time.sleep(0.3)
print("  终态 =", d["status"], "| error =", d.get("error"), "| 用时 %.0fs" % (time.time() - t0))

s = client.get("/api/chain/jobs/%s/series" % jid).json()
raw = s["raw"]
print("\n[2] /series.raw  长度 =", len(raw), "| 前 5 点 =", [round(v, 8) for v in raw[:5]])

print("\n[3] 官方金标准")
import numpy as np  # noqa: E402

gold_full = np.load(GOLD, allow_pickle=False)
gold = np.ravel(gold_full)[:112]
print("  文件 =", GOLD.name, "| 全长 =", np.ravel(gold_full).size, "| 取 [:112]")
print("  前 5 点 =", [round(float(v), 8) for v in gold[:5]])

print("\n[4] 逐点对拍（口径：raw[:112] vs gold[:112]）")
ours = np.asarray(raw, dtype=np.float64)
n = min(112, ours.size, gold.size)
diff = np.abs(ours[:n] - gold[:n])
print("  参与点数 =", n)
print("  最大绝对差 = %.6e" % diff.max())
print("  平均绝对差 = %.6e" % diff.mean())
print("  [%s] 最大差 < 1e-6" % ("OK" if diff.max() < 1e-6 else "!!"))

print("\n[5] 附：全 2352 点对拍（若 raw 长度够）")
if ours.size >= 2352:
    dd = np.abs(ours[:2352] - np.ravel(gold_full)[:2352])
    print("  最大绝对差 = %.6e  < 1e-6 ? %s" % (dd.max(), dd.max() < 1e-6))
else:
    print("  raw 只有 %d 点（zip 只进 2000 一年），跳过全长对拍" % ours.size)

ok = (d["status"] == "completed") and (diff.max() < 1e-6)
print("\nACCEPTANCE2 =", ok)
sys.exit(0 if ok else 1)
