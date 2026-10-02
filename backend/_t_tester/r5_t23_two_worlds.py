# -*- coding: utf-8 -*-
"""R5 Tester T23：**闭合因果链** —— 两套 .mat 让阶段①② 的落盘件互不相同。

T22 已证真推理**确定性**（同一输入三次跨进程同 sha）。
那么 E 段 2.107e+01 的差异只能来自**输入**。
本探针把「同一份 zip、同一份代码、只换 .mat」的两个世界各跑一次 onlyProjection，
比阶段① `series.npy` 与阶段② `normalized_window.npy`。

世界A = E2E 用的那套：`r5sandbox/摸库交付_2026-09-15/数据/`（真件 mat 6c8b3ed4）
世界B = 我 T15/T18/T19 用的那套：`…/03_夹具/rain/`（夹具 mat a8b25511）
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

SANDBOX = Path(r"C:\Users\Dong_\AppData\Local\Temp\r5sandbox")
BACKEND = SANDBOX / "backend"
DELIV = SANDBOX / "摸库交付_2026-09-15"
MAT_REAL = DELIV / "数据" / "CIOmode_1982_2017.mat"
MAT_FIX = DELIV / "03_夹具" / "rain" / "CIOmode_1982_2017.mat"
GOLD_REAL = DELIV / "数据" / "20_40_100_125" / "CIOproj_-20_20_40_120_1degree_pre1.npy"
GOLD_FIX = DELIV / "03_夹具" / "rain" / "20_40_100_125" / "CIOproj_-20_20_40_120_1degree_pre1.npy"
NC_REAL = DELIV / "数据" / "U850"
NC_FIX = DELIV / "03_夹具" / "U850"

sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

print("mat 真件 sha =", hashlib.sha256(MAT_REAL.read_bytes()).hexdigest())
print("mat 夹具 sha =", hashlib.sha256(MAT_FIX.read_bytes()).hexdigest())
print("NC 真件 数量 =", len(list(NC_REAL.glob("*.nc"))))
print("NC 夹具 数量 =", len(list(NC_FIX.glob("*.nc"))))

g_real = np.load(GOLD_REAL, allow_pickle=False)
g_fix = np.load(GOLD_FIX, allow_pickle=False)
print("\nGOLD 真件", g_real.shape, "范围 [%.4f, %.4f]" % (g_real.min(), g_real.max()))
print("GOLD 夹具", g_fix.shape, "范围 [%.4f, %.4f]" % (g_fix.min(), g_fix.max()))
dn = np.abs(g_real.ravel().astype(np.float64) - g_fix.ravel().astype(np.float64))
print("两个 GOLD 逐点最大差 = %.6e" % dn.max())
# 形状（归一化后）是否一样？若一样则"只是尺度差"，不一样则"数据本身就不同"
def nrm(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.min()) / (v.max() - v.min())
print("归一化到 [0,1] 后最大差 = %.6e  <- 若也很大，两者不是同一个投影的缩放"
      % np.abs(nrm(g_real) - nrm(g_fix)).max())


def run(mat: Path, nc_dir: Path, label: str):
    TMP = Path(tempfile.mkdtemp(prefix="t23_%s_" % label))
    CH = TMP / "chain_jobs"
    TR = TMP / "truth"
    CH.mkdir(parents=True, exist_ok=True)
    TR.mkdir(parents=True, exist_ok=True)
    os.environ["CHAIN_JOB_ROOT"] = str(CH)
    os.environ["TRUTH_DIR"] = str(TR)
    os.environ["CIOPROJ_MAT"] = str(mat)
    os.environ["CIOPROJ_ALLOW_MOCK"] = "1"

    ncs = sorted(nc_dir.glob("*.nc"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ncs:
            zf.write(p, "uwnd_5-9/" + p.name)
    zb = buf.getvalue()

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import importlib
    import chain as ch
    importlib.reload(ch)
    app = FastAPI()
    app.include_router(ch.router)
    client = TestClient(app)

    r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850&onlyProjection=true",
                    files={"file": ("cesm.uwnd.5-9.zip", zb, "application/zip")})
    if r.status_code != 202:
        return None, None, r.status_code, r.text[:200]
    jid = r.json()["jobId"]
    d = None
    for _ in range(1800):
        d = client.get("/api/chain/jobs/%s" % jid).json()
        if d["status"] in ("completed", "failed"):
            break
        time.sleep(0.3)
    if d["status"] != "completed":
        return None, None, d["status"], d.get("error")
    jd = CH / jid
    s = hashlib.sha256((jd / "series.npy").read_bytes()).hexdigest()
    n = hashlib.sha256((jd / "normalized_window.npy").read_bytes()).hexdigest()
    series = np.load(jd / "series.npy", allow_pickle=False)
    nw = np.load(jd / "normalized_window.npy", allow_pickle=False)
    return (s, n, series, nw), None, "completed", None


print("\n" + "=" * 74)
res = {}
for label, mat, ncd in (("REAL", MAT_REAL, NC_REAL), ("FIX", MAT_FIX, NC_FIX)):
    print("跑世界 %s（mat=%s, nc=%d 个）…" % (label, mat.name, len(list(ncd.glob('*.nc')))), flush=True)
    t0 = time.time()
    out, err, st, msg = run(mat, ncd, label.lower())
    print("  终态=%s 用时 %.0fs %s" % (st, time.time() - t0, msg or ""))
    if out:
        res[label] = out
        print("  series.npy sha            =", out[0])
        print("  normalized_window.npy sha =", out[1])
        print("  series 范围 [%.4f, %.4f]  nw 范围 [%.6f, %.6f]"
              % (out[2].min(), out[2].max(), out[3].min(), out[3].max()))

print("\n" + "=" * 74)
if "REAL" in res and "FIX" in res:
    a, b = res["REAL"], res["FIX"]
    print("[系列] 两个世界 series.npy 同 sha ?", a[0] == b[0])
    print("[系列] 逐点最大绝对差 = %.6e"
          % np.abs(np.ravel(a[2]).astype(np.float64) - np.ravel(b[2]).astype(np.float64)).max())
    print("[窗口] 两个世界 normalized_window.npy 同 sha ?", a[1] == b[1])
    print("[窗口] 逐点最大绝对差 = %.6e  <-- 这就是喂进 LSTM 的输入差"
          % np.abs(np.ravel(a[3]).astype(np.float64) - np.ravel(b[3]).astype(np.float64)).max())
    # 各世界 vs 各自 GOLD
    for label, gold in (("REAL", g_real), ("FIX", g_fix)):
        d = np.abs(np.ravel(res[label][2]).astype(np.float64)
                   - np.ravel(gold).astype(np.float64))
        print("[%s] series vs 自家 GOLD 最大差 = %.6e (<1e-6 ? %s)"
              % (label, d.max(), d.max() < 1e-6))
    print("\n对比：T15 夹具世界的 normalized_window = ab4217cf9208dd1be6e947f8b1517cce5d4677c8d3fdfab71bad574fb4798bd0")
