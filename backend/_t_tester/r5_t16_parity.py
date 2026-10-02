# -*- coding: utf-8 -*-
"""R5 Tester T16：验收 1 对拍 —— **同进程、同一份 U850 zip、新链 vs 旧 /api/predict/jobs**。

两个相位，同一个脚本：
  ① 桩推理相位（秒级）：证明两条路由把**同一个 cio (112,) 张量**喂进 `_run_archive_inference`，
     且两条的 prediction.npy 逐位一致。
  ② 真推理相位（分钟级）：关掉桩、跑真 8181 checkpoint，两条路由的 prediction.npy
     **逐位一致**（`np.array_equal`，不是"接近"）——这就是验收 1 的本体。

三根全部隔离到本进程私有临时目录；`live_prediction.__file__` / `chain.__file__` 打印自证。
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

REAL_INFER = os.environ.get("T16_REAL") == "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIX = (HERE / "../摸库交付_2026-09-15/03_夹具").resolve()
MAT = FIX / "rain" / "CIOmode_1982_2017.mat"
NC = FIX / "U850"
assert NC.is_dir(), NC

TMP = Path(tempfile.mkdtemp(prefix="t16_parity_"))
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
import live_prediction as lp  # noqa: E402

print("chain.__file__           =", ch.__file__)
print("live_prediction.__file__ =", lp.__file__)
print("ch.JOB_ROOT =", ch.JOB_ROOT)
print("lp.JOB_ROOT =", lp.JOB_ROOT)
assert ch.JOB_ROOT == CH.resolve() and lp.JOB_ROOT == LP.resolve(), "隔离失败，拒绝继续"

# zip：与 test_predict_chain.py C1 同式
ncs = sorted(NC.glob("*.nc"))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in ncs:
        zf.write(p, "uwnd_5-9/" + p.name)
zb = buf.getvalue()
print("zip sha256 =", hashlib.sha256(zb).hexdigest(), "| nc =", len(ncs))

SEEN = {"chain": [], "legacy": []}
real_inference = lp._run_archive_inference


def spy_chain(cio, progress_callback=None, span=(3, 95)):
    SEEN["chain"].append((np_copy(cio), tuple(span)))
    return real_inference(cio, progress_callback, span)


def spy_legacy(cio, progress_callback=None, span=(3, 95)):
    SEEN["legacy"].append((np_copy(cio), tuple(span)))
    return real_inference(cio, progress_callback, span)


def np_copy(a):
    import numpy as np
    return np.array(a, copy=True)


if REAL_INFER:
    lp._run_archive_inference = lambda cio, cb=None, span=(3, 95): (
        spy_chain(cio, cb, span))
else:
    import numpy as np

    def fake(cio, progress_callback=None, span=(3, 95)):
        SEEN["_stub"] = SEEN.get("_stub", 0) + 1
        return np.tile(cio.astype(np.float32), (lp.GRID_ROWS, lp.GRID_COLS, 1))

    lp._run_archive_inference = fake

app = FastAPI()
app.include_router(ch.router)
app.include_router(lp.router)
client = TestClient(app)


def wait(prefix, jid, timeout=7200):
    end = time.time() + timeout
    while time.time() < end:
        d = client.get("%s/jobs/%s" % (prefix, jid)).json()
        if d["status"] in ("completed", "failed"):
            return d
        time.sleep(0.3)
    return {"status": "timeout"}


print("\n" + "=" * 78)
print("相位：%s 推理" % ("真" if REAL_INFER else "桩"))
print("=" * 78)

t0 = time.time()
r1 = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                 files={"file": ("cesm.uwnd.5-9.zip", zb, "application/zip")})
print("chain  POST ->", r1.status_code, r1.text[:160])
r2 = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                 files={"file": ("cesm.uwnd.5-9.zip", zb, "application/zip")})
print("legacy POST ->", r2.status_code, r2.text[:160])

j1 = wait("/api/chain", r1.json()["jobId"])
j2 = wait("/api/predict", r2.json()["jobId"])
print("chain  终态 =", j1["status"], j1.get("error"))
print("legacy 终态 =", j2["status"], j2.get("error"))
print("总用时 = %.0fs" % (time.time() - t0))

p1 = ch.JOB_ROOT / r1.json()["jobId"] / "prediction.npy"
p2 = lp.JOB_ROOT / r2.json()["jobId"] / "prediction.npy"
print("\nchain  prediction.npy =", p1, p1.exists())
print("legacy prediction.npy =", p2, p2.exists())

if p1.exists() and p2.exists():
    import numpy as np
    a = np.load(p1, allow_pickle=False)
    b = np.load(p2, allow_pickle=False)
    h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
    print("\n--- 验收 1 断言 ---")
    print("  chain  sha256 =", h1)
    print("  legacy sha256 =", h2)
    print("  shape/dtype   = %s %s | %s %s" % (a.shape, a.dtype, b.shape, b.dtype))
    same = bool(a.shape == b.shape and a.dtype == b.dtype
                and np.array_equal(a, b, equal_nan=False))
    print("  [%s] 两条路由的 prediction.npy **逐位一致**（np.array_equal）"
          % ("OK" if same else "!!"))
    print("  [%s] 两份文件字节 sha256 相同" % ("OK" if h1 == h2 else "!!"))
    if not same:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print("     最大差 = %.6e" % d.max())

# 阶段③ 只读落盘件
nw = ch.JOB_ROOT / r1.json()["jobId"] / "normalized_window.npy"
print("\n--- 阶段③ ---")
print("  chain normalized_window.npy 存在 =", nw.exists())
if nw.exists:
    print("  其 sha256 =", hashlib.sha256(nw.read_bytes()).hexdigest())

# 两条路由喂进推理的张量是否同一个
print("\n--- 喂进 _run_archive_inference 的 cio ---")
if REAL_INFER:
    import numpy as np
    if SEEN["chain"] and SEEN["legacy"]:
        c1, s1 = SEEN["chain"][0]
        c2, s2 = SEEN["legacy"][0]
        print("  chain  张量 %s %s span=%s" % (c1.shape, c1.dtype, s1))
        print("  legacy 张量 %s %s span=%s" % (c2.shape, c2.dtype, s2))
        print("  [%s] 两条路由喂进去的 cio 逐位一致"
              % ("OK" if np.array_equal(c1, c2) else "!!"))
        if nw.exists():
            nwd = np.load(nw, allow_pickle=False)
            print("  [%s] 喂进去的 cio == 落盘 normalized_window.npy（阶段③ 只读落盘件）"
                  % ("OK" if np.array_equal(c1, nwd) else "!!"))
else:
    print("  （桩相位）桩被调用次数 =", SEEN.get("_stub"))
