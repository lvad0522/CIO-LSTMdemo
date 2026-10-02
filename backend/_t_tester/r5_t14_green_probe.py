# -*- coding: utf-8 -*-
"""R5 Tester T14：GREEN 边针对性复验（自包含，不 import test_chain_api 以免被其顶层改根）。

口径与 test_chain_api.py 的 A6 段**逐字相同**：同样的 LEAK_MARKERS / _DRIVE_RE /
递归扫描语义 / matPath==basename 断言 / sandboxRoot 不在断言。
"""
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIX = (HERE / "../摸库交付_2026-09-15/03_夹具").resolve()
MAT = FIX / "rain" / "CIOmode_1982_2017.mat"
NC = FIX / "U850"
os.environ["CIOPROJ_MAT"] = str(MAT)
os.environ["CIOPROJ_ALLOW_MOCK"] = "1"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import live_prediction as lp  # noqa: E402

# ---- 与 test_chain_api.py:64-121 逐字相同的扫描口径 ----
LEAK_MARKERS = (
    str(HERE.resolve()),
    str(HERE.parent.resolve()),
    "backend\\", "backend/",
    "uploads\\", "uploads/",
)
_DRIVE_RE = re.compile(r"[A-Za-z]:[\\/]")


def _leak_hits(node, path=()):
    hits = []
    if isinstance(node, dict):
        for key, value in node.items():
            hits.extend(_leak_hits(value, path + (str(key),)))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            hits.extend(_leak_hits(value, path + ("[%d]" % index,)))
    elif isinstance(node, str):
        merged = node.replace("\\", "/")
        if any(m.replace("\\", "/") in merged for m in LEAK_MARKERS) \
                or _DRIVE_RE.search(node):
            hits.append((".".join(path), node))
    return hits


print("live_prediction.__file__ =", lp.__file__)
print("lp.JOB_ROOT              =", lp.JOB_ROOT)
print("MAT                      =", MAT)
print("LEAK_MARKERS             =", LEAK_MARKERS)

ncs = sorted(NC.glob("*.nc"))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in ncs:
        zf.write(p, "uwnd_5-9/" + p.name)
zb = buf.getvalue()


def fake_inference(cio, progress_callback=None, span=(3, 95)):
    if progress_callback:
        progress_callback(span[0], "桩推理开始")
        progress_callback(span[1], "桩推理结束")
    import numpy as np
    return np.tile(cio.astype(np.float32), (lp.GRID_ROWS, lp.GRID_COLS, 1))


lp._run_archive_inference = fake_inference

app = FastAPI()
app.include_router(lp.router)
client = TestClient(app)

r = client.post("/api/predict/jobs?year=2000&lead=pre1&which=u850",
                files={"file": ("cesm.uwnd.5-9.zip", zb, "application/zip")})
print("\nPOST ->", r.status_code)
created = r.json()
print("--- 202 受理响应 projection ---")
print("   ", created.get("projection"))
print("    202 递归扫命中 =", _leak_hits(created), "（A6 段注释说明：202 本来就瘦，故必为空）")

jid = created["jobId"]
for _ in range(900):
    d = client.get("/api/predict/jobs/%s" % jid).json()
    if d["status"] in ("completed", "failed"):
        break
    time.sleep(0.2)

print("\n=== 终态响应 ===")
print("  status =", d["status"], "| error =", d.get("error"))
proj = d.get("projection") or {}
print("  projection.matPath        =", repr(proj.get("matPath")))
print("  projection.matSource      =", repr(proj.get("matSource")))
print("  'sandboxRoot' in proj     =", "sandboxRoot" in proj)
print("  projection 全部键         =", sorted(proj))
hits = _leak_hits(d)
print("  终态递归扫命中            =", hits)

ok1 = len(hits) == 0
ok2 = proj.get("matPath") == MAT.name
ok3 = "sandboxRoot" not in proj
print("\n=== 断言（与 A6 段同口径）===")
print("  [%s] 终态响应递归扫不到任何绝对路径" % ("OK" if ok1 else "!!"))
print("  [%s] 资产身份行照常显示: matPath=%r（期望 %r）"
      % ("OK" if ok2 else "!!", proj.get("matPath"), MAT.name))
print("  [%s] sandboxRoot 键已删" % ("OK" if ok3 else "!!"))
print("\nGREEN =", ok1 and ok2 and ok3)
sys.exit(0 if (ok1 and ok2 and ok3) else 1)
