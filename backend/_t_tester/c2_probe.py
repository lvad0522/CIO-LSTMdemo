# -*- coding: utf-8 -*-
"""C-2 独立复核：**递归**展开 /api/chain 响应体，找服务器绝对路径。

不复用开发者的跑法，也不只查 `jobDir` 这一个键名：
把响应体当任意 JSON 走进每一个 dict/list，对**每个字符串值**做标记匹配，
并打印它出现在响应里的完整 JSON 路径（如 result.verification.truthPath）。

用法:  python c2_probe.py
"""
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent            # backend/_t_tester
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

REPO = BACKEND.parent
DATA = REPO / "摸库交付_2026-09-15" / "数据"

# 沙盒建在**仓库外**的系统临时区：本探针一个字节都不落进 repo（比 backend/_tmp_* 更干净）
import tempfile                                          # noqa: E402
TMP = Path(tempfile.gettempdir()) / "chain_c2_probe"
# 三个 env 都必须在 import chain / live_prediction **之前**设好（两模块 import 期求值）
os.environ["CHAIN_JOB_ROOT"] = str(TMP / "chain_jobs")
os.environ["PREDICTION_JOB_ROOT"] = str(TMP / "predict_jobs")
os.environ["TRUTH_DIR"] = str(TMP / "truth")
os.environ["CIOPROJ_MAT"] = str(DATA / "CIOmode_1982_2017.mat")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for sub in ("chain_jobs", "predict_jobs", "truth/pre1/2000"):
    (TMP / sub).mkdir(parents=True, exist_ok=True)

import numpy as np                                     # noqa: E402
import chain as ch                                     # noqa: E402
import live_prediction as lp                           # noqa: E402

# 自检：确认真的隔离到真实任务库之外（否则这个探针会踩真实库）
REAL_UPLOADS = BACKEND / "uploads"
for name, root in (("chain", ch.JOB_ROOT), ("predict", lp.JOB_ROOT)):
    assert REAL_UPLOADS not in root.parents, \
        "%s.JOB_ROOT 没隔离到真实 uploads/ 之外: %s" % (name, root)
    assert REPO not in root.parents, \
        "%s.JOB_ROOT 落在仓库里（应为系统临时区）: %s" % (name, root)
print("隔离: chain.JOB_ROOT=%s" % ch.JOB_ROOT)
print("隔离: lp.JOB_ROOT   =%s" % lp.JOB_ROOT)

from fastapi import FastAPI                            # noqa: E402
from fastapi.testclient import TestClient              # noqa: E402

app = FastAPI()
app.include_router(ch.router)
client = TestClient(app)

# ---- 标记：任何"看着像服务器绝对路径"的东西
MARKERS = {
    "Windows 盘符绝对路径": re.compile(r"[A-Za-z]:[\\/]"),
    "仓库目录名": re.compile(re.escape(REPO.name)),
    "backend + 分隔符": re.compile(r"backend[\\/]"),
    "uploads": re.compile(r"uploads", re.I),
    "chain_jobs": re.compile(r"chain_jobs", re.I),
    "prediction_jobs": re.compile(r"prediction_jobs", re.I),
    "BACKEND 绝对路径字面量": re.compile(re.escape(str(BACKEND).replace("\\", "\\\\"))),
}
# 明确不算泄漏的东西（前端本来就要用的相对 URL）
ALLOW = re.compile(r"^/api/chain/jobs/[0-9a-f]+/download/")


def walk(node, path="$.", hits=None):
    """递归走遍 JSON：每个字符串值都过一遍标记。"""
    if hits is None:
        hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, "%s.%s" % (path, k), hits)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, "%s[%d]" % (path, i), hits)
    elif isinstance(node, str):
        if ALLOW.match(node):
            return hits
        for label, rx in MARKERS.items():
            m = rx.search(node)
            if m:
                hits.append({"jsonPath": path, "marker": label,
                             "match": m.group(0), "value": node[:160]})
                break
    return hits


def build_zip(nc_dir, inner="uwnd_5-9"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(nc_dir.glob("*.nc")):
            zf.write(p, "%s/%s" % (inner, p.name))
    return buf.getvalue()


def wait(job_id, timeout=300):
    import time
    end = time.time() + timeout
    while time.time() < end:
        d = client.get("/api/chain/jobs/%s" % job_id).json()
        if d.get("status") in ("completed", "failed"):
            return d
        time.sleep(0.1)
    return {"status": "timeout"}


real_inference = lp._run_archive_inference


def fake_inference(cio, progress_callback=None, span=(3, 95)):
    return np.tile(np.asarray(cio, dtype=np.float32), (lp.GRID_ROWS, lp.GRID_COLS, 1))


lp._run_archive_inference = fake_inference

ZIP = build_zip(DATA / "U850")
print("真 zip: %d 字节, %d 个 nc" % (len(ZIP), len(sorted((DATA / "U850").glob("*.nc")))))

REPORT = {"post_jobs": [], "get_jobs": [], "other_endpoints": {}}

# ---- ① 全新真 zip 建任务，扫 POST /jobs 的**完整响应体**
r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                files={"file": ("cesm.uwnd.5-9.zip", ZIP, "application/zip")})
print("\nPOST /api/chain/jobs -> %s" % r.status_code)
created = r.json()
REPORT["post_jobs"] = walk(created)
print("  POST /jobs 顶层键: %s" % sorted(created))
print("  POST /jobs 递归命中: %s" % (REPORT["post_jobs"] or "无"))

jid = created["jobId"]

# ---- 让实况场可用（result.verification 才会填满）——这正是"路径可能挪到别的字段"的地方
job1 = wait(jid)
if job1["status"] != "completed":
    print("!! 建任务没跑完: %s" % job1.get("error"))
    sys.exit(3)
norm = np.load(ch.JOB_ROOT / jid / "normalized_window.npy", allow_pickle=False)
truth_dir = TMP / "truth" / "pre1" / "2000"
truth_dir.mkdir(parents=True, exist_ok=True)
np.save(truth_dir / "pre_1_2000(1)_20_40_100_125_0.25_real2.npy",
        np.tile(norm.astype(np.float64), (lp.GRID_ROWS, lp.GRID_COLS, 1)),
        allow_pickle=False)

# ---- ② 再建一个任务（有实况），扫 GET /jobs/{id} 的完整响应体
r2 = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850",
                 files={"file": ("cesm.uwnd.5-9.zip", ZIP, "application/zip")})
jid2 = r2.json()["jobId"]
job2 = wait(jid2)
print("\nGET /api/chain/jobs/%s -> status=%s" % (jid2[:8], job2.get("status")))
REPORT["get_jobs"] = walk(job2)
ver = (job2.get("result") or {}).get("verification") or {}
print("  verification.available = %s" % ver.get("available"))
print("  verification 键 = %s" % sorted(ver))
print("  GET /jobs/{id} 递归命中: %s" % json.dumps(REPORT["get_jobs"],
                                                  ensure_ascii=False, indent=2))

# ---- ③ 顺手扫其余端点（路径也可能藏在别处）
for suffix in ("series", "completeness", "spectrum", "pearson",
               "truth/preview", "preview", "grid"):
    rr = client.get("/api/chain/jobs/%s/%s" % (jid2, suffix))
    if rr.status_code != 200:
        REPORT["other_endpoints"][suffix] = "HTTP %s" % rr.status_code
        continue
    try:
        body = rr.json()
    except Exception:
        REPORT["other_endpoints"][suffix] = "非 JSON（二进制下载）"
        continue
    REPORT["other_endpoints"][suffix] = walk(body)
print("\n其余端点递归命中: %s" % json.dumps(REPORT["other_endpoints"],
                                             ensure_ascii=False, indent=2))

out = Path(__file__).resolve().parent / "c2_probe_result.json"
out.write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n结果写入 %s" % out)

total = len(REPORT["post_jobs"]) + len(REPORT["get_jobs"]) + sum(
    len(v) for v in REPORT["other_endpoints"].values() if isinstance(v, list))
print("=" * 70)
print("绝对路径外泄命中总数 = %d" % total)
lp._run_archive_inference = real_inference
