# -*- coding: utf-8 -*-
"""独立 CIO 诊断 API 契约测试。

运行：python test_cio_diagnostics.py

任务根隔离（R5-T2）：本套件会真跑 `POST /api/cio/jobs`，任务目录一律落在本进程
专属临时区，**绝不写**真 `backend/uploads/cio_jobs/`；跑完有末置自检进退出判定。
"""

import atexit
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ==================================================== 任务根隔离（C-4 同形状 / R5-T2）
# `cio_diagnostics.JOB_ROOT` 在 **import 期求值**（cd:29-33），env 必须在下面那行
# `import cio_diagnostics` **之前**设好 —— 运行期再改模块属性补不住 import 期派生值。
REAL_UPLOADS = Path(__file__).resolve().parent / "uploads"
TMP_BASE = Path(tempfile.mkdtemp(prefix="r5t2_cio_diag_%d_" % os.getpid()))
JOB_ROOT = TMP_BASE / "cio_jobs"
TRUTH_ROOT = TMP_BASE / "truth"
for _d in (JOB_ROOT, TRUTH_ROOT, TMP_BASE / "prediction_jobs"):
    _d.mkdir(parents=True, exist_ok=True)
os.environ["CIO_JOB_ROOT"] = str(JOB_ROOT)
os.environ["PREDICTION_JOB_ROOT"] = str(TMP_BASE / "prediction_jobs")
os.environ["TRUTH_DIR"] = str(TRUTH_ROOT)
atexit.register(shutil.rmtree, TMP_BASE, ignore_errors=True)   # 异常退出也清理

import cio_diagnostics


app = FastAPI()
app.include_router(cio_diagnostics.router)
client = TestClient(app)


# ------------------------------------------------ 真工作区零写入的自检装置
REAL_JOB_DIRS = (REAL_UPLOADS / "chain_jobs",
                 REAL_UPLOADS / "prediction_jobs",
                 REAL_UPLOADS / "cio_jobs")


def _inside(path, base):
    """`path` 是否（严格地）落在 `base` 之下。"""
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


# 守卫用的是**模块内实际求出的** JOB_ROOT（env 没生效就会命中）：否则硬编码的
# `uploads/cio_jobs` 会让本套件一边判绿、一边往真工作区里倒桩产物（R5-T2 实锤）。
if any(_inside(cio_diagnostics.JOB_ROOT, real) for real in REAL_JOB_DIRS):
    print("⚠  任务根没隔离到临时区，自检会往真工作区里写文件：")
    print("     cio_diagnostics.JOB_ROOT = %s" % cio_diagnostics.JOB_ROOT)
    print("   拒绝继续（C-4）—— 这不是通过。")
    sys.exit(2)


def _snapshot_tree(base):
    """目录名全集 + 各条目 mtime + **根目录 mtime**（见兄弟套件同名函数说明）。"""
    plain = str(base)
    if not base.is_dir():
        return {"key": plain, "exists": False, "root_mtime": None, "entries": {}}
    entries = {}
    for path in sorted(base.rglob("*")):
        try:
            entries[str(path)] = path.stat().st_mtime
        except OSError:
            continue
    try:
        root_mtime = base.stat().st_mtime
    except OSError:
        root_mtime = None
    return {"key": plain, "exists": True, "root_mtime": root_mtime,
            "entries": entries}


def _no_write_diff(before, after):
    """比对跑前/跑后快照，返回 (是否零差异, 差异描述)。"""
    if before.get("exists") != after.get("exists"):
        return False, "存在性变了：%r -> %r" % (before["exists"], after["exists"])
    if not before.get("exists"):
        return True, "目录本就不存在（无既有件，亦未新建）"
    b, a = before["entries"], after["entries"]
    parts = []
    stale = sorted(set(b) - set(a))
    fresh = sorted(set(a) - set(b))
    touched = [p for p in sorted(set(b) & set(a)) if b[p] != a[p]]
    if stale:
        parts.append("%d 个条目被删：%s" % (len(stale), stale[:3]))
    if fresh:
        parts.append("%d 个条目新增：%s" % (len(fresh), fresh[:3]))
    if touched:
        parts.append("%d 个条目 mtime 被改：%s" % (len(touched), touched[:3]))
    if before.get("root_mtime") != after.get("root_mtime"):
        parts.append("根目录 mtime 变了（建了又删？）")
    return (not parts), ("；".join(parts) if parts else "无差异")


# 跑前快照：这之后才是全部用例体
REAL_SNAPSHOT_BEFORE = {str(d): _snapshot_tree(d) for d in REAL_JOB_DIRS}


def wait_job(job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/cio/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("CIO job timed out")


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def test_static_contracts() -> None:
    response = client.get("/api/cio/capabilities")
    assert response.status_code == 200, response.text
    capabilities = response.json()
    assert capabilities["channels"]["u850"]["available"] is True
    assert capabilities["channels"]["sst"]["available"] is False
    assert capabilities["jointProjection"]["available"] is False
    assert capabilities["spectrum"]["available"] is True
    assert capabilities["spectrum"]["jointAvailable"] is False
    assert capabilities["spectrum"]["blockSize"] == 112

    response = client.get("/api/cio/reference")
    assert response.status_code == 200, response.text
    reference = response.json()
    assert reference["startDate"] == "1982-01-01"
    assert reference["endDate"] == "2017-12-31"
    assert len(reference["dates"]) == len(reference["values"]) == 13149
    assert reference["dates"][0] == reference["startDate"]
    assert reference["dates"][-1] == reference["endDate"]

    response = client.get("/api/cio/mode/u850")
    assert response.status_code == 200, response.text
    mode = response.json()
    assert mode["rows"] == 41 and mode["cols"] == 81
    assert len(mode["data"]) == 41
    assert all(len(row) == 81 for row in mode["data"])
    assert mode["lat"][0] == -20 and mode["lat"][-1] == 20
    assert mode["lon"][0] == 40 and mode["lon"][-1] == 120
    assert abs(mode["explainedVariancePercent"] - 10.8) < 1e-9

    response = client.get("/api/cio/spectrum?confidence=0.99")
    assert response.status_code == 200, response.text
    spectrum = response.json()
    assert spectrum["available"] is False
    assert spectrum["confidence"] == 0.99
    assert spectrum["confidenceMethod"] is None
    assert "SST" in spectrum["reason"]


def test_npy_job_without_model() -> None:
    time_axis = np.arange(2352, dtype=np.float64)
    amplitudes = 1 + 0.03 * np.floor(time_axis / 112)
    raw = amplitudes * (
        np.sin(2 * np.pi * time_axis / 28)
        + 0.2 * np.sin(2 * np.pi * time_axis / 14)
    )
    response = client.post(
        "/api/cio/jobs",
        files={"file": ("CIOproj_pre1.npy", npy_bytes(raw), "application/octet-stream")},
    )
    assert response.status_code == 202, response.text
    job = wait_job(response.json()["jobId"])
    assert job["status"] == "completed", job
    assert job["projectionMode"] == "uploaded_cio"
    assert job["calendarKnown"] is False
    assert job["result"]["seriesLength"] == 2352

    response = client.get(f"/api/cio/jobs/{job['jobId']}/series")
    assert response.status_code == 200, response.text
    series = response.json()
    assert series["calendarKnown"] is False
    assert series["dates"] is None
    assert len(series["raw"]) == 2352
    assert len(series["normalizedWindow"]) == 112

    response = client.get(f"/api/cio/jobs/{job['jobId']}/completeness")
    assert response.status_code == 200, response.text
    completeness = response.json()
    assert completeness["calendarKnown"] is False
    assert completeness["daysPerYear"] is None
    assert completeness["blockSize"] == 112
    assert completeness["completeBlocks"] == 21

    spectra = {}
    for confidence in (0.90, 0.95, 0.99):
        response = client.get(
            f"/api/cio/jobs/{job['jobId']}/spectrum?confidence={confidence}"
        )
        assert response.status_code == 200, response.text
        spectrum = response.json()
        assert spectrum["available"] is True
        assert spectrum["source"] == "uploaded-cio"
        assert spectrum["blockSize"] == 112
        assert spectrum["blockCount"] == 21
        assert spectrum["actualPower"] is None
        assert spectrum["actualConfidence"] is None
        assert spectrum["confidence"] == confidence
        assert spectrum["confidenceMethod"] == "block-bootstrap-upper-quantile"
        lengths = {
            len(spectrum["periodDays"]),
            len(spectrum["projectedPower"]),
            len(spectrum["projectedConfidence"]),
        }
        assert len(lengths) == 1 and lengths.pop() > 0
        assert np.isfinite(spectrum["periodDays"]).all()
        assert np.isfinite(spectrum["projectedPower"]).all()
        assert np.isfinite(spectrum["projectedConfidence"]).all()
        spectra[confidence] = spectrum

    np.testing.assert_allclose(
        spectra[0.90]["projectedPower"], spectra[0.99]["projectedPower"]
    )
    assert np.any(
        np.asarray(spectra[0.90]["projectedConfidence"])
        != np.asarray(spectra[0.99]["projectedConfidence"])
    )


def test_spectrum_requires_a_complete_block() -> None:
    raw = np.linspace(-1.0, 1.0, 111, dtype=np.float64)
    spectrum = cio_diagnostics._spectrum_payload(
        raw, dates=None, confidence=0.95, source="uploaded-cio"
    )
    assert spectrum["available"] is False
    assert spectrum["blockSize"] == 112
    assert "112" in spectrum["reason"]


def test_sst_modes_fail_without_fallback() -> None:
    payload = npy_bytes(np.arange(112, dtype=np.float64))
    for kind in ("sst", "both"):
        response = client.post(
            f"/api/cio/jobs?which={kind}",
            files={"file": (f"{kind}.npy", payload, "application/octet-stream")},
        )
        assert response.status_code == 501, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "SST_NOT_AVAILABLE"


def test_real_uploads_untouched() -> None:
    """末置自检（C-4 同形状）：真工作区 uploads/ 本轮零写入，**进退出判定**。

    本套件真跑 `POST /api/cio/jobs`，是本轮 R5-T2 的泄漏主角；这里拿跑前/跑后
    快照比对证明没写，而不是"打印一句"。
    """
    print("  隔离自检（C-4）：任务根在临时区，不写 backend/uploads/")
    print("    CIO_JOB_ROOT            -> %s" % cio_diagnostics.JOB_ROOT)
    print("    PREDICTION_JOB_ROOT     -> %s" % os.environ.get("PREDICTION_JOB_ROOT"))
    print("    真工作区（仓库内）        -> %s" % REAL_UPLOADS)
    assert _inside(cio_diagnostics.JOB_ROOT, TMP_BASE), \
        "CIO_JOB_ROOT 环境覆盖没生效：%s" % cio_diagnostics.JOB_ROOT
    problems = []
    for real in REAL_JOB_DIRS:
        ok, why = _no_write_diff(REAL_SNAPSHOT_BEFORE[str(real)], _snapshot_tree(real))
        print("  [%s] 真工作区 %s 本轮一个字节都没被写%s"
              % ("OK" if ok else "!!", real.name, "" if ok else "   <- " + why))
        if not ok:
            problems.append("%s：%s" % (real.name, why))
    assert not problems, problems


if __name__ == "__main__":
    test_static_contracts()
    test_npy_job_without_model()
    test_spectrum_requires_a_complete_block()
    test_sst_modes_fail_without_fallback()
    test_real_uploads_untouched()
    print("test_cio_diagnostics: ALL PASS")
