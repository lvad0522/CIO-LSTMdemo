# -*- coding: utf-8 -*-
"""独立 CIO 诊断 API 契约测试。

运行：python test_cio_diagnostics.py
"""

import io
import time

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cio_diagnostics


app = FastAPI()
app.include_router(cio_diagnostics.router)
client = TestClient(app)


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


if __name__ == "__main__":
    test_static_contracts()
    test_npy_job_without_model()
    test_spectrum_requires_a_complete_block()
    test_sst_modes_fail_without_fallback()
    print("test_cio_diagnostics: ALL PASS")
