# -*- coding: utf-8 -*-
"""投影过程预览的独立 API 契约测试（不依赖 nc、模型权重或真实数据包）。

跑法：`cd backend; python test_projection_preview_contract.py`
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from fastapi import HTTPException

import chain
import cioproj


def _preview_arrays() -> dict:
    """造一份有正、负贡献且在全部 112 步精确闭合的最小诊断产物。"""
    shape = (cioproj.NLAT, cioproj.NLON, 112)
    raw = np.zeros(shape, dtype=np.float64)
    processed = np.zeros(shape, dtype=np.float64)
    weights = np.zeros((cioproj.NLAT, cioproj.NLON), dtype=np.float64)

    weights[0, 0] = 2.0
    weights[0, 1] = -4.0
    weights[0, 2] = 1.5
    processed[0, 0, :] = np.linspace(-1.0, 1.0, 112)
    processed[0, 1, :] = 0.5
    processed[0, 2, :] = 1.0
    contribution = processed * weights[:, :, None]
    projected = np.sum(contribution, axis=(0, 1), dtype=np.float64)
    return {
        "raw": raw,
        "processed": processed,
        "weights": weights,
        "dates": np.asarray(["2000-day-%03d" % i for i in range(112)]),
        "projected": projected,
    }


@contextmanager
def _temporary_preview_job(arrays: dict):
    """注册一个只完成阶段①的内存任务，并在临时目录写入诊断产物。"""
    old_root = chain.JOB_ROOT
    with chain._jobs_lock:
        old_jobs = dict(chain._jobs)
        chain._jobs.clear()

    with tempfile.TemporaryDirectory(prefix="projection-preview-contract-") as tmp:
        root = Path(tmp)
        job_id = "preview-contract"
        job_dir = root / job_id
        job_dir.mkdir()
        np.save(job_dir / "series.npy", arrays["projected"], allow_pickle=False)
        np.savez_compressed(job_dir / "projection_preview.npz", **arrays)
        chain.JOB_ROOT = root
        with chain._jobs_lock:
            chain._jobs[job_id] = {
                "jobId": job_id,
                "stageIndex": 1,
                "projectionMode": "u850_only",
                "calendarKnown": True,
                "projection": {"year": 2000, "lead": 3},
            }
        try:
            yield job_id, job_dir
        finally:
            chain.JOB_ROOT = old_root
            with chain._jobs_lock:
                chain._jobs.clear()
                chain._jobs.update(old_jobs)


def _expect_http_error(call, status_code: int, detail: str) -> None:
    try:
        call()
    except HTTPException as exc:
        assert exc.status_code == status_code
        assert detail in str(exc.detail)
    else:
        raise AssertionError("预期 HTTPException(%d)，调用却成功" % status_code)


def test_valid_preview_returns_float64_statistics_and_average_abs():
    arrays = _preview_arrays()
    with _temporary_preview_job(arrays) as (job_id, _job_dir):
        payload = chain.chain_projection_preview(job_id, time_index=7)
        later_payload = chain.chain_projection_preview(job_id, time_index=89)

    contribution = arrays["processed"] * arrays["weights"][:, :, None]
    expected_average = np.mean(np.abs(contribution), axis=2, dtype=np.float64)
    selected = contribution[:, :, 7]
    expected_positive = float(np.sum(selected[selected > 0], dtype=np.float64))
    expected_negative = float(np.sum(selected[selected < 0], dtype=np.float64))
    expected_net = float(np.sum(selected, dtype=np.float64))

    assert payload["totalSteps"] == 112
    assert payload["date"] == arrays["dates"][7]
    assert payload["dates"] == arrays["dates"].tolist()
    assert payload["selectedContribution"]["positiveSum"] == expected_positive
    assert payload["selectedContribution"]["negativeSum"] == expected_negative
    assert payload["selectedContribution"]["net"] == expected_net
    assert payload["selectedContribution"]["projectedCio"] == arrays["projected"][7]
    assert abs(payload["selectedContribution"]["closureError"]) <= 2e-4
    assert abs(
        payload["selectedContribution"]["positiveSum"]
        + payload["selectedContribution"]["negativeSum"]
        - payload["selectedContribution"]["net"]
    ) <= 2e-4

    window = payload["windowContribution"]
    assert (window["rows"], window["cols"]) == (cioproj.NLAT, cioproj.NLON)
    np.testing.assert_allclose(
        np.asarray(window["averageAbs"], dtype=np.float64), expected_average,
        rtol=0, atol=0,
    )
    assert window["scaleMin"] == 0.0
    assert window["scaleMax"] == float(np.max(expected_average))
    assert window["unit"] == "CIO 贡献"
    assert len(window["lat"]) == cioproj.NLAT
    assert len(window["lon"]) == cioproj.NLON

    # 三张图的色标必须由全窗口，而非当前日期重算。
    assert [stage["scaleMax"] for stage in payload["stages"]] == [
        stage["scaleMax"] for stage in later_payload["stages"]
    ]


def test_preview_rejects_time_index_outside_fixed_112_step_window():
    with _temporary_preview_job(_preview_arrays()) as (job_id, _job_dir):
        _expect_http_error(
            lambda: chain.chain_projection_preview(job_id, time_index=112),
            422,
            "必须小于 112",
        )


def test_preview_rejects_malformed_dates_or_projected_lengths():
    arrays = _preview_arrays()
    arrays["raw"] = arrays["raw"][:-1, :, :]
    with _temporary_preview_job(arrays) as (job_id, _job_dir):
        _expect_http_error(
            lambda: chain.chain_projection_preview(job_id, time_index=0),
            500,
            "raw 必须是 41×81×112",
        )

    arrays = _preview_arrays()
    arrays["dates"] = arrays["dates"][:-1]
    with _temporary_preview_job(arrays) as (job_id, _job_dir):
        _expect_http_error(
            lambda: chain.chain_projection_preview(job_id, time_index=0),
            500,
            "dates 必须与 112 点时间轴等长",
        )

    arrays = _preview_arrays()
    arrays["projected"] = arrays["projected"][:-1]
    with _temporary_preview_job(arrays) as (job_id, _job_dir):
        _expect_http_error(
            lambda: chain.chain_projection_preview(job_id, time_index=0),
            500,
            "projected 必须与 112 点时间轴等长",
        )


def test_preview_rejects_nonfinite_or_nonclosing_artifact():
    arrays = _preview_arrays()
    arrays["processed"][0, 0, 0] = np.nan
    with _temporary_preview_job(arrays) as (job_id, _job_dir):
        _expect_http_error(
            lambda: chain.chain_projection_preview(job_id, time_index=0),
            500,
            "processed 包含 NaN 或 Inf",
        )

    arrays = _preview_arrays()
    arrays["projected"] = arrays["projected"] + 3e-4
    with _temporary_preview_job(arrays) as (job_id, _job_dir):
        _expect_http_error(
            lambda: chain.chain_projection_preview(job_id, time_index=0),
            500,
            "全窗口闭合误差",
        )

    arrays = _preview_arrays()
    arrays["processed"][0, 0, :] = 1e308
    arrays["weights"][0, 0] = 1e308
    arrays["projected"] = np.zeros(112, dtype=np.float64)
    with _temporary_preview_job(arrays) as (job_id, _job_dir):
        _expect_http_error(
            lambda: chain.chain_projection_preview(job_id, time_index=0),
            500,
            "格点贡献包含 NaN 或 Inf",
        )


if __name__ == "__main__":
    test_valid_preview_returns_float64_statistics_and_average_abs()
    test_preview_rejects_time_index_outside_fixed_112_step_window()
    test_preview_rejects_malformed_dates_or_projected_lengths()
    test_preview_rejects_nonfinite_or_nonclosing_artifact()
    print("projection preview contract: OK")
