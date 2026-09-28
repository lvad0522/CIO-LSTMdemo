# -*- coding: utf-8 -*-
"""诊断开关回归：开启中间场采样不得改变默认 U850 投影值。"""

import numpy as np

import cioproj


def test_diagnostics_do_not_change_projection():
    rng = np.random.default_rng(20260921)
    u850 = rng.normal(size=(cioproj.NLAT, cioproj.NLON, 4))
    dom = np.array([1, 2, 3, 4], dtype=int)
    consts = {
        "Umean": 0.25,
        "Ustd": 1.75,
        "e_u850": rng.normal(size=cioproj.N_U850),
    }

    baseline = cioproj.project_u850(u850, dom, consts)
    diagnosed, preview = cioproj.project_u850(
        u850, dom, consts, diagnostic_indices=[0, 2]
    )

    np.testing.assert_array_equal(diagnosed, baseline)
    assert preview["raw"].shape == (cioproj.NLAT, cioproj.NLON, 2)
    assert preview["processed"].shape == (cioproj.NLAT, cioproj.NLON, 2)
    assert preview["weights"].shape == (cioproj.NLAT, cioproj.NLON)
    assert preview["raw"].dtype == np.float64
    assert preview["processed"].dtype == np.float64
    assert preview["weights"].dtype == np.float64

    # 诊断场与投影序列必须共享 float64 数值边界；不能先落成 float32 再靠
    # API 上转来伪装精度。这里独立按格点求和，覆盖 F-order 权重还原后的闭合。
    contribution = preview["processed"] * preview["weights"][:, :, None]
    diagnostic_net = np.sum(contribution, axis=(0, 1), dtype=np.float64)
    np.testing.assert_allclose(
        diagnostic_net,
        diagnosed[[0, 2]],
        rtol=0,
        atol=2e-4,
    )


if __name__ == "__main__":
    test_diagnostics_do_not_change_projection()
    print("diagnostic projection parity: OK")
