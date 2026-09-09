"""Mock 数据生成器 —— 复刻前端 mock/data.js 的逻辑。

有真实数据后，切换 main.py 中 from mock_data import ... 改为 from real_data import ...。
"""

import math
import random

S2S_MODELS = [
    "BOM", "CMA", "CNRM", "ECCC", "ECMWF",
    "HMCR", "ISAC", "JMA", "KMA", "NCEP", "UKMO",
]


class ValidationError(ValueError):
    """参数校验错误，main.py 中捕获后转为 HTTP 400。"""


VALID_REGIONS = {"india", "eastasia"}


def _check_region(region):
    if region not in VALID_REGIONS:
        raise ValidationError(f"无效区域: '{region}'，可选: india, eastasia")


class SeededRandom:
    """确定性伪随机，保证每次生成数据一致（复刻 JS 的 seededRandom）。"""

    def __init__(self, seed):
        self.s = seed

    def random(self):
        self.s = (self.s * 9301 + 49297) % 233280
        return self.s / 233280


def gen_cio_time_series(filter_low=0.02, filter_high=0.1, cio_region='tropical', length=2240):
    """生成 CIO 指数时序曲线，参数改变时波形和 seed 都会变化。"""
    seed = hash((round(filter_low, 3), round(filter_high, 3), cio_region)) % 1000000
    rng = SeededRandom(seed)
    # 滤波参数影响主导周期：低频越低周期越长，高频越高短周期越强
    low_period = max(50, int(1.0 / filter_low)) if filter_low > 0 else 300
    high_period = max(5, int(1.0 / filter_high)) if filter_high > 0 else 10
    data = []
    for t in range(length):
        trend = math.sin(t / low_period * math.pi) * 0.5
        wave1 = math.sin(t / high_period * math.pi * 2) * 0.3
        wave2 = math.sin(t / (high_period // 2 + 1) * math.pi * 2) * 0.1
        noise = (rng.random() - 0.5) * 0.2
        data.append(trend + wave1 + wave2 + noise)
    return data


def gen_cio_correlation_map(significance=0.95, cio_region='tropical'):
    """生成 CIO-降水相关空间分布，参数改变时空间格局变化。"""
    rows, cols = 40, 80
    seed = hash((round(significance, 3), cio_region)) % 1000000
    rng = SeededRandom(seed)
    # 显著性越高，信号越强、噪声越小
    signal_strength = 0.5 + significance * 0.2
    noise_amp = 0.25 - significance * 0.1
    data = []
    for i in range(rows):
        row = []
        for j in range(cols):
            lat_factor = math.sin((i / rows) * math.pi)
            lon_factor = math.sin((j / cols) * math.pi * 0.8 + 0.5)
            v = lat_factor * lon_factor * signal_strength + (rng.random() - 0.5) * noise_amp
            row.append(max(-1, min(1, v)))
        data.append(row)
    return {"data": data, "rows": rows, "cols": cols}


def gen_skill_map(region="india"):
    _check_region(region)
    rows = 50 if region == "india" else 25
    cols = 80 if region == "india" else 60
    rng = SeededRandom(hash(region) % 1000000)
    data = []
    for i in range(rows):
        row = []
        for j in range(cols):
            lat_center = math.sin((i / rows) * math.pi)
            lon_center = math.sin((j / cols) * math.pi * 0.7 + 0.3)
            v = 0.15 + lat_center * lon_center * 0.55 + (rng.random() - 0.5) * 0.12
            row.append(max(0, min(0.85, v)))
        data.append(row)
    return {"data": data, "rows": rows, "cols": cols}


def gen_time_series(region="india", grid_i=25, grid_j=40, lead="pre6"):
    seed = grid_i * 1000 + grid_j + hash((region, lead)) % 1000000
    rng = SeededRandom(seed)
    length = 112
    # lead 越长，预测误差越大
    lead_factor = 1.0 if lead == "pre6" else 1.8
    real = []
    pred = []
    for t in range(length):
        base = (3 + math.sin(t / 15 * math.pi * 2) * 2.5
                + math.sin(t / 5 * math.pi * 2) * 1.0
                + (rng.random() - 0.5) * 0.5)
        real.append(max(0, base))
        pred.append(max(0, base + (rng.random() - 0.5) * 1.2 * lead_factor))
    return {"real": real, "pred": pred}


def gen_s2s_comparison(lead="pre6", region="india"):
    seed = hash((lead, region)) % 1000000
    rng = SeededRandom(seed)
    # LSTM+CIO 的 r 随 lead 和 region 变化
    lead_factor = 1.0 if lead == "pre6" else 0.85
    region_factor = 1.0 if region == "india" else 0.92
    lstm_r = round(0.62 * lead_factor * region_factor, 3)
    models = [{"name": name, "r": round(0.12 + rng.random() * 0.24, 3)}
              for name in S2S_MODELS]
    models.append({"name": "LSTM+CIO", "r": lstm_r})
    return models


def gen_results_table(year=2019, lead="pre6", region="india"):
    seed = hash((year, lead, region)) % 1000000
    rng = SeededRandom(seed)
    table = []
    for i in range(6):
        table.append({
            "experiment": i + 1,
            "pearsonR": round(0.45 + rng.random() * 0.30, 3),
            "rmse": round(2.0 + rng.random() * 1.5, 2),
            "mae": round(1.5 + rng.random() * 1.0, 2),
        })
    return table
