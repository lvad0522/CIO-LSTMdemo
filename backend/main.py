"""CIO+LSTM 降水预测 — 后端 API 服务。

启动：python main.py          # 开发模式 http://localhost:8000
"""

import io
import logging
import os

import numpy as np
from fastapi import FastAPI, Query, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger("main")

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 数据源切换 ──────────────────────────────────────────────
# 真实数据集已接入（Dataset/），region 仅支持 eastasia（20-40°N, 100-125°E）
# CIO 模块 mock 已关停（2026-08-25）：cioTS/cioCorr 返回 null，避免伪造数据误导
# s2s 数据集无对应数据，由 real_data 内部 mock 降级并带 source 标记
from real_data import (
    ValidationError,
    DataNotFoundError,
    gen_skill_map,
    gen_time_series,
    gen_s2s_comparison,
    gen_results_table,
)

app = FastAPI(title="CIO+LSTM 降水预测 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValidationError)
def validation_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(DataNotFoundError)
def data_not_found_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.on_event("startup")
def verify_dataset_on_startup():
    """启动时执行数据集完整性扫描，输出缺失/意外文件统计（不阻断启动）。"""
    from real_data import verify_dataset
    result = verify_dataset()
    missing = result["missing"]
    logger.info(
        "数据集完整性校验: 期望 %d 文件, 缺失 %d, 意外文件 %d",
        result["total_expected"], len(missing), result["malformed"],
    )
    for path in missing[:10]:
        logger.warning("数据集缺失: %s", path)
    if len(missing) > 10:
        logger.warning("... 其余 %d 个缺失文件省略", len(missing) - 10)


def parse_lead(lead: str) -> int:
    """解析 lead 参数（'pre6' → 6），严格校验 pre1~pre20（拒绝前导零如 pre01）。"""
    if not isinstance(lead, str) or len(lead) > 5:
        raise ValidationError(f"无效提前期参数: '{lead}'，格式应为 pre1~pre20")
    digits = lead[3:] if lead.startswith("pre") else ""
    if not digits.isdigit():
        raise ValidationError(f"无效提前期参数: '{lead}'，格式应为 pre1~pre20")
    n = int(digits)
    if not 1 <= n <= 20 or (len(digits) > 1 and digits.startswith("0")):
        raise ValidationError(f"无效提前期: '{lead}'，可选 pre1~pre20（不允许前导零）")
    return n


@app.get("/")
def root():
    return {"status": "ok", "service": "CIO+LSTM 降水预测 API"}


@app.get("/api/run")
def run_prediction(
    year: int = Query(2019, description="预测年份 2000-2019"),
    lead: str = Query("pre6", description="提前时间 pre1~pre20"),
    region: str = Query("eastasia", description="区域（数据集仅支持 eastasia）"),
    filter_low: float = Query(0.02, description="CIO 滤波低频"),
    filter_high: float = Query(0.1, description="CIO 滤波高频"),
    cio_region: str = Query("tropical", description="CIO 海域范围"),
    significance: float = Query(0.95, description="显著性置信水平"),
):
    """执行一次完整的预测流程，返回所有结果。"""
    if region != "eastasia":
        raise ValidationError(
            f"无效区域: '{region}'。数据集仅支持 eastasia（20-40°N, 100-125°E）"
        )
    lead_n = parse_lead(lead)
    return {
        # CIO 模块：mock 已关停，无真实 SST+Uwind 计算前返回 null（前端显示"暂无数据"提示）
        "cioTS": None,
        "cioCorr": None,
        "skillMap": gen_skill_map(year, lead_n),
        "timeSeries": gen_time_series(year, lead_n, 40, 50),  # 默认格点：网格中心
        "s2s": gen_s2s_comparison(lead, region),
        "table": gen_results_table(year, lead_n),
    }


@app.get("/api/grid")
def grid_query(
    i: int = Query(40, description="格点行索引 (0-80)"),
    j: int = Query(50, description="格点列索引 (0-100)"),
    region: str = Query("eastasia", description="区域（数据集仅支持 eastasia）"),
    year: int = Query(2019, description="预测年份 2000-2019"),
    lead: str = Query("pre6", description="提前时间 pre1~pre20"),
):
    """查询某个格点的预测 vs 真实时序。"""
    if region != "eastasia":
        raise ValidationError(
            f"无效区域: '{region}'。数据集仅支持 eastasia（20-40°N, 100-125°E）"
        )
    lead_n = parse_lead(lead)
    return gen_time_series(year, lead_n, i, j)


@app.get("/api/s2s")
def s2s_comparison():
    """返回 S2S 模式对比数据。"""
    return gen_s2s_comparison()


@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """接收用户上传的 .npy 数据文件，保存到 backend/uploads/ 并返回数组摘要。"""
    if not file.filename or not file.filename.lower().endswith(".npy"):
        raise HTTPException(400, "仅支持 .npy 文件")
    content = await file.read()
    if not content:
        raise HTTPException(400, "文件内容为空")

    try:
        arr = np.load(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "npy 解析失败，请确认是有效的 NumPy .npy 文件")
    if arr.ndim == 0:
        raise HTTPException(400, "npy 内容为空（0 维标量）")

    safe_name = os.path.basename(file.filename)
    with open(os.path.join(UPLOAD_DIR, safe_name), "wb") as f:
        f.write(content)

    return {
        "status": "ok",
        "filename": file.filename,
        "saved_to": f"backend/uploads/{safe_name}",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
