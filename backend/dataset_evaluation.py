"""Historical prediction/truth evaluation, always for one explicitly selected fold."""
import csv
import io
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
import real_data as D
import verification as V

router = APIRouter(prefix="/api/dataset", tags=["dataset-evaluation"])

def arrays(year, lead, fold):
    D._check_lead_year(lead, year)
    allowed = D._display_folds(lead, year)
    if fold not in allowed:
        raise HTTPException(422, "该实验折不属于本站展示口径")
    p = D._load_npy(lead, year, fold, "predict2")
    t = D._load_npy(lead, year, fold, "real2")
    if p.shape != t.shape or p.shape != (81, 101, 112):
        raise HTTPException(422, "预测与实况形状不匹配")
    return p, t

@router.get("/{year}/{lead}/{fold}/preview")
def preview(year: int, lead: int, fold: int, time_index: int = Query(0, ge=0, lt=112)):
    p, _ = arrays(year, lead, fold)
    return frame(p, time_index)

@router.get("/{year}/{lead}/{fold}/truth/preview")
def truth_preview(year: int, lead: int, fold: int, time_index: int = Query(0, ge=0, lt=112)):
    _, t = arrays(year, lead, fold)
    return frame(t, time_index)

def frame(values, index):
    v = values[:, :, index]
    finite = v[np.isfinite(v)]
    if not finite.size:
        raise HTTPException(422, "当前日期没有有效数据")
    return dict(timeIndex=index, rows=81, cols=101, min=float(finite.min()), max=float(finite.max()),
                data=[[float(x) if np.isfinite(x) else None for x in row] for row in v])

@router.get("/{year}/{lead}/{fold}/focus-region")
def focus(year: int, lead: int, fold: int):
    p, t = arrays(year, lead, fold)
    # Match the original r channel of this fold, never the pointwise fold maximum.
    r = D._load_npy(lead, year, fold, "pearson2")[:, :, 2]
    result = V.focus_region_summary(p, t, r)
    result["dates"] = [f"{year}-{m:02d}-{d:02d}" for m in range(6, 10) for d in range(2, 30)]
    return result

@router.get("/{year}/{lead}/{fold}/download/focus-region")
def download_focus(year: int, lead: int, fold: int):
    data = focus(year, lead, fold)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["date", "region_mean_prediction_mm_per_day", "region_mean_truth_mm_per_day"])
    writer.writerows(zip(data["dates"], data["pred"], data["truth"]))
    return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=focus_pre{lead}_{year}_fold{fold}.csv"})
