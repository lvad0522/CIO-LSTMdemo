# -*- coding: utf-8 -*-
"""目标年窗口契约（target-window-contract）。

`/api/chain/jobs/{id}/series` 的 `targetWindow` 与 `/projection-preview` 的
窗口元信息/口径文案。**本文件的存在理由是"窗口起点必须真的按年表定位"**：

  * 判据全部落在**跨来源**上，不是自己和自己比 —— 阶段① 的真年表
    （`dates.json` 的 2352 个真实日期）与阶段② 的定位结果
    （`normalization.selectedRange`）是两条独立路径，逐值相等才算同源；
  * 失败态一律 `available=false + reason`，`start` 必须是 **null**：
    任何"静默退回第 0 点"的实现（`start = 0`）在本文件里必红（见 A3/B2/C4）。

分块：
  A. `.zip`（真年表）：字段齐全、起点 1344（= 12 × 112）、与 selectedRange
     逐值相等、dates 为真实 112 天、monthDay 与真实日期互相印证
  B. `.npy`（均匀年表）：dates=null 但 available=true，monthDay 仍由
     `cioproj.window(lead)` 取
  C. 定位失败三态（缺年 / 该年天数≠112 / 越界）：200 不 500，带 reason，
     且不静默退回 0 点（start/end/dates/monthDay 全 null）+ 正向对照
  D. 只由阶段① 算出：摘掉任务状态里的 `normalization`，targetWindow 逐值不变
  E. `/projection-preview`：targetYear/dates/window/calendarKnown + 三处口径文案

跑法:  python test_target_window.py
"""
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

try:                                    # Windows 控制台默认 GBK，中文断言读不清
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

FAILS = []


def check(cond, label, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "!!", label,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        FAILS.append(label)
    return cond


def find_data():
    """真数据包：U850/ + CIOmode .mat（本文件只认真件）。"""
    env = os.environ.get("CIOPROJ_REAL")
    cands = ([Path(env)] if env else []) + [
        (HERE / rel).resolve()
        for rel in ("../摸库交付_2026-09-15/数据", "../../摸库交付_2026-09-15/数据")
    ]
    for cand in cands:
        if (cand / "U850").is_dir() and (cand / "CIOmode_1982_2017.mat").is_file():
            return cand
    return None


print("test_target_window: 目标年窗口契约自检")
print("=" * 78)

DATA = find_data()
if DATA is None:
    print("⚠  没找到真数据包（摸库交付_2026-09-15/数据/）—— 什么都没验证。")
    print("=" * 78)
    print("判定: 跳过（退出码 2 —— 这不是通过）")
    sys.exit(2)

NC = DATA / "U850"
MAT = DATA / "CIOmode_1982_2017.mat"
os.environ["CIOPROJ_MAT"] = str(MAT)

# 任务目录隔离到临时区（pid 唯一化：并行实例互不干扰），必须在 import 前设好 ——
# `chain.JOB_ROOT` 在 import 期求值。
TMP_BASE = HERE / ("_tmp_target_window_%d" % os.getpid())
shutil.rmtree(TMP_BASE, ignore_errors=True)
JOB_ROOT = TMP_BASE / "chain_jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["CHAIN_JOB_ROOT"] = str(JOB_ROOT)

import chain as ch                                   # noqa: E402
import cioproj                                       # noqa: E402
import live_prediction as lp                         # noqa: E402

ch.JOB_ROOT = JOB_ROOT                               # 与 env 同值，双保险

REAL_JOB_DIR = HERE / "uploads" / "chain_jobs"
if REAL_JOB_DIR.is_dir() and str(JOB_ROOT.resolve()).startswith(
        str(REAL_JOB_DIR.resolve())):
    print("⚠  任务目录没隔离到临时区，拒绝继续（这不是通过）")
    sys.exit(2)

from fastapi import FastAPI                          # noqa: E402
from fastapi.testclient import TestClient            # noqa: E402

app = FastAPI()
app.include_router(ch.router)
client = TestClient(app)
PREFIX = "/api/chain/jobs"

# 只跑①②：本文件验的是窗口元信息，不该加载任何 .pt（也就没理由依赖权重包）。
# 桩既是"用了 onlyProjection 就绝不会被调用"的守卫，也让误调用不会拖成分钟级。
STUB_CALLED = {"n": 0}


def stub_inference(cio, progress_callback=None, span=(3, 95), **kw):
    STUB_CALLED["n"] += 1
    return np.zeros((lp.GRID_ROWS, lp.GRID_COLS, lp.N_STEPS), dtype=np.float32)


real_inference = lp._run_archive_inference
lp._run_archive_inference = stub_inference

TARGET_WINDOW_KEYS = {"available", "year", "lead", "start", "end", "dates",
                      "monthDay", "source", "reason"}


def wait(job_id, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        data = client.get("%s/%s" % (PREFIX, job_id)).json()
        if data.get("status") in ("completed", "failed"):
            return data
        time.sleep(0.1)
    return {"status": "timeout"}


def build_zip(nc_paths, inner="uwnd_5-9"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in nc_paths:
            zf.write(path, "%s/%s" % (inner, path.name))
    return buf.getvalue()


def npy_bytes(array):
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def expected_month_days(lead_index):
    """`cioproj.window(lead)` 口径下的 (月, 日) 序列 —— **独立于被测实现**的
    复算：直接照 `cioproj.load_field` 的取日规则（每日序 2..29）走。"""
    cioproj.window(lead_index)
    return [[month, day] for month in cioproj.MONTHS
            for day in range(2, 30) if cioproj.in_window(month, day, lead_index)]


def series_window(job_id):
    response = client.get("%s/%s/series" % (PREFIX, job_id))
    return response, (response.json() if response.status_code == 200 else None)


def inject_calendar(job_id, days_per_year):
    """把 `daysPerYear` 塞进任务状态（阶段① 年表就是从这里读的）。

    走 `_update_job` 这一条写状态的既有通道，不直接改 `_window_calendar` ——
    被测的仍是真实分支逻辑。
    """
    state = dict(ch._get_job(job_id).get("projection") or {})
    ch._update_job(job_id, projection={**state, "daysPerYear": days_per_year})


def restore_calendar(job_id, state):
    ch._update_job(job_id, projection=state)


NCS = sorted(NC.glob("*.nc"))
ZIP_BYTES = build_zip(NCS)
print("数据根 = %s（%d 个 nc）" % (DATA, len(NCS)))

try:
    # ======================================= A. `.zip` 入口：真年表定位
    print("A. `.zip` 入口（真年表）：targetWindow 与阶段② 同源")
    # (pre3, 2012)：2012 之前有 12 个完整年 ⇒ 起点 = 12 × 112 = 1344（非 0，
    # 于是"起点写死 0"的实现必红）。年份可用性走真实受理闸门，不写死放行。
    r = client.post("%s?year=2012&lead=pre3&which=u850&onlyProjection=true" % PREFIX,
                    files={"file": ("cesm.uwnd.5-9.zip", ZIP_BYTES, "application/zip")})
    if r.status_code != 202:
        print("⚠  (pre3,2012) 未受理（%s）：本机权重/数据盘面不满足前提 —— "
              "这不是通过" % r.status_code)
        print("   %s" % r.text[:200])
        print("=" * 78)
        print("判定: 跳过（退出码 2 —— 这不是通过）")
        sys.exit(2)
    zip_job_id = r.json()["jobId"]
    zip_job = wait(zip_job_id)
    check(zip_job["status"] == "completed" and zip_job["stageIndex"] == 2,
          "A0：`.zip` 只跑①② 的链路完成（completed / stageIndex=2）",
          (zip_job.get("status"), zip_job.get("stageIndex"), zip_job.get("error")))
    check(STUB_CALLED["n"] == 0, "A0：onlyProjection 从未调用推理（未加载 .pt）",
          STUB_CALLED["n"])

    job_dir = JOB_ROOT / zip_job_id
    series_on_disk = np.load(job_dir / "series.npy", allow_pickle=False)
    dates_on_disk = ch.diagnostics._read_json(job_dir / "dates.json")
    norm = zip_job.get("normalization") or {}
    r, doc = series_window(zip_job_id)
    check(r.status_code == 200, "A1：/series -> 200", r.text[:160])

    tw = (doc or {}).get("targetWindow")
    check(isinstance(tw, dict), "A1：/series 新增 targetWindow 对象",
          sorted(doc or {}))
    if not isinstance(tw, dict):
        raise SystemExit(1)
    check(set(tw) == TARGET_WINDOW_KEYS,
          "A1：targetWindow 字段齐全（available/year/lead/start/end/dates/"
          "monthDay/source/reason），不多不少", sorted(set(tw) ^ TARGET_WINDOW_KEYS))
    check(tw["available"] is True and tw["reason"] is None,
          "A1：定位成功 -> available=true / reason=null",
          (tw["available"], tw["reason"]))
    check(tw["year"] == 2012 and tw["lead"] == "pre3",
          "A1：year/lead 取自任务**实际**组合（不是查询参数默认值）",
          (tw["year"], tw["lead"]))

    # --- A2：与阶段② 同源（单一定位路径），逐值相等
    locate = lp.locate_year_window(
        (zip_job.get("projection") or {}).get("daysPerYear"),
        tw["year"], int(series_on_disk.size))
    check(tw["start"] == 1344 and tw["end"] == 1344 + 112,
          "A2：起点 = 12 × 112 = 1344，窗口左闭右开 [1344, 1456)",
          (tw["start"], tw["end"]))
    check(tw["start"] == locate == norm.get("windowStart")
          and tw["end"] == norm.get("selectedRange", [None, None])[1]
          and [tw["start"], tw["end"]] == norm.get("selectedRange"),
          "A2：targetWindow 与阶段② selectedRange **逐值相等**"
          "（locate_year_window 单一定位路径）",
          (tw["start"], tw["end"], locate, norm.get("selectedRange"),
           norm.get("windowStart")))
    check(tw["start"] != 0,
          "A2：起点**不是** 0（把窗口起点退回第 0 点的实现会在这里红）", tw["start"])

    # --- A3：dates = 该年 112 天真实日期
    expect_dates = dates_on_disk[tw["start"]:tw["end"]]
    check(tw["dates"] == expect_dates and len(tw["dates"]) == 112,
          "A3：dates = dates.json[start:end] 的 112 个真实日期",
          (len(tw["dates"] or []), (tw["dates"] or [None])[:1],
           (expect_dates or [None])[:1]))
    check(tw["dates"][0] == "2012-05-27" and tw["dates"][-1] == "2012-09-26",
          "A3：窗口首尾 = 2012-05-27 … 2012-09-26（pre3 口径："
          "(5,30-3)…(9,29-3)）",
          (tw["dates"][0], tw["dates"][-1]))

    # --- A4：monthDay = window(lead) 口径，且与真实日期互相印证
    expect_md = expected_month_days(3)
    check(tw["monthDay"] == expect_md and len(tw["monthDay"]) == 112,
          "A4：monthDay = cioproj.window(3) 口径的 112 个 (月, 日)",
          (len(tw["monthDay"] or []), (tw["monthDay"] or [None])[:2],
           expect_md[:2]))
    check([[int(d[5:7]), int(d[8:10])] for d in tw["dates"]] == tw["monthDay"],
          "A4：真实日期推出的 (月, 日) 与窗口口径**逐值相等**"
          "（两条独立推导不许分叉）")
    check(tw["source"] == "u850-calendar"
          and tw["source"] == ch._window_calendar(zip_job_id, series_on_disk)["source"],
          "A4：source 与阶段② 实际选中的年表同源（u850-calendar）",
          (tw["source"], ch._window_calendar(zip_job_id, series_on_disk)["source"]))

    # ======================================= B. `.npy` 入口：均匀年表
    #
    # 组合**动态取**（权重组是下载态，不写死某个 (lead, year)）：只要 year ≥ 2001
    # 起点就非 0，"退回第 0 点"的实现照样红。pre1 只有冻结归档 2000，所以这里
    # 必须挑别的 lead —— 这也顺带证明 targetWindow 吃的是任务实际组合。
    print("B. `.npy` 入口（均匀年表）：无 dates 也能给 monthDay")
    catalog = client.get("/api/chain/models").json()
    _avail = [(item["lead"], y["year"]) for item in catalog.get("leads", [])
              for y in item["years"] if y.get("available") and y["year"] >= 2001]
    if not _avail:
        print("⚠  本机没有任何 year≥2001 的可用权重组 —— 这不是通过")
        print("=" * 78)
        print("判定: 跳过（退出码 2 —— 这不是通过）")
        sys.exit(2)
    _lead, _year = next((c for c in _avail if c == ("pre3", 2007)), _avail[0])
    _lead_index = int(_lead.removeprefix("pre"))
    _expect_start = (_year - lp.SERIES_EPOCH_YEAR) * lp.N_STEPS
    print("  [i] `.npy` 用例组合 = %s/%d（均匀年表起点预期 %d）"
          % (_lead, _year, _expect_start))
    seed = np.random.default_rng(20260922).normal(size=2352)
    r = client.post("%s?year=%d&lead=%s&which=cio&onlyProjection=true"
                    % (PREFIX, _year, _lead),
                    files={"file": ("CIO.npy", npy_bytes(seed))})
    check(r.status_code == 202, "B1：`.npy` 建任务 -> 202", r.text[:160])
    npy_job_id = r.json()["jobId"]
    npy_job = wait(npy_job_id)
    check(npy_job["status"] == "completed", "B1：`.npy` 链路跑完（①②）",
          npy_job.get("error"))
    r, doc = series_window(npy_job_id)
    tw_npy = (doc or {}).get("targetWindow") or {}
    check(doc.get("dates") is None and doc.get("sampleIndices") is not None,
          "B1：`.npy` 入口 /series 的 dates 为 null（旧口径不动）",
          (doc.get("dates"), type(doc.get("sampleIndices")).__name__))
    check(tw_npy.get("available") is True and tw_npy.get("dates") is None,
          "B2：dates 为 null 时 available 仍可为 true，targetWindow.dates=null",
          tw_npy)
    check(tw_npy.get("start") == _expect_start
          and tw_npy.get("end") == _expect_start + lp.N_STEPS,
          "B2：均匀年表定位 %d -> [%d, %d)（同样不许退回 0）"
          % (_year, _expect_start, _expect_start + lp.N_STEPS),
          (tw_npy.get("start"), tw_npy.get("end")))
    check(tw_npy.get("monthDay") == expected_month_days(_lead_index)
          and len(tw_npy.get("monthDay") or []) == 112,
          "B2：monthDay 由 cioproj.window(%d) 取（无 dates 也给得出来）"
          % _lead_index,
          (tw_npy.get("monthDay") or [None])[:2])
    check(tw_npy.get("source") == "uniform-calendar"
          and tw_npy.get("lead") == _lead and tw_npy.get("year") == _year,
          "B2：source 标明均匀年表（uniform-calendar）+ 组合取自任务实际值",
          (tw_npy.get("source"), tw_npy.get("lead"), tw_npy.get("year")))
    check([tw_npy.get("start"), tw_npy.get("end")]
          == (npy_job.get("normalization") or {}).get("selectedRange"),
          "B2：`.npy` 入口同样与阶段② selectedRange 逐值相等",
          (tw_npy.get("start"), tw_npy.get("end"),
           (npy_job.get("normalization") or {}).get("selectedRange")))

    # ======================================= C. 定位失败三态
    print("C. 定位失败三态：available=false + reason，不 500 也不退回 0 点")
    _saved_projection = dict(ch._get_job(npy_job_id).get("projection") or {})
    _cases = [
        ("缺该年", {2000: 112, 2001: 112}, ["不在序列年表里", str(_year)]),
        ("该年天数≠112", {2000: 112, _year: 84}, ["84", "112"]),
        ("越界", {2000: 2296, _year: 112}, ["超出序列长度", "2352"]),
    ]
    for _label, _table, _needles in _cases:
        inject_calendar(npy_job_id, _table)
        r, doc = series_window(npy_job_id)
        tw_bad = (doc or {}).get("targetWindow") or {}
        check(r.status_code == 200,
              "C：%s -> 200（不抛 500）" % _label,
              "%s %s" % (r.status_code, r.text[:160]))
        check(tw_bad.get("available") is False
              and all(n in (tw_bad.get("reason") or "") for n in _needles),
              "C：%s -> available=false 且 reason 说清原因（%s）"
              % (_label, "/".join(_needles)), tw_bad.get("reason"))
        check(tw_bad.get("start") is None and tw_bad.get("end") is None
              and tw_bad.get("dates") is None
              and tw_bad.get("monthDay") is None,
              "C：%s -> 定位失败时 start/end/dates/monthDay 全 null"
              "（**不静默退回第 0 点**）" % _label,
              {k: tw_bad.get(k) for k in
               ("start", "end", "dates", "monthDay")})
        check(tw_bad.get("year") == _year and tw_bad.get("lead") == _lead,
              "C：%s -> 失败时仍报出目标组合（可诊断）" % _label,
              (tw_bad.get("year"), tw_bad.get("lead")))
    # 正向对照：状态还回去之后必须**恢复正常** —— 证明上面三条红是注入的
    # 年表造成的，不是 `available=false` 恒真。
    restore_calendar(npy_job_id, _saved_projection)
    r, doc = series_window(npy_job_id)
    tw_ok = (doc or {}).get("targetWindow") or {}
    check(tw_ok.get("available") is True and tw_ok.get("start") == _expect_start,
          "C4：正向对照 —— 年表还原后 available=true / start=%d"
          "（失败分支不是恒假）" % _expect_start,
          {k: tw_ok.get(k) for k in ("available", "start", "reason")})

    # --- C5：日期索引与年表不一致（dates.json 盖不到窗口）。
    # 年表说 2012 的窗口是 [1344, 1456)，而 dates.json 只到 1349 —— 此时**不许**
    # 切一段半截日期当"真实日期"上屏（那会让 x 轴少 107 天还自称真实）。
    _dates_file = JOB_ROOT / zip_job_id / "dates.json"
    _saved_dates_bytes = _dates_file.read_bytes()
    try:
        _dates_file.write_text(
            json.dumps(dates_on_disk[:tw["start"] + 5]), encoding="utf-8")
        r, doc = series_window(zip_job_id)
        tw_short = (doc or {}).get("targetWindow") or {}
        check(r.status_code == 200 and tw_short.get("available") is False
              and "覆盖不到" in (tw_short.get("reason") or "")
              and tw_short.get("dates") is None and tw_short.get("start") is None,
              "C5：日期索引盖不到窗口 -> available=false + reason，"
              "不切半截日期冒充真实日期",
              (r.status_code, tw_short.get("available"), tw_short.get("reason")))
    finally:
        _dates_file.write_bytes(_saved_dates_bytes)
    r, doc = series_window(zip_job_id)
    tw_restored = (doc or {}).get("targetWindow") or {}
    check(tw_restored == tw,
          "C5：dates.json 还原后 targetWindow 逐值复原（对照）",
          {k: (tw_restored.get(k), tw.get(k)) for k in TARGET_WINDOW_KEYS
           if tw_restored.get(k) != tw.get(k)})

    # ======================================= D. 只由阶段① 算出
    print("D. targetWindow 只由阶段① 数据算出（不读 normalization）")
    _saved_norm = ch._get_job(zip_job_id).get("normalization")
    ch._update_job(zip_job_id, normalization=None)
    try:
        check(ch._get_job(zip_job_id).get("normalization") is None,
              "D1：摘除生效（任务状态里 normalization 已为 None）",
              ch._get_job(zip_job_id).get("normalization"))
        r, doc = series_window(zip_job_id)
        tw_no_norm = (doc or {}).get("targetWindow") or {}
        check(tw_no_norm == tw,
              "D1：摘掉 normalization 后 targetWindow **逐值不变**"
              "（不读阶段② 结果）",
              {k: (tw_no_norm.get(k), tw.get(k)) for k in TARGET_WINDOW_KEYS
               if tw_no_norm.get(k) != tw.get(k)})
    finally:
        ch._update_job(zip_job_id, normalization=_saved_norm)

    # ======================================= E. /projection-preview
    print("E. /projection-preview：窗口元信息与三处口径文案")
    r = client.get("%s/%s/projection-preview?time_index=0" % (PREFIX, zip_job_id))
    check(r.status_code == 200, "E1：/projection-preview -> 200", r.text[:160])
    pv = r.json()
    for _key in ("targetYear", "dates", "window", "calendarKnown"):
        check(_key in pv, "E1：响应补了 %s" % _key, sorted(pv))
    check(pv.get("targetYear") == 2012 and pv.get("calendarKnown") is True,
          "E1：targetYear/calendarKnown 取自任务实际状态",
          (pv.get("targetYear"), pv.get("calendarKnown")))
    check(pv.get("dates") == tw["dates"] and len(pv.get("dates") or []) == 112,
          "E1：dates = 目标年 112 天真实日期，与 /series 的 targetWindow 同源",
          (len(pv.get("dates") or []), (pv.get("dates") or [None])[:1]))
    check(pv.get("date") == pv["dates"][pv["timeIndex"]],
          "E1：既有的 date 仍等于 dates[timeIndex]（口径没被改坏）",
          (pv.get("date"), pv["dates"][pv["timeIndex"]]))
    check(pv.get("window") == [[5, 27], [9, 26]],
          "E1：window = cioproj.window(pre3) = [[5,27],[9,26]]", pv.get("window"))

    stages = {item["key"]: item for item in pv.get("stages", [])}
    check(sorted(stages) == ["contribution", "input", "processed"],
          "E2：三张卡片的 key 不变（前端按它取图）", sorted(stages))
    _in, _mid, _con = (stages.get("input") or {}, stages.get("processed") or {},
                       stages.get("contribution") or {})
    # ① 输入场：必须标明是**目标年真实 U850 场**
    _in_text = "%s %s" % (_in.get("title"), _in.get("description"))
    check("目标年" in _in_text and "真实" in _in_text and "U850" in _in_text
          and "尚未" in _in_text,
          "E2：输入卡片标明「目标年真实 U850 场」（尚未预处理）", _in_text)
    # ② 中间量：必须标明**不是 U850 原始场** + 解释日序平均
    _mid_text = "%s %s" % (_mid.get("title"), _mid.get("description"))
    check("不是 U850 原始场" in _mid_text,
          "E2：中间量卡片标明「不是 U850 原始场」", _mid_text)
    check("日序平均" in _mid_text and ("平均" in _mid_text),
          "E2：中间量卡片解释了日序平均的含义（同一日序跨年求平均的当日气候态）",
          _mid_text)
    # ③ 贡献图：必须标明**等于中间量乘权重**
    _con_text = "%s %s" % (_con.get("title"), _con.get("description"))
    check("等于" in _con_text and "中间量" in _con_text and "权重" in _con_text,
          "E2：贡献图卡片标明「等于中间量乘权重」", _con_text)
    check(pv.get("contributionSum") == float(np.sum(
        np.asarray(_con["data"], dtype=np.float64))),
          "E2：口径文案改了，数值口径没动（contributionSum 仍等于贡献图求和）",
          pv.get("contributionSum"))

    # ======================================= F. 既有字段不许漂
    print("F. 旧字段逐字保留")
    r, doc = series_window(zip_job_id)
    check(set(doc) == {"projectionMode", "calendarKnown", "dates",
                       "sampleIndices", "raw", "normalizedWindow", "targetWindow"},
          "/series 字段集 = 旧 6 键 + targetWindow", sorted(doc))
    check(doc["projectionMode"] == "u850_only" and doc["calendarKnown"] is True
          and doc["sampleIndices"] is None and len(doc["raw"]) == 2352
          and len(doc["normalizedWindow"]) == 112,
          "旧 6 键取值照常", {k: doc.get(k) for k in
                              ("projectionMode", "calendarKnown", "sampleIndices")})

finally:
    lp._run_archive_inference = real_inference
    shutil.rmtree(TMP_BASE, ignore_errors=True)

print("=" * 78)
if FAILS:
    print("失败 %d 项：" % len(FAILS))
    for item in FAILS:
        print("  -", item)
    sys.exit(1)
print("判定: 全部通过")
