# -*- coding: utf-8 -*-
"""Tester 独立验证 ③ ——  验收 3 的**因果性**证明：阶段③ 真的只读落盘件。

不用 monkeypatch 探针，用"改文件 / 挪文件 → 看下游跟不跟着变"：

  实验 A（改② 的产物）：同一个任务 fork 出两份，B 的 series.npy 乘 1.001 后
     只跑②③；若③ 真的读落盘 normalized_window.npy，B 的 prediction 必须与 A 不同。
  实验 B（挪走原料）：把 fork A 的 raw/ 整个移走、只留 normalized_window.npy，
     再跑③；若③ 会自己重算投影，它必须失败/找不到 raw 里的 nc。
  实验 C（改② 的产物再跑③）：把 A 的 normalized_window.npy 乘 1.0005 重跑③，
     预测必须跟着变 —— 证明③ 消费的就是这个文件。

跑法:  cd backend && python _t_tester/causality_stage3.py
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PRED_ROOT = HERE / "_cau_pred"
TRUTH_ROOT = HERE / "_cau_truth"
for p in (PRED_ROOT, TRUTH_ROOT):
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
os.environ["PREDICTION_JOB_ROOT"] = str(PRED_ROOT)
os.environ["TRUTH_DIR"] = str(TRUTH_ROOT)
os.environ["CIOPROJ_MAT"] = str(BACKEND / "assets" / "CIOmode_1982_2017.mat")
os.environ.pop("CIOPROJ_ALLOW_MOCK", None)

from fastapi.testclient import TestClient            # noqa: E402
import chain as ch                                   # noqa: E402
import main                                          # noqa: E402

ZIP = (BACKEND / "uploads" / "prediction_jobs"
       / "30cc03890812494fbf3e101333926ce2" / "raw_fields.zip")
assert ZIP.is_file(), ZIP
zip_bytes = ZIP.read_bytes()

RES = {}
FAILS = []
PASSES = []


def check(cond, label, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "!!", label,
                           "" if cond else "   <- " + str(detail)))
    (PASSES if cond else FAILS).append(label)
    return bool(cond)


def wait(client, jid, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get("/api/chain/jobs/%s" % jid).json()
        if j.get("status") in ("completed", "failed"):
            return j
        time.sleep(2)
    return {"status": "timeout"}


def fork(dst: Path, src: Path, job_id: str = ""):
    """克隆一个已跑完① 的任务目录到新 jobId（raw/ 用硬链接省磁盘，只读不改）。"""
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("series.npy", "dates.json", "completeness.json", "raw_fields.zip"):
        s = src / f
        if s.is_file():
            shutil.copy2(s, dst / f)
    if (src / "raw").is_dir():
        shutil.copytree(src / "raw", dst / "raw", copy_function=os.link)


def register(job_id, job_dir):
    with ch._jobs_lock:
        ch._jobs[job_id] = {
            "jobId": job_id, "status": "running", "stage": "fork", "progress": 50,
            "stageIndex": 2, "filename": "fork.zip", "inputKind": "raw-zip",
            "projectionMode": "u850_only", "createdAt": time.time(),
            "jobDir": str(job_dir), "projection": {"lead": 1, "year": 2000,
                                                   "warning": None},
        }


client = TestClient(main.app)

print("因果性实验：阶段③ 是否只读落盘件")
print("=" * 78)

# ---------- 先跑一次 ①（onlyProjection 会在② 后停下，留给我们 fork） ----------
t0 = time.time()
r = client.post("/api/chain/jobs?year=2000&lead=pre1&which=u850&onlyProjection=true",
                files={"file": ("cesm.uwnd.anom.daily.5-9.zip", zip_bytes,
                                "application/zip")})
assert r.status_code == 202, r.text[:300]
base_id = r.json()["jobId"]
base = wait(client, base_id)
print("[cau] ① 完成 %s 用时 %.1fs" % (base["status"], time.time() - t0))
assert base["status"] == "completed", base.get("error")
BASE_DIR = ch.JOB_ROOT / base_id
series0 = np.load(BASE_DIR / "series.npy", allow_pickle=False)
norm0 = np.load(BASE_DIR / "normalized_window.npy", allow_pickle=False)
print("[cau] series=%s norm=%s" % (series0.shape, norm0.shape))

# ---------- 实验 A：fork A（干净）与 fork B（series 被改） ----------
a_id, b_id = "testerforkA0000000000000000000a", "testerforkB0000000000000000000b"
A = ch.JOB_ROOT / a_id
B = ch.JOB_ROOT / b_id
shutil.rmtree(A, ignore_errors=True)
shutil.rmtree(B, ignore_errors=True)
fork(A, BASE_DIR, a_id)
fork(B, BASE_DIR, b_id)

sB = series0.copy()
# 探针必须是**非均匀**扰动：`_prepare_cio` 用的是 min-max 归一化
#   (base - min) / (max - min)
# 整段乘一个常数在数学上会被完全约掉（实测逐位差 0），拿它当探针测不出任何东西。
# 这里只改一个点，改变 min/max 跨度，一定会让归一化窗口变化。
sB[5] += 0.5 * float(series0.max() - series0.min())
np.save(B / "series.npy", sB, allow_pickle=False)
check(not np.array_equal(np.load(B / "series.npy"), series0),
      "实验A：B 的 series.npy 确实与 A 不同（单点非均匀扰动）")

for jid, d in ((a_id, A), (b_id, B)):
    register(jid, d)
    ch._run_stage2(jid, d, {"warning": None})
    n = np.load(d / "normalized_window.npy", allow_pickle=False)
    RES["norm_" + jid[-1]] = {"shape": list(n.shape), "dtype": str(n.dtype),
                              "sum": float(n.sum())}
    print("[cau] fork %s ② 完成 norm sum=%.6f" % (jid[-1], float(n.sum())))

nA = np.load(A / "normalized_window.npy", allow_pickle=False)
nB = np.load(B / "normalized_window.npy", allow_pickle=False)
check(np.array_equal(nA, norm0),
      "实验A：干净的 fork A ② 产物与 ① 后原始件逐位一致（fork 没引入偏差）")
check(not np.array_equal(nA, nB),
      "实验A：series 单点扰动 → ② 产物跟着变（② 读的是落盘 series.npy）")
print("     norm A[:3]=%s  norm B[:3]=%s  maxdiff=%.3e"
      % (nA[:3], nB[:3], float(np.max(np.abs(nA.astype(float) - nB.astype(float))))))

# ---------- 实验 B：把 A 的 raw/ 挪走，只留 normalized_window.npy，跑③ ----------
tB = time.time()
moved = A / "_moved_raw"
os.rename(A / "raw", moved)
check(not (A / "raw").exists() and moved.is_dir(), "实验B：raw/ 已移出任务目录")
ch._run_stage3(a_id, A, "pre1", 2000)
jobA = ch._get_job(a_id)
# 注意：本脚本手工 register 的 fork 任务不会走 `_run_chain_job` 的收尾，
# 所以 status 一直是 "running"；判据用 stageIndex 与产物本身。
RES["stage3_after_raw_moved"] = {"stageIndex": jobA.get("stageIndex"),
                                 "error": jobA.get("error"),
                                 "seconds": round(time.time() - tB, 1)}
check(jobA.get("stageIndex") == 3 and (A / "prediction.npy").is_file(),
      "实验B：raw/ 不在的情况下 ③ 仍跑完（没有去 raw/ 重算投影）",
      jobA.get("error"))
check(jobA.get("error") is None, "实验B：③ 无 error", jobA.get("error"))
predA = np.load(A / "prediction.npy", allow_pickle=False)
RES["predA"] = {"shape": list(predA.shape), "dtype": str(predA.dtype),
                "sum": float(predA.sum()),
                "isnan": int(np.isnan(predA).sum())}
print("[cau] fork A ③ 完成 用时 %.1fs sum=%.4f" % (time.time() - tB,
                                                   float(predA.sum())))
os.rename(moved, A / "raw")     # 还原，不破坏现场

# ---------- 实验 C：改 A 的 normalized_window.npy，③ 必须跟着变 ----------
tC = time.time()
np.save(A / "normalized_window.npy",
        (norm0.astype(np.float64) * 1.0005).astype(np.float32), allow_pickle=False)
ch._run_stage3(a_id, A, "pre1", 2000)
jobA2 = ch._get_job(a_id)
predA2 = np.load(A / "prediction.npy", allow_pickle=False)
RES["stage3_after_norm_edit"] = {"stageIndex": jobA2.get("stageIndex"),
                                 "seconds": round(time.time() - tC, 1),
                                 "sum": float(predA2.sum())}
check(jobA2.get("stageIndex") == 3 and jobA2.get("error") is None,
      "实验C：改完 normalized_window 后 ③ 正常跑完", jobA2.get("error"))
check(not np.array_equal(predA2, predA),
      "实验C：改了 normalized_window.npy → prediction.npy 跟着变"
      "（③ 消费的就是这个文件）",
      "maxdiff=%.6f" % float(np.max(np.abs(predA2.astype(float)
                                          - predA.astype(float)))))
print("[cau] 实验C 后 sum=%.4f（改前 %.4f）maxdiff=%.6f"
      % (float(predA2.sum()), float(predA.sum()),
         float(np.max(np.abs(predA2.astype(float) - predA.astype(float))))))

print("\n" + "=" * 78)
print(json.dumps(RES, ensure_ascii=False, indent=2))
print("PASS %d / FAIL %d" % (len(PASSES), len(FAILS)))
for f in FAILS:
    print("   !!", f)
(HERE / "causality_result.json").write_text(
    json.dumps({"result": RES, "pass": PASSES, "fail": FAILS},
               ensure_ascii=False, indent=2), encoding="utf-8")
sys.exit(1 if FAILS else 0)
