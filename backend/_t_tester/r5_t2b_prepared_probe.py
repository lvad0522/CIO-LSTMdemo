# -*- coding: utf-8 -*-
"""R5 Tester T2b：只读探针 —— 用 _prepare_cio 后的 112 点做 tile，比对磁盘件。

关键修正：_run_job 里喂给 _run_archive_inference 的是 **prepare() 之后的 cio (112 点)**，
不是 raw cio_input.npy。所以 stub 应该 tile 的是 prepared cio，不是原始输入。
"""
import hashlib
import io
import os
import sys

# 控制台 UTF-8（Windows 中文）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

UP = BACKEND / "uploads"
ROWS, COLS, T = 81, 101, 112

import live_prediction as lp  # noqa: E402

NEW = {
    "4342f0ff": "4342f0ffe9014aeb90a10a482441f164",
    "d75b585a": "d75b585a3c0541e69136e2b5e8823398",
    "e28c48df": "e28c48df659d4f5395fc3694c8085711",
    "f060de56": "f060de56ac91496784569a573cd57e8f",
    "f189516c": "f189516c5f51469d82f15d1acd7a46ef",
    "base":     "c40bd2a5ebfb4433b375d1f223017bf4",
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def nbytes(a):
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


print("=" * 78)
print("T2b: prepared-cio tile 假设检验")
print("=" * 78)

disk = {}
for k, d in NEW.items():
    disk[k] = np.load(UP / "prediction_jobs" / d / "prediction.npy", allow_pickle=False)
print("\n所有磁盘 prediction.npy 的 sha256（再确认一次）:")
for k, a in disk.items():
    print("  %-9s %s" % (k, sha(nbytes(a))))

ref_vec = disk["4342f0ff"][0, 0, :]
print("\n空间常数向量 ref = disk[0,0,:] 前 5 点:", np.round(ref_vec[:5], 8))
print("ref min=%.8f max=%.8f" % (ref_vec.min(), ref_vec.max()))

print("\n[逐个 npy 任务] _prepare_cio(cio_input) -> tile -> 比磁盘")
for k in ("d75b585a", "e28c48df", "f060de56", "f189516c"):
    d = NEW[k]
    raw = np.load(UP / "prediction_jobs" / d / "cio_input.npy", allow_pickle=False)
    cio, norm = lp._prepare_cio(raw)
    stub = np.tile(cio.astype(np.float32), (ROWS, COLS, 1))
    ok_arr = bool(stub.shape == disk[k].shape and np.array_equal(stub, disk[k]))
    ok_byt = sha(nbytes(stub)) == sha(nbytes(disk[k]))
    print("  %-9s raw=%-8s prepared=%s mode=%s" % (k, raw.shape, cio.shape, norm.get("mode")))
    print("            tile(prepared) vs disk: array_equal=%s bytes_equal=%s" % (ok_arr, ok_byt))
    if not ok_arr:
        v = cio.astype(np.float32)
        print("            prepared 前 5 点:", np.round(v[:5], 8))
        print("            与 ref 向量 array_equal =", bool(np.array_equal(v, ref_vec)))

print("\n[zip 任务 4342f0ff] 无 cio_input.npy，比对 ref 向量与其它人的 prepared")
print("  4342f0ff 有 raw/ 子目录:",
      sorted(p.name for p in (UP / "prediction_jobs" / NEW["4342f0ff"]).iterdir()))
print("  raw/ 内:", sorted(p.name for p in (UP / "prediction_jobs" / NEW["4342f0ff"] / "raw").iterdir())[:5],
      "... 共", len(list((UP / "prediction_jobs" / NEW["4342f0ff"] / "raw").iterdir())), "项")
