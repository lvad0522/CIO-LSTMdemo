# -*- coding: utf-8 -*-
"""R5 Tester T2：只读探针 —— 判定 6 个新增目录里 prediction.npy 的来源。

假设 H_stub: 这些 prediction.npy 是 test_predict_chain.py 的
             `fake_inference` 桩产出 = np.tile(cio, (81, 101, 1)).astype(float32)
假设 H_real: 是真 LSTM 推理产出。

只读：不写 backend/uploads 下任何字节。
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
UP = BACKEND / "uploads"

ROWS, COLS, T = 81, 101, 112
NEW_PRED = {
    "4342f0ff": UP / "prediction_jobs" / "4342f0ffe9014aeb90a10a482441f164" / "prediction.npy",
    "d75b585a": UP / "prediction_jobs" / "d75b585a3c0541e69136e2b5e8823398" / "prediction.npy",
    "e28c48df": UP / "prediction_jobs" / "e28c48df659d4f5395fc3694c8085711" / "prediction.npy",
    "f060de56": UP / "prediction_jobs" / "f060de56ac91496784569a573cd57e8f" / "prediction.npy",
    "f189516c": UP / "prediction_jobs" / "f189516c5f51469d82f15d1acd7a46ef" / "prediction.npy",
    "base":     UP / "prediction_jobs" / "c40bd2a5ebfb4433b375d1f223017bf4" / "prediction.npy",
}
CIO_IN = {
    "d75b585a": UP / "prediction_jobs" / "d75b585a3c0541e69136e2b5e8823398" / "cio_input.npy",
    "e28c48df": UP / "prediction_jobs" / "e28c48df659d4f5395fc3694c8085711" / "cio_input.npy",
    "f060de56": UP / "prediction_jobs" / "f060de56ac91496784569a573cd57e8f" / "cio_input.npy",
    "f189516c": UP / "prediction_jobs" / "f189516c5f51469d82f15d1acd7a46ef" / "cio_input.npy",
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def npy_bytes(a, allow_pickle=False):
    import io
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=allow_pickle)
    return buf.getvalue()


print("=" * 78)
print("T2 只读探针：新增目录 prediction.npy 来源判定")
print("=" * 78)

print("\n[1] prediction.npy 自身结构（判 shape/dtype + 是否沿空间常数）")
preds = {}
for k, p in NEW_PRED.items():
    a = np.load(p, allow_pickle=False)
    preds[k] = a
    # 沿空间轴是否常数：所有 (i,j) 是否等于 (0,0)
    flat_ref = a[0, 0, :]
    spatial_const = bool(np.all(a == flat_ref))
    print("  %-9s shape=%-14s dtype=%-9s min=%.6f max=%.6f mean=%.6f 空间常数=%s"
          % (k, a.shape, a.dtype, a.min(), a.max(), a.mean(), spatial_const))

print("\n[2] 逐位比对：6 个 prediction.npy 的 sha256")
for k, a in preds.items():
    print("  %-9s %s" % (k, sha(npy_bytes(a))))

print("\n[3] H_stub 检验：stub = np.tile(cio, (81,101,1)).astype(float32)")
for k, p in CIO_IN.items():
    raw = np.load(p, allow_pickle=False)
    pred_on_disk = preds[k]
    stub = np.tile(np.asarray(raw, dtype=np.float32), (ROWS, COLS, 1))
    same_arr = bool(stub.shape == pred_on_disk.shape
                    and np.array_equal(stub, pred_on_disk))
    same_bytes = sha(npy_bytes(stub)) == sha(npy_bytes(pred_on_disk))
    print("  %-9s cio=%-10s -> stub 与磁盘 prediction.npy: array_equal=%s bytes_equal=%s"
          % (k, raw.shape, same_arr, same_bytes))
    if not same_arr:
        # 给点诊断：差在哪
        d = np.abs(stub.astype(np.float64) - pred_on_disk.astype(np.float64))
        print("            最大差=%.6e  均值差=%.6e" % (d.max(), d.mean()))

print("\n[4] baseline c40bd2a5 的空间常数性 + 与 4342f0ff 逐位")
b = preds["base"]
print("  base 空间常数=%s" % bool(np.all(b == b[0, 0, :])))
print("  base npy 字节 sha=%s" % sha(npy_bytes(b)))
print("  4342f0ff == base (array) ? %s" % bool(np.array_equal(b, preds["4342f0ff"])))

print("\n[5] 结论")
all_stub = True
for k, p in CIO_IN.items():
    raw = np.load(p, allow_pickle=False)
    stub = np.tile(np.asarray(raw, dtype=np.float32), (ROWS, COLS, 1))
    if not (stub.shape == preds[k].shape and np.array_equal(stub, preds[k])):
        all_stub = False
if all_stub:
    print("  >>> 4 个 npy 任务的 prediction.npy 与 fake_inference 桩输出【逐位一致】")
    print("  >>> 支持 H_stub：由 test_predict_chain.py（其 C2 段打桩）写出")
else:
    print("  >>> 至少一个不匹配 H_stub，见上")
