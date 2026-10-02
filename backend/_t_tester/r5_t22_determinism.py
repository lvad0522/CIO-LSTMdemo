# -*- coding: utf-8 -*-
"""R5 Tester T22：**确定性判别实验** —— 同一个 prepared cio 喂进真推理，跑两遍。

T15 的真推理作业（t16_parity_25stbknw）留下了它的阶段②落盘件 `normalized_window.npy`
（sha ab4217cf…），真推理输出 = 63084c26…。本探针把**同一个 112 点向量**再喂一遍，
看是否复现 63084c26。若不复现 → 真推理**跨进程不确定**，E 段"逐位一致"判据先天会抖。

不写 backend/ 下任何字节；只读真仓库的模型包。
用法：python r5_t22_determinism.py [第几遍]
"""
import hashlib
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROUND = sys.argv[1] if len(sys.argv) > 1 else "1"
REPO = Path(r"D:\大三上\科研\降水预测-新任务")
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))

T15 = Path(r"C:\Users\Dong_\AppData\Local\Temp\t16_parity_25stbknw")
NW = T15 / "chain_jobs" / "4b1d6b6cdcd2444d91fce21690573b10" / "normalized_window.npy"
EXPECTED = "63084c26ceb6a9effcb470deffac4ee48490e47b6b3a9cf34bd51a21153138bd"

import numpy as np  # noqa: E402
import live_prediction as lp  # noqa: E402

print("round        =", ROUND)
print("lp.__file__  =", lp.__file__)
print("  sha256     =", hashlib.sha256(Path(lp.__file__).read_bytes()).hexdigest())
print("MODEL_ARCHIVE=", lp.MODEL_ARCHIVE, "| is_file =", lp.MODEL_ARCHIVE.is_file())

cio = np.load(NW, allow_pickle=False)
print("输入 cio     =", cio.shape, cio.dtype, "| sha256 =",
      hashlib.sha256(NW.read_bytes()).hexdigest())
assert cio.shape == (112,)

print("\n真推理中（约 1-2 分钟）…")
t0 = time.time()
out = lp._run_archive_inference(cio, lambda p, s: None, (33, 97))
print("用时 = %.0fs" % (time.time() - t0))

buf = Path(os.environ["TEMP"]) / ("t22_round%s.npy" % ROUND)
np.save(buf, out, allow_pickle=False)
h = hashlib.sha256(buf.read_bytes()).hexdigest()
print("\n本遍 sha256 =", h)
print("T15  sha256 =", EXPECTED)
print("[%s] 跨进程逐位可复现" % ("OK" if h == EXPECTED else "!!"))
if h != EXPECTED:
    ref = np.load(T15 / "chain_jobs" / "4b1d6b6cdcd2444d91fce21690573b10" / "prediction.npy")
    d = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    print("     与 T15 的最大差 = %.6e  （E 段报的是 2.107e+01）" % d.max())
    print("     不等的格点数 = %d / %d" % (int((d > 0).sum()), d.size))
