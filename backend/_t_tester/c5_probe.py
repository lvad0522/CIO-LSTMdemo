# -*- coding: utf-8 -*-
"""C-5 独立复核：chain.JOB_ROOT 的 env 覆盖 + 默认路径逐字不变 + 不牵连旧路由。

子进程模式（每个 case 一个新解释器，避免 import 期缓存互相污染）：
    python c5_probe.py child      # 按当前 env 打印两个 JOB_ROOT
    python c5_probe.py routes     # 打印 app 的全部路由路径
父进程跑 4 个 env 组合 + 1 个路由表。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
PY = sys.executable


def child_env(chain_root=None, predict_root=None):
    env = dict(os.environ)
    for k in ("CHAIN_JOB_ROOT", "PREDICTION_JOB_ROOT"):
        env.pop(k, None)
    if chain_root:
        env["CHAIN_JOB_ROOT"] = chain_root
    if predict_root:
        env["PREDICTION_JOB_ROOT"] = predict_root
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def run(mode, arg=None, **kw):
    cmd = [PY, str(Path(__file__).resolve()), mode]
    if arg:
        cmd.append(arg)
    out = subprocess.run(cmd,
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(BACKEND),
                         env=child_env(**kw))
    if out.returncode != 0:
        raise SystemExit("child %s 失败 rc=%s\n%s\n%s"
                         % (mode, out.returncode, out.stdout[-2000:], out.stderr[-2000:]))
    return json.loads(out.stdout.strip().splitlines()[-1])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "child":
    sys.path.insert(0, str(BACKEND))
    import chain as ch
    import live_prediction as lp
    print(json.dumps({
        "chain": str(ch.JOB_ROOT),
        "predict": str(lp.JOB_ROOT),
        "chain_old_default": str(lp.BACKEND_DIR / "uploads" / "chain_jobs"),
        "predict_old_default": str(lp.BACKEND_DIR / "uploads" / "prediction_jobs"),
    }, ensure_ascii=False))
    raise SystemExit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "routes":
    sys.path.insert(0, str(BACKEND))
    import main
    # 本机 fastapi 0.139 把 include_router 包成 _IncludedRouter（无 .path/.routes），
    # 遍历 app.routes 看不见子路由；用 openapi() 拿权威的完整路径表。
    print(json.dumps(sorted(main.app.openapi()["paths"]), ensure_ascii=False))
    raise SystemExit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "chainroutes":
    # 指定任意 chain.py（含变异体/返工前副本），打印它的 route 表
    sys.path.insert(0, str(BACKEND))
    import importlib.util
    spec = importlib.util.spec_from_file_location("chain_under_test", sys.argv[2])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chain_under_test"] = mod
    spec.loader.exec_module(mod)
    out = sorted("%s %s" % (",".join(sorted(r.methods)), r.path)
                 for r in mod.router.routes)
    print(json.dumps(out, ensure_ascii=False))
    raise SystemExit(0)

FAILS = []


def check(cond, label, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "!!", label,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        FAILS.append(label)


TMP = Path(os.environ.get("TEMP", "/tmp"))
A, B = str(TMP / "c5_chain_a"), str(TMP / "c5_predict_b")

print("C-5. chain.JOB_ROOT env 覆盖")
c1 = run("child")
print("  ① 都不设: chain=%s" % c1["chain"])
print("            predict=%s" % c1["predict"])
check(c1["chain"] == c1["chain_old_default"],
      "不设 env 时 chain.JOB_ROOT 与旧值**逐字相同**",
      "%r != %r" % (c1["chain"], c1["chain_old_default"]))
check(c1["predict"] == c1["predict_old_default"],
      "不设 env 时 lp.JOB_ROOT 与旧值逐字相同",
      "%r != %r" % (c1["predict"], c1["predict_old_default"]))

c2 = run("child", chain_root=A)
check(c2["chain"] == A, "设 CHAIN_JOB_ROOT 后 chain.JOB_ROOT 生效", c2["chain"])
check(c2["predict"] == c1["predict"],
      "设 CHAIN_JOB_ROOT **不牵连** lp.JOB_ROOT（旧路由）",
      "%r != %r" % (c2["predict"], c1["predict"]))

c3 = run("child", predict_root=B)
check(c3["predict"] == B, "设 PREDICTION_JOB_ROOT 后 lp.JOB_ROOT 生效", c3["predict"])
check(c3["chain"] == c1["chain"],
      "设 PREDICTION_JOB_ROOT **不牵连** chain.JOB_ROOT（两 env 独立）",
      "%r != %r" % (c3["chain"], c1["chain"]))

c4 = run("child", chain_root=A, predict_root=B)
check(c4["chain"] == A and c4["predict"] == B,
      "两个 env 同时设：各归各的", c4)

print("\nC-5b. 三个路由族都在（旧 /api/cio 与 /api/predict 原样保留）")
paths = run("routes")
fam = {}
for p in paths:
    for f in ("/api/chain", "/api/cio", "/api/predict"):
        if p.startswith(f):
            fam[f] = fam.get(f, 0) + 1
print("  openapi 路径总数 = %d；各族端点数 = %s" % (len(paths), fam))
for f in ("/api/chain", "/api/cio", "/api/predict"):
    check(fam.get(f, 0) > 0, "%s 路由族存在（%d 个端点）" % (f, fam.get(f, 0)))
check(fam.get("/api/chain") == 13, "/api/chain 端点数 = 13", fam.get("/api/chain"))
check(fam.get("/api/cio") == 9, "/api/cio 端点数 = 9", fam.get("/api/cio"))
check(fam.get("/api/predict") == 8, "/api/predict 端点数 = 8", fam.get("/api/predict"))
for p in ("/api/predict/jobs", "/api/cio/jobs", "/api/chain/jobs"):
    check(p in paths, "端点 %s 在" % p, "")
(Path(__file__).resolve().parent / "c5_routes.json").write_text(
    json.dumps(paths, ensure_ascii=False, indent=1), encoding="utf-8")

print("\nC-5c. /api/chain 路由面回归：返工前后逐条相同（返工不该改动公开契约）")
MUT = BACKEND.parent / ".harness/tmp/site-chain/mut/chain.py"
cur = run("chainroutes", str(BACKEND / "chain.py"))
old = run("chainroutes", str(MUT))
check(cur == old, "/api/chain 路由表与返工前完全一致（%d 条）" % len(cur),
      set(cur) ^ set(old))

print()
print("=" * 70)
if FAILS:
    print("C-5 失败 %d 项: %s" % (len(FAILS), FAILS))
    raise SystemExit(1)
print("C-5 全部通过")
