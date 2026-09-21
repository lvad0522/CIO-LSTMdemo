# -*- coding: utf-8 -*-
"""退役守卫：`main.app` 上 13 条旧任务路由不可达、4 条资产路由仍可用。

背景（RLR-5）：三个兄弟套件（`test_chain_api.py` / `test_predict_chain.py` /
`test_cio_diagnostics.py`）全部**自建内存 app**，不经 `main.py` —— 所以"摘挂载"
这个动作**不会让任何套件变红**，退役本身在此之前是**零覆盖**。本套件补上这块：
它是唯一**额外 `import main`** 并直接对 `main.app` 断言的自检。

覆盖 11 条（对应 `tasks.md` 第 4 节；4.3 拆为 4.3a / 4.3b，另加 4.10）：
  4.2 负向·静态：`openapi()["paths"]` 无 `/api/predict/*`、无 `/api/cio/jobs*`
  4.3a 负向·动态（**同方法**探针）：`TestClient(main.app)` 对 **13 条**旧路径**逐条按
      各自 HTTP 方法**真打，全部须 **404**。**判别力有限，此处口径必须说准**：13 条里
      只有 2 条 POST 路径（`/api/predict/jobs` / `/api/cio/jobs`）真有判别力 —— 它们是
      **受理端点**（`file: UploadFile = File(...)`），缺 file 返回 422、受理成功返回 202，
      **怎么都不会是 404**。另外 **11 条 GET 路径的同方法 404 什么都
      证明不了**：`status_code == 404` **无法区分**「路由不存在」与「路由存在、但 handler
      自报"任务不存在"」（`_get_job` / `_require_*` 抛 `HTTPException(404)`）。
      实测反例（Reviewer R-M3，`.harness/tmp/retire-legacy-job-routes/
      reviewer_probe_hidden_route_r2.py`）：把 11 条 GET 旧路由以 `include_in_schema=False`
      + **真实 handler** 挂回 `main.app`（无双挂载、不碰资产面）时，加 4.10/4.3b 之前的
      本套件 **10/10 全绿**并打印"判定: 全部通过"，而同一次运行里
      `POST /api/predict/jobs/deadbeef -> 405` 正证明该路径上**确有 route 对象**。
      （回头看清楚：R-M2 能被抓住是**特例**——它那条探针 handler 恰好返回 200；
      R-M1 的"有牙"也是假象——唯一红的断言是 `POST /api/predict/jobs 实测 422` 那条
      **双挂载**，7 条 GET 探针当时全过，而它们打的路径那时**确实被挂载着**。）
  4.3b 负向·动态（**交叉方法**探针）：对每条旧路径发一个它退役前**未注册**的方法，
      须**恰好 404**。判别力来自 HTTP 语义的**不对称**：路径存在但方法不允许 ⇒ **405**；
      路径不存在 ⇒ **404**。于是"404 且非 405"是可判定的，而同方法 404 不是。
      **覆盖全部 13 条**（4.3a 只覆盖到 2 条），且**必须带对照断言**（一个已知存在的
      路径在错方法下真返 405），否则"本套件里没出现 405"这个负结论不可判定。
  4.10 负向·**结构**（有效路由表）：递归 walk `main.app` 的**有效路由**，断言不存在任何
      path 以 `/api/predict/` 或 `/api/cio/jobs` 开头的 route **对象**。**不依赖 HTTP
      语义**，故 `include_in_schema=False` 藏不住它 —— 这正是 4.2（查 openapi 表扫不到）
      与 4.3a（同方法打 404 分不清）之间那块盲区的结构性判据，也是 R-M3 的直接克星。
  4.4 正向：4 条资产路由**在表里**且**真 200**
  4.5 反向：oracle 未被误伤 —— `lp.router` 恰 8 条、`cd.router` 恰 9 条，
      且 4 条资产路径是 `cd.router.routes` 的**子集**（是被**复制**而非被**移动**）
  4.7 `asset_router` 自身组成：恰 4 条、prefix 正确、与 `CIO_ASSET_PATHS` 相等、
      且与 `cd.router.routes` 是**同一批对象**（`is` 比较）
  4.8 实现唯一性：每条资产 path 在 `main.app` 上**恰好 1 个** route，其
      `endpoint is cd.<对应 handler>` 且 `endpoint.__module__ == "cio_diagnostics"`
      —— 本条能拦住"只摘 line 40 不摘 line 41"的双挂载反例（那时每条资产会有 2 份）
  4.9 4.3 清单自身的完备性：硬编码的 13 条与退役前公开面**集合相等**，且每条
      方法与旧 router 装饰器一致（防"清单被改窄 ⇒ 断言恒真"）
  附 A 链零回归·静态：13 条 `/api/chain/*` 一条不少（防止摘挂载摘过头）
  附 B oracle 挂载点仍可执行：自建内存 app 挂 `cd.router` + `lp.router`，
      17 条旧契约路径**全在**其 openapi 表里（spec delta「oracle 挂载点仍可执行」的
      可执行形态，**纯静态、不发请求**，故零副作用）

跑法：python test_legacy_route_retirement.py      （也可 pytest 收集）
"""

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:                                    # Windows 控制台默认 GBK，中文断言读不清
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ==================================================== 任务根隔离
# `import main` 会连带 import `cio_diagnostics` / `live_prediction` / `chain`，
# 三者的 JOB_ROOT 都在 **import 期求值**。本套件只做 GET/404 断言，本不该写任何
# 目录；把任务根先重定向到本进程专属临时区，是为了让"本套件零写入"成为**结构性**
# 保证而不是"我记得没写"。env 必须在 import main **之前**设好。
TMP_BASE = Path(tempfile.mkdtemp(prefix="rlr_guard_%d_" % os.getpid()))
for _name in ("cio_jobs", "prediction_jobs", "chain_jobs"):
    (TMP_BASE / _name).mkdir(parents=True, exist_ok=True)
(TMP_BASE / "truth").mkdir(parents=True, exist_ok=True)
os.environ["CIO_JOB_ROOT"] = str(TMP_BASE / "cio_jobs")
os.environ["PREDICTION_JOB_ROOT"] = str(TMP_BASE / "prediction_jobs")
os.environ["CHAIN_JOB_ROOT"] = str(TMP_BASE / "chain_jobs")
os.environ["TRUTH_DIR"] = str(TMP_BASE / "truth")
atexit.register(shutil.rmtree, TMP_BASE, ignore_errors=True)

from fastapi import FastAPI                       # noqa: E402
from fastapi.testclient import TestClient         # noqa: E402

import cio_diagnostics as cd                      # noqa: E402
import live_prediction as lp                      # noqa: E402

# 对 `main.app` 的断言必须 import 真 app —— 这正是本套件与三个兄弟套件的分界。
import main                                       # noqa: E402

ASSET_PATHS = (
    "/api/cio/capabilities",
    "/api/cio/reference",
    "/api/cio/mode/u850",
    "/api/cio/spectrum",
)
LEGACY_PREDICT_PREFIX = "/api/predict/"
LEGACY_CIO_JOBS_PREFIX = "/api/cio/jobs"
CHAIN_PATHS = (
    "/api/chain/jobs",
    "/api/chain/jobs/{job_id}",
    "/api/chain/jobs/{job_id}/completeness",
    "/api/chain/jobs/{job_id}/download/cio",
    "/api/chain/jobs/{job_id}/download/normalized",
    "/api/chain/jobs/{job_id}/download/pearson",
    "/api/chain/jobs/{job_id}/download/prediction",
    "/api/chain/jobs/{job_id}/grid",
    "/api/chain/jobs/{job_id}/pearson",
    "/api/chain/jobs/{job_id}/preview",
    "/api/chain/jobs/{job_id}/series",
    "/api/chain/jobs/{job_id}/spectrum",
    "/api/chain/jobs/{job_id}/truth/preview",
)


# ------------------------------------------------------------------ 工具
def _effective_routes(app):
    """展开 app 上的**有效**路由，穿透 fastapi 0.139 的 `_IncludedRouter` 包装。

    实测（fastapi 0.139.0）：`app.routes` 顶层的子路由是 `_IncludedRouter`，
    它的 `path` 是 `None`，真正的 `APIRoute` 藏在 `.original_router.routes` 里。
    不穿透就会"看不到任何子路由"——而那种情形下 4.8 会因**找不到**资产路由而变红
    （fail-closed），不会静默判绿。
    """
    pending = list(app.routes)
    seen = set()
    while pending:
        route = pending.pop()
        inner = getattr(route, "original_router", None)
        if inner is not None:
            if id(inner) in seen:               # 防御：同一 router 被 include 两次
                continue
            seen.add(id(inner))
            pending.extend(inner.routes)
            continue
        yield route


def _handler_for(router, path):
    """`router` 上 path 对应的 handler（endpoint）对象。"""
    hits = [r for r in router.routes if getattr(r, "path", None) == path]
    assert len(hits) == 1, (
        "%s 上 path=%r 的路由应恰为 1 条，实测 %d 条" % (router, path, len(hits))
    )
    return hits[0].endpoint


def _client():
    return TestClient(main.app)


# ------------------------------------------------------------------ 4.2 负向·静态
def test_paths_have_no_legacy_predict_routes():
    paths = list(main.app.openapi()["paths"])
    left = [p for p in paths if p.startswith(LEGACY_PREDICT_PREFIX)]
    assert not left, "`/api/predict/*` 应已全部退役，仍在 openapi 表内：%s" % left


def test_paths_have_no_legacy_cio_jobs_routes():
    paths = list(main.app.openapi()["paths"])
    left = [p for p in paths if p.startswith(LEGACY_CIO_JOBS_PREFIX)]
    assert not left, "`/api/cio/jobs*` 应已全部退役，仍在 openapi 表内：%s" % left


# ------------------------------------------------------------------ 4.3 负向·动态
# 13 条退役路径 = `cd.router`（5 条）+ `lp.router`（8 条）的全集，**逐条写死**
# 各自的 HTTP 方法。方法与路径都必须与旧 router 的装饰器**逐字一致**：
#   cio_diagnostics.py:543/633/645/665/674（POST + 4 GET）
#   live_prediction.py:883/949/954/977/1023/1074/1102/1118（POST + 7 GET）
# 以上 13 个行号是**照行号去核对**用的锚点（本注释唯一的存在理由），2026-09-21
# 收口轮已 `grep -n "^@router\."` 逐行复核：cio_diagnostics 那 5 个曾错写成
# 531/621/633/653/662（531 实为 `"period": None,`、653 实为 `np.load(...)`、662 实为 `}`），
# live_prediction 那 8 个逐个全对。**改动这两个文件后请同步复核**——锚点陈旧比没有更坏。
# 写成硬编码清单（而不是从 `cd.router.routes` 现取）是刻意的：清单必须**独立于
# 被测对象**，否则"旧 router 被删掉一条"会让清单跟着缩水、断言恒真。
LEGACY_PATHS = (
    ("POST", "/api/cio/jobs"),
    ("GET", "/api/cio/jobs/{job_id}"),
    ("GET", "/api/cio/jobs/{job_id}/series"),
    ("GET", "/api/cio/jobs/{job_id}/completeness"),
    ("GET", "/api/cio/jobs/{job_id}/spectrum"),
    ("POST", "/api/predict/jobs"),
    ("GET", "/api/predict/jobs/{job_id}"),
    ("GET", "/api/predict/jobs/{job_id}/preview"),
    ("GET", "/api/predict/jobs/{job_id}/grid"),
    ("GET", "/api/predict/jobs/{job_id}/pearson"),
    ("GET", "/api/predict/jobs/{job_id}/truth/preview"),
    ("GET", "/api/predict/jobs/{job_id}/download/pearson"),
    ("GET", "/api/predict/jobs/{job_id}/download"),
)
# `{job_id}` 的填充值。注意本套件已把 `*_JOB_ROOT` 全重定向到本进程专属临时区
# （见文件头），故不可能撞上真任务。
# **订正（2026-09-21 收口轮）**：这里原先接着写"'任务不存在'与'路由不存在'在旧 handler
# 里都是 404（`_require_*` 抛 HTTPException(404)），判别力不受影响"—— 前半句对，
# 后半句**说反了**：正因为两者都是 404，**同方法**探针在 11 条 GET 路径上才**不可区分**
# （R-M3 的根因）。判别力只能来自 4.3b（交叉方法 ⇒ 405/404 之分）与 4.10（结构判据）。
PROBE_JOB_ID = "deadbeef"


def _probe_path(template):
    return template.replace("{job_id}", PROBE_JOB_ID)


def _cross_method(template, retired_method):
    """返回 `template` 在**退役前未注册**的一个方法（GET ↔ POST 互换），并就地自证。

    自证不可省：若某路径日后同时注册了 GET 与 POST，互换出来的方法就会是**已注册**的，
    那时"路径挂回 ⇒ 405"判据失效（挂回后它会 200/202 而非 405），断言会以无法归因的
    方式变红或（更糟）失去判别力。故先断言它确实不在旧 router 的方法集里。
    """
    cross = "GET" if retired_method == "POST" else "POST"
    registered = set()
    for r in list(cd.router.routes) + list(lp.router.routes):
        if getattr(r, "path", None) == template:
            registered |= set(r.methods)
    assert cross not in registered, (
        "交叉方法 %s 在旧 router 的同路径 %s 上**已被注册**（%s）—— 『路径存在 ⇒ 405』"
        "判据失效，4.3b 不再可判别" % (cross, template, sorted(registered))
    )
    return cross


def test_legacy_paths_return_404_on_retired_method():
    """4.3a：13 条退役路径**逐条按各自退役前的方法**真打，全部须 404。

    命名如实：本条打的是 13 条**路径**（不再是"2 条 POST"）。但**判别力与其覆盖面
    不成比例**——只有 2 条 POST 路径有判别力（旧 handler 对不存在的任务返回
    422 / 202）；11 条 GET 路径的同方法 404 是**歧义读数**（路由不在 ⇒ 404，路由在
    而 handler 自报"任务不存在" ⇒ 也是 404），见文件头 4.3a 段的实测反例 R-M3。
    真正覆盖全 13 条的动态判据是 4.3b（交叉方法），结构性判据是 4.10。
    """
    with _client() as client:
        for method, template in LEGACY_PATHS:
            path = _probe_path(template)
            r = client.request(method, path)
            assert r.status_code == 404, (
                "%s %s 应返回 404（真退役），实测 %d —— 422 / 405 说明该路由仍被挂载"
                "（典型成因：main.py 只摘了 include live_prediction，漏摘"
                " include cio_diagnostics）；200 说明它以 include_in_schema=False "
                "的隐藏挂载溜了回来 —— openapi 表看不见、真打可达"
                % (method, path, r.status_code)
            )


def test_legacy_paths_return_404_on_unregistered_method():
    """4.3b：**交叉方法**探针 —— 对每条旧路径发一个退役前**未注册**的方法，须恰好 404。

    判别式（与 4.3a 的本质差别）：Starlette 先按 **path** 匹配，命中后若方法不允许
    返回 **405**；path 根本不存在才 **404**。故「405 / 404 之分」正好回答"这条路径上
    到底有没有 route 对象"——**同方法 404 回答不了**（路由不在 ⇒ 404；路由在而 handler
    自报"任务不存在" ⇒ 也是 404）。本条**覆盖全部 13 条**，是 4.3a 判别力薄弱的正面补偿。

    对照断言**不可省**：先证明"405 这个读数在本 app 上确实会出现"，否则"13 条都没
    出现 405"这个负结论**不可判定**（也可能是这个 app 压根不产生 405）。
    """
    with _client() as client:
        # 对照 1：路径存在、只注册了 GET ⇒ 405（`chain.py:546` 是 POST /jobs，
        # `chain.py:619` 的 /jobs/{job_id} 只有 GET；路径参数形态与 11 条旧 GET 同形）
        c1 = client.request("POST", "/api/chain/jobs/%s" % PROBE_JOB_ID)
        assert c1.status_code == 405, (
            "对照组 POST /api/chain/jobs/<id> 应返回 405（该路径存在、只注册了 GET），"
            "实测 %d —— 本 app 若不产生 405，下面的 404 断言全部失去判别力"
            % c1.status_code
        )
        # 对照 2：资产路径与退役与否无关，恒在 ⇒ 另一个必须为 405 的读数
        c2 = client.request("POST", ASSET_PATHS[0])
        assert c2.status_code == 405, (
            "对照组 POST %s 应返回 405（资产面 GET-only、且必然挂载），实测 %d"
            % (ASSET_PATHS[0], c2.status_code)
        )
        for method, template in LEGACY_PATHS:
            cross = _cross_method(template, method)
            path = _probe_path(template)
            r = client.request(cross, path)
            assert r.status_code == 404, (
                "交叉方法 %s %s 应返回 404（该 path 上不应存在任何 route 对象），实测 %d "
                "—— 405 说明路径被挂回（方法不匹配）；200 / 202 / 422 说明它连 %s 都受理了；"
                "三者都可能是 `include_in_schema=False` 的隐藏挂载（openapi 表看不见、真打可达）"
                % (cross, path, r.status_code, cross)
            )


def test_legacy_paths_table_is_complete():
    """4.3 的硬编码清单必须恰好覆盖 13 条退役路径 —— 防止清单被悄悄改窄。

    "清单必须独立于被测对象"（见 `LEGACY_PATHS` 上方注释）的反面代价：清单本身
    可能被改窄而无人发现。这里用**另一份**独立常量（`CHAIN_PATHS` 的兄弟清单，
    即 `cd.router` / `lp.router` 的契约面）做集合相等核对。
    """
    oracle_legacy = {
        "/api/cio/jobs",
        "/api/cio/jobs/{job_id}",
        "/api/cio/jobs/{job_id}/completeness",
        "/api/cio/jobs/{job_id}/series",
        "/api/cio/jobs/{job_id}/spectrum",
        "/api/predict/jobs",
        "/api/predict/jobs/{job_id}",
        "/api/predict/jobs/{job_id}/download",
        "/api/predict/jobs/{job_id}/download/pearson",
        "/api/predict/jobs/{job_id}/grid",
        "/api/predict/jobs/{job_id}/pearson",
        "/api/predict/jobs/{job_id}/preview",
        "/api/predict/jobs/{job_id}/truth/preview",
    }
    listed = {t for _m, t in LEGACY_PATHS}
    assert listed == oracle_legacy, (
        "4.3 的清单与退役前公开面**不相等**：改窄会让断言恒真。缺 %s；多 %s"
        % (sorted(oracle_legacy - listed), sorted(listed - oracle_legacy))
    )
    # 每条路径的方法还须与旧 router 的装饰器逐字一致 —— 方法写错（如 GET 写成
    # POST）会让守卫在错误的方法上打，退役路径照样漏网。
    expected_methods = {}
    for r in list(cd.router.routes) + list(lp.router.routes):
        if r.path in oracle_legacy:
            for m in r.methods:
                expected_methods.setdefault(r.path, set()).add(m)
    for method, template in LEGACY_PATHS:
        assert method in expected_methods.get(template, set()), (
            "%s %s 的方法与旧 router 不一致（旧 router 上该方法为 %s）"
            % (method, template, sorted(expected_methods.get(template, ())))
        )


# ------------------------------------------------------------------ 4.4 正向
def test_asset_routes_present_and_200():
    paths = set(main.app.openapi()["paths"])
    missing = [p for p in ASSET_PATHS if p not in paths]
    assert not missing, "资产路由应从 openapi 表消失却仍在：%s" % missing
    with _client() as client:
        for path in ASSET_PATHS:
            r = client.get(path)
            assert r.status_code == 200, (
                "%s 应真 200（前端在用），实测 %d" % (path, r.status_code)
            )


# ------------------------------------------------------------------ 4.5 反向·oracle 未被误伤
def test_oracle_routers_untouched():
    assert len(lp.router.routes) == 8, (
        "`live_prediction.router` 应原样保留 8 条（差分 oracle 的挂载点），实测 %d 条"
        % len(lp.router.routes)
    )
    assert len(cd.router.routes) == 9, (
        "`cio_diagnostics.router` 应原样保留 9 条全集（差分 oracle 的挂载点），"
        "实测 %d 条" % len(cd.router.routes)
    )
    cd_paths = {r.path for r in cd.router.routes}
    assert set(ASSET_PATHS) <= cd_paths, (
        "4 条资产路径必须是 `cd.router.routes` 的**子集**（资产是被**复制**进"
        "asset_router 而非被**移走**）——缺：%s" % (set(ASSET_PATHS) - cd_paths)
    )


# ------------------------------------------------------------------ 4.7 asset_router 自身组成
def test_asset_router_composition():
    asset_router = getattr(cd, "asset_router", None)
    assert asset_router is not None, (
        "`cio_diagnostics.asset_router` 不存在 —— 模块底部（import 期）的组装缺失"
    )
    assert len(asset_router.routes) == 4, (
        "asset_router 应恰为 4 条，实测 %d 条：%s"
        % (len(asset_router.routes), [r.path for r in asset_router.routes])
    )
    assert asset_router.prefix == "/api/cio", (
        "asset_router.prefix 应为 '/api/cio'，实测 %r" % asset_router.prefix
    )
    got = sorted(r.path for r in asset_router.routes)
    assert got == sorted(cd.CIO_ASSET_PATHS), (
        "asset_router 的 path 集合应与 CIO_ASSET_PATHS 相等\n  期望 %s\n  实测 %s"
        % (sorted(cd.CIO_ASSET_PATHS), got)
    )
    bleed = [r.path for r in asset_router.routes if "/jobs" in r.path]
    assert not bleed, "资产面不该含任何任务路由，实测混入：%s" % bleed
    # 「资产端点不重复实现」= 实现**本体**唯一：两处必须是同一批 handler 对象
    for path in cd.CIO_ASSET_PATHS:
        a = _handler_for(asset_router, path)
        b = _handler_for(cd.router, path)
        assert a is b, (
            "%s 的 handler 在 asset_router 与 cd.router 里不是同一个对象"
            "（%r vs %r）—— 资产被重新实现了第二份" % (path, a, b)
        )


# ------------------------------------------------------------------ 4.8 实现唯一性（挂载面）
def test_asset_routes_mounted_exactly_once_on_main_app():
    effective = [r for r in _effective_routes(main.app)
                 if getattr(r, "path", None) is not None]
    for path in ASSET_PATHS:
        hits = [r for r in effective if r.path == path]
        assert len(hits) == 1, (
            "%s 在 main.app 上应恰有 1 个 route，实测 %d 个 —— 2 个说明**双挂载**"
            "（asset_router 与 cd.router 同时被 include）；0 个说明资产面整体掉了"
            % (path, len(hits))
        )
        route = hits[0]
        expected = _handler_for(cd.router, path)
        assert route.endpoint.__module__ == "cio_diagnostics", (
            "%s 的 endpoint.__module__ 应为 'cio_diagnostics'，实测 %r"
            % (path, route.endpoint.__module__)
        )
        assert route.endpoint is expected, (
            "%s 的 endpoint 不是 cio_diagnostics 里的 %r（实测 %r）—— "
            "资产端点出现了第二份实现" % (path, expected, route.endpoint)
        )


# ------------------------------------------------------------------ 4.10 负向·结构（有效路由表）
def test_no_legacy_routes_in_effective_route_table():
    """4.10：`main.app` 的**有效路由表**里不得存在任何旧任务路径的 route **对象**。

    为什么必须有这条（收口轮 C010 的补牙）：本套件原有三块负向判据都有盲区 ——
    4.2 查 `openapi()["paths"]`，而 `include_in_schema=False` 的**隐藏挂载不进这张表**；
    4.3a（同方法探针）对 11 条 GET 路径**不可区分**（404 语义歧义）。
    本条只看 route 对象本身，**不依赖任何 HTTP 语义** —— 上述两种盲区都绕不过去。

    实测反例（Reviewer R-M3，`.harness/tmp/retire-legacy-job-routes/
    reviewer_probe_hidden_route_r2.py`）：11 条 GET 旧路由以 `include_in_schema=False`
    + **真实 handler** 挂回 `main.app` 时，加上本条之前的守卫 **10/10 全绿**，
    而 `POST /api/predict/jobs/deadbeef -> 405` 证明该路径上确有 route 对象。
    加上本条与 4.3b 后该形态转红。

    复用 4.8 的 `_effective_routes`（fastapi 0.139 的子路由藏在 `_IncludedRouter.
    original_router` 里，顶层不可见），**不另写一份 walk**。

    **已知边界（本人实测，不藏）**：**子应用挂载**形态（`app.mount("/api", 子app)`，
    记为 M-4）内部对 walk 不可见 —— 本条**不覆盖**它；但对它 **fail-closed**
    （见下方 `opaque` 断言）：该形态会让本条**变红并提示扩展 walk**，不会静默判绿。
    同一边界也适用于 4.8 与附 A（它们共用同一个 walk）。
    """
    effective = [r for r in _effective_routes(main.app)
                 if getattr(r, "path", None) is not None]
    # walk 健全性：若 walk 失效（展开不出子路由），下面的负向断言会**恒真**。
    # 用两条无论如何都必须在的 route 钉住它，把"walk 坏了"与"退役没做"区分开
    # （附 A 也有同类自检，但本条必须**自足**——否则 walk 回归时本条会静默判绿）。
    for anchor in ("/api/cio/capabilities", "/api/chain/jobs/{job_id}"):
        assert any(r.path == anchor for r in effective), (
            "递归 walk 未在 main.app 上展开出 %s —— walk 失效，本条断言已失去判别力"
            % anchor
        )
    # **fail-closed**：walk 只认得 `_IncludedRouter.original_router` 这一种包装。
    # 若有人改用**子应用**挂回旧路由（`app.mount("/api", 子app)` / `router.mount(...)`），
    # walk 只会看到 `Mount.path == "/api"` —— 它既不匹配两个退役前缀、也不会展开内部，
    # 本条会**静默判绿**。实测（收口轮，自建最小 app 复现 M-4 形态）：
    # 该形态下 `GET /api/predict/jobs/x -> 200`，而 walk 探出的 path 只有
    # `['/api', '/docs', '/openapi.json', ...]`。故这里**显式拒绝**任何 walk 展不开的
    # 容器路由：宁可红并提示"请扩展 walk"，也不能绿。
    opaque = sorted({type(r).__name__ for r in _effective_routes(main.app)
                     if getattr(r, "routes", None) is not None})
    assert not opaque, (
        "main.app 上出现 walk **展不开**的容器路由：%s —— 其内部路由对 4.10 不可见"
        "（如 `app.mount('/api', 子app)`）。本条 fail-closed：请扩展 `_effective_routes` "
        "以穿透该容器类型，**不要在此处放行**" % opaque
    )
    left = sorted({r.path for r in effective
                   if r.path.startswith(LEGACY_PREDICT_PREFIX)
                   or r.path.startswith(LEGACY_CIO_JOBS_PREFIX)})
    assert not left, (
        "main.app 的有效路由表中不应残留任何旧任务路径（**含 `include_in_schema=False` "
        "的隐藏挂载**——它们不进 openapi 表却真可达）：%s" % left
    )


# ------------------------------------------------------------------ 附 A 链零回归·静态
def test_chain_routes_still_mounted():
    paths = set(main.app.openapi()["paths"])
    missing = [p for p in CHAIN_PATHS if p not in paths]
    assert not missing, (
        "13 条 `/api/chain/*` 应一条不少（摘挂载摘过头的反例）：缺 %s" % missing
    )
    # walk 自身的健全性：上面若因 walk 失效而"看不到子路由"，这里会一起红——
    # 否则 `missing` 恒为空、断言恒真（"保证不保证"）。
    effective = [r for r in _effective_routes(main.app)
                 if getattr(r, "path", None) is not None]
    for path in CHAIN_PATHS:
        assert any(r.path == path for r in effective), (
            "walk 未在 main.app 上展开出 %s —— 递归 walk 失效，本套件的挂载面断言"
            "已失去判别力" % path
        )


# ------------------------------------------------------------------ 附 B oracle 挂载点仍可执行
def test_inmemory_oracle_app_exposes_all_legacy_routes():
    """自建内存 app 上旧契约端点仍按原样可执行（spec delta 的 Scenario）。

    **纯静态**：只查自建 app 的 openapi 表，不发请求 —— 因此不触发任何任务创建、
    不写 `uploads/`。发出的请求由 `test_predict_chain.py` / `test_cio_diagnostics.py`
    / `test_chain_api.py` 承担（它们全绿即在证伪"oracle 被误伤"）。
    """
    oracle = FastAPI()
    oracle.include_router(cd.router)
    oracle.include_router(lp.router)
    paths = set(oracle.openapi()["paths"])
    expected = set(ASSET_PATHS) | {
        "/api/cio/jobs",
        "/api/cio/jobs/{job_id}",
        "/api/cio/jobs/{job_id}/completeness",
        "/api/cio/jobs/{job_id}/series",
        "/api/cio/jobs/{job_id}/spectrum",
        "/api/predict/jobs",
        "/api/predict/jobs/{job_id}",
        "/api/predict/jobs/{job_id}/download",
        "/api/predict/jobs/{job_id}/download/pearson",
        "/api/predict/jobs/{job_id}/grid",
        "/api/predict/jobs/{job_id}/pearson",
        "/api/predict/jobs/{job_id}/preview",
        "/api/predict/jobs/{job_id}/truth/preview",
    }
    assert expected <= paths, (
        "自建内存 app 上应能挂出全部 17 条旧契约路径（oracle 可执行性），缺：%s"
        % sorted(expected - paths)
    )
    # 反向：`main.app` 上这些旧任务路径必须**不在**——两张表互为对照，
    # 防止"退役改在自建 app 上而 main.app 未变"这类张冠李戴。
    main_paths = set(main.app.openapi()["paths"])
    assert not (expected - set(ASSET_PATHS)) & main_paths, (
        "旧任务路径不应出现在 main.app 上：%s"
        % sorted((expected - set(ASSET_PATHS)) & main_paths)
    )


# ------------------------------------------------------------------ 跑法
def _run_all():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    print("=" * 78)
    print("退役守卫：main.app 上旧任务路由不可达 / 资产路由可用 / oracle 未被误伤")
    print("-" * 78)
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                            # noqa: BLE001
            failures.append((name, exc))
            print("FAIL  %s" % name)
            print("      %s: %s" % (type(exc).__name__, exc))
        else:
            print("ok    %s" % name)
    print("-" * 78)
    print("通过 %d / %d" % (len(tests) - len(failures), len(tests)))
    if failures:
        print("失败 %d 项：%s" % (len(failures), [n for n, _ in failures]))
        return 1
    print("判定: 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
