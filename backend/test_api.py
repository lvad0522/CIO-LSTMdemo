"""API 测试脚本 —— 验证后端所有接口正常（真实数据集接入后）。

用法：
  1. 确保后端已启动（python main.py）
  2. 新开终端运行本脚本：python test_api.py
"""

import io
import os
import sys
import json
import subprocess
import urllib.request
import urllib.error

# Windows GBK 控制台兼容（避免 ✓ 等 Unicode 字符编码崩溃）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

# 被测后端地址。**同一份代码可以同时起两种口径的站点**（DISPLAY_MODE 环境变量），
# 所以地址可用 API_BASE 覆盖，好逐个站点验：
#   API_BASE=http://127.0.0.1:8001 python test_api.py    # fold1 站点
API = os.environ.get("API_BASE", "http://localhost:8000")
passed = 0
failed = 0

# 展示口径由 real_data.DISPLAY_MODE 决定（**两条分支只差这一个常量**）。
# 本测试两条分支共用：期望值按当前口径算，不写死行数/折号。
import real_data

DISPLAY_MODE = real_data.DISPLAY_MODE
TABLE_ROWS = 1 if DISPLAY_MODE == "fold1" else 6


def test(name, method="GET", path="/", expect_status=200, checks=None):
    """发起 HTTP 请求并验证返回。"""
    global passed, failed
    url = f"{API}{path}"
    try:
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        status = e.code
        body = str(e)
    except Exception as e:
        status = 0
        body = str(e)

    ok = status == expect_status
    if ok and checks:
        try:
            for key, expected in checks.items():
                actual = body
                for k in key.split("."):
                    actual = actual[k]
                if callable(expected):
                    ok = expected(actual)
                else:
                    ok = actual == expected
                if not ok:
                    print(f"  ✗ 检查 {key}: 期望 {expected}, 实际 {actual}")
                    break
        except (KeyError, TypeError, IndexError) as e:
            print(f"  ✗ 检查异常: {e}, body keys={list(body.keys()) if isinstance(body, dict) else 'N/A'}")
            ok = False

    if ok:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name} (状态 {status})")
        if isinstance(body, str):
            print(f"    错误: {body[:300]}")
        failed += 1


def main():
    global passed, failed
    print("=" * 50)
    print(f"CIO+LSTM API 测试 — {API}（真实数据集）")
    print("=" * 50)

    # ── 1. 根路径 ──
    print("\n[1] 根路径")
    test("GET / 返回 ok", checks={"status": "ok"})

    # ── 2. 主预测接口（真实数据）──
    print("\n[2] 主预测接口 /api/run（真实数据）")
    required_keys = ["cioTS", "cioCorr", "skillMap", "timeSeries", "s2s", "table"]
    url = f"{API}/api/run?year=2019&lead=pre6"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    for key in required_keys:
        assert key in data, f"缺少字段: {key}"
    assert data["cioTS"] is None, "CIO mock 已关停，cioTS 应为 null"
    assert data["cioCorr"] is None, "CIO mock 已关停，cioCorr 应为 null"
    # S2S 走真实数据（新绘图资料/corr-2003.nc）：每个 lead 一行，不是 mock 的 12 个模型
    assert len(data["s2s"]) == 20, f"s2s 应为 20 行（lead 1-20）, 实际 {len(data['s2s'])}"
    assert len(data["table"]) == TABLE_ROWS, \
        f"table 应为 {TABLE_ROWS} 行（口径 {DISPLAY_MODE}）, 实际 {len(data['table'])}"
    assert data["skillMap"]["rows"] == 81, f"skillMap rows 应为 81, 实际 {data['skillMap']['rows']}"
    assert data["skillMap"]["cols"] == 101, f"skillMap cols 应为 101, 实际 {data['skillMap']['cols']}"
    assert len(data["timeSeries"]["real"]) == 112
    assert len(data["timeSeries"]["pred"]) == 112
    print("  ✓ 返回全部 6 个字段且结构正确（cioTS/cioCorr=null，skillMap 81×101 真实网格）")
    passed += 1

    # 技能地图含负值（真实数据特征）
    flat_r = [v for row in data["skillMap"]["data"] for v in row]
    assert min(flat_r) < 0, f"真实技能地图应含负值, 实际 min={min(flat_r)}"
    assert max(flat_r) > 0
    print(f"  ✓ 技能地图含负值: r 范围 [{min(flat_r):.3f}, {max(flat_r):.3f}]")
    passed += 1

    # source 标记
    assert data["skillMap"]["source"] == "dataset"
    assert data["timeSeries"]["source"] == "dataset"
    assert data["table"][0]["source"] == "dataset"
    assert all(m["source"] == "dataset" for m in data["s2s"])
    print("  ✓ source 标记正确（dataset/mock 区分；cioTS/cioCorr 已关停 mock 为 null）")
    passed += 1

    # ── 3. 无 region 参数兼容（默认 eastasia）──
    print("\n[3] 无 region 参数兼容")
    test("无 region 参数 200（默认 eastasia）", path="/api/run?year=2019&lead=pre6")
    test("grid 无 region 参数 200", path="/api/grid?i=40&j=50")

    # ── 4. 区域限制 ──
    print("\n[4] 区域限制（仅 eastasia）")
    test("region=india 返回 400", path="/api/run?region=india", expect_status=400)
    test("region=india 的 grid 返回 400", path="/api/grid?i=40&j=50&region=india", expect_status=400)

    # ── 5. 参数校验 ──
    print("\n[5] 参数校验")
    test("lead=pre25 非法提前期 400", path="/api/run?lead=pre25", expect_status=400)
    test("lead=preX 非法格式 400", path="/api/run?lead=preX", expect_status=400)
    test("lead=pre01 前导零 400", path="/api/run?lead=pre01", expect_status=400)
    test("year=1999 非法年份 400", path="/api/run?year=1999", expect_status=400)
    test("year=2025 非法年份 400", path="/api/run?year=2025", expect_status=400)
    test("grid i=81 越界 400", path="/api/grid?i=81&j=50", expect_status=400)
    test("grid j=101 越界 400", path="/api/grid?j=101&i=40", expect_status=400)

    # ── 6. 格点查询（真实数据 + year/lead 参数）──
    print("\n[6] 格点查询 /api/grid")
    url = f"{API}/api/grid?i=40&j=50&year=2019&lead=pre6"
    with urllib.request.urlopen(url) as resp:
        ts = json.loads(resp.read().decode())
    assert "real" in ts and "pred" in ts
    assert len(ts["real"]) == 112 and len(ts["pred"]) == 112
    assert all(isinstance(v, (int, float)) for v in ts["real"])
    assert all(isinstance(v, (int, float)) for v in ts["pred"])
    assert ts["source"] == "dataset"
    print("  ✓ 返回 real/pred 各 112 天（折 1 原始序列），source=dataset")
    passed += 1

    # 显式 year/lead 与默认参数一致（i=40,j=50 是默认格点）
    url2 = f"{API}/api/grid"
    with urllib.request.urlopen(url2) as resp:
        ts2 = json.loads(resp.read().decode())
    assert ts["real"] == ts2["real"] and ts["pred"] == ts2["pred"]
    print("  ✓ 显式 (2019, pre6) 与默认参数结果一致")
    passed += 1

    # ── 7. 数据完整性（pre19/2002 real2 已用 pre1 同 fold 复制填补）──
    print("\n[7] pre19/2002 组合（real2 曾缺失，2026-08-25 已填补）")
    with urllib.request.urlopen(f"{API}/api/grid?i=40&j=50&year=2002&lead=pre19", timeout=30) as resp:
        ts19 = json.loads(resp.read().decode())
    assert len(ts19["real"]) == 112 and len(ts19["pred"]) == 112
    assert isinstance(ts19.get("r"), (int, float))
    # 填补一致性：真实值不依赖 lead，pre19 的 real 应与 pre1 逐值一致
    with urllib.request.urlopen(f"{API}/api/grid?i=40&j=50&year=2002&lead=pre1", timeout=30) as resp:
        ts1 = json.loads(resp.read().decode())
    assert ts1["real"] == ts19["real"], "pre19 填补的 real 应与 pre1/2002 逐值一致"
    print("  ✓ /api/grid pre19/2002 正常，real 与 pre1/2002 逐值一致")
    passed += 1

    # /api/run 整体依赖 table（需要 real2），填补后应正常返回 6 行
    with urllib.request.urlopen(f"{API}/api/run?year=2002&lead=pre19", timeout=30) as resp:
        run19 = json.loads(resp.read().decode())
    assert len(run19["table"]) == TABLE_ROWS, f"填补后 table 应为 {TABLE_ROWS} 行"
    print(f"  ✓ /api/run pre19/2002 正常返回，table {TABLE_ROWS} 行")
    passed += 1

    # ── 8. 实验统计表可复现 ──
    print("\n[8] 实验统计表")
    with urllib.request.urlopen(f"{API}/api/run?year=2015&lead=pre10") as resp:
        t1 = json.loads(resp.read().decode())["table"]
    with urllib.request.urlopen(f"{API}/api/run?year=2015&lead=pre10") as resp:
        t2 = json.loads(resp.read().decode())["table"]
    assert t1 == t2, "同一 (year, lead) 两次请求 table 应完全一致"
    expected_exps = [1] if DISPLAY_MODE == "fold1" else [1, 2, 3, 4, 5, 6]
    assert [row["experiment"] for row in t1] == expected_exps, \
        f"experiment 列应为 {expected_exps}（口径 {DISPLAY_MODE}）"
    assert all(0 < row["pearsonR"] < 1 for row in t1)  # 池化 r 应为正
    assert all(row["rmse"] > 0 and row["mae"] > 0 for row in t1)
    print(f"  ✓ table {TABLE_ROWS} 行可复现，r/RMSE/MAE 均为真实计算值")
    passed += 1

    # ── 9. S2S 对比 ──
    print("\n[9] S2S 对比 /api/s2s")
    url = f"{API}/api/s2s"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    # 真实数据：每个 lead 一行（1-20），每行含 11 个 S2S 模式 + S2S_Mean + LSTM
    # （mock 降级时才是 12 个模型的柱状结构——corr-2003.nc 在，走不到那条路）
    assert len(data) == 20, f"S2S 应为 20 行（每 lead 一行）, 实际 {len(data)}"
    assert [r["lead"] for r in data] == list(range(1, 21)), "lead 应为 1-20"
    assert all(r["source"] == "dataset" for r in data)
    assert all("LSTM" in r and "S2S_Mean" in r for r in data)
    # pre7 换单折后仍应出值（旧代码会因 fold≥2 缺文件而静默断线）
    assert data[6]["LSTM"] is not None, "pre7 的 LSTM 那格不应为 None"
    print("  ✓ 20 行（lead 1-20），含 S2S_Mean/LSTM，source=dataset；pre7 有值")
    passed += 1

    # ── 10. 上传接口 ──
    print("\n[10] 上传接口 /api/upload")
    arr = np.zeros((4, 5), dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    payload = buf.getvalue()
    boundary = "----testboundary123"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.npy"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{API}/api/upload", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    assert data["status"] == "ok"
    assert data["shape"] == [4, 5]
    assert data["dtype"] == "float32"
    print("  ✓ 上传 .npy 返回 shape/dtype（已实现，非 not_implemented）")
    passed += 1

    # ── 11. 数据集完整性校验（直接调用 real_data.verify_dataset）──
    print("\n[11] 数据集完整性校验")
    import real_data
    v = real_data.verify_dataset()
    # 期望文件数按**实际折数**实算，不写死：19 个 lead 恒 6 折，pre7 是修复批
    # （Dataset/pre7-new/），折 2~6 补齐中，补齐后本式自动跟着涨，无需改测试。
    # 见 摸库交付_2026-09-15/04_文档/pre7六折落位_给后端agent.md
    _pre7_folds = len(real_data._folds(7, 2000, "pearson2"))
    want_total = 19 * 20 * 6 * 3 + _pre7_folds * 20 * 3
    assert v["total_expected"] == want_total, \
        f"期望 {want_total} 文件（pre7 {_pre7_folds} 折）, 实际 {v['total_expected']}"
    assert len(v["missing"]) == 0, f"期望 0 缺失（pre19/2002 已填补）, 实际 {len(v['missing'])}"
    assert v["malformed"] == 300, f"期望 300 个意外文件, 实际 {v['malformed']}"
    print(f"  ✓ verify_dataset: {want_total} 期望（pre7 {_pre7_folds} 折）/ 0 缺失 / 300 畸形冗余")
    passed += 1
    # 口径修复（2026-09-17）：那 50 个目录的 pearson2 由"第二遍"的截断名 predict2 算出，
    # 故 predict2 优先解析到截断名（保证热力图 r 与曲线同源）；real2 与干净目录不变。
    # 详见 摸库交付_2026-09-15/04_文档/Dataset口径修复单_给后端agent.md
    p1 = real_data._file_path(3, 2000, 1, "predict2")
    p2 = real_data._file_path(3, 2000, 1, "real2")
    p3 = real_data._file_path(6, 2019, 1, "predict2")
    assert p1.endswith("pre_3_20001)_20_40_100_125_0.25_predict2.npy"), \
        f"脏目录 predict2 应取截断名, 实际 {p1}"
    assert p2.endswith("pre_3_2000(1)_20_40_100_125_0.25_real2.npy"), \
        f"real2 不应受修复影响, 实际 {p2}"
    assert p3.endswith("pre_6_2019(1)_20_40_100_125_0.25_predict2.npy"), \
        f"干净目录应仍是正常名, 实际 {p3}"
    arr = real_data._load_npy(3, 2000, 1, "predict2")
    assert not bool(np.isnan(arr).any()), "精确加载结果不应含 NaN"
    print("  ✓ 口径修复：脏目录 predict2 取截断名 / real2 与干净目录不变，加载无 NaN")
    passed += 1

    # pre7 修复批（2026-09-19 落 Dataset/pre7-new/，坏批 Dataset/pre7/ 不再被读）
    # 折数**不写死**：折 2~6 自跑重复实验补齐中（现 1 折），落齐后本块无需改动。
    # 见 摸库交付_2026-09-15/04_文档/pre7六折落位_给后端agent.md
    f7 = real_data._folds(7, 2000)
    assert f7, "pre7 应有折"
    assert 1 in f7, f"pre7 必须有折 1（fold1 口径依赖它）, 实际 {f7}"
    assert f7 == sorted(f7), f"折号应升序, 实际 {f7}"
    assert real_data._year_dir(7, 2000).replace("\\", "/").endswith("Dataset/pre7-new/2000"), \
        f"pre7 应读 Dataset/pre7-new/, 实际 {real_data._year_dir(7, 2000)}"
    assert real_data._folds(6, 2000) == [1, 2, 3, 4, 5, 6], "pre6 应仍是 6 折"
    assert real_data._folds(3, 2000, "predict2") == [1, 2, 3, 4, 5, 6], \
        "脏目录的截断名不应被算成折号"
    # 旧坏批一个字节都不许被读：pre7 的解析路径不得落在 Dataset/pre7/ 下
    assert "Dataset/pre7/" not in real_data._file_path(7, 2000, 1, "pearson2").replace("\\", "/"), \
        "pre7 不得再读坏批 Dataset/pre7/"
    assert len(real_data.gen_results_table(2000, 7)) == len(f7), "pre7 table 行数应等于折数"
    assert real_data.gen_skill_map(2000, 7)["rows"] == 81
    assert len(real_data.gen_time_series(2000, 7, 40, 50)["real"]) == 112
    print(f"  ✓ pre7 修复批：_folds(7,2000)={f7}，读 Dataset/pre7-new/，"
          f"skillMap/timeSeries/table 均不抛 DataNotFoundError")
    passed += 1

    # 展示口径（real_data.DISPLAY_MODE）：main=foldmax（各折逐格点取最大）/
    # feature/display-single-fold=fold1（只取折 1）。两条分支共用本测试，期望值按当前口径算。
    for lead, year in ((6, 2019), (1, 2000), (7, 2000), (19, 2002)):
        want = [1] if DISPLAY_MODE == "fold1" else real_data._folds(lead, year, "pearson2")
        assert real_data._display_folds(lead, year) == want, \
            f"pre{lead}/{year} 展示折应为 {want}（口径 {DISPLAY_MODE}）"
        want_r = np.fmax.reduce([real_data._load_npy(lead, year, f, "pearson2")[..., 2]
                                 for f in want])
        sm = np.array(real_data.gen_skill_map(year, lead)["data"])
        assert np.array_equal(sm, want_r), \
            f"pre{lead}/{year} skillMap 与展示口径不符（口径 {DISPLAY_MODE}）"
        ts = real_data.gen_time_series(year, lead, 40, 50)
        assert ts["r"] == round(float(want_r[40, 50]), 4), \
            f"pre{lead}/{year} 时序 r 应与热力图同格点同源"
        for key, kind in (("pred", "predict2"), ("real", "real2")):
            want_series = np.mean(np.stack(
                [real_data._load_npy(lead, year, f, kind)[40, 50, :] for f in want], 0), 0)
            assert np.allclose(ts[key], want_series), \
                f"pre{lead}/{year} {key} 与展示口径不符"
        assert len(real_data.gen_results_table(year, lead)) == len(want), \
            f"pre{lead}/{year} table 行数应等于展示折数"
    print(f"  ✓ 展示口径 {DISPLAY_MODE}：skillMap / 时序 r / pred·real / table 行数四处一致")
    passed += 1

    # _display_folds 的两条不变量（口径无关，两分支共用）：
    #   1) 展示折恒为**实有折的子集**——不许凭空造折
    #   2) fold1 口径下缺折 1 必须报错，**不许静默换折**：悄悄退到别的折就是悄悄换了
    #      口径，数字会变而没人知道（同 Dataset 口径修复单里"不许静默回退"的规矩）
    _orig_folds, _orig_mode = real_data._folds, real_data.DISPLAY_MODE
    try:
        real_data._folds = lambda lead, year, kind="pearson2": [2, 3]
        real_data.DISPLAY_MODE = "foldmax"
        assert real_data._display_folds(6, 2019) == [2, 3], "foldmax 应返回实有折全集"
        real_data.DISPLAY_MODE = "fold1"
        try:
            real_data._display_folds(6, 2019)
            raise AssertionError("fold1 口径下缺折 1 应抛 DataNotFoundError，不得静默换折")
        except real_data.DataNotFoundError:
            pass
        real_data._folds = lambda lead, year, kind="pearson2": []
        try:
            real_data._display_folds(6, 2019)
            raise AssertionError("该目录无任何折时应抛 DataNotFoundError")
        except real_data.DataNotFoundError:
            pass
    finally:
        real_data._folds, real_data.DISPLAY_MODE = _orig_folds, _orig_mode
    print("  ✓ _display_folds 守卫：展示折 ⊆ 实有折；fold1 缺折 1 报错不静默换折")
    passed += 1

    # ── 12. pre7 走 HTTP ──
    print("\n[12] pre7 修复批（HTTP）")
    with urllib.request.urlopen(f"{API}/api/run?year=2000&lead=pre7", timeout=30) as resp:
        run7 = json.loads(resp.read().decode())
    # 行数按当前口径与**实际折数**算，不写死（pre7 折数补齐中）
    want7_rows = 1 if DISPLAY_MODE == "fold1" else len(f7)
    assert len(run7["table"]) == want7_rows, \
        f"pre7 table 应 {want7_rows} 行（口径 {DISPLAY_MODE}, pre7 {len(f7)} 折）, 实际 {len(run7['table'])}"
    assert run7["skillMap"]["rows"] == 81 and run7["skillMap"]["cols"] == 101
    with urllib.request.urlopen(f"{API}/api/grid?i=40&j=50&year=2000&lead=pre7", timeout=30) as resp:
        g7 = json.loads(resp.read().decode())
    assert len(g7["real"]) == 112 and len(g7["pred"]) == 112
    print(f"  ✓ /api/run 与 /api/grid 对 pre7 正常（table {want7_rows} 行、序列 112 天）")
    passed += 1

    # ── 13. DISPLAY_MODE 环境变量覆盖（import 期生效，只能起子进程验）──
    # 覆盖是为了**同一份代码同时起两种口径的站点**（两个后端读同一份 Dataset/）。
    # 两条分支只差默认值：无 env 时必须等于本分支的默认口径。
    print("\n[13] DISPLAY_MODE 环境变量覆盖")

    def _mode_with(env_value):
        """在子进程里 import real_data，返回 (returncode, stdout, stderr)。"""
        env = dict(os.environ)
        if env_value is None:
            env.pop("DISPLAY_MODE", None)
        else:
            env["DISPLAY_MODE"] = env_value
        r = subprocess.run(
            [sys.executable, "-c", "import real_data; print(real_data.DISPLAY_MODE)"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, env=env, timeout=120,
        )
        return r.returncode, r.stdout.strip(), r.stderr

    rc, out, err = _mode_with(None)
    assert rc == 0, f"无 env 时应正常 import，实际 rc={rc}: {err[:200]}"
    assert out == DISPLAY_MODE, f"无 env 时应用本分支默认口径 {DISPLAY_MODE}, 实际 {out}"
    for mode in ("foldmax", "fold1"):
        rc, out, err = _mode_with(mode)
        assert rc == 0 and out == mode, f"DISPLAY_MODE={mode} 应生效, 实际 rc={rc} out={out!r}"
    # 非法值必须**报错退出**，不许静默回落（悄悄换口径比报错危险得多）
    rc, out, err = _mode_with("Fold1")
    assert rc != 0, f"非法 DISPLAY_MODE 应让进程报错退出, 实际 rc={rc} out={out!r}"
    assert "DISPLAY_MODE" in err, f"报错信息应点名 DISPLAY_MODE, 实际: {err[-300:]}"
    print(f"  ✓ 无 env 时用默认 {DISPLAY_MODE}；foldmax/fold1 均可覆盖；非法值报错不回落")
    passed += 1

    # ── 汇总 ──
    print("\n" + "=" * 50)
    total = passed + failed
    print(f"总计: {passed}/{total} 通过", end="")
    if failed:
        print(f", {failed} 失败 ✗")
        sys.exit(1)
    else:
        print(" ✓")


if __name__ == "__main__":
    main()
