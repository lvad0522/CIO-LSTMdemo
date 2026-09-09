"""API 测试脚本 —— 验证后端所有接口正常（真实数据集接入后）。

用法：
  1. 确保后端已启动（python main.py）
  2. 新开终端运行本脚本：python test_api.py
"""

import io
import sys
import json
import urllib.request
import urllib.error

# Windows GBK 控制台兼容（避免 ✓ 等 Unicode 字符编码崩溃）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

API = "http://localhost:8000"
passed = 0
failed = 0


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
    assert len(data["s2s"]) == 12, f"s2s 应为 12 个模型, 实际 {len(data['s2s'])}"
    assert len(data["table"]) == 6, f"table 应为 6 行, 实际 {len(data['table'])}"
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
    assert all(m["source"] == "mock" for m in data["s2s"])
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
    print("  ✓ 返回 real/pred 各 112 天（6-fold 平均），source=dataset")
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
    assert len(run19["table"]) == 6, "填补后 table 应为 6 行"
    print("  ✓ /api/run pre19/2002 正常返回，table 6 行")
    passed += 1

    # ── 8. 实验统计表可复现 ──
    print("\n[8] 实验统计表")
    with urllib.request.urlopen(f"{API}/api/run?year=2015&lead=pre10") as resp:
        t1 = json.loads(resp.read().decode())["table"]
    with urllib.request.urlopen(f"{API}/api/run?year=2015&lead=pre10") as resp:
        t2 = json.loads(resp.read().decode())["table"]
    assert t1 == t2, "同一 (year, lead) 两次请求 table 应完全一致"
    assert [row["experiment"] for row in t1] == [1, 2, 3, 4, 5, 6]
    assert all(0 < row["pearsonR"] < 1 for row in t1)  # 池化 r 应为正
    assert all(row["rmse"] > 0 and row["mae"] > 0 for row in t1)
    print("  ✓ table 6 行可复现，r/RMSE/MAE 均为真实计算值")
    passed += 1

    # ── 9. S2S 对比 ──
    print("\n[9] S2S 对比 /api/s2s")
    url = f"{API}/api/s2s"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    assert len(data) == 12
    names = [m["name"] for m in data]
    assert "LSTM+CIO" in names, "缺少 LSTM+CIO"
    assert all("r" in m for m in data)
    print("  ✓ 12 个模型，含 LSTM+CIO（mock 降级，source=mock）")
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
    assert v["total_expected"] == 7200, f"期望 7200 文件, 实际 {v['total_expected']}"
    assert len(v["missing"]) == 0, f"期望 0 缺失（pre19/2002 已填补）, 实际 {len(v['missing'])}"
    assert v["malformed"] == 300, f"期望 300 个意外文件, 实际 {v['malformed']}"
    print("  ✓ verify_dataset: 7200 期望 / 0 缺失（pre19/2002 real2 已填补）/ 300 畸形冗余")
    passed += 1
    # 精确加载不命中畸形文件（S11）
    p1 = real_data._file_path(3, 2000, 1, "predict2")
    p2 = real_data._file_path(3, 2000, 1, "real2")
    assert p1.replace("(", ")") != p1  # 畸形文件名缺左括号，正常路径必含
    arr = real_data._load_npy(3, 2000, 1, "predict2")
    assert not bool(np.isnan(arr).any()), "精确加载结果不应含 NaN"
    print("  ✓ 精确加载（pre3/2000 fold1）无 NaN，不命中畸形文件")
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
