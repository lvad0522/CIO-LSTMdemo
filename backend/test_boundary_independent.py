# -*- coding: utf-8 -*-
"""Tester 独立边界测试 —— 不依赖开发者 test_api.py。

覆盖：
A. /api/run 极端组合 year=2000&lead=pre1 / year=2019&lead=pre20
B. /api/run?lead=pre20&year=2015 的 table 与 skillMap 数值合理性
C. 连续 5 次不同 (year, lead) 请求后的内存稳定性（psutil 观察后端进程）
D. /api/grid 边界格点 i=0&j=0 / i=80&j=100
E. 非法 lead=pre0 → 400
F. /api/upload 传非 npy 文件 → 400
"""
import io
import json
import sys
import urllib.request
import urllib.error

import numpy as np
import psutil

API = "http://localhost:8000"
BACKEND_PID = 34544  # netstat 查到的监听进程（17:28 重启后的新进程）

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}  {detail}")


def get(path, timeout=60):
    try:
        with urllib.request.urlopen(f"{API}{path}", timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, str(e)


def main():
    print("=" * 60)
    print("A. /api/run 极端组合")
    print("=" * 60)

    # A1: year=2000, lead=pre1
    st, body = get("/api/run?year=2000&lead=pre1")
    ok = st == 200
    detail = f"HTTP {st}"
    if ok:
        sm = body["skillMap"]["data"]
        flat = [v for row in sm for v in row]
        detail = (f"skillMap {len(sm)}x{len(sm[0])}, r∈[{min(flat):.3f},{max(flat):.3f}], "
                  f"timeSeries len={len(body['timeSeries']['real'])}, table rows={len(body['table'])}")
        ok = (len(sm) == 81 and len(sm[0]) == 101 and len(body["timeSeries"]["real"]) == 112
              and len(body["table"]) == 6)
    record("A1 /api/run year=2000&lead=pre1 200", ok, detail)

    # A2: year=2019, lead=pre20
    st, body = get("/api/run?year=2019&lead=pre20")
    ok = st == 200
    detail = f"HTTP {st}"
    if ok:
        sm = body["skillMap"]["data"]
        flat = [v for row in sm for v in row]
        detail = (f"skillMap {len(sm)}x{len(sm[0])}, r∈[{min(flat):.3f},{max(flat):.3f}], "
                  f"timeSeries len={len(body['timeSeries']['real'])}, table rows={len(body['table'])}")
        ok = (len(sm) == 81 and len(sm[0]) == 101 and len(body["timeSeries"]["real"]) == 112
              and len(body["table"]) == 6)
    record("A2 /api/run year=2019&lead=pre20 200", ok, detail)

    # A3: /api/grid 极端组合
    st, body = get("/api/grid?i=0&j=0&year=2000&lead=pre1")
    ok = st == 200 and len(body["real"]) == 112 and len(body["pred"]) == 112
    record("A3 /api/grid year=2000&lead=pre1 200", ok, f"HTTP {st}, real={len(body.get('real', []))}")

    print()
    print("=" * 60)
    print("B. table 与 skillMap 数值合理性 (year=2015, lead=pre20)")
    print("=" * 60)

    st, body = get("/api/run?year=2015&lead=pre20")
    ok = st == 200
    detail = f"HTTP {st}"
    if ok:
        table = body["table"]
        rows = []
        for row in table:
            r, rmse, mae = row["pearsonR"], row["rmse"], row["mae"]
            rows.append(f"fold{row['experiment']}: r={r}, rmse={rmse}, mae={mae}")
            if not (-1.0 < r < 1.0):
                ok = False
            if not (rmse > 0 and mae > 0):
                ok = False
            if row["source"] != "dataset":
                ok = False
        detail = "; ".join(rows)
        sm = body["skillMap"]["data"]
        flat = [v for row in sm for v in row]
        detail += f" | skillMap r∈[{min(flat):.3f},{max(flat):.3f}]"
        if not all(-1.0 < v < 1.0 for v in flat):
            ok = False
            detail += " <- 存在 |r|>=1 的格点!"
    record("B1 table 6 行数值合理 (r∈(-1,1), rmse/mae>0, source=dataset)", ok, detail)

    print()
    print("=" * 60)
    print("C. 连续 5 次不同 (year, lead) 请求后内存稳定性")
    print("=" * 60)

    proc = psutil.Process(BACKEND_PID)
    mem_before = proc.memory_info().rss / 1024 / 1024
    print(f"  请求前 RSS: {mem_before:.1f} MB")
    combos = [("2000", "pre1"), ("2005", "pre10"), ("2010", "pre15"), ("2015", "pre20"), ("2019", "pre6")]
    for year, lead in combos:
        st, _ = get(f"/api/run?year={year}&lead={lead}")
        print(f"  /api/run?year={year}&lead={lead} -> HTTP {st}")
    mem_after = proc.memory_info().rss / 1024 / 1024
    delta = mem_after - mem_before
    print(f"  请求后 RSS: {mem_after:.1f} MB, 增量 {delta:+.1f} MB")
    # 5 组 (year,lead) 每组 6fold*3kind=18 文件，LRU 容量 16 组 ~171MB；增量应远小于 171MB
    ok = delta < 100
    record("C1 5 次不同组合请求后内存增量 < 100MB", ok, f"Δ={delta:+.1f} MB (请求前 {mem_before:.1f} → 请求后 {mem_after:.1f})")

    # C2: 同一组合重复请求（缓存命中路径）
    st1, body1 = get("/api/run?year=2015&lead=pre20")
    st2, body2 = get("/api/run?year=2015&lead=pre20")
    ok = st1 == 200 and st2 == 200 and body1["table"] == body2["table"]
    record("C2 重复请求 table 数值可复现", ok, f"HTTP {st1}/{st2}")

    print()
    print("=" * 60)
    print("D. /api/grid 边界格点")
    print("=" * 60)

    for i, j in [(0, 0), (80, 100)]:
        st, body = get(f"/api/grid?i={i}&j={j}&year=2019&lead=pre6")
        ok = st == 200 and len(body.get("real", [])) == 112 and len(body.get("pred", [])) == 112
        record(f"D1 /api/grid?i={i}&j={j} 边界格点 200", ok,
               f"HTTP {st}, real len={len(body.get('real', [])) if isinstance(body, dict) else 'N/A'}")

    # D2: 越界边界
    for i, j in [(-1, 50), (81, 50), (40, -1), (40, 101)]:
        st, body = get(f"/api/grid?i={i}&j={j}")
        record(f"D2 /api/grid?i={i}&j={j} 越界 400", st == 400, f"HTTP {st}: {body.get('detail', '')[:60]}")

    print()
    print("=" * 60)
    print("E. 非法 lead 参数")
    print("=" * 60)

    for lead in ["pre0", "pre21", "pre00", "pre01", "6", "pre-1", ""]:
        st, body = get(f"/api/run?lead={lead}")
        record(f"E1 /api/run?lead='{lead}' -> 400", st == 400, f"HTTP {st}: {body.get('detail', '')[:60]}")

    # E2: /api/grid 同样校验 lead
    st, body = get("/api/grid?i=40&j=50&lead=pre0")
    record("E2 /api/grid?lead=pre0 -> 400", st == 400, f"HTTP {st}: {body.get('detail', '')[:60]}")

    # E3: lead=pre20 合法，pre21 非法
    st, _ = get("/api/run?year=2019&lead=pre20")
    record("E3 /api/run?lead=pre20 合法 200", st == 200, f"HTTP {st}")

    print()
    print("=" * 60)
    print("F. /api/upload 非 npy 文件")
    print("=" * 60)

    def upload(filename, content, ctype="application/octet-stream"):
        boundary = "----testerboundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{API}/api/upload", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode())
            except Exception:
                return e.code, str(e)

    st, body = upload("evil.txt", b"not a numpy file at all")
    record("F1 /api/upload 传 .txt -> 400", st == 400, f"HTTP {st}: {body.get('detail', '')[:60]}")

    st, body = upload("data.csv", b"a,b,c\n1,2,3")
    record("F2 /api/upload 传 .csv -> 400", st == 400, f"HTTP {st}: {body.get('detail', '')[:60]}")

    # F3: .npy 后缀但内容非 npy -> 400（解析失败）
    st, body = upload("fake.npy", b"this is not a valid npy binary")
    record("F3 /api/upload fake.npy（内容非 npy）-> 400", st == 400, f"HTTP {st}: {body.get('detail', '')[:60]}")

    # F4: 合法 .npy -> 200
    buf = io.BytesIO()
    np.save(buf, np.zeros((2, 3), dtype=np.float64))
    st, body = upload("valid_test.npy", buf.getvalue())
    ok = st == 200 and body.get("shape") == [2, 3]
    record("F4 /api/upload 合法 .npy -> 200", ok, f"HTTP {st}: {body.get('shape')}")

    print()
    print("=" * 60)
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    print(f"边界测试汇总: {passed}/{total} 通过, {failed} 失败")
    if failed:
        for name, ok, detail in results:
            if not ok:
                print(f"  FAIL: {name} :: {detail}")
        sys.exit(1)
    print("全部通过 ✓")


if __name__ == "__main__":
    main()
