"""测试修复 Failed to fetch 问题 (P0-1: 健康检查, P0-2: CORS 通配).

这些测试验证：
1. main.py 的 CORS 配置支持所有来源（开发环境）
2. start.py 具有健康检查功能
"""
import sys
import os
import json

# ── 允许从 backend/ 导入 start.py ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, "backend")
if os.path.isdir(BACKEND_DIR):
    sys.path.insert(0, BACKEND_DIR)
# 也把 start.py 所在目录加入
sys.path.insert(0, os.path.dirname(ROOT) if os.path.isfile(os.path.join(ROOT, "start.py")) else ROOT)

import pytest


# ════════════════════════════════════════════════════════════════
# 测试 1: CORS 配置
# ════════════════════════════════════════════════════════════════

class TestCORSConfig:
    """验证后端 CORS 中间件配置正确."""

    def test_cors_allow_origins_allows_all(self):
        """CORS allow_origins 应包含 '*' 以支持 Vite 端口偏移."""
        from main import app

        cors_middleware = None
        for mw in app.user_middleware:
            if mw.cls.__name__ == "CORSMiddleware":
                cors_middleware = mw
                break

        assert cors_middleware is not None, "未找到 CORSMiddleware"

        opts = cors_middleware.kwargs
        origins = opts.get("allow_origins", [])
        assert "*" in origins, f"allow_origins 应包含 '*', 实际为 {origins}"

    def test_cors_allows_any_method(self):
        """CORS 应允许所有 HTTP 方法."""
        from main import app

        cors_middleware = None
        for mw in app.user_middleware:
            if mw.cls.__name__ == "CORSMiddleware":
                cors_middleware = mw
                break

        assert cors_middleware is not None
        opts = cors_middleware.kwargs
        methods = opts.get("allow_methods", [])
        assert "*" in methods, f"allow_methods 应包含 '*', 实际为 {methods}"

    def test_cors_allows_any_header(self):
        """CORS 应允许所有请求头."""
        from main import app

        cors_middleware = None
        for mw in app.user_middleware:
            if mw.cls.__name__ == "CORSMiddleware":
                cors_middleware = mw
                break

        assert cors_middleware is not None
        opts = cors_middleware.kwargs
        headers = opts.get("allow_headers", [])
        assert "*" in headers, f"allow_headers 应包含 '*', 实际为 {headers}"


# ════════════════════════════════════════════════════════════════
# 测试 2: start.py 健康检查功能
# ════════════════════════════════════════════════════════════════

class TestHealthCheckFunctions:
    """验证 start.py 中的健康检查函数."""

    def test_health_check_module_has_check_function(self):
        """start.py 应导出 health_check 或等效函数."""
        # 尝试导入 start.py 中的健康检查函数
        # 由于 start.py 不是标准模块名，我们直接检查文件内容
        start_path = os.path.join(os.path.dirname(BACKEND_DIR), "start.py") if os.path.isdir(BACKEND_DIR) else os.path.join(ROOT, "..", "start.py")
        if not os.path.exists(start_path):
            start_path = os.path.join(ROOT, "start.py")
        if not os.path.exists(start_path):
            # 尝试在项目根目录下找
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            start_path = os.path.join(project_root, "start.py")

        assert os.path.exists(start_path), f"找不到 start.py: {start_path}"

        with open(start_path, encoding="utf-8") as f:
            content = f.read()

        # 应该包含健康检查相关的函数或逻辑
        assert "health" in content.lower() or "check" in content.lower(), \
            "start.py 应包含健康检查相关逻辑"

    def test_health_check_function_exists(self):
        """start.py 应定义 health_check 函数用于检测后端是否存活."""
        start_path = os.path.join(os.path.dirname(BACKEND_DIR), "start.py") if os.path.isdir(BACKEND_DIR) else os.path.join(ROOT, "..", "start.py")
        if not os.path.exists(start_path):
            start_path = os.path.join(ROOT, "start.py")
        if not os.path.exists(start_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            start_path = os.path.join(project_root, "start.py")

        assert os.path.exists(start_path), f"找不到 start.py: {start_path}"

        with open(start_path, encoding="utf-8") as f:
            content = f.read()

        # 应定义一个可调用的健康检查函数
        assert "def health_check" in content or "def check_backend" in content or "def _check" in content, \
            "start.py 应定义 health_check/check_backend 函数"

    def test_health_check_can_detect_running_server(self):
        """health_check 应能正确检测正在运行的服务."""
        # 导入 start 模块中的健康检查函数
        start_path = os.path.join(os.path.dirname(BACKEND_DIR), "start.py") if os.path.isdir(BACKEND_DIR) else os.path.join(ROOT, "..", "start.py")
        if not os.path.exists(start_path):
            start_path = os.path.join(ROOT, "start.py")
        if not os.path.exists(start_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            start_path = os.path.join(project_root, "start.py")

        assert os.path.exists(start_path), f"找不到 start.py: {start_path}"

        # 动态导入 start 模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("launcher", start_path)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)

        # 找到健康检查函数
        health_fn = None
        for name in ["health_check", "check_backend", "check_server"]:
            fn = getattr(launcher, name, None)
            if callable(fn):
                health_fn = fn
                break

        assert health_fn is not None, "start.py 中未找到可调用的健康检查函数"

        # 检查 mock 服务器关闭时返回 False
        import socket
        # 找一个肯定没在监听的端口
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        free_port = find_free_port()
        result = health_fn(f"http://localhost:{free_port}")
        assert result is False, f"对未监听端口 {free_port} 的健康检查应返回 False, 实际返回 {result}"

    def test_health_check_returns_bool(self):
        """health_check 应始终返回 bool 值."""
        start_path = os.path.join(os.path.dirname(BACKEND_DIR), "start.py") if os.path.isdir(BACKEND_DIR) else os.path.join(ROOT, "..", "start.py")
        if not os.path.exists(start_path):
            start_path = os.path.join(ROOT, "start.py")
        if not os.path.exists(start_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            start_path = os.path.join(project_root, "start.py")

        assert os.path.exists(start_path)

        import importlib.util
        spec = importlib.util.spec_from_file_location("launcher", start_path)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)

        health_fn = None
        for name in ["health_check", "check_backend", "check_server"]:
            fn = getattr(launcher, name, None)
            if callable(fn):
                health_fn = fn
                break

        assert health_fn is not None
        result = health_fn("http://invalid-host-12345.xyz")
        assert isinstance(result, bool), f"health_check 应返回 bool, 实际返回 {type(result)}"


# ════════════════════════════════════════════════════════════════
# 测试 3: 前端端口检查
# ════════════════════════════════════════════════════════════════

class TestFrontendPortCheck:
    """验证 start.py 包含前端端口检查."""

    def test_frontend_port_check_exists(self):
        """start.py 应包含检查前端端口是否在监听的逻辑."""
        start_path = os.path.join(os.path.dirname(BACKEND_DIR), "start.py") if os.path.isdir(BACKEND_DIR) else os.path.join(ROOT, "..", "start.py")
        if not os.path.exists(start_path):
            start_path = os.path.join(ROOT, "start.py")
        if not os.path.exists(start_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            start_path = os.path.join(project_root, "start.py")

        assert os.path.exists(start_path)

        with open(start_path, encoding="utf-8") as f:
            content = f.read()

        # 应包含前端端口检查(5173 / Vite 端口)
        assert "5173" in content or "vite" in content.lower() or "frontend" in content.lower() or "前端" in content, \
            "start.py 应包含前端端口检查逻辑"


# ════════════════════════════════════════════════════════════════
# 测试 4: 集成测试 - CORS 头
# ════════════════════════════════════════════════════════════════

class TestCORSIntegration:
    """通过 TestClient 验证 CORS 头确实正确返回."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_cors_headers_returned(self, client):
        """任意来源的请求都应收到 CORS 头."""
        resp = client.get("/", headers={"Origin": "http://localhost:5173"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*", \
            f"应返回 access-control-allow-origin: *, 实际: {resp.headers.get('access-control-allow-origin')}"

    def test_cors_vite_port_offset(self, client):
        """Vite 端口偏移后(5174)也应正常工作."""
        resp = client.get("/", headers={"Origin": "http://localhost:5174"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*", \
            "端口偏移后的请求也应收到 CORS 头"

    def test_cors_arbitrary_origin(self, client):
        """任意来源都应有正确的 CORS 响应."""
        for origin in ["http://example.com", "http://localhost:3000", "null"]:
            resp = client.get("/api/run?year=2019&lead=pre6&region=india",
                              headers={"Origin": origin})
            assert resp.status_code == 200
            assert resp.headers.get("access-control-allow-origin") == "*", \
                f"来源 {origin} 应收到 access-control-allow-origin: *"


# ════════════════════════════════════════════════════════════════
# 测试 5: 后端 API 基本功能保持正常
# ════════════════════════════════════════════════════════════════

class TestAPIIntegrity:
    """验证修改 CORS 后 API 功能不受影响."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_run_endpoint(self, client):
        resp = client.get("/api/run?year=2019&lead=pre6&region=india")
        assert resp.status_code == 200
        data = resp.json()
        for key in ["cioTS", "cioCorr", "skillMap", "timeSeries", "s2s", "table"]:
            assert key in data, f"缺少字段: {key}"

    def test_validation_error_handler(self, client):
        """CORS 修改不应破坏异常处理器."""
        resp = client.get("/api/grid?i=-1&j=-1&region=india")
        assert resp.status_code == 400
