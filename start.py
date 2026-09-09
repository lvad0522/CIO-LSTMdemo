"""一键启动 CIO+LSTM 降水预测系统"""
import subprocess
import time
import os
import socket

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "网站前端框架")

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"


def health_check(url, timeout=5):
    """检查后端 HTTP 服务是否存活。

    返回 True 表示服务正常，False 表示不可达或超时。
    """
    try:
        import urllib.request
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_port(host, port, timeout=2):
    """检查指定端口的 TCP 是否在监听。

    返回 True 表示端口已开放，False 表示不可达。
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def main():
    procs = []

    try:
        print("=" * 45)
        print("CIO+LSTM 降水预测系统 - 启动中...")
        print("=" * 45)
        print()

        # 1. 启动后端
        print("[1/3] 启动后端 API 服务...")
        p = subprocess.Popen(
            ["python", "main.py"],
            cwd=BACKEND_DIR,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        procs.append(p)

        # 健康检查：等待后端就绪
        print("      等待后端就绪...", end=" ", flush=True)
        backend_ok = False
        for attempt in range(10):
            if health_check(BACKEND_URL):
                backend_ok = True
                break
            time.sleep(0.5)

        if backend_ok:
            print("OK")
        else:
            print("\033[91m失败\033[0m")
            print("\033[91m  ! 后端启动失败。请检查：")
            print("  ! 1. 是否已安装依赖: pip install -r backend/requirements.txt")
            print("  ! 2. 端口 8000 是否被占用")
            print("  ! 3. 控制台窗口中的错误信息\033[0m")

        # 2. 启动前端
        print("[2/3] 启动前端开发服务器...")
        p = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        procs.append(p)

        # 简单检查前端端口
        print("      检查前端端口...", end=" ", flush=True)
        time.sleep(3)
        frontend_ok = check_port("127.0.0.1", 5173) or check_port("127.0.0.1", 5174)
        if frontend_ok:
            print("OK")
        else:
            print("\033[93m未检测到(可能稍后启动)\033[0m")
            print("\033[93m  ! 如果前端无法访问，请检查：")
            print("  ! 1. 是否已安装 npm 依赖: cd 网站前端框架 && npm install")
            print("  ! 2. Vite 控制台窗口中的错误信息\033[0m")

        # 3. 打开浏览器
        print("[3/3] 打开浏览器...")
        subprocess.Popen(["start", FRONTEND_URL], shell=True)

        print()
        print("=" * 45)
        print(f"  后端: {BACKEND_URL}")
        print(f"  前端: {FRONTEND_URL}")
        print("=" * 45)
        print()
        print("按 Ctrl+C 停止所有服务")

        # 保持运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        for p in procs:
            p.kill()
        print("已停止")


if __name__ == "__main__":
    main()
