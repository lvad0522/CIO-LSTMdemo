"""模拟数据沙盒 —— 把假数据在**结构上**圈死，让它够不着后端。

## 规矩（两条，方向相反）

- **沙盒里**（路径位于被 `.MOCK_SANDBOX` 标记的目录之下）：一律按"假"处理，
  必须显式 `CIOPROJ_ALLOW_MOCK=1` 才可用，且结果如实标成 `mock`。
- **沙盒外**：一律按"必须是真的"处理 —— 正交性不合格就**拒绝**，
  `CIOPROJ_ALLOW_MOCK=1` 在这里**不构成任何放行**。

## 为什么要有"沙盒外"那半条

`CIOPROJ_ALLOW_MOCK` 是个环境变量，很容易被留在 shell 里忘记清掉。
如果只靠它放行，等于"设过一次就永久松开"。现在放行必须**同时**满足两个条件
（文件确实在沙盒里 + 显式开了开关），环境变量单独存在不构成放行。

> 2026-09-15 定：这条规则是为了回应"别让模拟数据影响后端"。此前
> `ALLOW_MOCK=1` 可以让**任意位置**的非正交 `.mat` 被接受，只要它不在默认
> 资产路径上；现在不行了。

## 标记文件

`.MOCK_SANDBOX` 的内容只给人看（写明这块是假的、谁造的）；代码只认**存在性**。
目前挂在 `摸库交付_2026-09-15/03_夹具/` 上。

注意标记**不能**挂到 `摸库交付_2026-09-15/` 那一层 —— 隔壁 `数据/` 是真的，
挂高了会把真数据一起误判成假。
"""

from __future__ import annotations

import os
from pathlib import Path

MARKER = ".MOCK_SANDBOX"

# 显式放行开关。仅在文件确实落在沙盒里时才有意义。
ALLOW_ENV = "CIOPROJ_ALLOW_MOCK"


def find_root(path) -> Path | None:
    """向上找最近的沙盒根；不在沙盒里返回 None。

    path 可以是文件也可以是目录（不要求存在，路径判断而已）。
    """
    p = Path(path).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / MARKER).is_file():
            return candidate
    return None


def is_sandboxed(path) -> bool:
    return find_root(path) is not None


def allow_flag_set() -> bool:
    """显式放行开关是否打开。**单独打开不构成放行**，见模块文档。"""
    return os.environ.get(ALLOW_ENV) == "1"
