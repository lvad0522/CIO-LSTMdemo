# CIO+LSTM 降水预测网站

网页端复现 CIO+LSTM 降水预测系统（东亚/印度季风季节内尺度技巧性预测演示）。

## 目录结构

| 目录 | 说明 |
|------|------|
| `backend/` | FastAPI 后端 API（端口 8000） |
| `网站前端框架/` | Vite + React 前端（端口 5173） |
| `start.py` | 一键启动脚本（自动拉起前后端并做健康检查） |

## 启动方式

前置依赖：Python 3.x、Node.js。

```bash
# 一键启动
python start.py

# 或分别启动：
# 终端 1 —— 后端
cd backend
pip install -r requirements.txt
python main.py          # → http://localhost:8000

# 终端 2 —— 前端
cd 网站前端框架
npm install
npm run dev             # → http://localhost:5173
```

浏览器打开 <http://localhost:5173/>

## ⚠ 展示口径：两个站点，别开错

同一份数据有两个展示口径，**页面长一样、只有数字不同、开错不报错**：

| 站点 | 口径 | 用途 |
|---|---|---|
| 对外 | **折 1**（一次真实训练） | 给人看、报数 |
| 对照 | **折最大**（逐格点挑最乐观折，**不是任何一次训练的能力**） | 只看放大效应 |

**同时开两个站点、怎么验自己开没开对、六个常见坑** —— 全部写在
**[`backend/两站点运行手册.md`](backend/两站点运行手册.md)**（保姆级，先读那篇）。

手动分别启动时注意两点：

- 后端要显式带 `DISPLAY_MODE=fold1`（或 `foldmax`），否则跟随 git 分支的默认值，
  带 `--reload` 的进程会在切分支时**悄悄换口径**；
- 前端换端口要用 `npx vite --port 5174 --strictPort`，**不要用 `npm run dev`**
  （该脚本是裸 `vite`，收不到端口参数）。

## 数据说明

- 真实数据目录 `Dataset/`（约 7200 个预计算 `.npy` 预测文件）**不入库**，需自行放回仓库根目录；缺失时依赖真实数据的接口（CIO/SkillMap 时序等）不可用，S2S 对比仍为 mock。
- 上传的 `.npy` 会落在 `backend/uploads/`（已 gitignore）。
