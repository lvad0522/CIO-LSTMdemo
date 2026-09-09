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

## 数据说明

- 真实数据目录 `Dataset/`（约 7200 个预计算 `.npy` 预测文件）**不入库**，需自行放回仓库根目录；缺失时依赖真实数据的接口（CIO/SkillMap 时序等）不可用，S2S 对比仍为 mock。
- 上传的 `.npy` 会落在 `backend/uploads/`（已 gitignore）。
