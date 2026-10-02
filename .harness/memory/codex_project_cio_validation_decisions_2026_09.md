---
name: Codex CIO validation visualization decisions
description: Codex 维护的 CIO 验证口径、覆盖摘要与同源代理约束
type: project
status: unverified
module: frontend/backend
---

当前频谱按每个 112 点投影窗口独立估计后平均，置信线使用分块 bootstrap；它不等同于论文的 SST+U850 全年连续谱。
数据覆盖只在顶部说明与 112 点窗口基准的比较，不作为数值质量评分，也不从无日期 .npy 推断年份。
开发期前端默认请求同源 `/api`，Vite 代理到 `127.0.0.1:8000`，避免应用内浏览器直接跨端口访问被拦截。
