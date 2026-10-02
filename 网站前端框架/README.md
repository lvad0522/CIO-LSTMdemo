# CIO + LSTM 前端

## 构建时必须指定后端

本项目同时运行 fold1 与 foldmax 两个站点。任何人或 Codex Agent 执行
`vite build` / `npm run build` 时，都必须显式提供 `VITE_API_BASE`；缺失时
`vite.config.js` 会直接终止构建，防止产物静默连接到错误后端。

PowerShell：

```powershell
$env:VITE_API_BASE = 'http://127.0.0.1:8001'; npm run build  # A：fold1
$env:VITE_API_BASE = 'http://127.0.0.1:8002'; npm run build  # B：foldmax
```

Bash：

```bash
VITE_API_BASE=http://127.0.0.1:8001 npm run build  # A：fold1
VITE_API_BASE=http://127.0.0.1:8002 npm run build  # B：foldmax
```

本地开发也应显式指定对应后端，例如：

```powershell
$env:VITE_API_BASE = 'http://127.0.0.1:8002'; npx vite --port 5175 --strictPort
```

## CIO-Rain 开场动画

约 4 秒：正投影地球由非洲方向旋转到东亚，标题与英文副标题渐显，最后画面多停留 0.5 秒后淡出。
大陆轮廓来自本地 Natural Earth 1:110m 数据，运行时无需外网。使用 `d3-geo`
计算投影和球面裁剪，Canvas 绘制；主界面同时挂载并加载。

每个标签页首次进入播放，`sessionStorage` 标记 `cio-rain.intro-seen.v1` 避免刷新重播。
可以点击 `Skip intro` / 按 Escape 跳过，或点击侧栏底部“重播开场”重新观看。
系统启用“减少动态效果”时直接进入主界面；资产加载失败也直接进入。

## Vite 模板说明

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
