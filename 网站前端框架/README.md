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

## 自适应布局与 OBS 录制

页面随窗口宽度布局，不再使用固定 1560×1040 画布或整页缩放。主内容可用宽度
不足 900px 时结果与投影卡片改为单列；视口宽度不超过 760px 时侧栏位于内容上方，
采用整页滚动。地图数据、海岸线、等值线和点击坐标共用原来的映射与比例。
切回宽屏时外层不保留窄屏的滚动偏移，侧栏从顶部显示；表格只在自己的容器内横向滚动。

OBS 可以录制 16:9 画布（例如 1920×1080），浏览器窗口无需保持 3:2。
使用窗口捕获并裁掉浏览器工具栏，再等比适配画布；避免拉伸地图。
布局修改不改变 A/B 的端口或 API 地址，构建仍必须显式设置 `VITE_API_BASE`。
当前 5174 是连接 8001 的开发服务，5175 是读取共享 `dist` 的预览服务；更新这个
`dist` 时使用 B 的 8002 地址。如果两站都改用生产构建，应分别使用独立输出目录，
避免 A 的构建覆盖 B 的产物。

## CIO-RainCast 开场动画

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
