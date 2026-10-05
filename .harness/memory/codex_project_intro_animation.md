---
name: CIO-RainCast globe intro and responsive layout
description: Approved opening animation, responsive OBS layout, A/B routing and browser verification.
type: project
status: verified
module: frontend
---

- User approved a Canvas orthographic globe rotating to East Asia; the current brand is CIO-RainCast with the exact subtitle “East Asian Summer Monsoon Intraseasonal Rainfall Prediction”. Land uses local Natural Earth 1:110m geometry and d3-geo.
- Since 2026-10-05 the user-approved OBS layout is responsive, replacing the fixed 1560×1040 scaled canvas: workspace below 900px stacks results; viewport at/below 760px stacks the sidebar and scrolls the whole page. IntroShell covers the viewport independently; the globe also fits short windows. App remains mounted underneath and blocked via inert/aria-hidden while playing.
- Duration is 4 seconds (user requested an extra 0.5-second final hold); first entry per tab plays, completion/skip stores sessionStorage `cio-rain.intro-seen.v1`. Refresh skips. Footer “重播开场 · Replay intro” replays without resetting research task state. Escape and Skip intro dismiss it.
- Reduced-motion preference skips intro and hides replay. Asset/context failures exit to main UI; all geographic assets are local, with source documented beside the GeoJSON. The East Asia glow is decorative.
- Vite build must always supply VITE_API_BASE: A=5174→8001/fold1, B=5175→8002/foldmax. B runs preview of the shared dist, so building that dist with A's API would misroute B; use separate output directories if building both. Responsive browser checks covered 1920×1080, 1560×1040, 1366×768, 1024×768, 768×1024, 375×812 and a 1280×540 intro. Default results and a real NetCDF projection sample loaded; map layers/clicks remained aligned. Screenshots: `tmp/cio-raincast-responsive-desktop.png`, original intro `tmp/cio-rain-intro-east-asia.jpg`.
