---
name: CIO-Rain globe intro
description: Approved opening animation design, local globe data, session behavior and verification entry.
type: project
status: verified
module: frontend
---

- User approved a Canvas orthographic globe rotating to East Asia, followed by CIO-Rain and the exact subtitle “East Asian Summer Monsoon Intraseasonal Rainfall Prediction”; the land uses local Natural Earth 1:110m geometry and d3-geo.
- IntroShell sits outside ViewportCanvas so the animation covers the viewport independently of the existing fixed app canvas. App remains mounted underneath; playing blocks its interaction via inert/aria-hidden.
- Duration is 4 seconds (user requested an extra 0.5-second final hold); first entry per tab plays, completion/skip stores sessionStorage `cio-rain.intro-seen.v1`. Refresh skips. Footer “重播开场 · Replay intro” replays without resetting research task state. Escape and Skip intro dismiss it.
- Reduced-motion preference skips intro and hides replay. Asset/context failures exit to main UI; all geographic assets are local, with source documented beside the GeoJSON. The East Asia glow is decorative.
- Vite build must always supply VITE_API_BASE (B=8002, A=8001). Verified actual 5175 globe/title frame, automatic dismissal, Skip, Escape, and refresh behavior; screenshot is `tmp/cio-rain-intro-east-asia.jpg`.
