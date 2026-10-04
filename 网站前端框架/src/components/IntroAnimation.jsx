import { useEffect, useRef } from 'react';
import { geoGraticule10, geoOrthographic, geoPath } from 'd3-geo';
import './IntroAnimation.css';

const DURATION = 4000;
const GRATICULE = geoGraticule10();
const clamp = value => Math.max(0, Math.min(1, value));
const smooth = value => { const t = clamp(value); return t * t * (3 - 2 * t); };

function paintGlobe(ctx, size, land, longitude, emphasis) {
  const center = size / 2;
  const radius = size * 0.448;
  const projection = geoOrthographic()
    .rotate([-longitude, -25])
    .translate([center, center])
    .scale(radius)
    .clipAngle(90);
  const path = geoPath(projection, ctx);
  ctx.clearRect(0, 0, size, size);

  const halo = ctx.createRadialGradient(center, center, radius * .93, center, center, radius * 1.1);
  halo.addColorStop(0, 'rgba(71, 164, 188, 0)');
  halo.addColorStop(.48, 'rgba(76, 155, 181, .14)');
  halo.addColorStop(1, 'rgba(71, 164, 188, 0)');
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, size, size);

  ctx.save();
  ctx.beginPath();
  path({ type: 'Sphere' });
  ctx.clip();
  const ocean = ctx.createRadialGradient(size * .31, size * .27, radius * .1, center, center, radius * 1.15);
  ocean.addColorStop(0, '#15425c');
  ocean.addColorStop(.62, '#0d3048');
  ocean.addColorStop(1, '#081e30');
  ctx.fillStyle = ocean;
  ctx.fillRect(0, 0, size, size);

  if (land) {
    ctx.beginPath();
    path(land);
    ctx.fillStyle = '#386773';
    ctx.fill();
    ctx.strokeStyle = 'rgba(167, 210, 212, .74)';
    ctx.lineWidth = .75;
    ctx.stroke();
  }

  ctx.beginPath();
  path(GRATICULE);
  ctx.strokeStyle = 'rgba(97, 169, 199, .24)';
  ctx.lineWidth = .55;
  ctx.stroke();

  const shade = ctx.createRadialGradient(size * .3, size * .26, 0, center, center, radius * 1.1);
  shade.addColorStop(0, 'rgba(199, 230, 230, .07)');
  shade.addColorStop(.67, 'rgba(3, 13, 25, 0)');
  shade.addColorStop(1, 'rgba(3, 13, 25, .49)');
  ctx.fillStyle = shade;
  ctx.fillRect(0, 0, size, size);

  if (emphasis > 0) {
    const [x, y] = projection([115, 28]);
    const glow = ctx.createRadialGradient(x, y, 0, x, y, radius * .24);
    glow.addColorStop(0, `rgba(124, 222, 215, ${.16 * emphasis})`);
    glow.addColorStop(1, 'rgba(124, 222, 215, 0)');
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, size, size);
  }
  ctx.restore();
  ctx.beginPath();
  path({ type: 'Sphere' });
  ctx.strokeStyle = 'rgba(136, 206, 223, .62)';
  ctx.lineWidth = 1;
  ctx.stroke();
}

export default function IntroAnimation({ onComplete }) {
  const overlayRef = useRef(null);
  const canvasRef = useRef(null);
  const skipRef = useRef(null);

  useEffect(() => {
    const overlay = overlayRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) { onComplete(); return undefined; }
    const controller = new AbortController();
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let land = null;
    let size = 600;
    let frameId;
    let finished = false;
    const started = performance.now();
    const previousFocus = document.activeElement;
    const finish = () => {
      if (finished) return;
      finished = true;
      onComplete();
    };
    const motionChanged = () => { if (motion.matches) finish(); };
    motion.addEventListener('change', motionChanged);
    skipRef.current?.focus({ preventScroll: true });

    const resize = () => {
      size = Math.max(1, canvas.getBoundingClientRect().width);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(size * dpr);
      canvas.height = Math.round(size * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    fetch(`${import.meta.env.BASE_URL}geo/intro-land.geojson`, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error('Globe asset unavailable');
        return response.json();
      })
      .then(data => { land = data; })
      .catch(error => { if (error.name !== 'AbortError') finish(); });

    const animate = now => {
      if (finished) return;
      const elapsed = now - started;
      const rotation = 1 - (1 - clamp(elapsed / 1900)) ** 3;
      const brand = smooth((elapsed - 1400) / 650);
      overlay.style.setProperty('--globe-opacity', String(smooth(elapsed / 450)));
      overlay.style.setProperty('--brand-opacity', String(brand));
      overlay.style.setProperty('--brand-offset', `${(1 - brand) * 12}px`);
      overlay.style.setProperty('--subtitle-opacity', String(smooth((elapsed - 1650) / 750)));
      overlay.style.opacity = String(1 - smooth((elapsed - 3450) / 550));
      paintGlobe(ctx, size, land, 15 + 100 * rotation, brand);
      if (elapsed >= DURATION) finish();
      else frameId = requestAnimationFrame(animate);
    };
    frameId = requestAnimationFrame(animate);
    return () => {
      finished = true;
      cancelAnimationFrame(frameId);
      observer.disconnect();
      controller.abort();
      motion.removeEventListener('change', motionChanged);
      if (previousFocus?.isConnected) previousFocus.focus({ preventScroll: true });
    };
  }, [onComplete]);

  return (
    <div
      className="intro-animation" ref={overlayRef}
      role="dialog" aria-modal="true" aria-label="CIO-RainCast opening animation"
      onKeyDown={event => { if (event.key === 'Escape') onComplete(); }}
    >
      <div className="intro-composition" lang="en">
        <div className="intro-globe">
          <canvas ref={canvasRef} aria-hidden="true" />
        </div>
        <div className="intro-wordmark">
          <h1>CIO-RainCast</h1>
          <p>
            <span>East Asian Summer Monsoon</span>
            <span>Intraseasonal Rainfall</span>
            <span>Prediction</span>
          </p>
        </div>
      </div>
      <button className="intro-skip" ref={skipRef} onClick={onComplete}>Skip intro <span aria-hidden="true">↗</span></button>
    </div>
  );
}
