import { useEffect, useRef } from 'react';

// 既有固定画布按浏览器视口等比缩放；尺寸仍只在 canvas.css 中定义。
export default function ViewportCanvas({ children }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const fit = () => {
      const cs = getComputedStyle(el);
      const w = parseFloat(cs.width);
      const h = parseFloat(cs.height);
      if (!w || !h) return;
      el.style.setProperty('--canvas-scale', String(Math.min(window.innerWidth / w, window.innerHeight / h)));
    };
    fit();
    window.addEventListener('resize', fit);
    return () => window.removeEventListener('resize', fit);
  }, []);

  return (
    <div className="app-viewport">
      <div className="app-canvas" ref={ref}>{children}</div>
    </div>
  );
}
