import { useCallback, useEffect, useState } from 'react';
import IntroAnimation from './IntroAnimation';

const SESSION_KEY = 'cio-rain.intro-seen.v1';

function shouldPlayIntro() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return false;
  try { return sessionStorage.getItem(SESSION_KEY) !== 'yes'; }
  catch { return true; }
}

export default function IntroShell({ children }) {
  const [playing, setPlaying] = useState(shouldPlayIntro);
  const complete = useCallback(() => {
    try { sessionStorage.setItem(SESSION_KEY, 'yes'); } catch { /* Storage may be unavailable. */ }
    setPlaying(false);
  }, []);

  useEffect(() => {
    const replay = () => {
      if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) setPlaying(true);
    };
    window.addEventListener('cio-rain:replay-intro', replay);
    return () => window.removeEventListener('cio-rain:replay-intro', replay);
  }, []);

  return (
    <>
      <div inert={playing} aria-hidden={playing || undefined}>{children}</div>
      {playing && <IntroAnimation onComplete={complete} />}
    </>
  );
}
