import { useState, useRef, useEffect } from 'react';

export default function RunButton({ onRun, running, disabled }) {
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const handleClick = () => {
    setProgress(0);
    let p = 0;
    intervalRef.current = setInterval(() => {
      p += 2 + Math.floor(Math.random() * 5);
      if (p >= 100) {
        p = 100;
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        setProgress(100);
        onRun();
        setTimeout(() => setProgress(0), 500);
      } else {
        setProgress(p);
      }
    }, 150);
  };

  const isRunning = running || (progress > 0 && progress < 100);

  return (
    <div className="panel-section">
      <button
        className="run-btn"
        onClick={handleClick}
        disabled={disabled || isRunning}
      >
        {isRunning ? '⏳ 运行中...' : '▶ 运行'}
      </button>
      {(progress > 0 || running) && (
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
          <span className="progress-text">{Math.round(progress)}%</span>
        </div>
      )}
    </div>
  );
}
