export default function RunButton({ onRun, running, disabled, progress = 0, stage = '' }) {
  return (
    <div className="panel-section">
      <button
        className="run-btn"
        onClick={onRun}
        disabled={disabled || running}
      >
        {running ? '⏳ 运行中...' : '▶ 运行'}
      </button>
      {running && (
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
          <span className="progress-text">{Math.round(progress)}%</span>
          {stage && <span className="progress-stage">{stage}</span>}
        </div>
      )}
    </div>
  );
}
