export default function ParamPanel({ module, params, onChange }) {
  if (module === 'cio') {
    return (
      <div className="panel-section">
        <h3 className="section-title">CIO参数设置</h3>
        <div className="param-group">
          <label>滤波低频 (周期天数)</label>
          <input type="number" min={0.01} max={0.5} step={0.01}
            value={params.cioFilterLow}
            onChange={e => onChange('cioFilterLow', +e.target.value)} />
          <span className="param-hint">默认 0.02 ≈ 50天周期</span>
        </div>
        <div className="param-group">
          <label>滤波高频 (周期天数)</label>
          <input type="number" min={0.01} max={0.5} step={0.01}
            value={params.cioFilterHigh}
            onChange={e => onChange('cioFilterHigh', +e.target.value)} />
          <span className="param-hint">默认 0.1 ≈ 10天周期</span>
        </div>
        <div className="param-group">
          <label>海域范围</label>
          <select value={params.cioRegion}
            onChange={e => onChange('cioRegion', e.target.value)}>
            <option value="tropical">热带印度洋 (20°S-20°N, 40°E-120°E)</option>
            <option value="north">北印度洋 (0°-20°N, 40°E-120°E)</option>
          </select>
        </div>
        <div className="param-group">
          <label>显著性置信水平</label>
          <select value={params.cioSignificance}
            onChange={e => onChange('cioSignificance', e.target.value)}>
            <option value="0.90">90%</option>
            <option value="0.95">95%</option>
            <option value="0.99">99%</option>
          </select>
        </div>
      </div>
    );
  }

  return (
    <div className="panel-section">
      <h3 className="section-title">LSTM预测参数</h3>
      <div className="param-group">
        <label>预测年份</label>
        <select value={params.predYear}
          onChange={e => onChange('predYear', +e.target.value)}>
          {Array.from({ length: 20 }, (_, i) => 2000 + i).map(y => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>
      </div>
      <div className="param-group">
        <label>提前预报时间 (lead time)</label>
        <select value={params.leadTime}
          onChange={e => onChange('leadTime', e.target.value)}>
          {Array.from({ length: 20 }, (_, i) => i + 1).map(n => (
            <option key={`pre${n}`} value={`pre${n}`}>提前{n}天 (pre{n})</option>
          ))}
        </select>
      </div>
      <div className="param-group">
        <label>预测区域</label>
        <select value={params.predRegion}
          onChange={e => onChange('predRegion', e.target.value)}>
          <option value="eastasia">东亚 (81×101)</option>
        </select>
        <span className="param-hint">数据集仅覆盖东亚 20-40°N, 100-125°E</span>
      </div>
    </div>
  );
}
