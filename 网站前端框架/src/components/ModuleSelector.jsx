export default function ModuleSelector({ active, onSelect, cioDisabled }) {
  return (
    <div className="panel-section">
      <h3 className="section-title">功能模块</h3>
      <div className="module-buttons">
        <button
          className={`module-btn ${active === 'cio' ? 'active' : ''}`}
          onClick={() => onSelect('cio')}
          disabled={cioDisabled}
          title={cioDisabled ? '使用默认CIO数据时无需计算，请先选择"自行上传SST+Uwind数据"' : ''}
        >
          <span>CIO计算与验证</span>
          {cioDisabled && <span className="badge-disabled">不可用</span>}
        </button>
        <button
          className={`module-btn ${active === 'lstm' ? 'active' : ''}`}
          onClick={() => onSelect('lstm')}
        >
          <span>LSTM降雨预测</span>
        </button>
      </div>
    </div>
  );
}
