import NpyUploader from './NpyUploader';

export default function DataSelector({ value, onChange, disabled }) {
  return (
    <div className="panel-section">
      <h3 className="section-title">数据来源</h3>
      <label className={`radio-label ${disabled ? 'disabled' : ''}`}>
        <input
          type="radio"
          name="dataSource"
          value="default"
          checked={value === 'default'}
          onChange={() => onChange('default')}
          disabled={disabled}
        />
        <span>使用系统默认CIO数据</span>
      </label>
      <label className={`radio-label ${disabled ? 'disabled' : ''}`}>
        <input
          type="radio"
          name="dataSource"
          value="upload"
          checked={value === 'upload'}
          onChange={() => onChange('upload')}
          disabled={disabled}
        />
        <span>自行上传SST + Uwind数据</span>
      </label>
      {value === 'upload' && !disabled && <NpyUploader />}
    </div>
  );
}
