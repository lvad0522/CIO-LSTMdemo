import NpyUploader from './NpyUploader';

const UPLOAD_KINDS = [
  { value: 'npy', label: 'CIO 序列 (.npy)' },
  { value: 'zip', label: '原始气象场 (.zip)' },
];

export default function DataSelector({
  value, onChange, disabled, cioFile, onCioFileChange,
  uploadKind = 'npy', onUploadKindChange,
}) {
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
        <span>自行上传数据（模型验证）</span>
      </label>
      {value === 'upload' && (
        <>
          <div className="upload-kind-tabs" role="group" aria-label="上传数据类别">
            {UPLOAD_KINDS.map(kind => (
              <button
                key={kind.value}
                type="button"
                className={`upload-kind-tab ${uploadKind === kind.value ? 'is-active' : ''}`}
                aria-pressed={uploadKind === kind.value}
                disabled={disabled}
                onClick={() => onUploadKindChange?.(kind.value)}
              >
                {kind.label}
              </button>
            ))}
          </div>
          <NpyUploader
            kind={uploadKind}
            file={cioFile}
            onFileChange={onCioFileChange}
            disabled={disabled}
          />
        </>
      )}
    </div>
  );
}
