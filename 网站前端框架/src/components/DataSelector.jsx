import NpyUploader from './NpyUploader';

// 2026-09-21 PM 决定：注释掉「CIO 序列 (.npy)」入口 —— 标准CIO 投影基（.mat）
// 由后端本地调用，用户只需上传原始气象场，无需自带 CIO 序列。
// 恢复方式：取消下一行注释并把 App.jsx 的 uploadKind 默认值改回 'npy'
//（NpyUploader 与 handleCioFileChange 的 .npy 分支都还在，未删）。
const UPLOAD_KINDS = [
  // { value: 'npy', label: 'CIO 序列 (.npy)' },
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
