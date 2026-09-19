import { useRef, useState } from 'react';

const CONFIG = {
  npy: {
    ext: '.npy',
    dropzone: '拖拽 CIO .npy 到此处，或点击选择文件',
    badType: '仅支持 .npy 文件（一维 projected CIO 序列）',
    contract: (
      <>
        接受长度 <strong>2352</strong>（官方 21 年 × 112）或 <strong>2240</strong>（训练长度）
        的一维 projected CIO 序列，两者都按训练口径「切前 2240 点 → 整段 min-max → 取首年 112 点」。
        更短的序列（≥112 点，例如缺月份的真实件）按整段自归一化并给出警告。
      </>
    ),
  },
  zip: {
    ext: '.zip',
    dropzone: '拖拽原始气象场 .zip 到此处，或点击选择文件',
    badType: '仅支持 .zip 文件（打包的原始气象场）',
    contract: (
      <>
        打包 <strong>5–9 月逐日原始场 nc</strong>（每个文件只含 2–29 日，28 天/月；
        21 年 × 5 月 = 105 个），上传后由后端现场投影为 CIO。
        文件名需含 <code>u850</code> / <code>uwnd</code> 关键字；zip 里必须含 2000 年。
      </>
    ),
  },
};

export default function NpyUploader({ kind = 'npy', file, onFileChange, disabled = false }) {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  const config = CONFIG[kind] || CONFIG.npy;

  const selectFile = (nextFile) => {
    if (!nextFile) return;
    const lowerName = nextFile.name.toLowerCase();
    const detectedKind = lowerName.endsWith('.npy')
      ? 'npy'
      : lowerName.endsWith('.zip')
        ? 'zip'
        : null;
    if (!detectedKind) {
      setError('仅支持 CIO .npy 或原始气象场 .zip 文件');
      onFileChange?.(null);
      return;
    }
    setError(null);
    onFileChange?.(nextFile, detectedKind);
  };

  return (
    <div className="panel-section">
      <h3 className="section-title">上传数据</h3>
      <div
        className={`upload-dropzone ${dragOver ? 'drag-over' : ''} ${disabled ? 'uploading' : ''}`}
        onClick={() => !disabled && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (!disabled) selectFile(e.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".npy,.zip"
          hidden
          disabled={disabled}
          onChange={(e) => selectFile(e.target.files?.[0])}
        />
        <span className="upload-hint">{config.dropzone}</span>
      </div>
      {file && (
        <div className="upload-result">
          <span className="upload-ok">✓ {file.name}</span>
          <span className="param-hint">
            {(file.size / 1024 / 1024).toFixed(2)} MB · 点击“运行”后上传并推理
          </span>
        </div>
      )}
      {error && <div className="upload-error">⚠️ {error}</div>}
      <p className="param-hint upload-contract">{config.contract}</p>
    </div>
  );
}
