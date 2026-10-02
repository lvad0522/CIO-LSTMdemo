import { useRef, useState } from 'react';

const CONFIG = {
  npy: {
    ext: '.npy',
    dropzone: '拖拽 CIO .npy 到此处，或点击选择文件',
    badType: '仅支持 .npy 文件（一维 projected CIO 序列）',
    contract: (
      <>
        <strong>整条链只需上传一次</strong>：这个文件会一路供到降水推理，中途不再要求重选。
        接受长度 <strong>2352</strong>（官方 21 年 × 112）或 <strong>2240</strong>（训练长度）
        的一维 projected CIO 序列，两者都按训练口径「切前 2240 点 → 整段 min-max → 取首年 112 点」。
        序列首年视作 2000 年、每年按 112 天折算；更短的序列（≥112 点，例如缺月份的真实件）
        按整段自归一化并给出警告。
      </>
    ),
  },
  zip: {
    ext: '.zip',
    dropzone: '拖拽原始气象场 .zip 到此处，或点击选择文件',
    badType: '仅支持 .zip 文件（打包的原始气象场）',
    contract: (
      <>
        <strong>整条链只需上传一次</strong>（投影 → 归一化 → 降水推理都用这一份）。
        将逐日 U850 距平 NetCDF（.nc / .nc4）打包为 zip，可按日、月、年或多年组织。
        系统读取时间坐标、识别风场与 850 hPa 层，调整维度顺序和经纬度网格，并统一风速单位。
        当前模型从目标季节中选取每月 2–29 日，共 112 点；其余日期不会送入模型，总年份数无需固定为 21 年。
        必须覆盖目标窗口和 20°S–20°N、40°E–120°E；缺日期、重复日期、小时数据或无法确认的变量会明确报错。
        不会把绝对风场自动当作距平；请在变量属性或文件名中标明 anomaly/anom。
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
            {(file.size / 1024 / 1024).toFixed(2)} MB · 点击“运行”后上传并处理
          </span>
        </div>
      )}
      {error && <div className="upload-error">⚠️ {error}</div>}
      <p className="param-hint upload-contract">{config.contract}</p>
    </div>
  );
}
