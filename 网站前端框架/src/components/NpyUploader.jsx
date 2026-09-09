import { useRef, useState } from 'react';

const API_BASE = 'http://localhost:8000';

export default function NpyUploader() {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const upload = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.npy')) {
      setError('仅支持 .npy 文件');
      setResult(null);
      return;
    }
    setUploading(true);
    setError(null);
    setResult(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `上传失败 (${res.status})`);
      setResult(data);
    } catch (err) {
      setError(err.message || '连接后端失败，请确认后端已启动');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="panel-section">
      <h3 className="section-title">上传数据</h3>
      <div
        className={`upload-dropzone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); upload(e.dataTransfer.files?.[0]); }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".npy"
          hidden
          onChange={(e) => upload(e.target.files?.[0])}
        />
        {uploading ? <span>上传中…</span> : (
          <span className="upload-hint">拖拽 .npy 到此处，或点击选择文件</span>
        )}
      </div>
      {result && (
        <div className="upload-result">
          <span className="upload-ok">✓ {result.filename}</span>
          <span className="param-hint">
            形状 ({result.shape.join(', ')}) · {result.dtype}
          </span>
        </div>
      )}
      {error && <div className="upload-error">⚠️ {error}</div>}
    </div>
  );
}
