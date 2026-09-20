// 阶段② 归一化窗口：把"② 落盘的那一份"直接画出来。
// 它**自己**去 /api/chain/jobs/{id}/series 取 normalizedWindow（design D12），
// 不复用① 的取数结果 —— 面板上看到的与阶段③ 读盘喂模型的是同一个端点字段。
import { useEffect, useState } from 'react';
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_BASE || '';

const MODE_LABELS = {
  'training-equivalent': '训练同口径：截取前 2240 点后整段 min-max',
  partial: '短序列：按现有长度整段 min-max（尺度与训练不完全一致）',
};

const TRAIN_SEQ_LEN = 2240;

function formatValue(value) {
  return Number.isFinite(value) ? value.toFixed(4) : '—';
}

export default function NormalizationPanel({ job }) {
  const [window_, setWindow] = useState(null);
  const [error, setError] = useState(null);
  const normalization = job?.normalization || {};
  const selectedRange = normalization.selectedRange || [];

  useEffect(() => {
    if (!job?.jobId) return undefined;
    let alive = true;
    fetch(`${API_BASE}/api/chain/jobs/${job.jobId}/series`)
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || `归一化窗口读取失败 (${response.status})`);
        }
        return payload;
      })
      .then(payload => { if (alive) setWindow(payload.normalizedWindow || []); })
      .catch((reason) => { if (alive) setError(reason.message || '归一化窗口读取失败'); });
    return () => { alive = false; };
  }, [job?.jobId]);

  const chartData = (window_ || []).map((value, index) => ({ index: index + 1, value }));
  // 截断口径写成"截前 2240"只在真的截过时才对：短序列是整段自归一
  const truncated = Number(normalization.usedLength) === TRAIN_SEQ_LEN
    && Number(normalization.inputLength) > TRAIN_SEQ_LEN;

  const summary = [
    `输入 ${normalization.inputLength ?? '—'} 点`,
    truncated ? `截前 ${TRAIN_SEQ_LEN}` : `整段 ${normalization.usedLength ?? '—'} 点`,
    `min-max[${formatValue(normalization.cioMin)}, ${formatValue(normalization.cioMax)}]`,
    selectedRange.length === 2 ? `取第 ${selectedRange[0] + 1}–${selectedRange[1]} 点` : null,
  ].filter(Boolean).join(' → ');

  return (
    <div className="result-card normalization-panel">
      <div className="result-header">
        <div>
          <h3 className="result-title" style={{ margin: 0 }}>归一化窗口</h3>
          <p className="metric-note" style={{ margin: '6px 0 0' }}>
            这一份就是阶段③ 从磁盘读走的张量；界面上看到的与喂进模型的是同一个文件。
          </p>
        </div>
        <span className="region-badge">{normalization.mode || '—'}</span>
      </div>

      <div className="normalization-summary">{summary || '正在读取归一化摘要…'}</div>
      <p className="metric-note">
        {MODE_LABELS[normalization.mode] || '正在读取归一化口径'}
      </p>
      {normalization.warning && <p className="smoke-warning">⚠️ {normalization.warning}</p>}
      {error && <div className="error-banner">{error}</div>}

      {chartData.length ? (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 22, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="index"
              minTickGap={24}
              label={{ value: '窗口内第几天', position: 'insideBottom', offset: -12 }}
            />
            <YAxis width={54} domain={[0, 1]} />
            <Tooltip
              labelFormatter={value => `第 ${value} 天`}
              formatter={value => [Number(value).toFixed(4), '归一化值']}
            />
            <Line
              type="linear"
              dataKey="value"
              stroke="#3b8734"
              dot={false}
              strokeWidth={1.5}
              name="喂进模型的 112 点"
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="cio-unavailable" role="status">
          <strong>正在读取归一化窗口</strong>
          <span>等待任务序列端点</span>
        </div>
      )}
    </div>
  );
}
