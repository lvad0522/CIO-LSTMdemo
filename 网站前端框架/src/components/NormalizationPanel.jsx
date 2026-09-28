// 阶段② LSTM 输入：把"② 落盘的那一份"直接画出来。
// 它**自己**去 /api/chain/jobs/{id}/series 取 normalizedWindow（design D12），
// 不复用① 的取数结果 —— 面板上看到的与阶段③ 读盘喂模型的是同一个端点字段。
import { Fragment, useEffect, useState } from 'react';
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_BASE || '';

const MODE_LABELS = {
  'training-equivalent': '训练同口径：前 2240 点统一确定 min-max 范围',
  partial: '短序列：按现有长度整段 min-max（尺度与训练不完全一致）',
  'extended-base': '为覆盖目标年扩大了 min-max 基准（与训练口径存在偏差）',
};

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
  const usedLength = normalization.usedLength ?? '—';
  const targetYear = normalization.targetYear ?? job?.projection?.year;
  const flow = [
    {
      label: '投影 CIO',
      value: `${normalization.inputLength ?? '—'} 点`,
      note: '阶段①得到的完整时间序列',
    },
    {
      label: '确定统一缩放范围',
      value: `${usedLength} 点`,
      note: `最小值 ${formatValue(normalization.cioMin)}，最大值 ${formatValue(normalization.cioMax)}`,
    },
    {
      label: targetYear ? `取 ${targetYear} 年` : '取目标年',
      value: selectedRange.length === 2
        ? `第 ${selectedRange[0] + 1}–${selectedRange[1]} 点`
        : '112 点',
      note: '每天缩放到 0–1',
    },
    {
      label: '送入 LSTM',
      value: `${chartData.length || 112} 天`,
      note: '阶段③实际读取的输入张量',
    },
  ];

  return (
    <div className="result-card normalization-panel">
      <div className="result-header">
        <div>
          <h3 className="result-title" style={{ margin: 0 }}>LSTM 实际输入（112天）</h3>
          <p className="metric-note" style={{ margin: '6px 0 0' }}>
            CIO 原值大小没有统一量纲，先按训练时的规则缩放到 0–1，再截取目标年送入模型。
          </p>
        </div>
        <span className="region-badge">{normalization.mode || '—'}</span>
      </div>

      <div className="normalization-flow" aria-label="准备 LSTM 输入流程">
        {flow.map((step, index) => (
          <Fragment key={step.label}>
            <div className="normalization-step">
              <span>{step.label}</span>
              <strong>{step.value}</strong>
              <small>{step.note}</small>
            </div>
            {index < flow.length - 1 && <span className="normalization-arrow" aria-hidden="true">→</span>}
          </Fragment>
        ))}
      </div>
      <p className="metric-note">
        {MODE_LABELS[normalization.mode] || '正在读取归一化口径'}
      </p>
      {normalization.warning && <p className="smoke-warning">{normalization.warning}</p>}
      {error && <div className="error-banner">{error}</div>}

      {chartData.length ? (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 22, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="index"
              minTickGap={24}
              label={{ value: '目标年输入的第几天', position: 'insideBottom', offset: -12 }}
            />
            <YAxis
              width={58}
              domain={[0, 1]}
              label={{ value: '缩放后的 CIO', angle: -90, position: 'insideLeft' }}
            />
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
              name="LSTM 实际输入"
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
