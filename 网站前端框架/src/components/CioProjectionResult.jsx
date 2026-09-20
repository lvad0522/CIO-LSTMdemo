import { Fragment } from 'react';

const INPUT_LABELS = {
  'cio-npy': '上传的 CIO 序列 (.npy)',
  'raw-zip': '原始 U850 气象场 (.zip)',
};

const NORMALIZATION_LABELS = {
  'training-equivalent': '训练同口径（截取前 2240 点后整段 min-max）',
  partial: '按现有序列整段 min-max（短序列）',
};

function formatWindow(window) {
  if (!Array.isArray(window) || window.length !== 2) return null;
  return `${window[0][0]}月${window[0][1]}日 - ${window[1][0]}月${window[1][1]}日`;
}

export default function CioProjectionResult({ job, onShowPrediction }) {
  const projection = job.projection || {};
  const normalization = job.normalization || {};
  const inputKind = job.inputKind || projection.inputKind;
  const years = projection.years || [];
  const selectedRange = normalization.selectedRange || [];
  const missingMonths = projection.missingMonths || [];

  const rows = [
    ['输入文件', job.filename],
    ['输入类别', INPUT_LABELS[inputKind] || inputKind],
    ['CIO 输入长度', normalization.inputLength ? `${normalization.inputLength} 点` : null],
    ['归一化使用长度', normalization.usedLength ? `${normalization.usedLength} 点` : null],
    ['归一化口径', NORMALIZATION_LABELS[normalization.mode] || normalization.mode],
    ['模型取用窗口', selectedRange.length === 2
      ? `第 ${selectedRange[0] + 1}-${selectedRange[1]} 点（112 点）`
      : null],
    ['CIO 原始范围', Number.isFinite(normalization.cioMin) && Number.isFinite(normalization.cioMax)
      ? `${normalization.cioMin.toFixed(6)} - ${normalization.cioMax.toFixed(6)}`
      : null],
    ['原始场投影窗口', formatWindow(projection.window)],
    ['投影年份', years.length ? `${years[0]}-${years[years.length - 1]}（${years.length} 年）` : null],
    ['投影序列长度', projection.seriesLength ? `${projection.seriesLength} 点` : null],
    ['缺失月份', missingMonths.length ? `${missingMonths.length} 个` : (projection.seriesLength ? '无' : null)],
    ['投影基', projection.matPath
      ? `${String(projection.matPath).replace(/^.*[\\/]/, '')}（${projection.matSource === 'real' ? '真实资产' : '夹具资产'}）`
      : null],
    ['任务耗时', job.durationSeconds != null ? `${job.durationSeconds} 秒` : null],
  ].filter(([, value]) => value != null && value !== '');

  return (
    <div className="result-card">
      <div className="result-header">
        <div>
          <h3 className="result-title" style={{ margin: 0 }}>CIO计算与验证</h3>
          <p className="metric-note" style={{ margin: '6px 0 0' }}>
            {inputKind === 'raw-zip'
              ? '原始气象场已完成 CIO 投影与训练口径校验'
              : '上传的 CIO 序列已通过格式与训练口径校验'}
          </p>
        </div>
        <span className="region-badge">已完成</span>
      </div>

      <div className="prediction-summary">
        {rows.map(([label, value]) => (
          <Fragment key={label}>
            <span>{label}</span><strong>{value}</strong>
          </Fragment>
        ))}
      </div>

      {(normalization.warning || projection.warning || job.warning) && (
        <p className="smoke-warning">
          {normalization.warning || projection.warning || job.warning}
        </p>
      )}

      <p className="chart-note">
        当前后端使用同一任务完成 CIO 校验和 LSTM 推理；CIO 页面展示投影与归一化结果，
        预测场和实况对比在 LSTM 页面查看。
      </p>
      <button type="button" className="export-btn" onClick={onShowPrediction}>
        查看 LSTM 预测结果
      </button>
    </div>
  );
}
