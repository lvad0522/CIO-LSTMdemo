import { Fragment, useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { getHeatColor } from '../utils/heatColor';

const API_BASE = import.meta.env.VITE_API_BASE || '';

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

function useCioDiagnostics(jobId, confidence) {
  const [data, setData] = useState({
    capabilities: null,
    reference: null,
    mode: null,
    series: null,
    completeness: null,
    spectrum: null,
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    // 任务级诊断走链路自己的端点；三个**资产端点**（capabilities/reference/
    // mode/u850）不属于任何一个任务，保持原样直接访问 /api/cio。
    const endpoints = {
      capabilities: '/api/cio/capabilities',
      reference: '/api/cio/reference',
      mode: '/api/cio/mode/u850',
      series: `/api/chain/jobs/${jobId}/series`,
      completeness: `/api/chain/jobs/${jobId}/completeness`,
      spectrum: `/api/chain/jobs/${jobId}/spectrum?confidence=${confidence}`,
    };
    Promise.all(Object.entries(endpoints).map(async ([key, path]) => {
      const response = await fetch(`${API_BASE}${path}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `${key} 请求失败 (${response.status})`);
      return [key, payload];
    }))
      .then(entries => {
        if (alive) setData(Object.fromEntries(entries));
      })
      .catch(reason => {
        if (alive) setError(reason.message || 'CIO 诊断数据加载失败');
      });
    return () => { alive = false; };
  }, [jobId, confidence]);

  return { data, error };
}

function EmptyPanel({ label, reason }) {
  return (
    <div className="cio-unavailable" role="status">
      <strong>{label}</strong>
      <span>{reason}</span>
    </div>
  );
}

function ModeMap({ mode }) {
  if (!mode) return <EmptyPanel label="正在读取 U850 模态" reason="等待真实 MAT 资产响应" />;
  const maxAbs = Math.max(Math.abs(mode.min), Math.abs(mode.max), 1e-12);
  return (
    <div className="cio-mode-map-wrap">
      <div className="cio-mode-map" aria-label="第一 CIO 模态 U850 空间载荷">
        {[...mode.data].reverse().map((row, rowIndex) => (
          <div className="cio-mode-row" key={mode.rows - 1 - rowIndex}>
            {row.map((value, colIndex) => (
              <span
                key={colIndex}
                className="cio-mode-cell"
                style={{ backgroundColor: getHeatColor(value, -maxAbs, maxAbs) }}
                title={`${mode.lon[colIndex]}°E, ${mode.lat[mode.rows - 1 - rowIndex]}°N: ${value.toFixed(5)}`}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="cio-axis-row"><span>40°E</span><span>80°E</span><span>120°E</span></div>
      <div className="cio-mode-legend">
        <span>{(-maxAbs).toFixed(3)}</span>
        <div className="correlation-gradient" />
        <span>0</span>
        <span>{maxAbs.toFixed(3)}</span>
      </div>
    </div>
  );
}

function coverageRows(completeness, length = 0) {
  if (!completeness) return [];
  const blockSize = completeness.blockSize || 112;
  if (!completeness.calendarKnown) {
    const completePoints = completeness.completeBlocks * blockSize;
    const percent = length ? completePoints / length * 100 : 0;
    const remainder = completeness.remainder
      ? `，另有 ${completeness.remainder} 点余数`
      : '，无余数';
    return [
      ['数据覆盖', `${length} 点 = ${completeness.completeBlocks}×${blockSize}${remainder}（完整分块占 ${percent.toFixed(1)}%）`],
      ['完整度基准', `每个投影窗口应有 ${blockSize} 点；仅校验长度分块，不代表日期或数值质量`],
    ];
  }
  const yearRows = completeness.daysPerYear || [];
  const actual = yearRows.reduce((sum, row) => sum + row.days, 0);
  const expected = yearRows.length * blockSize;
  const completeYears = yearRows.filter(row => row.days === blockSize).length;
  const percent = expected ? actual / expected * 100 : 0;
  return [
    ['数据覆盖', `${actual}/${expected} 天（${completeYears}/${yearRows.length} 年完整，${percent.toFixed(1)}%）`],
    ['完整度基准', `实际天数 / 年数×${blockSize} 天；每年按投影窗口应有 ${blockSize} 天，不代表数值质量`],
  ];
}

function formatPower(value) {
  if (!Number.isFinite(value) || value <= 0) return '-';
  return value >= 0.01 && value < 1000 ? value.toFixed(3) : value.toExponential(1);
}

const MAT_SOURCE_LABELS = {
  real: 'real（服务器真件，e·eᵀ 过正交判据）',
  mock: 'mock（模拟数据沙盒里的夹具件）',
};

function matIdentityText(projection, mode) {
  if (!projection?.matSource) return null;      // 取不到就不显示这一行，不摆空占位
  const parts = [projection.matPath];
  if (projection.nModes && projection.sstColumns && projection.u850Columns) {
    parts.push(`${projection.nModes} 模态 × [SST ${projection.sstColumns} | U850 ${projection.u850Columns}]`);
  }
  if (mode?.explainedVariancePercent != null) {
    parts.push(`第一模态解释方差 ${mode.explainedVariancePercent.toFixed(1)}%`);
  }
  parts.push(MAT_SOURCE_LABELS[projection.matSource] || projection.matSource);
  return parts.join(' · ');
}

export default function CioProjectionResult({ job, confidence = '0.95' }) {
  const { data, error } = useCioDiagnostics(job.jobId, confidence);
  const projection = job.projection || {};
  const normalization = job.normalization || {};
  const inputKind = job.inputKind || projection.inputKind;
  const selectedRange = normalization.selectedRange || [];

  const chartData = useMemo(() => {
    const series = data.series;
    if (!series) return [];
    const actualByDate = new Map(
      (data.reference?.dates || []).map((item, index) => [item, data.reference.values[index]]),
    );
    return series.raw.map((projected, index) => {
      const date = series.dates?.[index] || null;
      return {
        x: date || index + 1,
        actual: date ? actualByDate.get(date) : undefined,
        projected,
      };
    });
  }, [data.reference, data.series]);

  const zoomData = chartData.slice(0, Math.min(112, chartData.length));
  const spectrumData = useMemo(() => {
    const spectrum = data.spectrum;
    if (!spectrum?.available) return [];
    return spectrum.periodDays.map((period, index) => ({
      period,
      projectedPower: spectrum.projectedPower[index],
      projectedConfidence: spectrum.projectedConfidence[index],
      actualPower: spectrum.actualPower?.[index],
      actualConfidence: spectrum.actualConfidence?.[index],
    }));
  }, [data.spectrum]);
  const calendarKnown = Boolean(data.series?.calendarKnown);
  const lineLabel = job.projectionMode === 'u850_only'
    ? 'U850-only 投影（链路诊断）'
    : '上传的 projected CIO';
  const years = projection.years || [];
  const missingMonths = projection.missingMonths || [];
  // `.npy` 入口：① 真的没跑过，写清楚而不是留一片看着像"跑了但空"的字段
  const stage1Skipped = Boolean(projection.stage1Skipped);
  const rows = [
    ['标准CIO 身份', matIdentityText(projection, data.mode)],
    ['输入文件', job.filename],
    ['输入类别', INPUT_LABELS[inputKind] || inputKind],
    ['诊断口径', stage1Skipped
      ? '未运行 —— 直接提供了已投影序列'
      : (job.projectionMode === 'u850_only' ? 'U850-only（非论文联合 CIO）' : '上传序列')],
    ['CIO 输入长度', normalization.inputLength ? `${normalization.inputLength} 点` : `${job.result?.seriesLength || 0} 点`],
    ['归一化口径', NORMALIZATION_LABELS[normalization.mode] || normalization.mode],
    ['模型窗口', selectedRange.length === 2 ? `第 ${selectedRange[0] + 1}-${selectedRange[1]} 点` : null],
    ['日期索引', calendarKnown ? '真实日期' : '未知（使用样本序号）'],
    ['投影窗口', formatWindow(projection.window)],
    ['投影年份', years.length ? `${years[0]}-${years[years.length - 1]}（${years.length} 年）` : null],
    ['缺失月份', missingMonths.length ? `${missingMonths.length} 个` : (calendarKnown ? '无' : null)],
    ['任务耗时', job.durationSeconds != null ? `${job.durationSeconds} 秒` : null],
    ...coverageRows(data.completeness, job.result?.seriesLength || 0),
  ].filter(([, value]) => value != null && value !== '');

  return (
    <>
      {error && <div className="error-banner">{error}</div>}
      <div className="result-card">
        <div className="result-header">
          <div>
            <h3 className="result-title" style={{ margin: 0 }}>CIO 计算与链路诊断</h3>
            <p className="metric-note" style={{ margin: '6px 0 0' }}>
              当前启用真实 U850/上传 CIO 数据及其分块诊断谱；SST 联合投影尚未接入。
            </p>
          </div>
          <span className="region-badge">{job.projectionMode === 'u850_only' ? 'U850-only' : '序列检查'}</span>
        </div>
        {stage1Skipped && (
          <p className="chain-stage-skip">
            阶段① 未运行 —— 直接提供了已投影序列（`.npy` 入口跳过投影，从阶段② 开始）。
          </p>
        )}
        <div className="prediction-summary">
          {rows.map(([label, value]) => (
            <Fragment key={label}><span>{label}</span><strong>{value}</strong></Fragment>
          ))}
        </div>
        {job.warning && <p className="smoke-warning">{job.warning}</p>}
      </div>

      <div className="result-card">
        <div className="result-header">
          <div>
            <h3 className="result-title" style={{ margin: 0 }}>CIO 多序列诊断</h3>
            <p className="metric-note" style={{ margin: '6px 0 0' }}>
              对照论文 Figure 3 的结构；只绘制当前链路能证明的数据。
            </p>
          </div>
          <span className="region-badge">置信水平 {(Number(confidence) * 100).toFixed(0)}%</span>
        </div>
        <div className="cio-figure-grid">
          <section className="cio-panel cio-panel-wide">
            <span className="cio-panel-tag">(a)</span>
            <h4>完整输入/投影序列</h4>
            {chartData.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 22, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="x" minTickGap={45} tickFormatter={value => calendarKnown ? String(value).slice(0, 7) : value} />
                  <YAxis width={44} />
                  <Tooltip labelFormatter={value => calendarKnown ? value : `样本 ${value}`} />
                  {calendarKnown && <Line type="linear" dataKey="actual" name="实际 CIO（EOF PC1）" stroke="#3156a3" dot={false} connectNulls={false} strokeWidth={1.5} />}
                  <Line type="linear" dataKey="projected" name={lineLabel} stroke="#3b8734" dot={false} strokeWidth={1.2} />
                </LineChart>
              </ResponsiveContainer>
            ) : <EmptyPanel label="正在读取序列" reason="等待任务序列端点" />}
          </section>

          <section className="cio-panel">
            <span className="cio-panel-tag">(b)</span>
            <h4>首个 112 点窗口放大</h4>
            {zoomData.length ? (
              <ResponsiveContainer width="100%" height={230}>
                <LineChart data={zoomData} margin={{ top: 8, right: 12, bottom: 22, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="x" minTickGap={36} tickFormatter={value => calendarKnown ? String(value).slice(5) : value} />
                  <YAxis width={42} />
                  <Tooltip />
                  {calendarKnown && <Line type="linear" dataKey="actual" name="实际 CIO（EOF PC1）" stroke="#3156a3" dot={false} strokeWidth={1.8} />}
                  <Line type="linear" dataKey="projected" name={lineLabel} stroke="#3b8734" dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            ) : <EmptyPanel label="正在读取局部序列" reason="等待任务序列端点" />}
          </section>

          <section className="cio-panel">
            <span className="cio-panel-tag">(c)</span>
            <h4>U850-only 分块平均功率谱</h4>
            {data.spectrum?.available ? (
              <ResponsiveContainer width="100%" height={230}>
                <LineChart data={spectrumData} margin={{ top: 8, right: 12, bottom: 22, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="period"
                    type="number"
                    scale="log"
                    domain={[5, 112]}
                    ticks={[5, 10, 20, 50, 100]}
                    allowDataOverflow
                    tickFormatter={value => `${value}`}
                    label={{ value: '周期（天）', position: 'insideBottom', offset: -12 }}
                  />
                  <YAxis
                    type="number"
                    scale="log"
                    domain={['auto', 'auto']}
                    allowDataOverflow
                    width={58}
                    tickFormatter={formatPower}
                  />
                  <Tooltip
                    labelFormatter={value => `周期 ${Number(value).toFixed(1)} 天`}
                    formatter={(value, name) => [formatPower(Number(value)), name]}
                  />
                  {data.spectrum.actualPower && (
                    <Line type="linear" dataKey="actualPower" name="实际 CIO 功率" stroke="#3156a3" dot={false} strokeWidth={1.6} />
                  )}
                  {data.spectrum.actualConfidence && (
                    <Line type="linear" dataKey="actualConfidence" name={`实际 CIO ${(Number(confidence) * 100).toFixed(0)}% 上包络`} stroke="#3156a3" strokeDasharray="5 4" dot={false} strokeWidth={1.1} />
                  )}
                  <Line type="linear" dataKey="projectedPower" name={`${lineLabel}功率`} stroke="#3b8734" dot={false} strokeWidth={1.6} />
                  <Line type="linear" dataKey="projectedConfidence" name={`${(Number(confidence) * 100).toFixed(0)}% bootstrap 上包络`} stroke="#3b8734" strokeDasharray="5 4" dot={false} strokeWidth={1.1} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyPanel
                label="当前序列不足以计算分块功率谱"
                reason={data.spectrum?.reason || '正在读取任务级频谱'}
              />
            )}
            <p className="chart-note">
              {data.spectrum?.available
                ? `${data.spectrum.blockCount} 个 ${data.spectrum.blockSize} 天窗口独立估计后平均；虚线为 ${(Number(confidence) * 100).toFixed(0)}% 分块 bootstrap 上包络。诊断周期 5–112 天，不等同于论文 SST+U850 全年连续谱。`
                : '不跨年份缺口拼接序列，也不使用 Pearson 临界 r 代替谱置信线。'}
            </p>
          </section>
        </div>
        <div className="cio-line-key">
          {calendarKnown && <span><i className="is-actual" />实际 CIO（EOF PC1）</span>}
          <span><i className="is-projected" />{lineLabel}</span>
          <span className="is-pending"><i />SST+U850 与 20–100 天联合序列：待接入</span>
        </div>
      </div>

      <div className="result-card">
        <div className="result-header">
          <h3 className="result-title" style={{ margin: 0 }}>第一 CIO 模态 U850 空间载荷</h3>
          {data.mode && <span className="region-badge">解释方差 {data.mode.explainedVariancePercent.toFixed(1)}%</span>}
        </div>
        <ModeMap mode={data.mode} />
      </div>
    </>
  );
}
