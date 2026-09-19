import { Fragment, useEffect, useRef, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const CELL_SCALE = 6;

const KIND_LABELS = { u850: 'U850 距平场', sst: 'SST', both: 'SST + U850' };
const KIND_SOURCE_LABELS = { 'explicit-param': '界面指定', filename: '按文件名判定' };
const NORM_LABELS = {
  'training-equivalent': '训练同口径（切前 2240 点，再整段 min-max）',
  partial: '整段自归一化（短序列，见警告）',
};

function precipColor(value, maxAbs) {
  const t = Math.max(-1, Math.min(1, value / Math.max(maxAbs, 1e-6)));
  if (t >= 0) {
    const fade = Math.round(255 * (1 - t));
    return `rgb(220, ${fade}, ${fade})`;
  }
  const fade = Math.round(255 * (1 + t));
  return `rgb(${fade}, ${fade}, 220)`;
}

// r 图：以 0 为中心的蓝-白-红发散色标（与 /api/run 的 corrColor 同一观感）
function rColor(value) {
  const t = Math.max(-1, Math.min(1, value));
  const fade = Math.round(255 * (1 - Math.abs(t)));
  return t >= 0 ? `rgb(255, ${fade}, ${fade})` : `rgb(${fade}, ${fade}, 255)`;
}

const VIEW_LABELS = {
  prediction: '预测场',
  truth: '实况场',
};

export default function LivePredictionResult({ job, confidence = '0.95' }) {
  const [timeIndex, setTimeIndex] = useState(0);
  const [frame, setFrame] = useState(null);
  const [view, setView] = useState('prediction');
  const [selectedGrid, setSelectedGrid] = useState({ i: 40, j: 50 });
  const [gridSeries, setGridSeries] = useState(null);
  const [pearson, setPearson] = useState(null);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);
  const rCanvasRef = useRef(null);

  const verification = job.result?.verification;
  const pearsonAvailable = Boolean(verification?.available);

  // 预测场 / 实况场共用同一套画布与坐标，只是数据源不同
  useEffect(() => {
    const controller = new AbortController();
    const path = view === 'truth'
      ? `${API_BASE}/api/predict/jobs/${job.jobId}/truth/preview`
      : `${API_BASE}/api/predict/jobs/${job.jobId}/preview`;
    fetch(`${path}?time_index=${timeIndex}`, { signal: controller.signal })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `场读取失败 (${res.status})`);
        return data;
      })
      .then(setFrame)
      .catch((err) => {
        if (err.name !== 'AbortError') setError(err.message);
      });
    return () => controller.abort();
  }, [job.jobId, timeIndex, view]);

  // r 图与置信水平无关，只有阈值随 confidence 变 —— 每次改档位向后端要一次摘要
  useEffect(() => {
    if (!pearsonAvailable) return undefined;
    const controller = new AbortController();
    fetch(`${API_BASE}/api/predict/jobs/${job.jobId}/pearson?confidence=${confidence}`, {
      signal: controller.signal,
    })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `相关系数读取失败 (${res.status})`);
        return data;
      })
      .then(setPearson)
      .catch((err) => {
        if (err.name !== 'AbortError') setError(err.message);
      });
    return () => controller.abort();
  }, [job.jobId, confidence, pearsonAvailable]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/api/predict/jobs/${job.jobId}/grid?i=${selectedGrid.i}&j=${selectedGrid.j}`, {
      signal: controller.signal,
    })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `格点序列读取失败 (${res.status})`);
        return data;
      })
      .then(setGridSeries)
      .catch((err) => {
        if (err.name !== 'AbortError') setError(err.message);
      });
    return () => controller.abort();
  }, [job.jobId, selectedGrid]);

  useEffect(() => {
    if (!frame || !canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = frame.cols * CELL_SCALE;
    canvas.height = frame.rows * CELL_SCALE;
    const ctx = canvas.getContext('2d');
    const maxAbs = Math.max(Math.abs(frame.min), Math.abs(frame.max), 1e-6);
    frame.data.forEach((row, i) => {
      row.forEach((value, j) => {
        ctx.fillStyle = precipColor(value, maxAbs);
        ctx.fillRect(j * CELL_SCALE, i * CELL_SCALE, CELL_SCALE, CELL_SCALE);
      });
    });
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 1;
    ctx.strokeRect(
      selectedGrid.j * CELL_SCALE + 0.5,
      selectedGrid.i * CELL_SCALE + 0.5,
      CELL_SCALE - 1,
      CELL_SCALE - 1,
    );
  }, [frame, selectedGrid]);

  // r 图：通过显著性的格点上色，没通过的盖一层灰 —— 显著性直接画在图上
  useEffect(() => {
    if (!pearson || !rCanvasRef.current) return;
    const canvas = rCanvasRef.current;
    canvas.width = pearson.cols * CELL_SCALE;
    canvas.height = pearson.rows * CELL_SCALE;
    const ctx = canvas.getContext('2d');
    const threshold = pearson.criticalR;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    pearson.data.forEach((row, i) => {
      row.forEach((value, j) => {
        const x = j * CELL_SCALE;
        const y = i * CELL_SCALE;
        if (value === null || !Number.isFinite(value)) {
          ctx.fillStyle = '#2b2b2b';            // 无定义（时间维恒定）
        } else {
          ctx.fillStyle = rColor(value);
          ctx.fillRect(x, y, CELL_SCALE, CELL_SCALE);
          if (Math.abs(value) < threshold) {
            ctx.fillStyle = 'rgba(240, 240, 240, 0.82)';   // 未过显著性的淡化
          } else {
            return;
          }
        }
        ctx.fillRect(x, y, CELL_SCALE, CELL_SCALE);
      });
    });
  }, [pearson]);

  const handleCanvasClick = (event) => {
    if (!frame) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const j = Math.min(frame.cols - 1, Math.floor((event.clientX - rect.left) / rect.width * frame.cols));
    const i = Math.min(frame.rows - 1, Math.floor((event.clientY - rect.top) / rect.height * frame.rows));
    setSelectedGrid({ i, j });
  };

  const chartData = (gridSeries?.pred || []).map((value, index) => ({
    day: index + 1,
    prediction: value,
  }));
  const result = job.result || {};

  const projection = job.projection || {};
  const normalization = job.normalization || {};
  const years = projection.years || [];
  const perYear = Object.values(projection.daysPerYear || {});
  const uniformDays = perYear.length && perYear.every(v => v === perYear[0]) ? perYear[0] : null;
  const windowText = projection.window
    ? `${projection.window[0][0]}月${projection.window[0][1]}日 – ${projection.window[1][0]}月${projection.window[1][1]}日`
    : null;
  const missing = projection.missingMonths || [];
  const missingText = missing.length
    ? `${[...new Set(missing.map(m => `${m[1]} 月`))].join('、')}，共 ${missing.length} 个文件缺`
    : (projection.seriesLength ? '无' : null);

  // 只列能确定的行；.npy 入口没有投影信息，这些行自然消失
  const metaRows = [
    ['输入类别', KIND_LABELS[projection.kind]
      ? `${KIND_LABELS[projection.kind]}（${KIND_SOURCE_LABELS[projection.kindSource] || '—'}）`
      : null],
    ['投影窗口', windowText],
    ['投影序列', projection.seriesLength
      ? `${projection.seriesLength} 点 · ${years.length} 年`
        + (uniformDays ? ` × ${uniformDays} 天` : '')
      : null],
    ['缺月份', missingText],
    ['归一化', NORM_LABELS[normalization.mode] || normalization.mode || null],
    ['模态资产', projection.matPath
      ? `${String(projection.matPath).replace(/^.*[\\/]/, '')}（${projection.matSource === 'real' ? '服务器真件' : '本地夹具件'}）`
      : null],
  ].filter(([, value]) => value);

  return (
    <>
      {error && <div className="error-banner">⚠️ {error}</div>}
      <div className="result-row">
        <div className="result-col">
          <div className="result-card">
            <div className="result-header">
              <h3 className="result-title" style={{ margin: 0 }}>
                {view === 'truth' ? '实况降水距平场' : '真实模型预测降水场'}
              </h3>
              <div className="view-toggle">
                {['prediction', 'truth'].map(kind => (
                  <button
                    key={kind}
                    type="button"
                    className={`view-toggle-btn ${view === kind ? 'is-active' : ''}`}
                    onClick={() => setView(kind)}
                    disabled={kind === 'truth' && !pearsonAvailable}
                    title={kind === 'truth' && !pearsonAvailable
                      ? '本次任务没有匹配到实况场'
                      : undefined}
                  >
                    {VIEW_LABELS[kind]}
                  </button>
                ))}
              </div>
            </div>
            <p className="metric-note">
              时间步 {timeIndex + 1}/112 · 当前场范围 {frame ? `${frame.min.toFixed(3)} ～ ${frame.max.toFixed(3)}` : '加载中'}
              {view === 'truth' && pearson?.truthName ? ` · ${pearson.truthName}` : ''}
            </p>
            {view === 'truth' ? (
              <p className="chart-note">
                实况是 TRMM 观测，预测是模型输出，<strong>两者色标各自按当前帧的极值归一化</strong>，
                所以并排看时颜色深浅不能直接对比，只能比空间形态；定量比较看下方的相关系数图。
              </p>
            ) : null}
            <div className="prediction-canvas-wrap">
              <canvas
                ref={canvasRef}
                className="prediction-canvas"
                onClick={handleCanvasClick}
                title="点击格点查看预测时序"
              />
            </div>
            <div className="prediction-legend">
              <span>负距平</span><div className="prediction-gradient" /><span>正距平</span>
            </div>
            <label className="prediction-slider-label">
              预测时间步
              <input
                type="range"
                min="0"
                max="111"
                value={timeIndex}
                onChange={(event) => setTimeIndex(Number(event.target.value))}
              />
            </label>
          </div>
        </div>

        <div className="result-col">
          <div className="result-card">
            <h3 className="result-title">
              格点预测时序
              <span className="region-badge">({selectedGrid.j}, {selectedGrid.i})</span>
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" label={{ value: '时间步', position: 'insideBottom', offset: -5 }} fontSize={11} />
                <YAxis label={{ value: '预测降水距平', angle: -90, position: 'insideLeft' }} fontSize={11} />
                <Tooltip formatter={(value) => Number(value).toFixed(4)} />
                <Line type="monotone" dataKey="prediction" name="预测值" stroke="#e63946" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="result-card">
            <h3 className="result-title">模型运行摘要</h3>
            <div className="prediction-summary">
              {metaRows.map(([label, value]) => (
                <Fragment key={label}>
                  <span>{label}</span><strong>{value}</strong>
                </Fragment>
              ))}
              <span>输出形状</span><strong>{result.shape?.join(' × ')}</strong>
              <span>全局最小值</span><strong>{Number(result.min).toFixed(4)}</strong>
              <span>全局最大值</span><strong>{Number(result.max).toFixed(4)}</strong>
              <span>运行耗时</span><strong>{job.durationSeconds} 秒</strong>
            </div>
            {projection.matSource === 'mock' && (
              <p className="smoke-warning">
                ⚠️ 模态资产是本地夹具件（03_夹具/make_mock.py 造的假数据），
                结果只反映管道是否通畅，<strong>不是真实气象预测</strong>。
                正式使用请放入服务器真件 CIOmode_1982_2017.mat。
              </p>
            )}
            {job.warning && <p className="smoke-warning">⚠️ {job.warning}</p>}
            <p className="chart-note">
              {projection.inputKind === 'raw-zip'
                ? '链路：原始 U850 距平场 → 沿纬度坏高通等 6 步投影 → 训练同口径归一化 → 8181 个 checkpoint 前向推理 → prediction.npy。'
                : '链路：上传的 projected CIO 序列 → 训练同口径归一化 → 8181 个 checkpoint 前向推理 → prediction.npy。'}
            </p>
            <a
              className="download-prediction-btn"
              href={`${API_BASE}${result.downloadUrl}`}
              download
            >
              下载 prediction_pre1_2000.npy
            </a>
          </div>
        </div>
      </div>

      {verification && !pearsonAvailable && (
        <div className="result-row">
          <div className="result-col">
            <div className="result-card">
              <h3 className="result-title">预测技巧检验</h3>
              <p className="chart-note">本次未做相关系数检验：{verification.reason}</p>
            </div>
          </div>
        </div>
      )}

      {pearsonAvailable && (
        <div className="result-row">
          <div className="result-col">
            <div className="result-card">
              <div className="result-header">
                <h3 className="result-title" style={{ margin: 0 }}>
                  逐格点相关系数（预测 vs 实况）
                </h3>
                <span className="region-badge">n = {pearson?.n ?? 112}</span>
              </div>
              <p className="metric-note">
                {pearson
                  ? `每个格点沿 112 天算一个 r · 通过 ${(pearson.confidence * 100).toFixed(0)}% 显著性需 |r| ≥ ${pearson.criticalR}`
                  : '正在计算相关系数'}
              </p>
              <div className="prediction-canvas-wrap">
                <canvas
                  ref={rCanvasRef}
                  className="prediction-canvas"
                  title="灰色 = 未通过显著性检验；深灰 = 该格点预测恒定、r 无定义"
                />
              </div>
              <div className="prediction-legend">
                <span>r = -1</span>
                <div className="prediction-gradient" />
                <span>r = +1</span>
                <span className="legend-swatch legend-swatch-muted" />未过显著性
                <span className="legend-swatch legend-swatch-void" />无定义
              </div>
            </div>
          </div>

          <div className="result-col">
            <div className="result-card">
              <h3 className="result-title">技巧摘要</h3>
              {pearson ? (
                <div className="prediction-summary">
                  <span>平均相关系数</span><strong>{pearson.meanR.toFixed(4)}</strong>
                  <span>中位数</span><strong>{pearson.medianR.toFixed(4)}</strong>
                  <span>范围</span><strong>{pearson.minR.toFixed(3)} ～ {pearson.maxR.toFixed(3)}</strong>
                  <span>正相关格点</span><strong>{(pearson.positiveFraction * 100).toFixed(1)}%</strong>
                  <span>显著正相关（有技巧）</span>
                  <strong>{(pearson.significantPositiveFraction * 100).toFixed(1)}%</strong>
                  <span>显著负相关（反相关）</span>
                  <strong>{(pearson.significantNegativeFraction * 100).toFixed(1)}%</strong>
                  <span>r 无定义格点</span>
                  <strong>{pearson.undefinedPoints} / {pearson.totalPoints}</strong>
                  <span>实况场</span><strong>{pearson.truthName}</strong>
                </div>
              ) : (
                <p className="chart-note">正在读取摘要…</p>
              )}
              <p className="chart-note">
                降水距平场空间上很不均匀，展平成一个数会被大值区主导，所以按实验室口径取
                <strong>逐格点沿时间维</strong>的 r（`pearson2.npy` 第 3 通道就是这个量）。
                显著负相关表示该格点比气候态还差，不是技巧，故与显著正相关分开计。
              </p>
              {pearson?.downloadUrl && (
                <a
                  className="download-prediction-btn"
                  href={`${API_BASE}${pearson.downloadUrl}`}
                  download
                >
                  下载 pearson_r_pre1_2000.npy
                </a>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
