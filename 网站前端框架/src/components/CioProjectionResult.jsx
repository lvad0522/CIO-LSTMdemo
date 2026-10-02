import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { getHeatColor, HEAT_COLOR_NODES } from '../utils/heatColor';

const API_BASE = import.meta.env.VITE_API_BASE || '';

const INPUT_LABELS = {
  'cio-npy': '上传的 CIO 序列 (.npy)',
  'raw-zip': '原始 U850 气象场 (.zip)',
};

const NORMALIZATION_LABELS = {
  'training-equivalent': '训练同口径（前 2240 点统一确定 min-max 范围）',
  partial: '按现有序列整段 min-max（短序列）',
  'extended-base': '扩大 min-max 基准以覆盖目标年（有口径偏差）',
};

const DOMAIN_LON_TICKS = [40, 60, 80, 100, 120];
const DOMAIN_LAT_TICKS = [-20, -10, 0, 10, 20];
const COASTLINE_URL = '/geo/cio_domain_coastline.json';
const DIVERGING_GRADIENT = `linear-gradient(90deg, ${HEAT_COLOR_NODES
  .map(([position, color]) => `${color} ${position * 100}%`)
  .join(', ')})`;
const SEQUENTIAL_COLOR_NODES = [
  [0, '#f7fbff'],
  [0.16, '#deebf7'],
  [0.36, '#9ecae1'],
  [0.56, '#6baed6'],
  [0.74, '#3182bd'],
  [0.88, '#08519c'],
  [1, '#08306b'],
];
const SEQUENTIAL_GRADIENT = 'linear-gradient(90deg, #f7fbff 0%, #deebf7 16%, #9ecae1 36%, #6baed6 56%, #3182bd 74%, #08519c 88%, #08306b 100%)';

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

function toFiniteNumber(value) {
  if (value == null || value === '') return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function gridShape(data) {
  if (!Array.isArray(data) || data.length === 0 || !Array.isArray(data[0]) || data[0].length === 0) {
    return null;
  }
  const cols = data[0].length;
  if (!data.every(row => Array.isArray(row) && row.length === cols)) return null;
  return { rows: data.length, cols };
}

function coordinatePair(value) {
  return Array.isArray(value)
    && toFiniteNumber(value[0]) != null
    && toFiniteNumber(value[1]) != null;
}

function cleanLine(line) {
  if (!Array.isArray(line)) return null;
  const points = line
    .filter(coordinatePair)
    .map(point => [Number(point[0]), Number(point[1])]);
  return points.length > 1 ? points : null;
}

function geometryLines(geometry) {
  if (!geometry || typeof geometry !== 'object') return [];
  if (geometry.type === 'LineString') return [geometry.coordinates];
  if (geometry.type === 'MultiLineString' || geometry.type === 'Polygon') return geometry.coordinates || [];
  if (geometry.type === 'MultiPolygon') return (geometry.coordinates || []).flat();
  if (geometry.type === 'GeometryCollection') {
    return (geometry.geometries || []).flatMap(geometryLines);
  }
  return [];
}

function lineCollection(value) {
  if (!value) return [];
  if (Array.isArray(value)) {
    if (coordinatePair(value[0])) return [value];
    return value.flatMap(lineCollection);
  }
  if (value.type === 'Feature') return geometryLines(value.geometry);
  if (value.type === 'FeatureCollection') return (value.features || []).flatMap(lineCollection);
  if (value.type) return geometryLines(value);
  return [];
}

function normalizeCoastlines(payload) {
  const normalize = value => lineCollection(value).map(cleanLine).filter(Boolean);
  return {
    coast: normalize(payload?.coast),
    borders: normalize(payload?.borders),
  };
}

function useCioDomainCoastlines() {
  const [coastlines, setCoastlines] = useState(null);

  useEffect(() => {
    let alive = true;
    const controller = new AbortController();
    fetch(COASTLINE_URL, { signal: controller.signal })
      .then(async response => (response.ok ? response.json() : null))
      .then(payload => {
        if (alive) setCoastlines(normalizeCoastlines(payload));
      })
      .catch(reason => {
        if (reason.name !== 'AbortError' && alive) {
          // 海岸线只是地图参考层；静态资源缺失不能阻止数值诊断展示。
          setCoastlines({ coast: [], borders: [] });
        }
      });
    return () => {
      alive = false;
      controller.abort();
    };
  }, []);

  return coastlines;
}

function axisRange(axis, fallbackStart, fallbackEnd) {
  const values = Array.isArray(axis)
    ? axis.map(toFiniteNumber).filter(value => value != null)
    : [];
  const min = values.length ? Math.min(...values) : fallbackStart;
  const max = values.length ? Math.max(...values) : fallbackEnd;
  return {
    min,
    max: max > min ? max : fallbackEnd,
    first: values[0] ?? fallbackStart,
    last: values.at(-1) ?? fallbackEnd,
  };
}

function mapAxes(lat, lon) {
  return {
    lon: axisRange(lon, 40, 120),
    lat: axisRange(lat, -20, 20),
  };
}

function mapScale(data, requestedScale, colorMode) {
  const requested = toFiniteNumber(requestedScale);
  if (requested != null && requested > 0) return requested;
  let maximum = 0;
  data.forEach(row => row.forEach(value => {
    const numeric = toFiniteNumber(value);
    if (numeric != null) {
      maximum = Math.max(maximum, colorMode === 'diverging' ? Math.abs(numeric) : numeric);
    }
  }));
  return Math.max(maximum, 1e-12);
}

function hexToRgb(hex) {
  const integer = Number.parseInt(hex.slice(1), 16);
  return [(integer >> 16) & 255, (integer >> 8) & 255, integer & 255];
}

function interpolatedColor(position, nodes) {
  const normalized = Math.max(0, Math.min(1, position));
  for (let index = 0; index < nodes.length - 1; index += 1) {
    const [start, startColor] = nodes[index];
    const [end, endColor] = nodes[index + 1];
    if (normalized <= end) {
      const ratio = (normalized - start) / (end - start);
      const from = hexToRgb(startColor);
      const to = hexToRgb(endColor);
      const rgb = from.map((channel, channelIndex) => (
        Math.round(channel + (to[channelIndex] - channel) * ratio)
      ));
      return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    }
  }
  return nodes.at(-1)[1];
}

function sequentialColor(value, scale) {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return '#f0f0f0';
  return interpolatedColor(numeric / scale, SEQUENTIAL_COLOR_NODES);
}

function formatCoordinate(value, hemisphere) {
  const absolute = Math.abs(value);
  if (hemisphere === 'lat') {
    if (value === 0) return '0°';
    return `${absolute}°${value > 0 ? 'N' : 'S'}`;
  }
  return `${absolute}°${value >= 0 ? 'E' : 'W'}`;
}

function formatMapNumber(value, signed = false) {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return '—';
  const prefix = signed && numeric > 0 ? '+' : '';
  const absolute = Math.abs(numeric);
  if (absolute > 0 && (absolute >= 1e4 || absolute < 1e-3)) {
    return `${prefix}${numeric.toExponential(2)}`;
  }
  return `${prefix}${numeric.toPrecision(3)}`;
}

function visibleTicks(ticks, range) {
  const span = range.max - range.min;
  if (!(span > 0)) return [];
  return ticks
    .filter(tick => tick >= range.min - 1e-6 && tick <= range.max + 1e-6)
    .map(tick => ({
      value: tick,
      percent: (tick - range.min) / span * 100,
    }));
}

function CoastlineOverlay({ coastlines, axes, rows, cols }) {
  const coast = coastlines?.coast || [];
  const borders = coastlines?.borders || [];
  if (!coast.length && !borders.length) return null;
  const lonSpan = axes.lon.max - axes.lon.min;
  const latSpan = axes.lat.max - axes.lat.min;
  if (!(lonSpan > 0) || !(latSpan > 0)) return null;

  const pathForLine = line => line.map(([longitude, latitude], index) => {
    const x = 0.5 + (longitude - axes.lon.min) / lonSpan * Math.max(cols - 1, 1);
    const y = 0.5 + (axes.lat.max - latitude) / latSpan * Math.max(rows - 1, 1);
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(3)},${y.toFixed(3)}`;
  }).join(' ');

  return (
    <svg
      className="coordinate-map-coastline"
      viewBox={`0 0 ${cols} ${rows}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      focusable="false"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      <g className="coordinate-map-coast">
        {coast.map((line, index) => (
          <path key={`coast-${index}`} d={pathForLine(line)} fill="none" stroke="#304a53" strokeWidth="0.52" vectorEffect="non-scaling-stroke" />
        ))}
      </g>
      <g className="coordinate-map-borders">
        {borders.map((line, index) => (
          <path key={`border-${index}`} d={pathForLine(line)} fill="none" stroke="#597078" strokeWidth="0.34" strokeDasharray="1.15 0.85" vectorEffect="non-scaling-stroke" />
        ))}
      </g>
    </svg>
  );
}

function CoordinateMap({
  data,
  lat,
  lon,
  scaleMax,
  colorMode = 'diverging',
  coastlines,
  title,
  date,
  unit,
  className = '',
}) {
  const canvasRef = useRef(null);
  const shape = gridShape(data);
  const rows = shape?.rows;
  const cols = shape?.cols;
  const axes = mapAxes(lat, lon);
  const scale = shape ? mapScale(data, scaleMax, colorMode) : 1;
  const lonTicks = visibleTicks(DOMAIN_LON_TICKS, axes.lon);
  const latTicks = visibleTicks(DOMAIN_LAT_TICKS, axes.lat);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !rows || !cols) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    canvas.width = cols;
    canvas.height = rows;
    const latAscending = axes.lat.first <= axes.lat.last;
    const lonAscending = axes.lon.first <= axes.lon.last;
    data.forEach((row, sourceRow) => {
      row.forEach((value, sourceCol) => {
        const targetRow = latAscending ? rows - sourceRow - 1 : sourceRow;
        const targetCol = lonAscending ? sourceCol : cols - sourceCol - 1;
        context.fillStyle = colorMode === 'sequential'
          ? sequentialColor(value, scale)
          : getHeatColor(value, -scale, scale);
        context.fillRect(targetCol, targetRow, 1, 1);
      });
    });
  }, [axes.lat.first, axes.lat.last, axes.lon.first, axes.lon.last, colorMode, cols, data, rows, scale]);

  if (!shape) {
    return <EmptyPanel label="地图数据不可用" reason="未收到完整的纬度 × 经度网格" />;
  }

  const legendStart = colorMode === 'sequential' ? '0' : formatMapNumber(-scale);
  const legendEnd = formatMapNumber(scale);
  return (
    <figure className={`coordinate-map ${className}`.trim()}>
      <figcaption className="coordinate-map-meta">
        <span className="coordinate-map-name">{title}</span>
        {date && <span>日期：{date}</span>}
        {unit && <span>单位：{unit}</span>}
      </figcaption>
      <div className="coordinate-map-body">
        <div className="coordinate-map-lat-axis" aria-hidden="true">
          {latTicks.map(tick => (
            <span
              className={`coordinate-map-lat-tick${tick.percent >= 99.999 ? ' coordinate-map-lat-tick--top' : ''}${tick.percent <= 0.001 ? ' coordinate-map-lat-tick--bottom' : ''}`}
              key={tick.value}
              style={{ top: `${100 - tick.percent}%` }}
            >
              {formatCoordinate(tick.value, 'lat')}
            </span>
          ))}
        </div>
        <div className="coordinate-map-plot" style={{ position: 'relative', overflow: 'hidden' }}>
          <canvas
            ref={canvasRef}
            className="coordinate-map-canvas projection-heatmap"
            width={shape.cols}
            height={shape.rows}
            role="img"
            aria-label={`${title}；经度 ${formatCoordinate(axes.lon.min, 'lon')} 至 ${formatCoordinate(axes.lon.max, 'lon')}，纬度 ${formatCoordinate(axes.lat.min, 'lat')} 至 ${formatCoordinate(axes.lat.max, 'lat')}`}
          />
          <div className="coordinate-map-gridlines" aria-hidden="true" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
            {lonTicks.map(tick => <span className="coordinate-map-gridline coordinate-map-gridline-lon" key={`lon-${tick.value}`} style={{ left: `${tick.percent}%` }} />)}
            {latTicks.map(tick => <span className="coordinate-map-gridline coordinate-map-gridline-lat" key={`lat-${tick.value}`} style={{ top: `${100 - tick.percent}%` }} />)}
          </div>
          <CoastlineOverlay coastlines={coastlines} axes={axes} rows={shape.rows} cols={shape.cols} />
        </div>
      </div>
      <div className="coordinate-map-lon-axis" aria-hidden="true">
        {lonTicks.map(tick => (
          <span className="coordinate-map-lon-tick" key={tick.value} style={{ left: `${tick.percent}%` }}>
            {formatCoordinate(tick.value, 'lon')}
          </span>
        ))}
      </div>
      <div className="coordinate-map-colorbar" aria-label={`${title} 色标`}>
        <span>{legendStart}</span>
        <div
          className={`coordinate-map-colorbar-gradient coordinate-map-colorbar-gradient--${colorMode}`}
          style={{ background: colorMode === 'sequential' ? SEQUENTIAL_GRADIENT : DIVERGING_GRADIENT }}
        >
          {colorMode === 'diverging' && <span className="coordinate-map-colorbar-zero">0</span>}
        </div>
        <span>{legendEnd}</span>
      </div>
    </figure>
  );
}

function ProjectionHeatmap({ stage, coastlines, date }) {
  if (!stage?.data?.length) {
    return <EmptyPanel label="正在读取投影过程" reason="等待阶段①诊断产物" />;
  }
  return (
    <CoordinateMap
      data={stage.data}
      lat={stage.lat}
      lon={stage.lon}
      scaleMax={stage.scaleMax}
      coastlines={coastlines}
      title={stage.title}
      date={date}
      unit={stage.unit}
      className={`projection-coordinate-map projection-coordinate-map--${stage.key || 'stage'}`}
    />
  );
}

function ModeMap({ mode, coastlines }) {
  if (!mode) return <EmptyPanel label="正在读取 U850 模态" reason="等待真实 MAT 资产响应" />;
  const maxAbs = Math.max(Math.abs(toFiniteNumber(mode.min) || 0), Math.abs(toFiniteNumber(mode.max) || 0), 1e-12);
  return (
    <CoordinateMap
      data={mode.data}
      lat={mode.lat}
      lon={mode.lon}
      scaleMax={mode.scaleMax || maxAbs}
      coastlines={coastlines}
      title="EOF1 U850 投影权重"
      date="固定 EOF1 空间模板"
      unit="标准化权重"
      className="projection-coordinate-map projection-eof-coordinate-map"
    />
  );
}

function AverageContributionMap({ windowContribution, coastlines }) {
  const averageAbs = windowContribution?.averageAbs;
  if (!Array.isArray(averageAbs) || averageAbs.length === 0) {
    return <EmptyPanel label="正在读取平均投影贡献" reason="等待 112 天窗口统计" />;
  }
  const dates = windowContribution.dates || [];
  const date = dates.length > 1
    ? `${dates[0]} — ${dates[dates.length - 1]}（${dates.length} 天平均）`
    : '112 天窗口平均';
  return (
    <CoordinateMap
      data={averageAbs}
      lat={windowContribution.lat}
      lon={windowContribution.lon}
      scaleMax={windowContribution.scaleMax}
      colorMode="sequential"
      coastlines={coastlines}
      title="平均绝对投影贡献 |processed × EOF1|"
      date={date}
      unit="平均 |CIO 贡献|"
      className="projection-coordinate-map projection-average-coordinate-map"
    />
  );
}

function ContributionSummary({ preview }) {
  const selected = preview?.selectedContribution || {};
  const positive = toFiniteNumber(selected.positiveSum);
  const negative = toFiniteNumber(selected.negativeSum);
  const net = toFiniteNumber(selected.net) ?? toFiniteNumber(preview?.contributionSum);
  const projected = toFiniteNumber(selected.projectedCio) ?? toFiniteNumber(preview?.projectedCio);
  const closure = toFiniteNumber(selected.closureError) ?? (
    net != null && projected != null ? net - projected : null
  );
  const exceedsTolerance = closure != null && Math.abs(closure) > 2e-4;
  const stats = [
    ['正贡献总和', positive],
    ['负贡献总和', negative],
    ['净投影 CIO', net],
    ['投影序列 CIO', projected],
    ['闭合误差', closure],
  ];

  return (
    <section className="projection-contribution-summary" aria-label="2c 格点投影贡献统计">
      <h4>2c 格点贡献的数值闭合</h4>
      <dl className="projection-contribution-stats">
        {stats.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{formatMapNumber(value, label !== '闭合误差')}</dd>
          </div>
        ))}
      </dl>
      <p className={`projection-closure-note${exceedsTolerance ? ' is-warning' : ''}`} role={exceedsTolerance ? 'alert' : undefined}>
        {exceedsTolerance ? '数学闭合警告：' : '数学闭合检查：'}正贡献 + 负贡献 = 净贡献，且净贡献 − 投影 CIO = 闭合误差。
        {closure != null && (exceedsTolerance
          ? ` 当前误差 ${formatMapNumber(closure, true)}，超过 2×10⁻⁴ 阈值。`
          : ` 当前误差 ${formatMapNumber(closure, true)}，在 2×10⁻⁴ 阈值内。`)}
      </p>
    </section>
  );
}

function useProjectionPreview(jobId, enabled, timeIndex) {
  const requestKey = jobId && enabled ? `${jobId}:${timeIndex}` : null;
  const [result, setResult] = useState({ requestKey: null, preview: null, error: null });

  useEffect(() => {
    if (!requestKey) return undefined;
    const controller = new AbortController();
    fetch(`${API_BASE}/api/chain/jobs/${jobId}/projection-preview?time_index=${timeIndex}`, {
      signal: controller.signal,
    })
      .then(async response => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `投影过程读取失败 (${response.status})`);
        return payload;
      })
      .then(payload => { setResult({ requestKey, preview: payload, error: null }); })
      .catch(reason => {
        if (reason.name !== 'AbortError') {
          setResult({ requestKey, preview: null, error: reason.message || '投影过程读取失败' });
        }
      });
    return () => controller.abort();
  }, [jobId, requestKey, timeIndex]);

  return result.requestKey === requestKey
    ? { preview: result.preview, error: result.error }
    : { preview: null, error: null };
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
  const coastlines = useCioDomainCoastlines();
  const previewEnabled = job.projectionMode === 'u850_only'
    && (job.inputKind || job.projection?.inputKind) === 'raw-zip';
  const [previewIndex, setPreviewIndex] = useState(0);
  const { preview, error: previewError } = useProjectionPreview(
    job.jobId, previewEnabled, previewIndex,
  );
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

  const targetWindow = data.series?.targetWindow;
  const zoomData = targetWindow?.available
    ? chartData.slice(targetWindow.start, targetWindow.end)
    : [];
  const targetWindowDates = targetWindow?.dates;
  const targetWindowLabel = targetWindow?.available
    ? targetWindowDates?.length
      ? `（${targetWindowDates[0]} … ${targetWindowDates[targetWindowDates.length - 1]}）`
      : targetWindow.year != null
        ? `（${targetWindow.year} 年）`
        : ''
    : '';
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
  const adaptation = projection.adaptation;
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
    ['模型输入口径', NORMALIZATION_LABELS[normalization.mode] || normalization.mode],
    ['模型窗口', selectedRange.length === 2 ? `第 ${selectedRange[0] + 1}-${selectedRange[1]} 点` : null],
    ['日期索引', calendarKnown ? '真实日期' : '未知（使用样本序号）'],
    ['投影窗口', formatWindow(projection.window)],
    ['投影年份', years.length ? `${years[0]}-${years[years.length - 1]}（${years.length} 年）` : null],
    ['缺失月份', missingMonths.length ? `${missingMonths.length} 个` : (calendarKnown ? '无' : null)],
    ['数据适配', adaptation ? `${adaptation.files.length} 个 NetCDF · ${adaptation.sourceSamples} 个源样本 → ${adaptation.selectedSamples} 个窗口样本` : null],
    ['空间适配', adaptation ? '按经纬度插值至 1° 网格（41 × 81），850 hPa，风速统一为 m/s' : null],
    ['时间采样', adaptation?.sampling],
    ['未进入窗口的样本', adaptation ? `${adaptation.ignoredSamples} 个（窗口外日期或未选年份）` : null],
    ['缺失窗口日期', adaptation ? `${adaptation.missingDates.length} 个` : null],
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
        {adaptation && (
          <details className="metric-note">
            <summary>查看数据适配明细</summary>
            <p>{adaptation.preprocessing}</p>
            {adaptation.missingDates.length > 0 && <p>缺失日期：{adaptation.missingDates.join('、')}</p>}
            <ul>{adaptation.files.map(item => (
              <li key={item.file}>
                {item.file}：{item.variable} · {item.sourceDimensions.join(' × ')} · {item.sourceShape.join(' × ')}；
                {item.sourceUnit} → {item.targetUnit}（×{item.unitFactor}）；
                {item.dateStart} — {item.dateEnd}（{item.calendar}，{item.dateSource}）；
                选中 {item.selectedSamples}/{item.sourceSamples} 点
              </li>
            ))}</ul>
          </details>
        )}
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
            <h4>目标年 112 点窗口放大{targetWindowLabel}</h4>
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
            ) : targetWindow?.available === false ? (
              <EmptyPanel
                label="无法定位目标年窗口"
                reason={targetWindow.reason}
              />
            ) : (
              <EmptyPanel label="正在读取局部序列" reason="等待任务序列端点" />
            )}
          </section>

          <section className="cio-panel">
            <span className="cio-panel-tag">(c)</span>
            <h4>CIO 不同时间尺度的波动强度{data.spectrum?.available ? `（${data.spectrum.blockCount} 个窗口平均）` : ''}</h4>
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
                ? `${data.spectrum.blockCount} 个 ${data.spectrum.blockSize} 天窗口分别估计后平均；虚线为 ${(Number(confidence) * 100).toFixed(0)}% 分块 bootstrap 上包络。它只回答“哪些时间尺度的 CIO 波动更强”，不代表降水预测准确率。`
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

      {previewEnabled && (
        <div className="result-card projection-preview-card">
          <div className="result-header">
            <div>
              <h3 className="result-title" style={{ margin: 0 }}>U850 如何投影成一个 CIO 数值</h3>
              <p className="metric-note" style={{ margin: '6px 0 0' }}>
                同一天依次查看输入风场、训练口径预处理，以及每个格点对最终 CIO 的贡献。
              </p>
            </div>
            {preview && <span className="region-badge">{preview.date}</span>}
          </div>
          {preview && (
            <>
              <div className="projection-slider-row">
                <label htmlFor="projection-time-index">查看目标年中的第几天</label>
                <input
                  id="projection-time-index"
                  type="range"
                  min="0"
                  max={Math.max(0, preview.totalSteps - 1)}
                  value={Math.min(previewIndex, Math.max(0, preview.totalSteps - 1))}
                  onChange={event => setPreviewIndex(Number(event.target.value))}
                />
                <output>第 {preview.timeIndex + 1} / {preview.totalSteps} 天</output>
              </div>
              <div className="projection-stage-grid">
                {preview.stages.map(stage => (
                  <section className="projection-stage" key={stage.key}>
                    <h4>{stage.title}</h4>
                    <p>{stage.description}</p>
                    <ProjectionHeatmap stage={stage} coastlines={coastlines} date={preview.date} />
                  </section>
                ))}
              </div>
              <ContributionSummary preview={preview} />
              <p className="chart-note">
                贡献图不是新的气象变量：它把“预处理后的每个格点 × 第一模态权重”画出来，所有格点相加才得到右侧这个 CIO 数值。
              </p>
            </>
          )}
          {!preview && !previewError && (
            <EmptyPanel label="正在读取投影过程对照" reason="等待阶段①诊断产物" />
          )}
          {previewError && <EmptyPanel label="暂时没有投影过程对照" reason={previewError} />}
        </div>
      )}

      <div className="result-card projection-weight-card">
        <div className="result-header">
          <h3 className="result-title" style={{ margin: 0 }}>窗口贡献强度与 EOF1 U850 权重</h3>
          {data.mode && <span className="region-badge">解释方差 {data.mode.explainedVariancePercent.toFixed(1)}%</span>}
        </div>
        <p className="metric-note" style={{ margin: '0 0 10px' }}>
          左图汇总 112 点窗口中各格点的平均绝对 EOF 加权投影贡献；右图是计算 CIO 时使用的固定空间权重模板，二者分别表达贡献强度与投影权重。
        </p>
        <div className={`projection-weight-layout${previewEnabled ? '' : ' projection-weight-layout--eof-only'}`}>
          {previewEnabled && (
            <section className="projection-weight-main">
              <h4>112 点平均绝对 EOF 加权投影贡献</h4>
              {preview?.windowContribution ? (
                <AverageContributionMap windowContribution={preview.windowContribution} coastlines={coastlines} />
              ) : (
                <EmptyPanel
                  label={previewError ? '平均贡献图不可用' : '正在读取平均投影贡献'}
                  reason={previewError || '等待 112 天窗口统计'}
                />
              )}
            </section>
          )}
          <section className="projection-weight-aux">
            <h4>EOF1 U850 投影权重（固定空间权重模板，不是某一天的 U850 场）</h4>
            <ModeMap mode={data.mode} coastlines={coastlines} />
          </section>
        </div>
      </div>
    </>
  );
}
