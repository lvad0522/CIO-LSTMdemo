import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  Legend, ReferenceLine, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts';

// 导师 lead_corr_2003 配色（11 个 S2S 模式 + S2S Mean + LSTM）
const MODEL_COLORS = {
  BOM: '#808080', CMA: '#1a478a', CNRM: '#dc143c', ECCC: '#ffa500',
  ECMWF: '#000000', HMCR: '#800080', ISAC: '#1a478a', JMA: '#dc143c',
  KMA: '#008000', NCEP: '#32cd32', UKMO: '#ffd700',
  S2S_Mean: '#008000', LSTM: '#1a478a',
};
const MODEL_ORDER = [
  'BOM', 'CMA', 'CNRM', 'ECCC', 'ECMWF',
  'HMCR', 'ISAC', 'JMA', 'KMA', 'NCEP', 'UKMO',
];
// dash-dot（对应导师脚本的 '-.' 虚线）
const DASH_DOT = '5 3 1 3';

// 图例顺序：导师画线顺序 + S2S Mean + LSTM（4 列网格排布）
const LEGEND_ORDER = [
  ...MODEL_ORDER, 'S2S_Mean', 'LSTM',
];

export default function ModelCompareChart({ data }) {
  const isLineData = data && data.length > 0 && 'lead' in data[0];

  // ── 真实数据（折线图）：{lead, 11模式, S2S_Mean, LSTM} × 20 ──
  if (isLineData) {
    return (
      <div className="result-card">
        <h3 className="result-title">模型对比: LSTM vs S2S（2003 年, lead 1-20）</h3>
        <div style={{ position: 'relative' }}>
          {/* 图例：右上角 4 列网格，半透明白底避免遮挡曲线 */}
          <div style={{
            position: 'absolute', top: 4, right: 4, zIndex: 10,
            display: 'grid', gridTemplateColumns: 'repeat(4, auto)',
            columnGap: 10, rowGap: 2,
            background: 'rgba(255,255,255,0.88)',
            border: '1px solid #e0e0e0', borderRadius: 4,
            padding: '4px 8px', fontSize: 11,
            pointerEvents: 'none',
          }}>
            {LEGEND_ORDER.map((name) => (
              <span key={name} style={{ display: 'flex', alignItems: 'center', gap: 4, whiteSpace: 'nowrap' }}>
                <span style={{
                  display: 'inline-block', width: 16, height: 2,
                  background: name === 'S2S_Mean' || name === 'LSTM'
                    ? MODEL_COLORS[name]
                    : `repeating-linear-gradient(90deg, ${MODEL_COLORS[name]} 0 2px, transparent 2px 4px)`,
                }} />
                {name}
              </span>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={data} margin={{ top: 8, right: 12, bottom: 12, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="lead" type="number" domain={[1, 20]}
                     tickCount={20} fontSize={11} label={{ value: 'Lead Time', position: 'insideBottom', offset: -8, fontSize: 12 }} />
              <YAxis domain={[-0.4, 1]} fontSize={11} ticks={[-0.3, 0, 0.3, 0.5, 0.7, 1.0]}
                     label={{ value: 'Correlation', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip
                formatter={(v) => (v == null ? '—' : Number(v).toFixed(3))}
                labelFormatter={(label) => `leadtime: ${label}`}
                position={{ y: 80 }}
                offset={{ x: 0, y: 0 }}
                allowEscapeViewBox={{ x: false, y: true }}
              />
              <ReferenceLine y={0} stroke="#808080" strokeDasharray="4 4" />
              <ReferenceLine y={0.5} stroke="#808080" strokeDasharray="4 4" />
            {MODEL_ORDER.map((name) => (
              <Line key={name} type="monotone" dataKey={name}
                    stroke={MODEL_COLORS[name]} strokeWidth={1.2}
                    strokeDasharray={DASH_DOT} dot={false}
                    connectNulls={false} />
            ))}
            <Line type="monotone" dataKey="S2S_Mean" stroke="#008000"
                  strokeWidth={3} dot={false} />
            <Line type="monotone" dataKey="LSTM" stroke="#1a478a"
                  strokeWidth={4} dot={false} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
        </div>
        <p className="chart-note">
          数据来源: 真实 S2S 数据（corr-2003.nc / corr-avg-2003.nc）与 LSTM 蓝框区域平均重算
          {data[0]?.source === 'dataset' ? '（dataset）' : ''}
        </p>
      </div>
    );
  }

  // ── 降级数据（柱状图回退）：{name, r} × 12 ──
  return (
    <div className="result-card">
      <h3 className="result-title">模型对比: LSTM+CIO vs S2S</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ bottom: 50 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" angle={-35} textAnchor="end" fontSize={11} interval={0} />
          <YAxis domain={[0, 0.8]} label={{ value: 'Pearson r', angle: -90, position: 'insideLeft' }} />
          <Tooltip formatter={(v) => v.toFixed(3)} />
          <Bar dataKey="r" radius={[4, 4, 0, 0]} activeBar={false}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.name === 'LSTM+CIO' ? '#e63946' : '#1a478a'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="chart-note">数据来源: mock 降级（S2S 真实数据不可用）</p>
    </div>
  );
}
