import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const COLORS = {
  real: '#1a478a',
  pred: '#e63946',
};

function LegendLine({ dashed = false }) {
  return (
    <svg width="24" height="3" style={{ verticalAlign: 'middle', marginRight: 6 }}>
      <line
        x1="0"
        y1="2"
        x2="24"
        y2="2"
        stroke={dashed ? COLORS.pred : COLORS.real}
        strokeWidth="2"
        strokeDasharray={dashed ? '4 2' : undefined}
      />
    </svg>
  );
}

function CustomLegend() {
  return (
    <div className="custom-legend" style={{ textAlign: 'center', marginTop: 8, fontSize: 12 }}>
      <span style={{ display: 'inline-flex', alignItems: 'center', marginRight: 16 }}>
        <LegendLine />
        真实值
      </span>
      <span style={{ display: 'inline-flex', alignItems: 'center' }}>
        <LegendLine dashed />
        预测值
      </span>
    </div>
  );
}

export default function TimeSeriesChart({ data, gridLabel }) {
  const { real, pred } = data;
  const chartData = real.map((v, i) => ({
    day: i + 1,
    real: v,
    pred: pred[i],
  }));

  // 优先用后端返回的 r（与热力图同格点同源，口径由 real_data.DISPLAY_MODE 决定；
  // 此处**不做任何口径假设**），无 r 字段（mock 降级）时回退本地重算
  const r = data.r !== undefined ? data.r : calcR2(real, pred);

  return (
    <div className="result-card">
      <h3 className="result-title">
        降水时序对比
        <span className="region-badge">{gridLabel}</span>
      </h3>
      <p className="metric-note">Pearson r = {r.toFixed(3)}</p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" label={{ value: '天 (6-9月)', position: 'insideBottom', offset: -5 }} fontSize={11} />
          <YAxis label={{ value: '降水距平 (mm/天)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }} fontSize={11} />
          <Tooltip />
          <Legend content={<CustomLegend />} />
          <Line type="monotone" dataKey="real" stroke={COLORS.real} name="真实值" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="pred" stroke={COLORS.pred} name="预测值" dot={false} strokeWidth={2} strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function calcR2(real, pred) {
  if (!real.length) return 0;
  const meanR = real.reduce((a, b) => a + b, 0) / real.length;
  const meanP = pred.reduce((a, b) => a + b, 0) / pred.length;
  const num = real.reduce((s, v, i) => s + (v - meanR) * (pred[i] - meanP), 0);
  const den = Math.sqrt(real.reduce((s, v) => s + (v - meanR) ** 2, 0) *
    pred.reduce((s, v) => s + (v - meanP) ** 2, 0));
  return den > 0 ? num / den : 0;
}
