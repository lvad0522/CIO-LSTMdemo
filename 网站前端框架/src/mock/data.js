// 模拟数据生成器 —— 用确定性随机生成稳定的mock数据

const S2S_MODELS = [
  "BOM", "CMA", "CNRM", "ECCC", "ECMWF",
  "HMCR", "ISAC", "JMA", "KMA", "NCEP", "UKMO"
];

// 简易伪随机（保证每次刷新数据一致）
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

// 生成CIO指数时序（模拟20-100天震荡）
export function genCIOTimeSeries(length = 2240) {
  const rng = seededRandom(42);
  const data = [];
  for (let t = 0; t < length; t++) {
    const trend = Math.sin(t / 300 * Math.PI) * 0.5;
    const wave1 = Math.sin(t / 20 * Math.PI * 2) * 0.3;
    const wave2 = Math.sin(t / 7 * Math.PI * 2) * 0.1;
    const noise = (rng() - 0.5) * 0.2;
    data.push(trend + wave1 + wave2 + noise);
  }
  return data;
}

// 生成CIO-降水相关空间分布
export function genCIOCorrelationMap() {
  const rows = 40, cols = 80;
  const rng = seededRandom(123);
  const data = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      const latFactor = Math.sin((i / rows) * Math.PI);
      const lonFactor = Math.sin((j / cols) * Math.PI * 0.8 + 0.5);
      let v = latFactor * lonFactor * 0.7 + (rng() - 0.5) * 0.2;
      row.push(Math.max(-1, Math.min(1, v)));
    }
    data.push(row);
  }
  return { data, rows, cols };
}

// 生成预测技能空间分布
export function genSkillMap(region = "india") {
  const rows = region === "india" ? 50 : 25;
  const cols = region === "india" ? 80 : 60;
  const rng = seededRandom(456);
  const data = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      const latCenter = Math.sin((i / rows) * Math.PI);
      const lonCenter = Math.sin((j / cols) * Math.PI * 0.7 + 0.3);
      let v = 0.15 + latCenter * lonCenter * 0.55 + (rng() - 0.5) * 0.12;
      row.push(Math.max(0, Math.min(0.85, v)));
    }
    data.push(row);
  }
  return { data, rows, cols };
}

// 生成某格点的预测vs真实时序
export function genTimeSeries(region = "india", gridI = 25, gridJ = 40) {
  const rng = seededRandom(gridI * 1000 + gridJ);
  const length = 112;
  const real = [];
  const pred = [];
  for (let t = 0; t < length; t++) {
    const base = 3 + Math.sin(t / 15 * Math.PI * 2) * 2.5
      + Math.sin(t / 5 * Math.PI * 2) * 1.0
      + (rng() - 0.5) * 0.5;
    real.push(Math.max(0, base));
    pred.push(Math.max(0, base + (rng() - 0.5) * 1.2));
  }
  return { real, pred };
}

// S2S模型对比数据
export function genS2SComparison() {
  const rng = seededRandom(789);
  const lstmR = 0.62;
  return S2S_MODELS.map((name, i) => ({
    name,
    r: 0.12 + rng() * 0.24,
  })).concat([{ name: "LSTM+CIO", r: lstmR }]);
}

// 6次实验结果表格
export function genResultsTable() {
  const rng = seededRandom(101112);
  return Array.from({ length: 6 }, (_, i) => ({
    experiment: i + 1,
    pearsonR: +(0.45 + rng() * 0.30).toFixed(3),
    rmse: +(2.0 + rng() * 1.5).toFixed(2),
    mae: +(1.5 + rng() * 1.0).toFixed(2),
  }));
}

export { S2S_MODELS };
