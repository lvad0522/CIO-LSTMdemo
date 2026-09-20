export const CORRELATION_MIN = -1;
export const CORRELATION_MAX = 1;

// 与现有预测技能热力图一致的九节点色板。
export const HEAT_COLOR_NODES = [
  [0.00, '#3A5FCD'],
  [0.25, '#8AA9DE'],
  [0.45, '#D8E2F0'],
  [0.50, '#F7F7F7'],
  [0.60, '#FDDBC7'],
  [0.70, '#F4A582'],
  [0.80, '#D6604D'],
  [0.90, '#B2182B'],
  [1.00, '#67001F'],
];

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function lerpColor(t, from, to) {
  const a = hexToRgb(from);
  const b = hexToRgb(to);
  const rgb = a.map((value, index) => (
    Math.round(value + (b[index] - value) * t)
  ));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

export function heatNormPosition(value, min = CORRELATION_MIN, max = CORRELATION_MAX) {
  if (!(min < 0 && max > 0)) {
    throw new Error('heat color domain must cross zero');
  }
  if (value <= 0) return 0.5 * (value - min) / (0 - min);
  return 0.5 + 0.5 * value / max;
}

export function getHeatColor(
  value,
  min = CORRELATION_MIN,
  max = CORRELATION_MAX,
) {
  if (!Number.isFinite(value)) return '#f0f0f0';
  const position = Math.max(0, Math.min(1, heatNormPosition(value, min, max)));
  for (let index = 0; index < HEAT_COLOR_NODES.length - 1; index += 1) {
    const [p0, c0] = HEAT_COLOR_NODES[index];
    const [p1, c1] = HEAT_COLOR_NODES[index + 1];
    if (position <= p1) {
      return lerpColor((position - p0) / (p1 - p0), c0, c1);
    }
  }
  return HEAT_COLOR_NODES[HEAT_COLOR_NODES.length - 1][1];
}
