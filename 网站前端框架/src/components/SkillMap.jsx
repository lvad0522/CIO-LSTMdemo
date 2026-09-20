import { useEffect, useMemo, useState } from 'react';
import { getHeatColor } from '../utils/heatColor';

// 剖面图已注释停用（2026-08-25：热力图已完整展示 r 分布，单行剖面价值有限）；
// 如需恢复，取消本行注释并还原下方 barData / selectedI / cross-section 区块
// import { ComposedChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine, Cell, ResponsiveContainer } from 'recharts';

// ── 前端平滑（2026-08-30：与 skill_map_nature.py 的 gaussian_filter(sigma=0.6) 口径对齐）──
// scipy 核半径 = int(truncate*sigma+0.5) = int(2.9) = 2 → 5×5 核，本实现同构；
// masked 卷积（窗口内按有效格点权重重归一化，无 NaN 晕带——与模板的 NaN 扩散为有意分歧，
// 前端更优）；内部格点（距边 >2 格）与 scipy 数值精确一致（对拍 max err 4.4e-16），
// 边界 2 圈为 OOB 重归一化的有意分歧（scipy 用 reflect 镜像填充）；
// 仅作用于展示层：上色/tooltip/0.5 等值线用平滑场，点击联动永远用原始格点。
const SMOOTH_SIGMA = 0.6; // 可调常量（与模板对齐，勿随意改）

function gaussianSmooth(grid) {
  const rows = grid.length, cols = grid[0].length;
  const r = 2; // 核半径（5×5）
  const s2 = 2 * SMOOTH_SIGMA * SMOOTH_SIGMA;
  const k1 = [];
  let kSum = 0;
  for (let x = -r; x <= r; x++) {
    const v = Math.exp(-(x * x) / s2);
    k1.push(v);
    kSum += v;
  }
  for (let i = 0; i < k1.length; i++) k1[i] /= kSum;

  const out = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      let sum = 0, wsum = 0;
      for (let di = -r; di <= r; di++) {
        const ni = i + di;
        if (ni < 0 || ni >= rows) continue;   // 窗口越界视作无效格点
        for (let dj = -r; dj <= r; dj++) {
          const nj = j + dj;
          if (nj < 0 || nj >= cols) continue;
          const v = grid[ni][nj];
          if (Number.isNaN(v)) continue;      // NaN 不参与加权
          const w = k1[di + r] * k1[dj + r];
          sum += v * w;
          wsum += w;
        }
      }
      row.push(wsum > 0 ? sum / wsum : NaN);  // 全无效窗口 → NaN（除零 guard）
    }
    out.push(row);
  }
  // 按原始 NaN 掩膜重置（数据坑不因邻格插值而视觉消失）
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (Number.isNaN(grid[i][j])) out[i][j] = NaN;
    }
  }
  return out;
}

// ── 地理底图（2026-08-30：cartopy 50m 抽稀导出，public/geo/east_asia_coastline.json）──
// 结构 {"coast": [[lon,lat]...], "borders": [[lon,lat]...]}（线段数组）；
// URL 带版本号防生产构建缓存失效（重跑导出脚本后递增）。
const GEO_URL = '/geo/east_asia_coastline.json?v=1';

// ── 色标约定（与 CIO/Pearson 空间图共享）──
// 范围 [-1, 1]，零点位于色标中心；
// 中心 0 = 中性浅灰白 #F7F7F7（弱化无技能区）；
// 正值多层暖色：浅杏 #FDDBC7 → 珊瑚橙 #F4A582 → 暖红 #D6604D → 红 #B2182B → 深酒红 #67001F；
// 负值柔和雾霾蓝：#3A5FCD → #D8E2F0（不抢眼）。

// 旧版 [-0.3, 1] 色标保留作对照，当前不启用：
// const COLOR_MIN = -0.3;
// const COLOR_MAX = 1.0;
// const COLOR_NODES = [
//   [0.00, '#3A5FCD'],   // r=-0.30 雾霾蓝
//   [0.25, '#8AA9DE'],   // r=-0.15
//   [0.45, '#D8E2F0'],   // r=-0.03 浅雾霾蓝
//   [0.50, '#F7F7F7'],   // r= 0.00 中性浅灰白
//   [0.60, '#FDDBC7'],   // r= 0.20 浅杏
//   [0.70, '#F4A582'],   // r= 0.40 珊瑚橙
//   [0.80, '#D6604D'],   // r= 0.60 暖红
//   [0.90, '#B2182B'],   // r= 0.80 红
//   [1.00, '#67001F'],   // r= 1.00 深酒红
// ];
// function oldNormPos(v) {
//   if (v <= 0) return 0.5 * (v - COLOR_MIN) / (0 - COLOR_MIN);
//   return 0.5 + 0.5 * v / COLOR_MAX;
// }

// ── 0.5 等值线（Marching Squares 简化版）──
// 数据格点 (i, j)（i=0 为 20°N 南端）→ 显示坐标 x=j, y=rows-i（北在上），
// viewBox = "0 0 cols rows"，配合 preserveAspectRatio="none" 拉伸后与格子完全对齐。
const CONTOUR_LEVEL = 0.5;

function contourPathSegments(grid, threshold) {
  const rows = grid.length;
  const cols = grid[0].length;
  const segs = [];

  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < cols - 1; j++) {
      const v00 = grid[i][j], v10 = grid[i][j + 1];
      const v11 = grid[i + 1][j + 1], v01 = grid[i + 1][j];

      // 4 条边的 crossing 参数 t ∈ (0,1)（线性插值），无则 null
      const cross = (a, b) => {
        if (a === b) return null;
        const t = (threshold - a) / (b - a);
        return t > 0 && t < 1 ? t : null;
      };
      // 顶边 v00→v10：y = rows - i；底边 v01→v11：y = rows - i - 1
      // 左边 v00→v01：x = j；右边 v10→v11：x = j + 1
      const tTop = cross(v00, v10), tRight = cross(v10, v11);
      const tBottom = cross(v01, v11), tLeft = cross(v00, v01);

      const pts = [];
      if (tTop !== null) pts.push(['t', j + tTop, rows - i]);
      if (tRight !== null) pts.push(['r', j + 1, rows - i - tRight]);
      if (tBottom !== null) pts.push(['b', j + tBottom, rows - i - 1]);
      if (tLeft !== null) pts.push(['l', j, rows - i - tLeft]);

      if (pts.length === 2) {
        segs.push([pts[0][1], pts[0][2], pts[1][1], pts[1][2]]);
      } else if (pts.length === 4) {
        // 鞍点（对角同号）：取 (顶,左) + (底,右)，歧义解视觉可接受
        const d1 = v00 >= threshold && v11 >= threshold;
        const d2 = v10 >= threshold && v01 >= threshold;
        if (d1 || !d2) {
          segs.push([pts[0][1], pts[0][2], pts[3][1], pts[3][2]]);
          segs.push([pts[1][1], pts[1][2], pts[2][1], pts[2][2]]);
        } else {
          segs.push([pts[0][1], pts[0][2], pts[1][1], pts[1][2]]);
          segs.push([pts[2][1], pts[2][2], pts[3][1], pts[3][2]]);
        }
      }
    }
  }
  return segs;
}

function contourPathD(grid, threshold) {
  return contourPathSegments(grid, threshold)
    .map(s => `M${s[0]} ${s[1]}L${s[2]} ${s[3]}`)
    .join('');
}

// ── 蓝框（重点区口径 112-121°E, 23-27°N；2026-08-30 由 112-120/23-28 修正，
// 与统计计算区域/导师 our_result 口径一致）──
// 经度 100-125°E = 25° 分 cols 格、纬度 20-40°N = 20° 分 rows 格；
// 注意 dLon 是 25/cols（不是 20/cols），2026-08-28 曾算错导致蓝框右偏
// 81×101 网格下实际值：x=48.48, w=36.36, yTop=52.65, h=16.2（格点位于格元左下角约定）
function boxRect(grid) {
  const rows = grid.length, cols = grid[0].length;
  const lon0 = 100, dLon = 25 / cols;            // 经度每格度数（0.25°）
  const lat0 = 20, dLat = 20 / rows;             // 纬度每格度数（0.25°）
  const x = (112 - lon0) / dLon;
  const w = (121 - lon0) / dLon - x;
  const yTop = rows - (27 - lat0) / dLat;        // 27°N 显示行
  const yBot = rows - (23 - lat0) / dLat;        // 23°N 显示行
  return { x, y: yTop, width: w, height: yBot - yTop };
}

// 经纬度 → 格点坐标（与 boxRect 同公式；地理折线用，北在上）
function lonToX(lon, cols) { return ((lon - 100) / 25) * cols; }
function latToY(lat, rows) { return rows - ((lat - 20) / 20) * rows; }

// 地理折线 → SVG path d（每组线段单独 M...L 子路径，防跨组连笔；
// Array.isArray 守卫：JSON 可解析但结构异常时返回空串，静默降级不崩页）
function geoPathD(segs, cols, rows) {
  if (!Array.isArray(segs)) return '';
  return segs
    .filter(Array.isArray)
    .map(seg => seg.map(([lon, lat], k) =>
      `${k === 0 ? 'M' : 'L'}${lonToX(lon, cols).toFixed(2)} ${latToY(lat, rows).toFixed(2)}`
    ).join(''))
    .join('');
}

function MiniRow({ row, onClick }) {
  return (
    <div className="mini-row">
      {row.map((v, j) => (
        <div
          key={j}
          className="mini-cell"
          style={{
            backgroundColor: getHeatColor(v),
            width: `${100 / row.length}%`,
          }}
          onClick={() => onClick(j)}
          title={Number.isNaN(v) ? '无数据' : `r ≈ ${v.toFixed(3)}`}
        />
      ))}
    </div>
  );
}

export default function SkillMap({ data, region, onGridClick }) {
  const { data: gridData, rows, cols } = data;
  const [geoData, setGeoData] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch(GEO_URL)
      .then(r => { if (!r.ok) throw new Error('geo fetch failed'); return r.json(); })
      .then(d => { if (alive) setGeoData(d); })
      .catch(() => { /* 请求/解析失败均静默降级：仅省略地理线，不抛错 */ });
    return () => { alive = false; };
  }, []);
  // 平滑为纯展示层：上色/tooltip/等值线用平滑场，点击联动（传 i,j）不受影响
  const smoothed = useMemo(() => gaussianSmooth(gridData), [gridData]);
  const handleCellClick = (i, j) => {
    // 点击仅联动时序图（整列红色高亮曾造成大红线，2026-08-25 已移除）
    if (onGridClick) onGridClick(i, j, region);
  };
  const box = boxRect(gridData);

  return (
    <div className="result-card">
      <h3 className="result-title">
        预测技能空间分布
        <span className="region-badge">东亚</span>
      </h3>
      <div className="heatmap-container">
        <div className="map-body">
          {/* 纬度轴（左侧，40°N 顶 / 20°N 底，每 5°；北在上） */}
          <div className="lat-axis">
            {[40, 35, 30, 25, 20].map(lat => (
              <span
                key={lat}
                className={lat === 20 ? 'bottom' : ''}
                style={lat === 20 ? undefined : { top: `${((40 - lat) / 20) * 100}%` }}
              >{lat}°N</span>
            ))}
            <span className="lat-title">纬度 (°N)</span>
          </div>
          <div className="heatmap-scroll">
            {/* 数组行号 i=0 为 20°N（南）、i=80 为 40°N（北）；翻转渲染使北在上（与论文/plot_our 一致） */}
            {[...smoothed].reverse().map((row, k) => {
              const iData = smoothed.length - 1 - k;
              return (
                <MiniRow
                  key={iData}
                  row={row}
                  onClick={(j) => handleCellClick(iData, j)}
                />
              );
            })}
            {/* 地理线 + 0.5 等值线 + 蓝框 overlay（2026-08-28 新增，对齐 Figure 5/6；
                2026-08-30 加海岸线/国界折线，等值线数据源改平滑场）：
                viewBox 用格点角点坐标，preserveAspectRatio="none" 拉伸后与格子对齐；
                pointer-events: none 保证点击仍穿透到格子 */}
            <svg
              className="heatmap-overlay"
              viewBox={`0 0 ${cols} ${rows}`}
              preserveAspectRatio="none"
            >
              {geoData && (
                <>
                  {/* 海岸线（深黑粗线，地理参照；随格子同映射拉伸，非正形投影；
                      粗度弱于 0.5 等值线（1.2 vs 1.6），层次分明） */}
                  <path
                    d={geoPathD(geoData.coast || [], cols, rows)}
                    fill="none"
                    stroke="#222"
                    strokeWidth={1.2}
                    opacity={0.9}
                    vectorEffect="non-scaling-stroke"
                  />
                  {/* 国界（2026-08-30 起注释停用：用户确认不需要；JSON 数据保留，取消注释即可恢复） */}
                  {/* <path
                    d={geoPathD(geoData.borders || [], cols, rows)}
                    fill="none"
                    stroke="#555"
                    strokeWidth={0.6}
                    opacity={0.6}
                    vectorEffect="non-scaling-stroke"
                  /> */}
                </>
              )}
              {/* 0.5 等值线（第一视觉层次：纯黑最粗，比海岸线醒目；r=0.5 有用技能边界参考线） */}
              <path
                d={contourPathD(smoothed, CONTOUR_LEVEL)}
                fill="none"
                stroke="#000"
                strokeWidth={1.6}
                opacity={0.9}
                vectorEffect="non-scaling-stroke"
              />
              <rect
                x={box.x} y={box.y} width={box.width} height={box.height}
                fill="none"
                stroke="#0057b8"
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
              />
            </svg>
          </div>
        </div>
        {/* 经度轴（底部，100-125°E 每 5°，与论文/plot_our 的 x 轴一致） */}
        <div className="lon-axis">
          {[100, 105, 110, 115, 120, 125].map(lon => (
            <span key={lon} style={{ left: `${((lon - 100) / 25) * 100}%` }}>{lon}°E</span>
          ))}
          <span className="lon-title">经度 (°E)</span>
        </div>
        {/* 共享九节点色板图例：相关系数固定覆盖 [-1, 1]。 */}
        <div className="heatmap-legend">
          <span>-1</span>
          <div className="legend-gradient">
            <div className="legend-ticks">
              <span style={{ left: '50%' }}>0</span>
              <span style={{ left: '75%' }}>0.5</span>
            </div>
          </div>
          <span>1.0</span>
        </div>
      </div>
      {/* ── 剖面图（2026-08-25 起注释停用：热力图已完整展示 r 分布，单行剖面价值有限）──
      {selectedI !== null && (
        <div className="cross-section">
          <p className="cross-label">格点 ({selectedI}, {selectedJ}) 所在行的剖面</p>
          <ResponsiveContainer width="100%" height={80}>
            <ComposedChart data={barData}>
              80px 小图不用 CartesianGrid（31 条虚线网格太密太乱），只保留 0 基线实线作正负分界
              <ReferenceLine y={0} stroke="#999" />
              <XAxis dataKey="j" hide />
              <YAxis hide domain={[-maxAbs * 1.1, maxAbs * 1.1]} />
              <Tooltip formatter={(v) => v.toFixed(3)} labelFormatter={(j) => `第${j}列`} />
              <Bar dataKey="r" minPointSize={2}>
                {barData.map((d, idx) => (
                  <Cell key={idx} fill={idx === selectedJ ? '#e63946' : '#1a478a'} />
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
      */}
    </div>
  );
}
