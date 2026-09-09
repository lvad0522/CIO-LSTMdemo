import { useState, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import DataSelector from './components/DataSelector';
import ModuleSelector from './components/ModuleSelector';
import ParamPanel from './components/ParamPanel';
import RunButton from './components/RunButton';
import SkillMap from './components/SkillMap';
import TimeSeriesChart from './components/TimeSeriesChart';
import ModelCompareChart from './components/ModelCompareChart';
import ResultsTable from './components/ResultsTable';
import './App.css';

const API_BASE = 'http://localhost:8000';

const defaultParams = {
  cioFilterLow: 0.02,
  cioFilterHigh: 0.1,
  cioRegion: 'tropical',
  cioSignificance: '0.95',
  predYear: 2019,
  leadTime: 'pre6',
  predRegion: 'eastasia',  // 数据集仅支持东亚（20-40°N, 100-125°E）
};

export default function App() {
  const [dataSource, setDataSource] = useState('default');
  const [activeModule, setActiveModule] = useState('lstm');
  const [params, setParams] = useState(defaultParams);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  // 时序图当前格点（默认 (40,50) 与 /api/run 默认一致；点击热力图后更新）
  // 显示顺序统一为 (经度, 纬度) = (j, i)：横轴=经度、纵轴=纬度（2026-08-25）
  const [selectedGrid, setSelectedGrid] = useState({ i: 40, j: 50 });

  const handleParamChange = useCallback((key, value) => {
    setParams(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const query = new URLSearchParams({
        year: params.predYear,
        lead: params.leadTime,
        region: params.predRegion,
        filter_low: params.cioFilterLow,
        filter_high: params.cioFilterHigh,
        cio_region: params.cioRegion,
        significance: params.cioSignificance,
      }).toString();
      const res = await fetch(`${API_BASE}/api/run?${query}`);
      if (!res.ok) throw new Error(`API 返回 ${res.status}`);
      const data = await res.json();
      setResults(data);
    } catch (err) {
      setError(err.message || '连接后端失败，请确认后端已启动');
      console.error('请求失败:', err);
    } finally {
      setRunning(false);
    }
  }, [params]);

  const handleGridClick = useCallback(async (i, j, region) => {
    setSelectedGrid({ i, j });
    try {
      // 携带当前 year/lead，保证点击时序与显示地图组合一致
      const query = new URLSearchParams({ i, j, region, year: params.predYear, lead: params.leadTime }).toString();
      const res = await fetch(`${API_BASE}/api/grid?${query}`);
      if (!res.ok) {
        console.warn('格点查询返回', res.status);
        return;
      }
      const ts = await res.json();
      setResults(prev => prev ? { ...prev, timeSeries: ts } : prev);
    } catch (err) {
      console.error('格点查询失败:', err);
    }
  }, [params.predYear, params.leadTime]);

  const cioDisabled = dataSource === 'default';

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <h1>CIO + LSTM</h1>
          <p>季风降水预测交互展示</p>
        </div>

        <DataSelector
          value={dataSource}
          onChange={setDataSource}
          disabled={running}
        />

        <ModuleSelector
          active={activeModule}
          onSelect={setActiveModule}
          cioDisabled={cioDisabled}
        />

        <ParamPanel
          module={activeModule}
          params={params}
          onChange={handleParamChange}
        />

        <RunButton onRun={handleRun} running={running} disabled={false} />

        <div className="sidebar-footer">
          <p>基于 Zhou et al. (2024) GRL</p>
          <p>src/src/ 代码实现</p>
        </div>
      </aside>

      <main className="main-content">
        {error && (
          <div className="error-banner">
            <span>⚠️ {error}</span>
            <button onClick={() => setError(null)} className="error-close">✕</button>
          </div>
        )}
        {!results && (
          <div className="empty-state">
            <h2>选择参数后点击"运行"</h2>
            <p>左侧面板设置参数 → 点击运行按钮 → 结果将展示在这里</p>
          </div>
        )}

        {results && activeModule === 'cio' && dataSource === 'upload' && !results.cioTS && (
          <div className="empty-state">
            <h2>暂无 CIO 数据</h2>
            <p>CIO 指数与相关分布需要 SST+Uwind 输入计算，后端暂未接入真实计算（mock 模拟已关停）。</p>
          </div>
        )}

        {results && activeModule === 'cio' && dataSource === 'upload' && results.cioTS && (
          <div className="result-row">
            <div className="result-card" style={{ flex: 1 }}>
              <h3 className="result-title">CIO指数时序曲线</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={results.cioTS.filter((_, i) => i % 20 === 0).map((v, i) => ({ t: i * 20, v }))} margin={{ bottom: 30 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" label={{ value: '时间 (天)', position: 'insideBottom', offset: -5 }} />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="v" stroke="#1a478a" dot={false} name="CIO指数" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="result-card" style={{ flex: 1 }}>
              <h3 className="result-title">CIO-降水相关空间分布 (显著性: {params.cioSignificance})</h3>
              <div className="heatmap-container">
                <div className="heatmap-scroll" style={{ maxHeight: 260 }}>
                  {results.cioCorr.data.slice(0, 30).map((row, i) => (
                    <div key={i} className="mini-row">
                      {row.slice(0, 60).map((v, j) => (
                        <div key={j} className="mini-cell" style={{
                          backgroundColor: corrColor(v),
                          width: `${100 / 60}%`,
                        }} title={`r = ${v.toFixed(3)}`} />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
              <ColorBar />
            </div>
          </div>
        )}

        {results && activeModule === 'cio' && dataSource === 'default' && (
          <div className="empty-state">
            <h2>CIO计算与验证不可用</h2>
            <p>CIO 计算需要原始 SST+Uwind 数据输入，后端暂未接入真实计算，本模块暂时无数据可显示。</p>
          </div>
        )}

        {results && activeModule === 'lstm' && (
          <>
            <div className="result-row">
              <div className="result-col">
                <SkillMap
                  data={results.skillMap}
                  region={params.predRegion}
                  onGridClick={handleGridClick}
                />
              </div>
              <div className="result-col">
                <TimeSeriesChart
                  data={results.timeSeries}
                  gridLabel={`东亚 (${selectedGrid.j}, ${selectedGrid.i})`}
                />
                <ModelCompareChart data={results.s2s} />
              </div>
            </div>
            <ResultsTable data={results.table} />
          </>
        )}
      </main>
    </div>
  );
}

function corrColor(v) {
  const t = (v + 1) / 2;
  const r = Math.round(30 + t * 200);
  const b = Math.round(200 - t * 170);
  return `rgb(${r}, 80, ${b})`;
}

function ColorBar() {
  const stops = [-1, -0.5, 0, 0.5, 1];
  return (
    <div className="colorbar" title="Pearson 相关系数 r">
      <div
        className="colorbar-gradient"
        style={{
          background: `linear-gradient(to right, ${corrColor(-1)}, ${corrColor(0)}, ${corrColor(1)})`,
        }}
      />
      <div className="colorbar-ticks">
        {stops.map(v => <span key={v}>{v}</span>)}
      </div>
      <div className="colorbar-labels">
        <span>低 (r=-1)</span>
        <span>高 (r=1)</span>
      </div>
    </div>
  );
}
