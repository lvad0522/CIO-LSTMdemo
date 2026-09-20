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
import LivePredictionResult from './components/LivePredictionResult';
import CioProjectionResult from './components/CioProjectionResult';
import './App.css';

// 后端地址：默认 8000，可用 VITE_API_BASE 覆盖（如换端口起第二套后端时不改代码）
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

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
  const [cioFile, setCioFile] = useState(null);
  const [uploadKind, setUploadKind] = useState('npy');  // 'npy'（CIO 序列）| 'zip'（原始气象场）
  const [predictionJob, setPredictionJob] = useState(null);

  const handleParamChange = useCallback((key, value) => {
    setParams(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleDataSourceChange = useCallback((nextSource) => {
    setDataSource(nextSource);
    setResults(null);
    setPredictionJob(null);
    setError(null);
    if (nextSource === 'upload') {
      setActiveModule('lstm');
      setParams(prev => ({ ...prev, predYear: 2000, leadTime: 'pre1' }));
    }
  }, []);

  const handleCioFileChange = useCallback((file, detectedKind) => {
    setCioFile(file);
    if (detectedKind) setUploadKind(detectedKind);
    if (file) setActiveModule('lstm');
    setPredictionJob(null);
    setResults(null);
    setError(null);
  }, []);

  // 切换上传类别（.npy / .zip）时必须丢掉上一个文件，否则扩展名与类别对不上
  const handleUploadKindChange = useCallback((nextKind) => {
    setUploadKind(nextKind);
    setCioFile(null);
    setPredictionJob(null);
    setResults(null);
    setError(null);
  }, []);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      if (dataSource === 'upload') {
        if (!cioFile) {
          throw new Error(uploadKind === 'zip'
            ? '请先选择原始气象场 .zip 文件'
            : '请先选择一维 CIO .npy 文件');
        }

        setResults(null);
        setPredictionJob({
          status: 'uploading',
          stage: uploadKind === 'zip' ? '正在上传原始气象场' : '正在上传 CIO.npy',
          progress: 0,
        });
        const form = new FormData();
        form.append('file', cioFile);
        const query = new URLSearchParams({ year: 2000, lead: 'pre1' });
        // 原始场入口必须显式声明类别（后端会再和文件名对一遍，冲突即报错）
        if (uploadKind === 'zip') query.set('which', 'u850');
        const createRes = await fetch(`${API_BASE}/api/predict/jobs?${query}`, {
          method: 'POST',
          body: form,
        });
        const created = await createRes.json();
        if (!createRes.ok) throw new Error(created.detail || `任务创建失败 (${createRes.status})`);
        setPredictionJob(created);

        let job = created;
        for (let attempt = 0; attempt < 1800; attempt += 1) {
          if (job.status === 'completed') break;
          if (job.status === 'failed') throw new Error(job.error || '模型推理失败');
          await new Promise(resolve => setTimeout(resolve, 1000));
          const statusRes = await fetch(`${API_BASE}/api/predict/jobs/${created.jobId}`);
          job = await statusRes.json();
          if (!statusRes.ok) throw new Error(job.detail || `进度查询失败 (${statusRes.status})`);
          setPredictionJob(job);
        }
        if (job.status !== 'completed') throw new Error('模型推理等待超过 30 分钟');
        setPredictionJob(job);
        return;
      }

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
  }, [params, dataSource, cioFile, uploadKind]);

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
    <div className={`app ${running ? 'is-running' : ''}`}>
      <aside className="sidebar">
        <div className="logo">
          <h1>CIO + LSTM</h1>
          <p>季风降水预测交互展示</p>
        </div>

        <DataSelector
          value={dataSource}
          onChange={handleDataSourceChange}
          disabled={running}
          cioFile={cioFile}
          onCioFileChange={handleCioFileChange}
          uploadKind={uploadKind}
          onUploadKindChange={handleUploadKindChange}
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
          liveModelMode={dataSource === 'upload'}
        />

        <RunButton
          onRun={handleRun}
          running={running}
          disabled={dataSource === 'upload' && !cioFile}
          progress={predictionJob?.progress || 0}
          stage={predictionJob?.stage || ''}
        />

        <div className="sidebar-footer">
          <p>基于 Zhou et al. (2024) GRL</p>
          <p>src/src/ 代码实现</p>
        </div>
      </aside>

      <main className="main-content" aria-busy={running}>
        {error && (
          <div className="error-banner">
            <span>⚠️ {error}</span>
            <button onClick={() => setError(null)} className="error-close">✕</button>
          </div>
        )}
        {!results && !predictionJob && !running && (
          <div className="empty-state">
            <h2>选择参数后点击"运行"</h2>
            <p>左侧面板设置参数 → 点击运行按钮 → 结果将展示在这里</p>
          </div>
        )}

        {running && dataSource === 'upload' && (
          <div className="empty-state">
            <h2>{predictionJob?.stage || '正在创建模型任务'}</h2>
            <p>
              真实进度：{Math.round(predictionJob?.progress || 0)}% ·
              {uploadKind === 'zip'
                ? ' 前 30% 为原始场投影，其后为逐格点加载并运行模型'
                : ' 后端正在逐格点加载并运行模型'}
            </p>
          </div>
        )}

        {predictionJob?.status === 'completed' && dataSource === 'upload' && activeModule === 'lstm' && (
          <LivePredictionResult job={predictionJob} confidence={params.cioSignificance} />
        )}

        {predictionJob?.status === 'completed' && dataSource === 'upload' && activeModule === 'cio' && (
          <CioProjectionResult
            job={predictionJob}
            onShowPrediction={() => setActiveModule('lstm')}
          />
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

        {results && dataSource === 'default' && activeModule === 'lstm' && (
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
