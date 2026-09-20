import { Fragment, useState, useCallback } from 'react';
import DataSelector from './components/DataSelector';
import ModuleSelector from './components/ModuleSelector';
import ParamPanel from './components/ParamPanel';
import RunButton from './components/RunButton';
import SkillMap from './components/SkillMap';
import TimeSeriesChart from './components/TimeSeriesChart';
import ModelCompareChart from './components/ModelCompareChart';
import ResultsTable from './components/ResultsTable';
import CioProjectionResult from './components/CioProjectionResult';
import NormalizationPanel from './components/NormalizationPanel';
import LivePredictionResult from './components/LivePredictionResult';
import './App.css';

// 开发期默认走 Vite 同源代理；第二套后端仍可用 VITE_API_BASE 覆盖。
const API_BASE = import.meta.env.VITE_API_BASE || '';

const defaultParams = {
  cioFilterLow: 0.02,
  cioFilterHigh: 0.1,
  cioRegion: 'tropical',
  cioSignificance: '0.95',
  predYear: 2019,
  leadTime: 'pre6',
  predRegion: 'eastasia',  // 数据集仅支持东亚（20-40°N, 100-125°E）
};

// 阶段门控只看 stageIndex，不看 status（design D11）：某一段跑完就解锁它的产物区
const STAGE_PROJECTION = 1;
const STAGE_NORMALIZATION = 2;
const STAGE_PREDICTION = 3;
const STAGE_STEPS = [
  { n: STAGE_PROJECTION, label: '投影CIO' },
  { n: STAGE_NORMALIZATION, label: '归一化窗口' },
  { n: STAGE_PREDICTION, label: 'LSTM降水' },
];

export default function App() {
  const [dataSource, setDataSource] = useState('default');
  const [runScope, setRunScope] = useState('full');   // 'projection' | 'full'
  const [params, setParams] = useState(defaultParams);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  // 时序图当前格点（默认 (40,50) 与 /api/run 默认一致；点击热力图后更新）
  // 显示顺序统一为 (经度, 纬度) = (j, i)：横轴=经度、纵轴=纬度（2026-08-25）
  const [selectedGrid, setSelectedGrid] = useState({ i: 40, j: 50 });
  const [cioFile, setCioFile] = useState(null);
  const [uploadKind, setUploadKind] = useState('npy');  // 'npy'（CIO 序列）| 'zip'（原始气象场）
  // 整条链只有一个任务状态、一次轮询（design D11：上传模式不再有 cioJob/predictionJob）
  const [chainJob, setChainJob] = useState(null);

  const handleParamChange = useCallback((key, value) => {
    setParams(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleDataSourceChange = useCallback((nextSource) => {
    setDataSource(nextSource);
    setResults(null);
    setChainJob(null);
    setError(null);
    if (nextSource === 'upload') {
      setParams(prev => ({ ...prev, predYear: 2000, leadTime: 'pre1' }));
    }
  }, []);

  const handleCioFileChange = useCallback((file, detectedKind) => {
    setCioFile(file);
    if (detectedKind) setUploadKind(detectedKind);
    setChainJob(null);
    setResults(null);
    setError(null);
  }, []);

  // 切换上传类别（.npy / .zip）时必须丢掉上一个文件，否则扩展名与类别对不上
  const handleUploadKindChange = useCallback((nextKind) => {
    setUploadKind(nextKind);
    setCioFile(null);
    setChainJob(null);
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
        // 整条链 = 一次提交 + 一个任务 + 一个轮询端点；上传件只提交这一次
        setChainJob({
          status: 'uploading',
          stage: uploadKind === 'zip' ? '正在上传原始气象场' : '正在上传 CIO.npy',
          progress: 0,
          stageIndex: 0,
        });
        const form = new FormData();
        form.append('file', cioFile);
        const query = new URLSearchParams({
          year: '2000',
          lead: 'pre1',
          which: uploadKind === 'zip' ? 'u850' : 'cio',
        });
        if (runScope === 'projection') query.set('onlyProjection', 'true');
        const basePath = '/api/chain/jobs';
        const createRes = await fetch(`${API_BASE}${basePath}?${query}`, {
          method: 'POST',
          body: form,
        });
        const created = await createRes.json();
        if (!createRes.ok) throw new Error(created.detail || `任务创建失败 (${createRes.status})`);
        setChainJob(created);

        let job = created;
        for (let attempt = 0; attempt < 2700; attempt += 1) {   // 上限 ≥ 30 分钟
          if (job.status === 'completed') break;
          if (job.status === 'failed') throw new Error(job.error || '链路任务失败');
          await new Promise(resolve => setTimeout(resolve, 1000));
          const statusRes = await fetch(`${API_BASE}${basePath}/${created.jobId}`);
          job = await statusRes.json();
          if (!statusRes.ok) throw new Error(job.detail || `进度查询失败 (${statusRes.status})`);
          setChainJob(job);
        }
        if (job.status !== 'completed') throw new Error('链路任务等待超过 45 分钟');
        setChainJob(job);
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
  }, [params, dataSource, cioFile, uploadKind, runScope]);

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

  const uploadMode = dataSource === 'upload';
  const stageIndex = Number(chainJob?.stageIndex ?? 0);
  const stageStatus = chainJob?.status;
  const stageInFlight = stageStatus === 'running' || stageStatus === 'queued' || stageStatus === 'uploading';
  const stageFailed = stageStatus === 'failed';
  // `.npy` 入口跳过①；只跑投影的完成态下③ 是"本次未运行"而不是"待运行"
  const stage1Skipped = Boolean(chainJob?.projection?.stage1Skipped);
  const projectionOnlyDone = stageStatus === 'completed' && stageIndex === STAGE_NORMALIZATION;
  // 只跑投影时侧栏显示被锁死的投影口径，全链时显示预测口径与置信水平（design D13）
  const paramModule = uploadMode && runScope === 'projection' ? 'cio' : 'lstm';
  const stageLabel = (n) => {
    if (n === STAGE_PROJECTION && stage1Skipped) return '投影CIO（已跳过）';
    if (n === STAGE_PREDICTION && projectionOnlyDone) return 'LSTM降水（本次未运行）';
    return STAGE_STEPS[n - 1].label;
  };
  const stageState = (n) => {
    if (stageIndex >= n) return 'done';
    if (stageFailed) return stageIndex === n - 1 ? 'failed' : 'pending';
    if (projectionOnlyDone && n === STAGE_PREDICTION) return 'skipped';
    if (stageInFlight && stageIndex === n - 1) return 'active';
    return 'pending';
  };

  return (
    <div className="app">
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

        {uploadMode && <ModuleSelector value={runScope} onChange={setRunScope} />}

        <ParamPanel
          module={paramModule}
          params={params}
          onChange={handleParamChange}
          liveModelMode={uploadMode}
        />

        <RunButton
          onRun={handleRun}
          running={running}
          disabled={uploadMode && !cioFile}
          progress={chainJob?.progress || 0}
          stage={chainJob?.stage || ''}
        />

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

        {/* 顶部阶段条：只读 stageIndex，与三段产物区同一判据 */}
        {uploadMode && chainJob && (
          <div className="chain-stage-bar" aria-label="链路阶段进度">
            <div className="chain-stage-track">
              {STAGE_STEPS.map((step, index) => {
                const state = stageState(step.n);
                return (
                  <Fragment key={step.n}>
                    {index > 0 && (
                      <span
                        className={`chain-stage-link ${stageIndex >= step.n ? 'done' : ''}`}
                        aria-hidden="true"
                      />
                    )}
                    <span className={`chain-stage is-${state}`}>
                      <i className="chain-stage-dot">
                        {state === 'done' ? '✓' : state === 'skipped' ? '–' : step.n}
                      </i>
                      <span className="chain-stage-label">{stageLabel(step.n)}</span>
                    </span>
                  </Fragment>
                );
              })}
            </div>
            <div className="chain-stage-meta">
              <span className="chain-stage-text">{chainJob.stage || '等待链路执行'}</span>
              <span className="chain-stage-percent">{Math.round(Number(chainJob.progress) || 0)}%</span>
            </div>
            {stageFailed && chainJob.error && <p className="chain-stage-error">⚠️ {chainJob.error}</p>}
          </div>
        )}

        {uploadMode && !chainJob && !running && (
          <div className="empty-state">
            <h2>上传一次即可跑完整条链</h2>
            <p>
              选择 .zip（原始 U850 气象场）或 .npy（已投影 CIO 序列）→ 点击“运行” →
              投影 CIO → 归一化窗口 → LSTM 降水推理 由上到下依次解锁，中途不需要重新选文件。
            </p>
          </div>
        )}

        {/* 三段竖排：每段按 stageIndex 单独解锁，不必等整条链跑完 */}
        {uploadMode && stageIndex >= STAGE_PROJECTION && (
          <section className="chain-stage-group">
            <h2 className="chain-stage-heading">阶段① 投影 CIO</h2>
            <CioProjectionResult job={chainJob} confidence={params.cioSignificance} />
          </section>
        )}

        {uploadMode && stageIndex >= STAGE_NORMALIZATION && (
          <section className="chain-stage-group">
            <h2 className="chain-stage-heading">阶段② 归一化窗口</h2>
            <NormalizationPanel job={chainJob} />
          </section>
        )}

        {/* 只跑投影的任务 stageIndex 停在 2，这里天然不会渲染（验收 5） */}
        {uploadMode && runScope === 'full' && stageIndex >= STAGE_PREDICTION && (
          <section className="chain-stage-group">
            <h2 className="chain-stage-heading">阶段③ LSTM 降水预测</h2>
            <LivePredictionResult job={chainJob} confidence={params.cioSignificance} />
          </section>
        )}

        {results && !uploadMode && (
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

        {!results && !chainJob && !running && !uploadMode && (
          <div className="empty-state">
            <h2>选择参数后点击"运行"</h2>
            <p>左侧面板设置参数 → 点击运行按钮 → 结果将展示在这里</p>
          </div>
        )}
      </main>
    </div>
  );
}
