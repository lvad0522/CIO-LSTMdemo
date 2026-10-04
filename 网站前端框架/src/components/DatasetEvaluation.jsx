import LivePredictionResult from './LivePredictionResult';
const BASE = import.meta.env.VITE_DATA_API_BASE || import.meta.env.VITE_API_BASE || '';
export default function DatasetEvaluation({ year, lead, fold }) {
  const nLead = Number(String(lead).replace('pre', ''));
  const job = { jobId: `dataset-${year}-${lead}-${fold}`, result: { verification: { available: true } } };
  return <section>
    <LivePredictionResult job={job} apiRoot={`${BASE}/api/dataset/${year}/${nLead}/${fold}`} showSkill={false} showRunSummary={false} />
  </section>;
}
