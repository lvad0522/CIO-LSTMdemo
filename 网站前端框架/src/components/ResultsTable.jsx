import { useRef } from 'react';

export default function ResultsTable({ data }) {
  const tableRef = useRef(null);

  const handleExportCSV = () => {
    if (!data || data.length === 0) return;
    const headers = ['实验次数', 'Pearson r', 'RMSE (mm/day)', 'MAE (mm/day)'];
    const rows = data.map(d => [d.experiment, d.pearsonR, d.rmse, d.mae]);
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lstm_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!data || data.length === 0) return null;

  return (
    <div className="result-card">
      <div className="result-header">
        <h3 className="result-title" style={{ margin: 0 }}>实验结果详情</h3>
        <button className="export-btn" onClick={handleExportCSV}>📥 导出CSV</button>
      </div>
      <div className="table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              <th>实验</th>
              <th>Pearson r</th>
              <th>RMSE (mm/day)</th>
              <th>MAE (mm/day)</th>
            </tr>
          </thead>
          <tbody>
            {data.map(d => (
              <tr key={d.experiment}>
                <td>第{d.experiment}次</td>
                <td className={d.pearsonR > 0.5 ? 'good' : d.pearsonR > 0.3 ? 'ok' : ''}>
                  {d.pearsonR.toFixed(3)}
                </td>
                <td>{d.rmse.toFixed(2)}</td>
                <td>{d.mae.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
