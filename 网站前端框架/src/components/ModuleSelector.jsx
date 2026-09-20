// 「功能模块」→「运行范围」：一条链只有一次提交，这里的两个取值只是跑多深。
// 上传模式下两者**都始终可选**（没有禁用态、没有「不可用」徽标）：只跑投影
// 不再依赖另一个模块先跑过，它本身就是合法的终点。
const SCOPES = [
  {
    value: 'projection',
    label: '只跑投影',
    hint: '只跑阶段①②：投影 CIO 与按训练口径归一化，不加载任何 LSTM 模型',
  },
  {
    value: 'full',
    label: '投影 + 降水',
    hint: '跑完整三阶段：投影 CIO → 归一化窗口 → LSTM 逐格点降水推理',
  },
];

export default function ModuleSelector({ value = 'full', onChange }) {
  return (
    <div className="panel-section">
      <h3 className="section-title">运行范围</h3>
      <div className="module-buttons">
        {SCOPES.map(scope => (
          <button
            key={scope.value}
            type="button"
            className={`module-btn ${value === scope.value ? 'active' : ''}`}
            onClick={() => onChange?.(scope.value)}
            title={scope.hint}
          >
            <span>{scope.label}</span>
          </button>
        ))}
      </div>
      <p className="param-hint">整条链只需上传一次；两个范围共用同一份上传件。</p>
    </div>
  );
}
