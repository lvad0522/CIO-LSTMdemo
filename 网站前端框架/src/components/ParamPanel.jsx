import { Children, useEffect, useRef, useState } from 'react';

export default function ParamPanel({ module, params, onChange, liveModelMode = false }) {
  if (module === 'cio') {
    return (
      <div className="panel-section">
        <h3 className="section-title">CIO参数设置</h3>
        <p className="param-hint" style={{ marginBottom: 10 }}>
          前三项由已固化的投影基与权重锁死，<strong>改不了</strong>，这里如实列出实际值；
          只有显著性阈值是纯后处理，可调。
        </p>
        <LockedParam
          label="滤波频段"
          value="沿纬度 20 天高通"
          reason="改频段会改变投影出的 CIO，而 8181 个 checkpoint 正是用这个口径训练的。
                  实测本口径对服务器官方金标准 corr = 1.00000000，改动必须重训。"
        />
        <LockedParam
          label="海域范围"
          value="20°S–20°N, 40°E–120°E @1° = 41×81"
          reason="投影基 e 的形状（后 3321 列）锁死了区域。换区域要拿 1982–2017 的
                  原始 SST+U850 重做一次 EOF 分析，本仓库没有那套数据与代码。"
        />
        <LockedParam
          label="预测区域"
          value="20–40°N, 100–125°E @0.25° = 81×101"
          reason="由权重包 pre1_2000.tar.gz 里 8181 个格点的坐标决定。"
        />
        <div className="param-group">
          <label>显著性置信水平</label>
          <StyledSelect ariaLabel="显著性置信水平" value={params.cioSignificance}
            onChange={e => onChange('cioSignificance', e.target.value)}>
            <option value="0.90">90%</option>
            <option value="0.95">95%</option>
            <option value="0.99">99%</option>
          </StyledSelect>
          <span className="param-hint">
            只作用于「预测 vs 实况」相关系数图的显著性阈值（n=112 双尾 t 检验，
            90/95/99% 对应 |r| ≥ 0.156/0.186/0.243），不改变 r 本身。
            在「在线推理」结果页生效。
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="panel-section">
      <h3 className="section-title">LSTM预测参数</h3>
      <div className="param-group">
        <label>预测年份</label>
        <StyledSelect ariaLabel="预测年份" value={params.predYear}
          onChange={e => onChange('predYear', +e.target.value)}>
          {(liveModelMode ? [2000] : Array.from({ length: 20 }, (_, i) => 2000 + i)).map(y => (
            <option key={y} value={y}>{y}</option>
          ))}
        </StyledSelect>
      </div>
      <div className="param-group">
        <label>提前预报时间 (lead time)</label>
        <StyledSelect ariaLabel="提前预报时间" value={params.leadTime}
          onChange={e => onChange('leadTime', e.target.value)}>
          {(liveModelMode ? [1] : Array.from({ length: 20 }, (_, i) => i + 1)).map(n => (
            <option key={`pre${n}`} value={`pre${n}`}>提前{n}天 (pre{n})</option>
          ))}
        </StyledSelect>
        {liveModelMode && (
          <span className="param-hint">
            已接入的模型权重仅支持 2000 / pre1；原始场投影本身可算 pre1–pre20，
            权重到位后自动放开
          </span>
        )}
      </div>
      <div className="param-group">
        <label>预测区域</label>
        <StyledSelect ariaLabel="预测区域" value={params.predRegion}
          onChange={e => onChange('predRegion', e.target.value)}>
          <option value="eastasia">东亚 (81×101)</option>
        </StyledSelect>
        <span className="param-hint">数据集仅覆盖东亚 20-40°N, 100-125°E</span>
      </div>
      {liveModelMode && (
        <div className="param-group">
          <label>显著性置信水平</label>
          <StyledSelect ariaLabel="显著性置信水平" value={params.cioSignificance}
            onChange={e => onChange('cioSignificance', e.target.value)}>
            <option value="0.90">90%</option>
            <option value="0.95">95%</option>
            <option value="0.99">99%</option>
          </StyledSelect>
          <span className="param-hint">
            结果页「逐格点相关系数」的显著性阈值（n=112 双尾 t 检验）
          </span>
        </div>
      )}
    </div>
  );
}

/** 锁死的参数：显示实际生效的值 + 为什么改不了，而不是给一个点了没反应的控件。 */
function LockedParam({ label, value, reason }) {
  return (
    <div className="param-group param-locked">
      <label>{label}</label>
      <span className="param-locked-value">{value}</span>
      {reason && <span className="param-hint">{reason}</span>}
    </div>
  );
}

function StyledSelect({ ariaLabel, value, onChange, children }) {
  const options = Children.toArray(children);
  const selectedIndex = Math.max(0, options.findIndex(option => String(option.props.value) === String(value)));
  const selectedOption = options[selectedIndex] || options[0];
  const [open, setOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(selectedIndex);
  const rootRef = useRef(null);
  const triggerRef = useRef(null);
  const menuRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const handlePointerDown = event => {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener('pointerdown', handlePointerDown);
    return () => document.removeEventListener('pointerdown', handlePointerDown);
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const frame = requestAnimationFrame(() => {
      menuRef.current
        ?.querySelector('[aria-selected="true"]')
        ?.scrollIntoView({ block: 'nearest' });
    });
    return () => cancelAnimationFrame(frame);
  }, [open, selectedIndex]);

  const chooseOption = option => {
    if (!option) return;
    onChange({ target: { value: option.props.value } });
    setOpen(false);
    triggerRef.current?.focus();
  };

  const handleKeyDown = event => {
    if (event.key === 'Escape') {
      if (open) event.preventDefault();
      setOpen(false);
      triggerRef.current?.focus();
      return;
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (!open) {
        setHighlightedIndex(selectedIndex);
        setOpen(true);
        return;
      }
      const direction = event.key === 'ArrowDown' ? 1 : -1;
      setHighlightedIndex(index => Math.min(options.length - 1, Math.max(0, index + direction)));
      return;
    }
    if (event.key === 'Home' || event.key === 'End') {
      if (!open) return;
      event.preventDefault();
      setHighlightedIndex(event.key === 'Home' ? 0 : options.length - 1);
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (open) chooseOption(options[highlightedIndex]);
      else {
        setHighlightedIndex(selectedIndex);
        setOpen(true);
      }
    }
  };

  return (
    <div
      className={`styled-select ${open ? 'is-open' : ''}`}
      ref={rootRef}
      onKeyDown={handleKeyDown}
    >
      <button
        type="button"
        className="styled-select-trigger"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        ref={triggerRef}
        onClick={() => {
          setHighlightedIndex(selectedIndex);
          setOpen(current => !current);
        }}
      >
        <span className="styled-select-value">{selectedOption?.props.children}</span>
        <span className="styled-select-chevron" aria-hidden="true" />
      </button>

      {open && (
        <div className="styled-select-menu" role="listbox" aria-label={ariaLabel} ref={menuRef}>
          {options.map((option, index) => {
            const optionValue = String(option.props.value);
            const isSelected = optionValue === String(value);
            return (
              <button
                type="button"
                role="option"
                aria-selected={isSelected}
                tabIndex={-1}
                className={`styled-select-option ${isSelected ? 'is-selected' : ''} ${index === highlightedIndex ? 'is-highlighted' : ''}`}
                key={optionValue}
                onMouseEnter={() => setHighlightedIndex(index)}
                onClick={() => chooseOption(option)}
              >
                <span>{option.props.children}</span>
                {isSelected && <span className="styled-select-check" aria-hidden="true">✓</span>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
