import { Children, useEffect, useRef, useState } from 'react';

const UPLOAD_YEARS = Array.from({ length: 20 }, (_, index) => 2000 + index);
const UPLOAD_LEADS = Array.from({ length: 20 }, (_, index) => `pre${index + 1}`);

export default function ParamPanel({
  module,
  params,
  onChange,
  liveModelMode = false,
  uploadYear = 2000,
  uploadLead = 'pre1',
  onUploadYearChange,
  onUploadLeadChange,
  modelCatalog = null,
  modelCatalogError = false,
  modelCatalogLoading = false,
}) {
  const catalogLeads = modelCatalog?.leads || [];
  const uploadModelSelectors = liveModelMode ? (
    <UploadModelSelectors
      uploadYear={uploadYear}
      uploadLead={uploadLead}
      onUploadYearChange={onUploadYearChange}
      onUploadLeadChange={onUploadLeadChange}
      catalogLeads={catalogLeads}
      modelCatalogLoading={modelCatalogLoading}
      modelCatalogError={modelCatalogError}
    />
  ) : null;

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
          reason={liveModelMode
            ? '由当前所选权重组的格点坐标决定；不同权重组可能使用不同的有效格点范围。'
            : '由权重包 pre1_2000.tar.gz 里 8181 个格点的坐标决定。'}
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
        {uploadModelSelectors}
      </div>
    );
  }

  return (
    <div className="panel-section">
      <h3 className="section-title">LSTM预测参数</h3>
      {liveModelMode ? uploadModelSelectors : (
        <>
          <div className="param-group">
            <label>预测年份</label>
            <StyledSelect ariaLabel="预测年份" value={params.predYear}
              onChange={e => onChange('predYear', +e.target.value)}>
              {Array.from({ length: 20 }, (_, i) => 2000 + i).map(y => (
                <option key={y} value={y}>{y}</option>
              ))}
            </StyledSelect>
          </div>
          <div className="param-group">
            <label>提前预报时间 (lead time)</label>
            <StyledSelect ariaLabel="提前预报时间" value={params.leadTime}
              onChange={e => onChange('leadTime', e.target.value)}>
              {Array.from({ length: 20 }, (_, i) => i + 1).map(n => (
                <option key={`pre${n}`} value={`pre${n}`}>提前{n}天 (pre{n})</option>
              ))}
            </StyledSelect>
          </div>
        </>
      )}
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

function UploadModelSelectors({
  uploadYear,
  uploadLead,
  onUploadYearChange,
  onUploadLeadChange,
  catalogLeads,
  modelCatalogLoading,
  modelCatalogError,
}) {
  const selectedLead = catalogLeads.find(item => item.lead === uploadLead);

  return (
    <>
      <div className="param-group">
        <label>预测年份</label>
        <StyledSelect ariaLabel="预测年份" value={uploadYear} onChange={onUploadYearChange}>
          {UPLOAD_YEARS.map(year => {
            const entry = selectedLead?.years?.find(item => Number(item.year) === year);
            const disabled = !entry || !entry.available;
            const reason = disabled
              ? `暂不可用：${entry?.reason || '未找到该组权重'}`
              : '';
            return (
              <option key={year} value={year} disabled={disabled} title={disabled ? reason : undefined}>
                {disabled ? `${year}（${reason}）` : year}
              </option>
            );
          })}
        </StyledSelect>
      </div>
      <div className="param-group">
        <label>提前预报时间 (lead time)</label>
        <StyledSelect ariaLabel="提前预报时间" value={uploadLead} onChange={onUploadLeadChange}>
          {UPLOAD_LEADS.map((leadName, index) => {
            const leadItem = catalogLeads.find(item => item.lead === leadName);
            const available = leadItem?.years?.some(item => item.available) || false;
            const unavailableReasons = [...new Set((leadItem?.years || [])
              .filter(item => !item.available && item.reason)
              .map(item => item.reason))];
            const reason = available
              ? ''
              : `暂不可用：${leadItem
                ? (unavailableReasons.join('；') || '该 lead 没有可用权重组')
                : '未找到该 lead 权重'}`;
            return (
              <option
                key={leadName}
                value={leadName}
                disabled={!available}
                title={reason || undefined}
              >
                {reason
                  ? `提前${index + 1}天 (${leadName})（${reason}）`
                  : `提前${index + 1}天 (${leadName})`}
              </option>
            );
          })}
        </StyledSelect>
        <span className="param-hint">
          {modelCatalogLoading
            ? '正在获取可用模型组合清单…'
            : modelCatalogError
              ? '未能获取可用组合清单，已回落为 2000 / pre1；上传与运行仍可用。'
              : '固定预留 2000–2019 与 pre1–pre20；没有对应权重的选项会标为暂不可用。'}
        </span>
      </div>
    </>
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

  const isDisabled = option => Boolean(option?.props?.disabled);
  const hasDisabledOptions = options.some(isDisabled);
  const findEnabledIndex = (start, direction) => {
    if (!options.length) return -1;
    if (!hasDisabledOptions) {
      return Math.min(options.length - 1, Math.max(0, start));
    }
    let index = (start + options.length) % options.length;
    for (let count = 0; count < options.length; count += 1) {
      if (!isDisabled(options[index])) return index;
      index = (index + direction + options.length) % options.length;
    }
    return -1;
  };

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
    if (!option || isDisabled(option)) return;
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
        setHighlightedIndex(findEnabledIndex(selectedIndex, 1));
        setOpen(true);
        return;
      }
      const direction = event.key === 'ArrowDown' ? 1 : -1;
      setHighlightedIndex(index => findEnabledIndex(index + direction, direction));
      return;
    }
    if (event.key === 'Home' || event.key === 'End') {
      if (!open) return;
      event.preventDefault();
      setHighlightedIndex(findEnabledIndex(
        event.key === 'Home' ? 0 : options.length - 1,
        event.key === 'Home' ? 1 : -1,
      ));
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (open) chooseOption(options[highlightedIndex]);
      else {
        setHighlightedIndex(findEnabledIndex(selectedIndex, 1));
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
        title={selectedOption?.props.title}
        onClick={() => {
          setHighlightedIndex(findEnabledIndex(selectedIndex, 1));
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
                aria-disabled={isDisabled(option)}
                tabIndex={-1}
                disabled={isDisabled(option)}
                title={option.props.title}
                className={`styled-select-option ${isSelected ? 'is-selected' : ''} ${index === highlightedIndex ? 'is-highlighted' : ''} ${isDisabled(option) ? 'is-disabled' : ''}`}
                key={optionValue}
                onMouseEnter={() => !isDisabled(option) && setHighlightedIndex(index)}
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
