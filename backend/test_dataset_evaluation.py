import numpy as np
import pytest
from fastapi import HTTPException
import dataset_evaluation as E

def test_results_table_preserves_computed_precision(monkeypatch):
    truth = np.array([0.013579, 1.23456789, 2.7890123, 4.1357911], dtype=np.float64)
    prediction = np.array([0.1111111, 1.3333333, 2.9999999, 3.9876543], dtype=np.float64)
    monkeypatch.setattr(E.D, '_check_lead_year', lambda lead, year: None)
    monkeypatch.setattr(E.D, '_display_folds', lambda lead, year: [1])
    monkeypatch.setattr(E.D, '_load_npy', lambda lead, year, fold, kind: truth if kind == 'real2' else prediction)
    row = E.D.gen_results_table(2019, 6)[0]
    expected = {
        'pearsonR': float(np.corrcoef(truth, prediction)[0, 1]),
        'rmse': float(np.sqrt(np.mean((prediction-truth)**2))),
        'mae': float(np.mean(np.abs(prediction-truth))),
    }
    for key, value in expected.items():
        assert row[key] == value
        assert row[key] != round(value, 3)

@pytest.fixture
def data(monkeypatch):
    truth = np.broadcast_to(np.arange(112, dtype=float), (81,101,112)).copy()
    r = np.zeros((81,101,3)); r[:,:,2] = .7
    monkeypatch.setattr(E.D, '_check_lead_year', lambda lead, year: None)
    monkeypatch.setattr(E.D, '_display_folds', lambda lead, year: [1,2])
    monkeypatch.setattr(E.D, '_load_npy', lambda lead, year, fold, kind: {'predict2': truth+fold+1, 'real2':truth, 'pearson2':r}[kind])

def test_same_fold_evaluation_and_dates(data):
    for fold in (1,2):
        v=E.focus(2019,6,fold)
        assert v['rmse'] == pytest.approx(fold+1)
        assert v['meanPointR'] == pytest.approx(.7)
        assert v['dates'][0] == '2019-06-02'
        assert v['dates'][-1] == '2019-09-29'
        assert len(v['dates']) == 112
        assert E.preview(2019,6,fold,111)['data'][0][0] == 112+fold
        assert E.truth_preview(2019,6,fold,111)['data'][0][0] == 111

def test_scope_guard(data):
    with pytest.raises(HTTPException) as exc: E.preview(2019,6,3,0)
    assert exc.value.status_code == 422

def test_csv_complete(data):
    lines=E.download_focus(2019,6,1).body.decode().splitlines()
    assert len(lines)==113
    assert lines[1].startswith('2019-06-02,')
    assert lines[-1].startswith('2019-09-29,')
