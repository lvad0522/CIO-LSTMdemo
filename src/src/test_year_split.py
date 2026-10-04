import numpy as np
import pytest

from year_split import split_leave_one_year_out


@pytest.mark.parametrize("year_index", range(20))
def test_all_2000_2019_folds_keep_exactly_the_other_19_years(year_index):
    years = np.repeat(np.arange(2000, 2020), 112)
    labels = years * 10 + np.tile(np.arange(112), 20)
    train_x, train_y, test_x, test_y = split_leave_one_year_out(
        years, labels, year_index
    )
    target = 2000 + year_index
    assert test_x.size == test_y.size == 112
    assert np.all(test_x == target)
    assert train_x.size == train_y.size == 19 * 112
    actual_years, counts = np.unique(train_x, return_counts=True)
    assert set(actual_years) == set(range(2000, 2020)) - {target}
    assert np.all(counts == 112)
    np.testing.assert_array_equal(test_y, target * 10 + np.arange(112))
    for year in actual_years:
        np.testing.assert_array_equal(train_y[train_x == year], year * 10 + np.arange(112))


@pytest.mark.parametrize("size", [18 * 112, 22 * 112])
def test_other_year_counts_are_not_tied_to_a_special_last_year(size):
    series = np.arange(size)
    train_x, _, test_x, _ = split_leave_one_year_out(series, series, size // 112 - 1)
    assert train_x.size == size - 112
    np.testing.assert_array_equal(test_x, series[-112:])


@pytest.mark.parametrize("inputs,outputs,year_index", [
    (np.zeros(2240), np.zeros(2239), 17),
    (np.zeros(2239), np.zeros(2239), 17),
    (np.zeros(112), np.zeros(112), 0),
    (np.zeros(2240), np.zeros(2240), -1),
    (np.zeros(2240), np.zeros(2240), 20),
    (np.zeros(2240), np.zeros(2240), 17.5),
])
def test_invalid_or_incomplete_year_blocks_are_rejected(inputs, outputs, year_index):
    with pytest.raises(ValueError):
        split_leave_one_year_out(inputs, outputs, year_index)
