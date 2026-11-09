"""Edge cases in display formatting; no changes to scientific metrics."""
import json
import numpy as np
import pytest
from causallens.dashboard import effect_groups, format_metric, json_safe


@pytest.mark.parametrize('value', [None, np.nan, np.inf, -np.inf])
def test_missing_diagnostics_are_not_numbers(value):
    assert format_metric(value) == 'N/A'
    assert json.dumps(json_safe({'metric': value}), allow_nan=False) == '{"metric": null}'


def test_zero_is_a_valid_measured_value():
    assert format_metric(0, 2, '%') == '0.00%'
    assert json_safe({'metric': 0.0}) == {'metric': 0.0}


@pytest.mark.parametrize('values', [[0]*70+[1]*30, [0]*40+[1]*30+[2]*30, [2]*75])
def test_tied_effects_stay_together_without_demanding_three_groups(values):
    groups = effect_groups(values)
    assert len(groups) == len(values)
    assert not groups.isna().any()
    for value in set(values):
        assert len(set(groups[np.array(values) == value])) == 1


def test_distinct_effect_groups_keep_their_order():
    groups = effect_groups(np.arange(75))
    assert len(groups.categories) == 3
    assert groups.codes[0] < groups.codes[-1]
