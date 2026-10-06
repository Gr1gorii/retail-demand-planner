import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import apply_scaled_targets, attach_levels, fit_scaled_quantiles, recent_levels, select_on_first_split, validation_blocks
from src.inventory import MODELS
from tests.test_pipeline import BUSINESS


ROOT = Path(__file__).resolve().parents[1]


def sums(origins, items=('A', 'B')):
    records = []
    for item in items:
        for origin in origins:
            scale = 4.5 if item == 'A' else 9.0
            records.append({'fold': 1, 'item_id': item, 'origin_day': origin, 'target_end_day': origin + 8,
                            'protection_period_days': 9, 'sales': scale * 2, 'price_known': 2,
                            'volume_tier': 'low', 'residual_scale': scale, **{model: scale for model in MODELS}})
    return pd.DataFrame(records)


def test_recent_levels_use_only_the_previous_28_days_and_stay_separate_by_item():
    rows = pd.DataFrame({'item_id': ['A'] * 60 + ['B'] * 60, 'day': list(range(1, 61)) * 2,
                         'sales': [0] * 60 + [2] * 60})
    original = recent_levels(rows)
    rows.loc[rows.day.ge(40), 'sales'] = 999
    changed = recent_levels(rows)
    pd.testing.assert_frame_equal(original[original.origin_day.le(40)], changed[changed.origin_day.le(40)])
    frame = sums([40]).drop(columns=['residual_scale', 'volume_tier'])
    attached = attach_levels(frame, original).set_index('item_id')
    assert attached.loc['A', 'residual_scale'] == 1
    assert attached.loc['A', 'recent_zero_share'] == 1
    assert attached.loc['B', 'residual_scale'] == 18
    assert attached.loc['B', 'volume_tier'] == 'medium'


def test_scaled_quantile_is_pooled_within_tier_and_converted_back_per_item():
    first = sums([1, 2])
    quantiles = fit_scaled_quantiles(first, BUSINESS)
    assert quantiles.samples.eq(4).all() and quantiles['items'].eq(2).all()
    assert quantiles.scaled_quantile.eq(1).all()
    future = sums([20], items=('C',))
    future['residual_scale'] = 6.75
    future[MODELS] = 6.75
    target = apply_scaled_targets(future, quantiles, BUSINESS)
    assert target.safety_stock_lightgbm.iloc[0] == 6.75
    assert target.target_lightgbm.iloc[0] == 14
    future['sales'] = 99999
    assert apply_scaled_targets(future, quantiles, BUSINESS).target_lightgbm.equals(target.target_lightgbm)


def test_scaled_quantiles_do_not_pool_different_folds_or_tiers():
    first = sums([1])
    more = first.copy()
    more['volume_tier'] = 'high'
    more['sales'] = more.lightgbm + more.residual_scale * 3
    later = first.copy()
    later['fold'] = 2
    later['sales'] = later.lightgbm + later.residual_scale * 5
    quantiles = fit_scaled_quantiles(pd.concat([first, more, later]), BUSINESS)
    values = quantiles[quantiles.method.eq('lightgbm')].set_index(['fold', 'volume_tier']).scaled_quantile
    assert values.loc[(1, 'low')] == 1 and values.loc[(1, 'high')] == 3 and values.loc[(2, 'low')] == 5


def test_validation_splits_leave_complete_protection_periods_between_fit_and_checks():
    original = sums(range(100, 156))
    first, inner, selection, second, splits = validation_blocks(original)
    assert first.origin_day.nunique() == 28 and inner.origin_day.nunique() == 14
    assert selection.origin_day.nunique() == 6 and second.origin_day.nunique() == 3
    assert inner.target_end_day.max() < selection.origin_day.min()
    assert first.target_end_day.max() < second.origin_day.min()
    chosen = select_on_first_split(inner, selection, BUSINESS)
    quantiles = fit_scaled_quantiles(first, BUSINESS)
    original.loc[original.origin_day.ge(136), ['sales', *MODELS]] = 99999
    changed_first, changed_inner, changed_selection, _, _ = validation_blocks(original)
    pd.testing.assert_frame_equal(quantiles, fit_scaled_quantiles(changed_first, BUSINESS))
    pd.testing.assert_frame_equal(chosen, select_on_first_split(changed_inner, changed_selection, BUSINESS))


@pytest.mark.parametrize('defect', ['future_validation', 'missing_tier', 'wrong_costs'])
def test_scaled_targets_reject_unavailable_or_leaking_calibration(defect):
    quantiles = fit_scaled_quantiles(sums([1]), BUSINESS)
    future = sums([20])
    if defect == 'future_validation': quantiles['calibration_target_end'] = 20
    if defect == 'missing_tier': future['volume_tier'] = 'high'
    if defect == 'wrong_costs': quantiles['quantile'] = 0.8
    with pytest.raises(ValueError): apply_scaled_targets(future, quantiles, BUSINESS)


def test_saved_forecasts_are_unchanged_and_second_validation_precedes_test_orders():
    path = ROOT / 'results/forecast_sha256.json'
    if not path.exists(): pytest.skip('Run safety-stock calibration first')
    hashes = json.loads(path.read_text())
    for filename, digest in hashes.items():
        assert hashlib.sha256((ROOT / 'results' / filename).read_bytes()).hexdigest() == digest
    plan = json.loads((ROOT / 'results/calibration_plan.json').read_text())
    assert plan['second_validation_checked_before_test'] and not plan['refit_on_second_split']
    assert plan['completed_test_simulation_calls'] == 9
    splits = pd.read_csv(ROOT / 'results/validation_splits.csv').set_index('fold')
    for window in json.loads((ROOT / 'results/summary.json').read_text())['windows']:
        row = splits.loc[window['fold']]
        assert row.first_target_end < row.second_start
        assert row.second_end + 8 < window['warmup_start']
