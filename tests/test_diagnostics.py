import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inventory import MODELS, protection_totals
from src.metrics import protection_diagnostics, review_cycles


ROOT = Path(__file__).resolve().parents[1]


def test_nine_day_errors_allow_cancellation_only_within_an_item_cycle():
    records = []
    for item, actual in [('A', 2), ('B', 1)]:
        for horizon in range(1, 10):
            records.append({'fold': 1, 'item_id': item, 'dept_id': 'FOODS_1', 'origin_day': 20,
                            'day': 19 + horizon, 'forecast_horizon_days': horizon, 'sales': actual,
                            'price_known': 10, 'baseline': 1 if item == 'A' else 2,
                            'moving_average': actual, 'lightgbm': actual + (1 if horizon == 1 else -1 if horizon == 2 else 0)})
    targets = protection_totals(pd.DataFrame(records))
    for model in MODELS: targets[f'target_{model}'] = 18
    cycles = review_cycles(targets, [{'fold': 1, 'test_start': 20, 'test_end': 26}])
    scored = pd.DataFrame({'fold': [1], **{f'{field}_{model}': [0] for model in MODELS for field in ['loss', 'shortage_cost', 'holding_cost']}})
    table = protection_diagnostics(cycles, scored, 0.9).query('fold == 1').set_index('method')
    assert table.loc['baseline', 'protection_wape'] == pytest.approx(18 / 27)
    assert table.loc['baseline', 'protection_bias'] == 0
    assert table.loc['lightgbm', 'protection_wape'] == 0
    assert table.loc['moving_average', 'protection_wape'] == 0


def test_coverage_excludes_unknown_cycles_and_counts_equality_as_covered():
    targets = pd.DataFrame({'fold': 1, 'item_id': 'A', 'origin_day': [13, 20, 21, 27],
                            'target_end_day': [21, 28, 29, 35], 'protection_period_days': 9,
                            'sales': [999, 18, 999, np.nan], **{model: 18 for model in MODELS},
                            **{f'target_{model}': 18 for model in MODELS}})
    cycles = review_cycles(targets, [{'fold': 1, 'test_start': 20, 'test_end': 33}])
    assert cycles.origin_day.tolist() == [20, 27]
    assert cycles.exceeded_lightgbm.iloc[0] == False
    assert pd.isna(cycles.exceeded_lightgbm.iloc[1])
    scored = pd.DataFrame({'fold': [1, 1], 'warmup': [True, False],
                           **{f'{field}_{model}': [999, value] for model in MODELS for field, value in [('shortage_cost', 3), ('holding_cost', 2), ('loss', 5)]}})
    table = protection_diagnostics(cycles, scored, 0.9)
    assert table.complete_cycles.eq(1).all() and table.excluded_cycles.eq(1).all()
    assert table.coverage.eq(1).all() and table.exceedance.eq(0).all()
    assert table.target_coverage.eq(0.9).all()
    np.testing.assert_allclose(table.target_exceedance, 0.1)
    assert table.shortage_cost.eq(3).all() and table.holding_cost.eq(2).all() and table.loss.eq(5).all()


@pytest.mark.parametrize('defect', ['duplicate', 'wrong_period'])
def test_review_cycles_reject_ambiguous_sums(defect):
    rows = pd.DataFrame({'fold': 1, 'item_id': 'A', 'origin_day': 20, 'target_end_day': 28,
                         'protection_period_days': 9, 'sales': 18, **{model: 18 for model in MODELS},
                         **{f'target_{model}': 18 for model in MODELS}}, index=[0])
    if defect == 'duplicate': rows = pd.concat([rows, rows])
    if defect == 'wrong_period': rows['protection_period_days'] = 7
    with pytest.raises(ValueError):
        review_cycles(rows, [{'fold': 1, 'test_start': 20, 'test_end': 26}])


def test_saved_diagnostics_match_daily_targets_and_unchanged_loss_totals():
    summary = json.loads((ROOT / 'results/summary.json').read_text())
    daily = pd.read_parquet(ROOT / 'results/predictions.parquet')
    targets = pd.read_parquet(ROOT / 'results/targets.parquet').set_index(['fold', 'item_id', 'origin_day'])
    table = pd.read_csv(ROOT / 'results/protection_diagnostics.csv')
    parts = []
    for window in summary['windows']:
        rows = daily[daily.fold.eq(window['fold']) & daily.origin_day.isin(range(window['test_start'], window['test_end'] + 1, 7))]
        sums = rows.groupby(['fold', 'item_id', 'origin_day'], observed=True)[['sales', *MODELS]].sum(min_count=9)
        parts.append(sums)
        valid = sums.dropna(subset=['sales'])
        result = table[table.fold.eq(window['fold'])].set_index('method')
        assert result.complete_cycles.eq(5748 if window['fold'] < 4 else 4311).all()
        for model in MODELS:
            error = valid[model] - valid.sales
            assert result.loc[model, 'protection_wape'] == pytest.approx(error.abs().sum() / valid.sales.sum())
            assert result.loc[model, 'protection_bias'] == pytest.approx(error.sum() / valid.sales.sum())
            expected = valid.sales.gt(targets.loc[valid.index, f'target_{model}']).mean()
            assert result.loc[model, 'exceedance'] == pytest.approx(expected)
    pooled = pd.concat(parts).dropna(subset=['sales'])
    result = table[table.fold.eq(0)].set_index('method')
    for model in MODELS:
        error = pooled[model] - pooled.sales
        assert result.loc[model, 'protection_wape'] == pytest.approx(error.abs().sum() / pooled.sales.sum())
        for field in ['shortage_cost', 'holding_cost', 'loss']:
            assert result.loc[model, field] == pytest.approx(summary[f'{field}_{model}'])
