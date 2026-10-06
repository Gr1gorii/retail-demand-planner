import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.demand import flag_stockout_runs
from src.inventory import MODELS, calculate_targets, protection_totals, simulate_replenishment
from src.metrics import forecast_metrics, weekly_forecasts
from tests.test_pipeline import BUSINESS


ROOT = Path(__file__).resolve().parents[1]
RULE = {'minimum_zero_run': 7, 'history_days': 90, 'minimum_mean_sales': 1.0, 'minimum_positive_share': 0.8}


def forecasts(origins, fold=1):
    records = []
    for origin in origins:
        for horizon in range(1, 10):
            records.append({'fold': fold, 'item_id': 'A', 'dept_id': 'FOODS_1', 'origin_day': origin,
                            'day': origin + horizon - 1, 'forecast_horizon_days': horizon,
                            'sales': 2.0, 'price_known': 10.0, **{model: 2.0 for model in MODELS}})
    return pd.DataFrame(records)


def test_safety_stock_uses_residuals_of_the_nine_day_sum():
    calibration = protection_totals(forecasts(range(1, 5)))
    calibration['sales'] = [9.0, 18.0, 27.0, 36.0]
    target = protection_totals(forecasts([20]))
    target['sales'] = 10000
    result = calculate_targets(target, calibration, BUSINESS)
    q = 0.5 / (0.5 + 0.25 * 0.7 * 7 / 365)
    expected = np.quantile(np.array([-9, 0, 9, 18]), q)
    assert result.safety_stock_lightgbm.iloc[0] == pytest.approx(expected)
    assert result.target_lightgbm.iloc[0] == np.ceil(18 + expected)
    target['sales'] = 0
    assert calculate_targets(target, calibration, BUSINESS).target_lightgbm.equals(result.target_lightgbm)


def test_safety_stock_does_not_pool_different_folds():
    calibration = pd.concat([protection_totals(forecasts([1], 1)), protection_totals(forecasts([1], 2))])
    calibration.loc[calibration.fold.eq(2), 'sales'] = 90
    target = protection_totals(pd.concat([forecasts([20], 1), forecasts([20], 2)]))
    assert calculate_targets(target, calibration, BUSINESS).target_lightgbm.tolist() == [18, 90]


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'wrong_target', 'future_validation'])
def test_incomplete_or_leaking_protection_forecasts_fail(defect):
    rows = forecasts([20])
    calibration = protection_totals(forecasts([1]))
    if defect == 'missing': rows = rows[rows.forecast_horizon_days.ne(9)]
    if defect == 'duplicate': rows = pd.concat([rows, rows.iloc[[0]]])
    if defect == 'wrong_target': rows.loc[0, 'day'] += 1
    if defect == 'future_validation': calibration['target_end_day'] = 20
    with pytest.raises(ValueError):
        calculate_targets(protection_totals(rows), calibration, BUSINESS)


def test_two_day_delivery_weekly_review_and_inventory_position():
    rows = pd.DataFrame({'fold': 1, 'item_id': 'A', 'day': range(20, 34), 'sales': 2.0, 'sell_price': 10.0, 'price_known': 10.0, **{model: 2.0 for model in MODELS}})
    targets = calculate_targets(protection_totals(forecasts([20, 27])), protection_totals(forecasts([1])), BUSINESS)
    window = {'fold': 1, 'warmup_start': 20, 'test_start': 27, 'test_end': 33}
    scored = simulate_replenishment(rows, targets, BUSINESS, [window])
    assert scored.order_lightgbm.tolist() == [18, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0]
    assert scored.receipt_lightgbm.tolist() == [0, 0, 18, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0]
    assert scored.closing_lightgbm.iloc[6] == 8
    assert scored.position_lightgbm.iloc[7] == 8
    assert scored.on_order_lightgbm.iloc[7] == 10
    assert scored.shortage_lightgbm.sum() == 4
    assert scored.warmup.sum() == 7
    assert scored.loc[~scored.warmup, 'shortage_lightgbm'].sum() == 0
    assert scored.loss_lightgbm.sum() == pytest.approx(20 + 110 * 10 * 0.7 * 0.25 / 365)
    for model in MODELS:
        previous = scored[f'closing_{model}'].shift().fillna(0)
        np.testing.assert_allclose(previous + scored[f'receipt_{model}'] - scored[f'sold_{model}'], scored[f'closing_{model}'])
        assert scored[f'order_{model}'].equals(scored.order_lightgbm)


def test_open_orders_are_deducted_from_order_up_to_level():
    business = {**BUSINESS, 'lead_time_days': 8, 'review_period_days': 1}
    rows = pd.DataFrame({'fold': 1, 'item_id': 'A', 'day': range(20, 30), 'sales': 0.0, 'sell_price': 10.0, 'price_known': 10.0, **{model: 2.0 for model in MODELS}})
    targets = calculate_targets(protection_totals(forecasts(range(20, 30))), protection_totals(forecasts([1])), business)
    scored = simulate_replenishment(rows, targets, business, [{'fold': 1, 'warmup_start': 20, 'test_start': 20, 'test_end': 29}])
    assert scored.order_lightgbm.sum() == 18
    assert scored.receipt_lightgbm.iloc[8] == 18
    assert scored.position_lightgbm.iloc[1] == 18


def test_weekly_wape_preserves_item_grain_and_fixed_review_origin():
    rows = forecasts([20, 27])
    rows.loc[rows.forecast_horizon_days.le(7), 'sales'] = 1.0
    weeks = weekly_forecasts(rows, [{'fold': 1, 'test_start': 20, 'test_end': 33}])
    assert len(weeks) == 2 and weeks.sales.eq(7).all()
    assert forecast_metrics(weeks)['wape_lightgbm'] == 1


def test_stockout_flags_do_not_use_declining_history_inside_the_run():
    regular = np.full(160, 2.0)
    regular[100:140] = 0
    regular[149:155] = 0
    intermittent = np.where(np.arange(160) % 2 == 0, 2.0, 0.0)
    intermittent[100:140] = 0
    rows = pd.DataFrame({'item_id': ['A'] * 160 + ['B'] * 160, 'day': np.tile(np.arange(1, 161), 2), 'date': np.tile(pd.date_range('2020-01-01', periods=160), 2), 'sales': np.r_[regular, intermittent], 'price_known': 2.0})
    flags = flag_stockout_runs(rows, RULE)
    assert flags.query("item_id == 'A' and day >= 101 and day <= 140").likely_stockout.all()
    assert flags.likely_stockout.sum() == 40
    rows.price_known = np.nan
    assert not flag_stockout_runs(rows, RULE).likely_stockout.any()


def test_saved_replenishment_results_match_independent_cost_and_stock_equations():
    summary = json.loads((ROOT / 'results/summary.json').read_text())
    if summary.get('protection_period_days') != 9: pytest.skip('Run the replenishment pipeline first')
    rows = pd.read_parquet(ROOT / 'results/simulation.parquet')
    full = pd.read_parquet(ROOT / 'results/simulation_with_warmup.parquet').sort_values(['fold', 'item_id', 'day'])
    config = summary['config']['business']
    assert summary['currency'] == 'USD' and 'usd_to_eur' not in config
    assert len(rows) == 160944 and rows.likely_stockout.sum() == 2915
    for model in MODELS:
        loss = rows.price_usd * (rows[f'shortage_{model}'] * (config['gross_margin'] + config['shortage_penalty_share']) + rows[f'closing_{model}'] * (1 - config['gross_margin']) * config['annual_holding_rate'] / 365)
        assert loss.sum() == pytest.approx(summary[f'loss_{model}'])
        previous = full.groupby(['fold', 'item_id'])[f'closing_{model}'].shift().fillna(0)
        np.testing.assert_allclose(previous + full[f'receipt_{model}'] - full[f'sold_{model}'], full[f'closing_{model}'])
        assert full.loc[~full.review_day, f'order_{model}'].eq(0).all()
        assert ((rows[model] - rows.sales).sum() / rows.sales.sum()) == pytest.approx(summary[f'bias_{model}'])
    grid = pd.read_csv(ROOT / 'results/sensitivity_windows.csv')
    assert len(grid) == 9 * 5 * 3
    assert set(grid.fold) == {0, 1, 2, 3, 4}
    horizon = pd.read_csv(ROOT / 'results/feature_audit.csv')
    assert horizon.minimum_lag_days.ge(horizon.forecast_horizon_days).all()
