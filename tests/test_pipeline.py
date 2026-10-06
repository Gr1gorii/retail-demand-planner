import numpy as np
import pandas as pd
import pytest

from src.backtest import make_windows
from src.features import FEATURES, assign_segments, feature_availability, make_features
from src.inventory import critical_ratio, protection_period


BUSINESS = {'lead_time_days': 2, 'review_period_days': 7, 'gross_margin': 0.3, 'shortage_penalty_share': 0.2, 'annual_holding_rate': 0.25}


def sample_panel():
    days = np.arange(1, 101)
    return pd.DataFrame({'item_id': ['A'] * 100 + ['B'] * 100, 'dept_id': 'FOODS_1',
                         'day': np.tile(days, 2), 'sales': np.r_[days, days * 2].astype(float),
                         'sell_price': np.r_[np.full(100, 2.0), np.full(100, 3.0)],
                         'date': np.tile(pd.date_range('2020-01-01', periods=100), 2),
                         'wday': 1, 'month': 1, 'year': 2020, 'snap': 0, 'event': 'none', 'is_event': 0})


@pytest.mark.parametrize('horizon', range(1, 10))
def test_direct_features_stop_before_each_forecast_origin(horizon):
    panel = sample_panel()
    original = make_features(panel, horizon)
    panel.loc[panel.day >= 60, ['sales', 'sell_price']] = 9999
    changed = make_features(panel, horizon)
    pd.testing.assert_frame_equal(original.loc[original.day < 60 + horizon, FEATURES], changed.loc[changed.day < 60 + horizon, FEATURES])
    row = original.query("item_id == 'A' and day == @target", local_dict={'target': 59 + horizon}).iloc[0]
    assert row.lag_1 == 59
    assert row.lag_7 == 53
    assert row.mean_7 == 56
    assert row.moving_average_28 == 45.5
    assert row.seasonal_naive == 59 + horizon - 7 * int(np.ceil(horizon / 7))
    assert original.query("item_id == 'B' and day == 1").lag_1.isna().all()
    audit = feature_availability(horizon)
    assert audit.minimum_lag_days.ge(horizon).all()
    assert 'likely_stockout' not in FEATURES


@pytest.mark.parametrize('horizon', [0, 10, 1.5])
def test_invalid_direct_horizons_fail(horizon):
    with pytest.raises(ValueError, match='1 to 9'):
        make_features(sample_panel(), horizon)


@pytest.mark.parametrize('horizon', [1, 9])
def test_price_fill_does_not_use_future_prices(horizon):
    panel = sample_panel()
    panel.loc[(panel.item_id == 'A') & (panel.day < 40), 'sell_price'] = np.nan
    frame = make_features(panel, horizon)
    assert frame.query("item_id == 'A' and day < @first", local_dict={'first': 40 + horizon}).price_known.isna().all()
    assert frame.query("item_id == 'A' and day == @first", local_dict={'first': 40 + horizon}).price_known.iloc[0] == 2


def test_windows_finish_validation_before_warmup_and_score_the_same_days():
    config = {'folds': 4, 'fold_days': 28, 'calibration_days': 56, 'train_days': 730, 'warmup_days': 28, 'business': BUSINESS}
    windows = make_windows(1941, config)
    days = []
    for window in windows:
        assert window['train_end'] < window['calibration_start']
        assert window['calibration_end'] + 8 < window['warmup_start']
        assert window['calibration_target_end'] == window['warmup_start'] - 1
        assert window['train_end'] - window['train_start'] + 1 == 730
        assert window['calibration_end'] - window['calibration_start'] + 1 == 56
        assert window['test_start'] - window['warmup_start'] == 28
        days.extend(range(window['test_start'], window['test_end'] + 1))
    assert len(days) == len(set(days)) == 112 and max(days) == 1941


def test_segments_ignore_future_observations():
    panel = sample_panel()
    original = assign_segments(panel[panel.day < 60])
    panel.loc[panel.day >= 60, 'sales'] = 0
    pd.testing.assert_frame_equal(original, assign_segments(panel[panel.day < 60]))


@pytest.mark.parametrize('values', [{'lead_time_days': -1}, {'review_period_days': 0}, {'review_period_days': 8}, {'annual_holding_rate': 0}, {'gross_margin': 1}, {'shortage_penalty_share': -1}])
def test_invalid_business_inputs_fail(values):
    with pytest.raises(ValueError):
        critical_ratio({**BUSINESS, **values})


def test_ratio_uses_one_review_period_of_daily_holding_cost():
    assert protection_period(BUSINESS) == 9
    assert critical_ratio(BUSINESS) == pytest.approx(0.5 / (0.5 + 7 * 0.25 * 0.7 / 365))
