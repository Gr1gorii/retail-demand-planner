import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features import FEATURES, assign_segments, feature_availability, make_features


def make_windows(last_day, config):
    protection = config['business']['lead_time_days'] + config['business']['review_period_days']
    feature_availability(protection)
    warmup = config.get('warmup_days', 28)
    first = last_day - config['folds'] * config['fold_days'] + 1
    windows = []
    for fold in range(1, config['folds'] + 1):
        start = first + (fold - 1) * config['fold_days']
        warmup_start = start - warmup
        calibration_end = warmup_start - protection
        calibration_start = calibration_end - config['calibration_days'] + 1
        windows.append({
            'fold': fold, 'train_start': calibration_start - config['train_days'],
            'train_end': calibration_start - 1,
            'calibration_start': calibration_start, 'calibration_end': calibration_end,
            'calibration_target_end': warmup_start - 1,
            'warmup_start': warmup_start, 'test_start': start,
            'test_end': start + config['fold_days'] - 1,
        })
    if windows[0]['train_start'] <= 36:
        raise ValueError('Not enough history for shifted features and validation')
    return windows


def fit_model(train, config):
    if train.sales.isna().any():
        raise ValueError('Training targets must be observed before the forecast origin')
    model = LGBMRegressor(
        objective='poisson', n_estimators=config['trees'], learning_rate=0.05,
        num_leaves=31, min_child_samples=100, colsample_bytree=0.9,
        reg_lambda=1, random_state=config['seed'], n_jobs=4, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    model.fit(train[FEATURES], train.sales)
    return model


def predict(model, frame, origins, fold, horizon, segments):
    feature_availability(horizon)
    rows = frame[frame.day.isin(np.asarray(origins) + horizon - 1)]
    cols = ['item_id', 'dept_id', 'day', 'date', 'sales', 'sell_price', 'price_known', 'is_event']
    result = rows[cols].copy()
    result['fold'] = fold
    result['origin_day'] = result.day - horizon + 1
    result['forecast_horizon_days'] = horizon
    result['baseline'] = rows.seasonal_naive.to_numpy()
    result['moving_average'] = rows.moving_average_28.to_numpy()
    result['lightgbm'] = np.maximum(model.predict(rows[FEATURES]), 0)
    result.loc[result.price_known.isna(), ['baseline', 'moving_average', 'lightgbm']] = 0
    return result.merge(segments, on='item_id', validate='many_to_one')


def run_backtest(panel, config, model_dir):
    last = int(panel.loc[panel.sales.notna(), 'day'].max())
    windows = make_windows(last, config)
    next_config = {**config, 'folds': 1, 'fold_days': 1}
    future_window = make_windows(last + 1, next_config)[0]
    future_window['fold'] = 0
    predictions, calibrations, future, future_calibration, importance = [], [], [], [], []
    for horizon in range(1, 10):
        print(f'Building features for h={horizon}', flush=True)
        frame = make_features(panel, horizon)
        for window in windows + [future_window]:
            fold = window['fold']
            train = frame[frame.day.between(window['train_start'], window['train_end'])]
            segments = assign_segments(panel[panel.day < window['warmup_start']])
            print(f'h={horizon}, fold={fold}: training {len(train):,} rows', flush=True)
            model = fit_model(train, config)
            calibration = predict(model, frame, range(window['calibration_start'], window['calibration_end'] + 1), fold, horizon, segments)
            origins = [last + 1] if fold == 0 else range(window['warmup_start'], window['test_end'] + 1)
            forecast = predict(model, frame, origins, fold, horizon, segments)
            if fold == 0:
                future.append(forecast)
                future_calibration.append(calibration)
            else:
                predictions.append(forecast)
                calibrations.append(calibration)
                importance.append(pd.DataFrame({'feature': FEATURES, 'gain': model.booster_.feature_importance(importance_type='gain'), 'fold': fold, 'horizon': horizon}))
            model.booster_.save_model(str(model_dir / f'fold_{fold}_h_{horizon}.txt'))
        del frame
    return (pd.concat(predictions, ignore_index=True), pd.concat(calibrations, ignore_index=True),
            pd.concat(future, ignore_index=True), pd.concat(future_calibration, ignore_index=True),
            pd.concat(importance, ignore_index=True), windows)
