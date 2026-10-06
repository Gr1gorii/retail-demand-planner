import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.backtest import fit_model, make_windows, predict, run_backtest
from src.calibration import recent_levels
from src.data import load_data, source_hashes
from src.demand import censoring_summary, flag_stockout_runs
from src.features import assign_segments, feature_availability, make_features
from src.inventory import MODELS, calculate_orders, calculate_targets, critical_ratio, protection_period, protection_totals, simulate, simulate_replenishment
from src.metrics import forecast_metrics, grouped_metrics, protection_diagnostics, result_table, review_cycles, summarize, weekly_forecasts
from src.recalibrate import recalibrate


def sensitivity_grid(daily, forecasts, calibration, weekly, business, windows, grid):
    results, detailed = [], []
    for penalty in grid['shortage_penalties']:
        for rate in grid['annual_holding_rates']:
            scenario = {**business, 'shortage_penalty_share': penalty, 'annual_holding_rate': rate}
            shortage = business['gross_margin'] + penalty
            holding = rate * (1 - business['gross_margin']) / 365
            targets = calculate_targets(forecasts, calibration, scenario)
            simulation = simulate_replenishment(daily, targets, scenario, windows)
            scored = simulation[~simulation.warmup]
            weekly_metrics = forecast_metrics(weekly, 'weekly_')
            fields = {'shortage_cost_share': shortage, 'holding_cost_share': holding,
                      'annual_holding_rate': rate, 'shortage_penalty_share': penalty,
                      'overage_cost_share': holding * scenario['review_period_days'],
                      'cost_ratio': shortage / (holding * scenario['review_period_days']),
                      'quantile': critical_ratio(scenario)}
            results.append({**fields, **summarize(scored), **weekly_metrics})
            table = result_table(scored, weekly)
            for key, value in fields.items():
                table[key] = value
            detailed.append(table)
    return pd.DataFrame(results), pd.concat(detailed, ignore_index=True)


def daily_reference(panel, config, model_dir):
    daily_config = {**config, 'warmup_days': 0, 'business': config['daily_sensitivity']}
    windows = make_windows(int(panel.loc[panel.sales.notna(), 'day'].max()), daily_config)
    frame = make_features(panel, 1)
    forecasts, validations = [], []
    for window in windows:
        print(f"Daily sensitivity, fold={window['fold']}", flush=True)
        train = frame[frame.day.between(window['train_start'], window['train_end'])]
        model = fit_model(train, config)
        segments = assign_segments(panel[panel.day < window['test_start']])
        forecasts.append(predict(model, frame, range(window['test_start'], window['test_end'] + 1), window['fold'], 1, segments))
        validations.append(predict(model, frame, range(window['calibration_start'], window['calibration_end'] + 1), window['fold'], 1, segments))
        model.booster_.save_model(str(model_dir / f"daily_fold_{window['fold']}.txt"))
    predictions, calibration = pd.concat(forecasts), pd.concat(validations)
    rows = simulate(calculate_orders(predictions, calibration, config['daily_sensitivity']), config['daily_sensitivity'])
    weekly = rows.groupby(['fold', 'item_id', rows.groupby(['fold', 'item_id']).cumcount() // 7], observed=True)[['sales', *MODELS]].sum().reset_index()
    return rows, result_table(rows, weekly)


def run(raw_dir, config_path, output_dir):
    started = time.time()
    config = json.loads(Path(config_path).read_text())
    if config['forecast_horizons'] != list(range(1, 10)):
        raise ValueError('This experiment requires all direct horizons from 1 to 9')
    period = protection_period(config['business'])
    if period != 9 or config['business']['lead_time_days'] != 2 or config['business']['review_period_days'] != 7:
        raise ValueError('The main experiment requires two-day lead time and weekly review')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / 'models'
    model_dir.mkdir(exist_ok=True)
    print(f"Loading {config['store']} / {config['category']}", flush=True)
    panel = load_data(raw_dir, config['store'], config['category'], 9)
    known = make_features(panel, 1)
    diagnostics = flag_stockout_runs(known, config['stockout_check'])
    del known
    predictions, calibration, future, future_calibration, importance, windows = run_backtest(panel, config, model_dir)
    predictions = predictions.merge(diagnostics.drop(columns='date'), on=['item_id', 'day'], validate='many_to_one')
    daily = predictions[predictions.forecast_horizon_days.eq(1)].copy()
    forecasts = protection_totals(predictions)
    calibration_sums = protection_totals(calibration)
    targets = calculate_targets(forecasts, calibration_sums, config['business'])
    full = simulate_replenishment(daily, targets, config['business'], windows)
    scored = full[~full.warmup].copy()
    weekly = weekly_forecasts(predictions, windows)
    table = result_table(scored, weekly)
    cycles = review_cycles(targets, windows, config['business']['review_period_days'])
    protection_diagnostics(cycles, scored, critical_ratio(config['business'])).to_csv(output / 'protection_diagnostics.csv', index=False)
    cycles.to_parquet(output / 'review_cycles.parquet', index=False)
    grid, grid_windows = sensitivity_grid(daily, forecasts, calibration_sums, weekly, config['business'], windows, config['cost_grid'])
    for name, rows in [('predictions', predictions), ('calibration', calibration), ('daily_forecasts', daily), ('protection_forecasts', forecasts), ('protection_calibration', calibration_sums), ('targets', targets), ('simulation', scored), ('simulation_with_warmup', full), ('weekly_forecasts', weekly), ('future', future), ('future_calibration', future_calibration)]:
        rows.to_parquet(output / f'{name}.parquet', index=False)
    table.to_csv(output / 'window_results.csv', index=False)
    grid.to_csv(output / 'sensitivity.csv', index=False)
    grid_windows.to_csv(output / 'sensitivity_windows.csv', index=False)
    pd.concat([feature_availability(horizon) for horizon in range(1, 10)]).to_csv(output / 'feature_audit.csv', index=False)
    importance.to_csv(output / 'importance.csv', index=False)
    for column in ['fold', 'volume', 'pattern', 'dept_id', 'is_event', 'likely_stockout']:
        grouped_metrics(scored, column).to_csv(output / f'by_{column}.csv', index=False)
    grouped_metrics(scored, 'item_id').sort_values('savings_moving_average').to_csv(output / 'by_item.csv', index=False)
    scored.groupby('date', observed=True)[['sales', *MODELS, 'loss_baseline', 'loss_moving_average', 'loss_lightgbm', 'savings', 'savings_moving_average']].sum().reset_index().to_csv(output / 'daily.csv', index=False)
    horizon_results = []
    for window in windows:
        rows = predictions[predictions.fold.eq(window['fold']) & predictions.origin_day.between(window['test_start'], window['test_end']) & predictions.day.le(window['test_end'])]
        for horizon, values in rows.groupby('forecast_horizon_days'):
            metrics = forecast_metrics(values)
            for model in MODELS:
                horizon_results.append({'fold': window['fold'], 'horizon': int(horizon), 'method': model, 'rows': len(values), 'wape': metrics[f'wape_{model}'], 'bias': metrics[f'bias_{model}']})
    pd.DataFrame(horizon_results).to_csv(output / 'by_horizon.csv', index=False)
    censoring = {**censoring_summary(scored), 'rule': config['stockout_check'], 'unflagged_metrics': summarize(scored[~scored.likely_stockout])}
    (output / 'censoring_summary.json').write_text(json.dumps(censoring, indent=2))
    pd.DataFrame([{'fold': fold, **censoring_summary(rows)} for fold, rows in scored.groupby('fold')]).to_csv(output / 'censoring_by_fold.csv', index=False)
    scored.loc[scored.likely_stockout, ['fold', 'item_id', 'day', 'date', 'zero_run_days', 'pre_run_mean', 'sales']].to_csv(output / 'stockout_flags.csv', index=False)
    future_sums = protection_totals(future)
    future_validation = protection_totals(future_calibration)
    recommendations = calculate_targets(future_sums, future_validation, config['business'])
    recommendations['on_hand'] = 0
    recommendations['on_order'] = 0
    recommendations['order_qty'] = recommendations.target_lightgbm
    recommendations.sort_values('order_qty', ascending=False).to_csv(output / 'recommendations.csv', index=False)
    legacy, legacy_table = daily_reference(panel, config, model_dir)
    legacy.to_parquet(output / 'daily_sensitivity.parquet', index=False)
    legacy_table.rename(columns={'holding_cost': 'excess_cost', 'weekly_wape': 'weekly_aggregated_h1_wape', 'weekly_bias': 'weekly_aggregated_h1_bias'}).to_csv(output / 'daily_sensitivity.csv', index=False)
    dates = panel.drop_duplicates('day').set_index('day').date
    for window in windows:
        for key in list(window):
            if key != 'fold':
                window[key + '_date'] = str(dates.loc[window[key]].date())
    fold_metrics = grouped_metrics(scored, 'fold')
    summary = {**summarize(scored), **forecast_metrics(weekly, 'weekly_'),
               'store': config['store'], 'category': config['category'],
               'start': str(scored.date.min().date()), 'end': str(scored.date.max().date()),
               'forecast_date': str(dates.loc[int(future.origin_day.iloc[0])].date()),
               'currency': 'USD', 'forecast_horizons': list(range(1, 10)), 'protection_period_days': 9,
               'critical_ratio': critical_ratio(config['business']), 'windows': windows, 'config': config,
               'source_sha256': source_hashes(raw_dir), 'censoring': censoring,
               'daily_sensitivity': summarize(legacy),
               'gain_in_every_window_vs_both': bool((fold_metrics.savings.gt(0) & fold_metrics.savings_moving_average.gt(0)).all()),
               'zero_sales_share': float(scored.sales.eq(0).mean()), 'missing_test_prices': int(scored.sell_price.isna().sum()),
               'runtime_seconds': round(time.time() - started, 1)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    levels = recent_levels(panel)
    levels = levels[levels.origin_day.between(windows[0]['calibration_start'], int(future.origin_day.max()))]
    summary = recalibrate(output, levels)
    print(json.dumps({key: summary[key] for key in ['savings', 'savings_moving_average', 'wape_lightgbm', 'weekly_wape_lightgbm', 'runtime_seconds']}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    run(args.raw_dir, args.config, args.output)
