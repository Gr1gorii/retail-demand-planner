import json
import hashlib
import shutil
import time
from pathlib import Path

import pandas as pd

from src.calibration import apply_scaled_targets, attach_levels, fit_scaled_quantiles, select_on_first_split, validation_blocks
from src.inventory import MODELS, critical_ratio, protection_totals, simulate_replenishment
from src.metrics import coverage_segments, coverage_table, forecast_metrics, grouped_metrics, protection_diagnostics, result_table, review_cycles, summarize


def recalibrate(output, levels):
    started = time.time()
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    if 'safety_stock_calibration' in summary:
        raise ValueError('Calibration already run. Use a fresh pipeline output to start another experiment.')
    config = summary['config']
    business = config['business']
    settings = {'method': 'tier_scaled', 'scale_days': 28, 'scale_floor_units': 1.0,
                'first_origins': 28, 'gap_origins': 8, 'second_origins': 20,
                'selection_fit_origins': 14, 'selection_score_origins': 6}
    unchanged_files = list((output / 'models').glob('*.txt')) + [output / f'{name}.parquet' for name in
                      ['predictions', 'calibration', 'daily_forecasts', 'protection_forecasts', 'protection_calibration', 'weekly_forecasts', 'future', 'future_calibration', 'daily_sensitivity']]
    hashes = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in unchanged_files}
    (output / 'forecast_sha256.json').write_text(json.dumps(hashes, indent=2))
    frames = {name: pd.read_parquet(output / f'{name}.parquet') for name in
              ['daily_forecasts', 'protection_forecasts', 'protection_calibration', 'weekly_forecasts', 'future', 'future_calibration', 'targets']}
    calibration = pd.concat([frames['protection_calibration'], protection_totals(frames['future_calibration'])], ignore_index=True)
    calibration = attach_levels(calibration, levels)
    first, inner, selection, second, splits = validation_blocks(calibration)
    forecasts = attach_levels(frames['protection_forecasts'], levels)
    future = attach_levels(protection_totals(frames['future']), levels)
    selection_table = select_on_first_split(inner, selection, business)
    scenarios, checks, quantile_tables = [], [], []
    for penalty in config['cost_grid']['shortage_penalties']:
        for rate in config['cost_grid']['annual_holding_rates']:
            case = {**business, 'shortage_penalty_share': penalty, 'annual_holding_rate': rate}
            quantiles = fit_scaled_quantiles(first, case)
            validation = apply_scaled_targets(second, quantiles, case)
            check = coverage_table(validation[validation.fold.ne(0)], critical_ratio(case))
            for table in [check, quantiles]:
                table['annual_holding_rate'] = rate
                table['shortage_penalty_share'] = penalty
            checks.append(check)
            quantile_tables.append(quantiles)
            scenarios.append((case, quantiles))
    splits.to_csv(output / 'validation_splits.csv', index=False)
    selection_table.to_csv(output / 'validation_selection.csv', index=False)
    pd.concat(checks, ignore_index=True).to_csv(output / 'validation_check.csv', index=False)
    pd.concat(quantile_tables, ignore_index=True).to_csv(output / 'calibration_quantiles.csv', index=False)
    levels.to_parquet(output / 'safety_levels.parquet', index=False)
    manifest = {'settings': settings, 'first_validation_selection': selection_table.to_dict('records'),
                'second_validation_checked_before_test': True, 'refit_on_second_split': False,
                'planned_test_simulation_calls': 9, 'forecast_source_sha256': summary['source_sha256']}
    (output / 'calibration_plan.json').write_text(json.dumps(manifest, indent=2))
    archive = output / 'uncalibrated'
    archive.mkdir(exist_ok=True)
    archived_names = ['summary.json', 'targets.parquet', 'simulation.parquet', 'simulation_with_warmup.parquet',
                      'review_cycles.parquet', 'protection_diagnostics.csv', 'window_results.csv', 'sensitivity.csv',
                      'sensitivity_windows.csv', 'recommendations.csv', 'by_fold.csv', 'by_item.csv']
    for name in archived_names:
        if not (archive / name).exists(): shutil.copy2(output / name, archive / name)
    old_cycles = review_cycles(attach_levels(frames['targets'], levels), summary['windows'])
    coverage_segments(old_cycles, summary['critical_ratio']).to_csv(archive / 'coverage_segments.csv', index=False)
    grid, details = [], []
    for case, quantiles in scenarios:
        targets = apply_scaled_targets(forecasts, quantiles[quantiles.fold.ne(0)], case)
        full = simulate_replenishment(frames['daily_forecasts'], targets, case, summary['windows'])
        scored = full[~full.warmup].copy()
        cycles = review_cycles(targets, summary['windows'])
        diagnostics = protection_diagnostics(cycles, scored, critical_ratio(case))
        table = result_table(scored, frames['weekly_forecasts']).merge(
            diagnostics.drop(columns=['loss', 'shortage_cost', 'holding_cost']), on=['fold', 'method'], validate='one_to_one')
        shortage = business['gross_margin'] + case['shortage_penalty_share']
        holding = case['annual_holding_rate'] * (1 - business['gross_margin']) / 365
        fields = {'annual_holding_rate': case['annual_holding_rate'], 'shortage_penalty_share': case['shortage_penalty_share'],
                  'shortage_cost_share': shortage, 'holding_cost_share': holding,
                  'overage_cost_share': holding * 7, 'cost_ratio': shortage / (holding * 7), 'quantile': critical_ratio(case)}
        pooled = diagnostics[diagnostics.fold.eq(0)].set_index('method')
        grid.append({**fields, **summarize(scored), **forecast_metrics(frames['weekly_forecasts'], 'weekly_'),
                     **{f'exceedance_{model}': float(pooled.loc[model, 'exceedance']) for model in MODELS}})
        for key, value in fields.items(): table[key] = value
        details.append(table)
        print(f"Scored annual rate {case['annual_holding_rate']:.0%}, penalty {case['shortage_penalty_share']:.0%}", flush=True)
        if case == business:
            for name, rows in [('targets', targets), ('simulation', scored), ('simulation_with_warmup', full), ('review_cycles', cycles)]:
                rows.to_parquet(output / f'{name}.parquet', index=False)
            diagnostics.to_csv(output / 'protection_diagnostics.csv', index=False)
            result_table(scored, frames['weekly_forecasts']).to_csv(output / 'window_results.csv', index=False)
            coverage_segments(cycles, critical_ratio(case)).to_csv(output / 'coverage_segments.csv', index=False)
            for column in ['fold', 'volume', 'pattern', 'dept_id', 'is_event', 'likely_stockout']:
                grouped_metrics(scored, column).to_csv(output / f'by_{column}.csv', index=False)
            grouped_metrics(scored, 'item_id').sort_values('savings_moving_average').to_csv(output / 'by_item.csv', index=False)
            scored.groupby('date', observed=True)[['sales', *MODELS, 'loss_baseline', 'loss_moving_average', 'loss_lightgbm', 'savings', 'savings_moving_average']].sum().reset_index().to_csv(output / 'daily.csv', index=False)
            metrics = summarize(scored)
            summary.update(metrics)
            summary['censoring']['unflagged_metrics'] = summarize(scored[~scored.likely_stockout])
            (output / 'censoring_summary.json').write_text(json.dumps(summary['censoring'], indent=2))
            summary['safety_stock_calibration'] = {**settings, 'second_validation_checked_before_test': True,
                                                  'test_exceedance': pooled.exceedance.to_dict(),
                                                  'coverage_misses_target': bool(pooled.exceedance.gt(1 - critical_ratio(case)).any()),
                                                  'selected_methods': selection_table[selection_table.selected].set_index('fold').method.to_dict()}
            folds = grouped_metrics(scored, 'fold')
            summary['gain_in_every_window_vs_both'] = bool((folds.savings.gt(0) & folds.savings_moving_average.gt(0)).all())
    pd.DataFrame(grid).to_csv(output / 'sensitivity.csv', index=False)
    pd.concat(details, ignore_index=True).to_csv(output / 'sensitivity_windows.csv', index=False)
    future_targets = apply_scaled_targets(future, fit_scaled_quantiles(first[first.fold.eq(0)], business), business)
    selected = selection_table.loc[selection_table.fold.eq(0) & selection_table.selected, 'method'].iloc[0]
    future_targets['method'] = selected
    future_targets['on_hand'] = 0
    future_targets['on_order'] = 0
    future_targets['order_qty'] = future_targets[f'target_{selected}']
    future_targets.sort_values('order_qty', ascending=False).to_csv(output / 'recommendations.csv', index=False)
    summary['config']['safety_stock'] = settings
    summary['calibration_runtime_seconds'] = round(time.time() - started, 1)
    assert all(hashlib.sha256((output / path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    manifest['completed_test_simulation_calls'] = len(scenarios)
    (output / 'calibration_plan.json').write_text(json.dumps(manifest, indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
