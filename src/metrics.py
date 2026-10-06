import numpy as np
import pandas as pd

from src.inventory import MODELS


def forecast_metrics(rows, prefix=''):
    result = {}
    demand = rows.sales.sum()
    for model in MODELS:
        error = rows[model] - rows.sales
        result[f'{prefix}mae_{model}'] = float(error.abs().mean())
        result[f'{prefix}wape_{model}'] = float(error.abs().sum() / demand) if demand else np.nan
        result[f'{prefix}bias_{model}'] = float(error.sum() / demand) if demand else np.nan
    return result


def summarize(rows):
    result = {'sku': int(rows.item_id.nunique()), 'rows': len(rows), 'demand': float(rows.sales.sum()), **forecast_metrics(rows)}
    for model in MODELS:
        for field in ['loss', 'shortage_cost', 'holding_cost', 'shortage']:
            result[f'{field}_{model}'] = float(rows[f'{field}_{model}'].sum())
        result[f'fill_rate_{model}'] = float(1 - result[f'shortage_{model}'] / result['demand']) if result['demand'] else np.nan
        if f'closing_{model}' in rows:
            result[f'average_stock_{model}'] = float(rows[f'closing_{model}'].mean())
            result[f'ordered_units_{model}'] = float(rows[f'order_{model}'].sum())
    result['savings'] = result['loss_baseline'] - result['loss_lightgbm']
    result['savings_share'] = result['savings'] / result['loss_baseline'] if result['loss_baseline'] else np.nan
    result['savings_moving_average'] = result['loss_moving_average'] - result['loss_lightgbm']
    result['savings_share_moving_average'] = result['savings_moving_average'] / result['loss_moving_average'] if result['loss_moving_average'] else np.nan
    return result


def grouped_metrics(rows, column):
    return pd.DataFrame([{column: name, **summarize(group)} for name, group in rows.groupby(column, observed=True)])


def weekly_forecasts(predictions, windows):
    parts = []
    for window in windows:
        reviews = range(window['test_start'], window['test_end'] + 1, 7)
        rows = predictions[predictions.fold.eq(window['fold']) & predictions.origin_day.isin(reviews) & predictions.forecast_horizon_days.le(7)]
        keys = ['fold', 'item_id', 'origin_day']
        counts = rows.groupby(keys, observed=True).size()
        if counts.ne(7).any() or rows.sales.isna().any():
            raise ValueError('Weekly evaluation needs seven known targets per item and review')
        parts.append(rows.groupby(keys, observed=True)[['sales', *MODELS]].sum().reset_index())
    return pd.concat(parts, ignore_index=True)


def result_table(scored, weekly):
    records = []
    for fold in [0] + sorted(scored.fold.unique().tolist()):
        rows = scored if fold == 0 else scored[scored.fold.eq(fold)]
        weeks = weekly if fold == 0 else weekly[weekly.fold.eq(fold)]
        metrics = {**summarize(rows), **forecast_metrics(weeks, 'weekly_')}
        for model in MODELS:
            records.append({'fold': fold, 'method': model, 'daily_wape': metrics[f'wape_{model}'],
                            'weekly_wape': metrics[f'weekly_wape_{model}'], 'bias': metrics[f'bias_{model}'],
                            'weekly_bias': metrics[f'weekly_bias_{model}'], 'loss': metrics[f'loss_{model}'],
                            'shortage_cost': metrics[f'shortage_cost_{model}'], 'holding_cost': metrics[f'holding_cost_{model}'],
                            'fill_rate': metrics[f'fill_rate_{model}']})
    return pd.DataFrame(records)


def review_cycles(targets, windows, review_period_days=7):
    parts = []
    for window in windows:
        reviews = range(window['test_start'], window['test_end'] + 1, review_period_days)
        parts.append(targets[targets.fold.eq(window['fold']) & targets.origin_day.isin(reviews)])
    cycles = pd.concat(parts, ignore_index=True).copy()
    if cycles.duplicated(['fold', 'item_id', 'origin_day']).any():
        raise ValueError('Review cycles must be unique per item, window and origin')
    if not cycles.protection_period_days.eq(9).all() or not cycles.target_end_day.eq(cycles.origin_day + 8).all():
        raise ValueError('Protection diagnostics require nine-day sums from one origin')
    cycles['complete'] = cycles.sales.notna()
    for model in MODELS:
        cycles[f'exceeded_{model}'] = cycles.sales.gt(cycles[f'target_{model}']).astype('boolean').where(cycles.complete)
    return cycles


def protection_diagnostics(cycles, scored, quantile):
    if 'warmup' in scored:
        scored = scored[~scored.warmup]
    records = []
    for fold in [0] + sorted(scored.fold.unique().tolist()):
        reviews = cycles if fold == 0 else cycles[cycles.fold.eq(fold)]
        complete = reviews[reviews.complete]
        rows = scored if fold == 0 else scored[scored.fold.eq(fold)]
        errors = forecast_metrics(complete)
        for model in MODELS:
            exceeded = int(complete[f'exceeded_{model}'].sum())
            rate = exceeded / len(complete) if len(complete) else np.nan
            records.append({'fold': fold, 'method': model, 'protection_wape': errors[f'wape_{model}'],
                            'protection_bias': errors[f'bias_{model}'], 'complete_cycles': len(complete),
                            'excluded_cycles': len(reviews) - len(complete), 'exceeded_cycles': exceeded,
                            'exceedance': rate, 'coverage': 1 - rate, 'target_coverage': quantile,
                            'target_exceedance': 1 - quantile,
                            **{field: float(rows[f'{field}_{model}'].sum()) for field in ['shortage_cost', 'holding_cost', 'loss']}})
    return pd.DataFrame(records)


def coverage_table(cycles, quantile):
    records = []
    for fold in [0] + sorted(cycles.fold.unique().tolist()):
        rows = cycles if fold == 0 else cycles[cycles.fold.eq(fold)]
        for model in MODELS:
            complete = rows[rows.sales.notna()]
            count = int(complete.sales.gt(complete[f'target_{model}']).sum())
            rate = count / len(complete) if len(complete) else np.nan
            records.append({'fold': fold, 'method': model, 'complete_cycles': len(complete),
                            'excluded_cycles': len(rows) - len(complete), 'exceeded_cycles': count,
                            'exceedance': rate, 'coverage': 1 - rate,
                            'target_coverage': quantile, 'target_exceedance': 1 - quantile})
    return pd.DataFrame(records)


def coverage_segments(cycles, quantile):
    parts = []
    for dimension in ['volume_tier', 'zero_share_band']:
        for segment, rows in cycles.groupby(dimension, observed=True):
            table = coverage_table(rows, quantile)
            table['dimension'] = dimension
            table['segment'] = segment
            means = rows[rows.sales.notna()].groupby('fold').recent_zero_share.mean()
            table['mean_zero_share'] = table.fold.map(means)
            table.loc[table.fold.eq(0), 'mean_zero_share'] = rows[rows.sales.notna()].recent_zero_share.mean()
            parts.append(table)
    return pd.concat(parts, ignore_index=True)
