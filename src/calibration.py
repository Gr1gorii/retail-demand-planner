import numpy as np
import pandas as pd

from src.inventory import MODELS, critical_ratio, protection_period


def recent_levels(panel):
    rows = panel.sort_values(['item_id', 'day']).copy()
    groups = rows.groupby('item_id', observed=True).sales
    rows['recent_ma28'] = groups.transform(lambda sales: sales.shift(1).rolling(28).mean())
    rows['recent_zero_share'] = groups.transform(lambda sales: sales.eq(0).shift(1).rolling(28).mean())
    rows['volume_tier'] = pd.cut(rows.recent_ma28, [-1, 1, 5, np.inf], labels=['low', 'medium', 'high']).astype(str)
    rows['zero_share_band'] = pd.cut(rows.recent_zero_share, [-1e-9, 0, 0.25, 0.5, 0.75, 1],
                                    labels=['0%', '(0%, 25%]', '(25%, 50%]', '(50%, 75%]', '(75%, 100%]']).astype(str)
    return rows[['item_id', 'day', 'recent_ma28', 'recent_zero_share', 'volume_tier', 'zero_share_band']].rename(columns={'day': 'origin_day'})


def attach_levels(sums, levels, scale_floor=1.0):
    result = sums.merge(levels, on=['item_id', 'origin_day'], validate='many_to_one')
    if len(result) != len(sums) or result.recent_ma28.isna().any() or result.recent_zero_share.isna().any():
        raise ValueError('Every origin needs 28 fully observed prior sales days')
    if scale_floor <= 0:
        raise ValueError('The residual scale floor must be positive')
    result['residual_scale'] = (result.recent_ma28 * 9).clip(lower=scale_floor)
    return result


def validation_blocks(calibration):
    first, selection_fit, selection_check, second, dates = [], [], [], [], []
    for fold, rows in calibration.groupby('fold', observed=True):
        origins = np.sort(rows.origin_day.unique())
        if len(origins) != 56 or not np.all(np.diff(origins) == 1):
            raise ValueError('Calibration requires 56 consecutive validation origins')
        first.append(rows[rows.origin_day.isin(origins[:28])])
        selection_fit.append(rows[rows.origin_day.isin(origins[:14])])
        selection_check.append(rows[rows.origin_day.isin(origins[22:28])])
        second.append(rows[rows.origin_day.isin(origins[36:56:7])])
        dates.append({'fold': int(fold), 'first_start': int(origins[0]), 'first_end': int(origins[27]),
                      'first_target_end': int(origins[27] + 8), 'selection_fit_end': int(origins[13]),
                      'selection_start': int(origins[22]), 'selection_end': int(origins[27]),
                      'second_start': int(origins[36]), 'second_end': int(origins[55]),
                      'second_review_origins': ','.join(map(str, origins[36:56:7]))})
    return tuple(pd.concat(parts, ignore_index=True) for parts in [first, selection_fit, selection_check, second]) + (pd.DataFrame(dates),)


def fit_scaled_quantiles(first, business):
    if first.sales.isna().any() or first.protection_period_days.ne(protection_period(business)).any():
        raise ValueError('Scaled residuals need fully observed protection sums')
    ratio = critical_ratio(business)
    records = []
    for model in MODELS:
        rows = first.assign(scaled_error=(first.sales - first[model]) / first.residual_scale)
        for (fold, tier), group in rows.groupby(['fold', 'volume_tier'], observed=True):
            records.append({'fold': int(fold), 'volume_tier': tier, 'method': model,
                            'scaled_quantile': float(group.scaled_error.quantile(ratio)), 'quantile': ratio,
                            'samples': len(group), 'items': int(group.item_id.nunique()),
                            'calibration_target_end': int(group.target_end_day.max())})
    return pd.DataFrame(records)


def apply_scaled_targets(forecasts, quantiles, business):
    if forecasts.protection_period_days.ne(protection_period(business)).any():
        raise ValueError('Targets must cover the protection period')
    if not np.isfinite(forecasts.residual_scale).all() or forecasts.residual_scale.le(0).any():
        raise ValueError('Residual scales must be finite and positive')
    if not np.allclose(quantiles['quantile'], critical_ratio(business), rtol=1e-12, atol=1e-12):
        raise ValueError('Safety-stock quantiles do not match the business costs')
    result = forecasts.copy()
    for model in MODELS:
        selected = quantiles[quantiles.method.eq(model)]
        matched = result[['fold', 'volume_tier', 'origin_day']].merge(selected, on=['fold', 'volume_tier'], how='left', validate='many_to_one')
        if matched.scaled_quantile.isna().any() or matched.calibration_target_end.ge(matched.origin_day).any():
            raise ValueError('Missing tier calibration or validation outcomes reach an order origin')
        correction = matched.scaled_quantile.to_numpy() * result.residual_scale.to_numpy()
        result[f'safety_stock_{model}'] = correction
        result[f'target_{model}'] = np.ceil(np.maximum(result[model] + correction, 0) - 1e-9).astype(int)
        result.loc[result.price_known.isna(), f'target_{model}'] = 0
    return result


def select_on_first_split(selection_fit, selection_check, business):
    quantiles = fit_scaled_quantiles(selection_fit, business)
    targets = apply_scaled_targets(selection_check, quantiles, business)
    ratio = critical_ratio(business)
    records = []
    for fold, rows in targets.groupby('fold', observed=True):
        for model in MODELS:
            error = rows.sales - rows[f'target_{model}']
            pinball = np.where(error >= 0, ratio * error, (ratio - 1) * error)
            records.append({'fold': int(fold), 'method': model, 'mean_pinball_units': float(pinball.mean()), 'cycles': len(rows)})
    table = pd.DataFrame(records)
    winners = table.loc[table.groupby('fold').mean_pinball_units.idxmin(), ['fold', 'method']].set_index('fold').method
    table['selected'] = table.method.eq(table.fold.map(winners))
    return table
