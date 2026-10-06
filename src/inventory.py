import numpy as np
import pandas as pd


MODELS = ['baseline', 'moving_average', 'lightgbm']


def protection_period(business):
    lead = business['lead_time_days']
    review = business['review_period_days']
    if not isinstance(lead, int) or not isinstance(review, int) or lead < 0 or review < 1 or lead + review > 9:
        raise ValueError('Lead time must be nonnegative and review positive, with protection at most 9 days')
    return lead + review


def critical_ratio(business):
    penalty = business['shortage_penalty_share']
    if penalty < 0:
        raise ValueError('Shortage penalty must be nonnegative')
    if 'annual_holding_rate' in business:
        protection_period(business)
        margin = business['gross_margin']
        rate = business['annual_holding_rate']
        if not 0 < margin < 1 or rate <= 0:
            raise ValueError('Gross margin must be between 0 and 1 and annual carrying rate positive')
        underage = margin + penalty
        overage = rate * (1 - margin) * business['review_period_days'] / 365
    else:
        purchase = business['purchase_share']
        if not 0 < purchase < 1:
            raise ValueError('Purchase cost must be between 0 and 1')
        underage = 1 - purchase + penalty
        salvage = business['salvage_share']
        if not 0 <= salvage < purchase:
            raise ValueError('Salvage must be nonnegative and below purchase cost')
        overage = purchase - salvage
    return underage / (underage + overage)


def protection_totals(predictions, period=9):
    if not 1 <= period <= 9:
        raise ValueError('Protection must be from 1 to 9 days')
    keys = ['fold', 'item_id', 'origin_day']
    required = keys + ['day', 'forecast_horizon_days', 'sales', 'price_known', *MODELS]
    if not set(required).issubset(predictions):
        raise ValueError('Protection forecasts need explicit origins, targets and horizons')
    rows = predictions[predictions.forecast_horizon_days.le(period)].copy()
    if rows.duplicated(keys + ['forecast_horizon_days']).any():
        raise ValueError('Duplicate horizon at a forecast origin')
    if rows.day.ne(rows.origin_day + rows.forecast_horizon_days - 1).any():
        raise ValueError('Target day does not match the forecast origin and horizon')
    counts = rows.groupby(keys, observed=True).forecast_horizon_days.agg(['count', 'min', 'max'])
    if counts['count'].ne(period).any() or counts['min'].ne(1).any() or counts['max'].ne(period).any():
        raise ValueError('Every order needs all horizons covering lead time plus review period')
    if not np.isfinite(rows[MODELS].to_numpy()).all() or rows[MODELS].lt(0).any().any():
        raise ValueError('Forecasts must be finite and nonnegative')
    sums = rows.groupby(keys, observed=True)[['sales', *MODELS]].sum(min_count=period).reset_index()
    first = rows[rows.forecast_horizon_days.eq(1)][keys + ['price_known', 'dept_id']]
    sums = sums.merge(first, on=keys, validate='one_to_one')
    sums['target_end_day'] = sums.origin_day + period - 1
    sums['protection_period_days'] = period
    return sums


def calculate_targets(forecasts, calibration, business):
    period = protection_period(business)
    for frame in [forecasts, calibration]:
        if 'protection_period_days' not in frame or frame.protection_period_days.ne(period).any():
            raise ValueError('The summed forecast must cover the complete protection period')
    if calibration.sales.isna().any():
        raise ValueError('Safety stock needs fully observed validation sums')
    if 'target_end_day' not in calibration or 'origin_day' not in forecasts:
        raise ValueError('Safety stock needs explicit validation target end dates and order origins')
    latest = calibration.groupby('fold').target_end_day.max()
    earliest = forecasts.groupby('fold').origin_day.min()
    if latest.reindex(earliest.index).isna().any() or latest.reindex(earliest.index).ge(earliest).any():
        raise ValueError('Every validation sum must finish before the first order origin')
    result = forecasts.copy()
    ratio = critical_ratio(business)
    for model in MODELS:
        residuals = calibration.assign(error=calibration.sales - calibration[model])
        offsets = residuals.groupby(['fold', 'item_id'], observed=True).error.quantile(ratio)
        keys = result.set_index(['fold', 'item_id']).index
        correction = offsets.reindex(keys).to_numpy()
        if not np.isfinite(correction).all():
            raise ValueError('Validation residuals are missing for some items')
        result[f'safety_stock_{model}'] = correction
        target = np.maximum(result[model].to_numpy() + correction, 0)
        result[f'target_{model}'] = np.ceil(target - 1e-9).astype(int)
        result.loc[result.price_known.isna(), f'target_{model}'] = 0
    return result


def simulate_replenishment(daily, targets, business, windows):
    period = protection_period(business)
    if targets.protection_period_days.ne(period).any():
        raise ValueError('Order targets do not cover the protection period')
    critical_ratio(business)
    lead, review = business['lead_time_days'], business['review_period_days']
    results = []
    for window in windows:
        start, end, fold = window['warmup_start'], window['test_end'], window['fold']
        rows = daily[daily.fold.eq(fold) & daily.day.between(start, end)].sort_values(['day', 'item_id']).copy()
        items = sorted(rows.item_id.unique())
        days = end - start + 1
        if len(rows) != days * len(items) or rows.duplicated(['day', 'item_id']).any() or rows.sales.isna().any():
            raise ValueError('Simulation needs one observed sale per item and day')
        if (window['test_start'] - start) % review:
            raise ValueError('Warm-up must contain complete review cycles')
        price = rows.sell_price.fillna(rows.price_known).fillna(0).to_numpy().reshape(days, -1)
        demand = rows.sales.to_numpy().reshape(days, -1)
        if (price < 0).any() or ((price == 0) & (demand > 0)).any():
            raise ValueError('Positive observed sales need a valid USD price')
        selected = targets[targets.fold.eq(fold)].set_index(['origin_day', 'item_id'])
        for model in MODELS:
            on_hand = np.zeros(len(items))
            scheduled = np.zeros((days + lead + 1, len(items)))
            fields = {field: np.zeros((days, len(items))) for field in ['opening', 'receipt', 'order', 'position', 'closing', 'on_order', 'sold', 'shortage', 'shortage_cost', 'holding_cost', 'loss', 'target', 'safety_stock']}
            for index in range(days):
                day = start + index
                received = scheduled[index].copy()
                on_hand += received
                fields['opening'][index] = on_hand
                pending = scheduled[index + 1:].sum(axis=0)
                position = on_hand + pending
                fields['position'][index] = position
                if index % review == 0:
                    keys = pd.MultiIndex.from_product([[day], items], names=['origin_day', 'item_id'])
                    decision = selected.reindex(keys)
                    if decision[f'target_{model}'].isna().any():
                        raise ValueError('Missing full-protection forecast on a review day')
                    target = decision[f'target_{model}'].to_numpy()
                    quantity = np.maximum(target - position, 0)
                    fields['order'][index] = quantity
                    fields['target'][index] = target
                    fields['safety_stock'][index] = decision[f'safety_stock_{model}'].to_numpy()
                    scheduled[index + lead] += quantity
                    if lead == 0:
                        on_hand += quantity
                        received += quantity
                sold = np.minimum(on_hand, demand[index])
                shortage = demand[index] - sold
                on_hand -= sold
                fields['receipt'][index] = received
                fields['closing'][index] = on_hand
                fields['on_order'][index] = scheduled[index + 1:].sum(axis=0)
                fields['sold'][index] = sold
                fields['shortage'][index] = shortage
                fields['shortage_cost'][index] = shortage * price[index] * (business['gross_margin'] + business['shortage_penalty_share'])
                fields['holding_cost'][index] = on_hand * price[index] * (1 - business['gross_margin']) * business['annual_holding_rate'] / 365
                fields['loss'][index] = fields['shortage_cost'][index] + fields['holding_cost'][index]
            for field, values in fields.items():
                rows[f'{field}_{model}'] = values.reshape(-1)
        rows['price_usd'] = price.reshape(-1)
        rows['review_day'] = (rows.day - start) % review == 0
        rows['warmup'] = rows.day < window['test_start']
        rows['savings'] = rows.loss_baseline - rows.loss_lightgbm
        rows['savings_moving_average'] = rows.loss_moving_average - rows.loss_lightgbm
        results.append(rows)
    return pd.concat(results, ignore_index=True)


def calculate_orders(predictions, calibration, business):
    if business.get('lead_time_days', 0) != 0 or business.get('review_period_days', 1) != 1:
        raise ValueError('The legacy daily policy requires lead time 0 and review period 1')
    for rows in [predictions, calibration]:
        if 'forecast_horizon_days' not in rows or rows.forecast_horizon_days.ne(1).any():
            raise ValueError('The daily policy requires explicit forecast horizon 1')
    result = predictions.copy()
    ratio = critical_ratio(business)
    for model in MODELS:
        errors = calibration.assign(error=calibration.sales - calibration[model])
        offsets = errors.groupby(['fold', 'item_id'], observed=True).error.quantile(ratio)
        correction = offsets.reindex(result.set_index(['fold', 'item_id']).index).to_numpy()
        if not np.isfinite(correction).all():
            raise ValueError('Missing calibration errors')
        result[f'adjustment_{model}'] = correction
        result[f'order_{model}'] = np.floor(np.maximum(result[model] + correction, 0) + 0.5 + 1e-9).astype(int)
        result.loc[result.price_known.isna(), f'order_{model}'] = 0
    return result


def simulate(orders, business):
    result = orders.copy()
    price = result.sell_price.fillna(result.price_known).fillna(0)
    if result.sales.isna().any() or (price < 0).any():
        raise ValueError('The daily simulation needs observed sales and valid USD prices')
    result['price_usd'] = price
    for model in MODELS:
        quantity = result[f'order_{model}']
        shortage = np.maximum(result.sales - quantity, 0)
        excess = np.maximum(quantity - result.sales, 0)
        result[f'shortage_{model}'] = shortage
        result[f'excess_{model}'] = excess
        result[f'shortage_cost_{model}'] = shortage * price * (1 - business['purchase_share'] + business['shortage_penalty_share'])
        result[f'holding_cost_{model}'] = excess * price * (business['purchase_share'] - business['salvage_share'])
        result[f'loss_{model}'] = result[f'shortage_cost_{model}'] + result[f'holding_cost_{model}']
    result['savings'] = result.loss_baseline - result.loss_lightgbm
    result['savings_moving_average'] = result.loss_moving_average - result.loss_lightgbm
    return result
