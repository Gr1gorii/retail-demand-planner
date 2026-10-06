import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.i18n import number, translate
from src.calibration import apply_scaled_targets, attach_levels, fit_scaled_quantiles, validation_blocks
from src.inventory import MODELS, critical_ratio, protection_totals, simulate_replenishment
from src.metrics import coverage_segments, forecast_metrics, grouped_metrics, protection_diagnostics, result_table, review_cycles, summarize


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
BLUE, GRAY, ORANGE = '#255885', '#83909c', '#ba7638'
st.set_page_config(page_title='CA_1 / FOODS', layout='wide', initial_sidebar_state='collapsed')
_, language_column = st.columns([3, 1])
with language_column:
    language_name = st.selectbox('Language / Lingua', ['English', 'Italiano'], index=1 if st.query_params.get('lang') == 'it' else 0, key='language')
language = 'it' if language_name == 'Italiano' else 'en'
language_changed = st.session_state.get('active_language') != language
st.session_state['active_language'] = language
st.query_params['lang'] = language


def t(text):
    return translate(text, language)


def money(value):
    return f'$ {number(value, language=language)}'


def percent(value):
    return f'{number(value * 100, 1, language)}%'


def decimal(value):
    return number(value, 2, language)


def translated_selectbox(label, options, key, labels):
    widget_key = f'{key}_{language}'
    if language_changed:
        st.session_state[widget_key] = st.session_state.get(key, options[0])
    selected = st.selectbox(t(label), options, format_func=lambda value: labels.get(value, value), key=widget_key)
    st.session_state[key] = selected
    return selected


@st.cache_data
def load_results(updated):
    frames = {name: pd.read_parquet(RESULTS / f'{name}.parquet') for name in ['daily_forecasts', 'protection_forecasts', 'protection_calibration', 'weekly_forecasts', 'future', 'future_calibration', 'safety_levels']}
    return json.loads((RESULTS / 'summary.json').read_text()), frames


@st.cache_data
def score_scenario(daily, forecasts, calibration, levels, business, windows):
    first, _, _, _, _ = validation_blocks(attach_levels(calibration, levels))
    targets = apply_scaled_targets(attach_levels(forecasts, levels), fit_scaled_quantiles(first, business), business)
    full = simulate_replenishment(daily, targets, business, windows)
    return full[~full.warmup], targets


def style_chart(figure, height=340):
    figure.update_layout(template='plotly_white', height=height, margin=dict(l=15, r=20, t=25, b=15), font=dict(family='Arial', size=13, color='#202a35'), legend=dict(orientation='h', y=1.14, x=0), hovermode='x unified', separators=',.' if language == 'it' else '.,')
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor='#e7eaee', zerolinecolor='#c9cfd6')
    return figure


def show_metrics_table(frame):
    display = frame.rename(columns={'fold': t('Window'), 'method': t('Method'), 'daily_wape': t('Daily WAPE'), 'weekly_wape': t('Weekly WAPE'), 'bias': t('Daily bias'), 'weekly_bias': t('Weekly bias'), 'loss': t('Loss, USD'), 'shortage_cost': t('Shortage, USD'), 'holding_cost': t('Holding, USD'), 'fill_rate': t('Fill rate')})
    display[t('Method')] = display[t('Method')].map(method_labels)
    formats = {t(name): percent for name in ['Daily WAPE', 'Weekly WAPE', 'Daily bias', 'Weekly bias', 'Fill rate']}
    formats.update({t(name): decimal for name in ['Loss, USD', 'Shortage, USD', 'Holding, USD']})
    st.dataframe(display.style.format(formats), hide_index=True, width='stretch')


def show_protection_tables(frame):
    names = {'fold': t('Window'), 'method': t('Method'), 'protection_wape': t('9-day WAPE'),
             'protection_bias': t('9-day bias'), 'shortage_cost': t('Shortage, USD'), 'holding_cost': t('Holding, USD'),
             'loss': t('Loss, USD'), 'complete_cycles': t('Complete cycles'), 'excluded_cycles': t('Excluded cycles'),
             'exceeded_cycles': t('Exceeding cycles'), 'exceedance': t('Demand > target'), 'coverage': t('Coverage'),
             'target_coverage': t('Target service'), 'target_exceedance': t('Target exceedance')}
    display = frame.rename(columns=names)
    display[t('Window')] = display[t('Window')].map(lambda value: t('All') if value == 0 else str(value))
    display[t('Method')] = display[t('Method')].map(method_labels)
    for fields in [['fold', 'method', 'protection_wape', 'protection_bias', 'shortage_cost', 'holding_cost', 'loss'],
                   ['fold', 'method', 'complete_cycles', 'excluded_cycles', 'exceeded_cycles', 'exceedance', 'coverage', 'target_coverage', 'target_exceedance']]:
        formats = {names[field]: decimal for field in ['shortage_cost', 'holding_cost', 'loss'] if field in fields}
        formats.update({names[field]: lambda value: f'{number(value * 100, 2, language)}%' for field in ['protection_wape', 'protection_bias', 'exceedance', 'coverage', 'target_coverage', 'target_exceedance'] if field in fields})
        st.dataframe(display[[names[field] for field in fields]].style.format(formats), hide_index=True, width='stretch')


if not (RESULTS / 'summary.json').exists():
    st.title(t('Demand forecast and order quantity'))
    st.info(t('Run first: python -m src.pipeline'))
    st.stop()
summary, frames = load_results((RESULTS / 'summary.json').stat().st_mtime_ns)
if summary.get('protection_period_days') != 9:
    st.info(t('The replenishment results are being prepared. Run the updated pipeline first.'))
    st.stop()
business = summary['config']['business'].copy()
with st.sidebar:
    st.header(t('Business assumptions'))
    business['gross_margin'] = st.slider(t('Gross margin'), 0.1, 0.7, business['gross_margin'], 0.05, key='margin')
    business['annual_holding_rate'] = st.slider(t('Annual carrying rate'), 0.05, 1.0, business['annual_holding_rate'], 0.05, key='carrying')
    business['shortage_penalty_share'] = st.slider(t('Shortage penalty'), 0.0, 1.0, business['shortage_penalty_share'], 0.05, key='penalty')
    st.caption(t('Holding cost uses unit cost: annual rate × selling price × (1 - gross margin) / 365.'))
    st.write(t('Safety-stock quantile: {value}').format(value=percent(critical_ratio(business))))
rows, targets = score_scenario(frames['daily_forecasts'], frames['protection_forecasts'], frames['protection_calibration'], frames['safety_levels'], business, summary['windows'])
metrics = {**summarize(rows), **forecast_metrics(frames['weekly_forecasts'], 'weekly_')}
cycles = review_cycles(targets, summary['windows'], business['review_period_days'])
diagnostics = protection_diagnostics(cycles, rows, critical_ratio(business))
method_labels = {'baseline': t('Seasonal naive'), 'moving_average': t('MA28'), 'lightgbm': 'LightGBM'}
colors = {method_labels['baseline']: GRAY, method_labels['moving_average']: ORANGE, 'LightGBM': BLUE}
st.title(t('Demand forecast and order quantity'))
st.caption(t('{store} · {category} · {items} items · {start} - {end} · four 28-day windows').format(store=summary['store'], category=summary['category'], items=number(summary['sku'], language=language), start=summary['start'], end=summary['end']))
st.info(t('Historical simulation in USD. Lead time 2 days, review every 7 days, forecast coverage 9 days. Stock carries forward; deliveries enter stock before opening.'))
effect_tab, order_tab, sensitivity_tab, error_tab, method_tab = st.tabs([t('Performance'), t('Order recommendations'), t('Sensitivity'), t('Error analysis'), t('Method and assumptions')])

with effect_tab:
    st.subheader(t('Replenishment result'))
    best_method = min(MODELS, key=lambda model: metrics[f'loss_{model}'])
    st.write(t('Lowest simulated cost: {method} with tier-scaled safety stock.').format(method=method_labels[best_method]))
    st.caption(t('Forecasts are unchanged. Scaled residuals are pooled by volume tier using the first validation block only. All methods are checked on the second block before test scoring.'))
    columns = st.columns(3)
    columns[0].metric(t('Saved vs naive'), money(metrics['savings']), percent(metrics['savings_share']))
    columns[1].metric(t('LGBM vs MA28'), money(metrics['savings_moving_average']), percent(metrics['savings_share_moving_average']))
    columns[2].metric(t('Best cost'), money(metrics[f'loss_{best_method}']))
    st.warning(t('Main limitation: pooled coverage can hide missed targets by window and intermittency. Default LightGBM exceedance is 0.65% overall vs a 0.67% target, but 0.73% in window 1, 0.90% in window 4 and 2.00% for items with over 75% zero-sales days.'))
    st.caption(t('Costs use the sidebar assumptions. Suspected stockouts: {share} of evaluated item-days. Hidden lost sales are unknown and are not imputed.').format(share=percent(summary['censoring']['flagged_share'])))
    table = result_table(rows, frames['weekly_forecasts'])
    show_metrics_table(table[table.fold.eq(0)].drop(columns='fold'))
    left, right = st.columns(2)
    with left:
        st.subheader(t('Shortage and holding costs, USD'))
        costs = pd.DataFrame([{t('Method'): method_labels[model], t('Cause'): t(label), t('Loss, USD'): metrics[f'{field}_{model}']} for model in MODELS for label, field in [('Shortage', 'shortage_cost'), ('Holding', 'holding_cost')]])
        figure = px.bar(costs, x=t('Method'), y=t('Loss, USD'), color=t('Cause'), barmode='stack', color_discrete_map={t('Shortage'): BLUE, t('Holding'): ORANGE})
        st.plotly_chart(style_chart(figure), width='stretch')
    with right:
        st.subheader(t('Effect by period'))
        folds = grouped_metrics(rows, 'fold')
        gains = folds.melt(id_vars='fold', value_vars=['savings', 'savings_moving_average'], var_name=t('Comparison'), value_name=t('Saved, USD'))
        gains[t('Comparison')] = gains[t('Comparison')].replace({'savings': t('vs seasonal naive'), 'savings_moving_average': t('vs MA28')})
        figure = px.bar(gains, x='fold', y=t('Saved, USD'), color=t('Comparison'), barmode='group', color_discrete_map={t('vs seasonal naive'): GRAY, t('vs MA28'): BLUE}, labels={'fold': t('Window')})
        figure.update_xaxes(dtick=1)
        st.plotly_chart(style_chart(figure), width='stretch')
    st.subheader(t('Results in each window'))
    show_metrics_table(table[table.fold.ne(0)])
    holds = (folds.savings.gt(0) & folds.savings_moving_average.gt(0)).all()
    st.write(t('Loss reduction holds in all four windows against both baselines.') if holds else t('Loss reduction does not hold in every window against both baselines.'))
    st.subheader(t('Nine-day protection diagnostics'))
    st.caption(t('I score item-level h = 1..9 sums issued together at scored weekly reviews. Forecast errors exclude safety stock. Shortage and holding costs retain all 28 scored days in each window.'))
    show_protection_tables(diagnostics)
    st.subheader(t('Coverage by recent demand'))
    st.caption(t('Volume: low <= 1, medium > 1 to 5, high > 5 units/day. Volume and zero-sales share use the 28 observed days before each review. The scale is max(9 × MA28, 1 unit).'))
    segment_dimension = translated_selectbox('Coverage group', ['volume_tier', 'zero_share_band'], 'coverage_group', {'volume_tier': t('Sales volume'), 'zero_share_band': t('Zero-sales share')})
    segment_rows = coverage_segments(cycles, critical_ratio(business))
    segment_rows = segment_rows[segment_rows.fold.eq(0) & segment_rows.dimension.eq(segment_dimension)][['segment', 'method', 'complete_cycles', 'exceedance', 'target_exceedance']]
    segment_rows['segment'] = segment_rows.segment.map(t)
    segment_rows['method'] = segment_rows.method.map(method_labels)
    segment_rows = segment_rows.rename(columns={'segment': t('Segment'), 'method': t('Method'), 'complete_cycles': t('Complete cycles'), 'exceedance': t('Demand > target'), 'target_exceedance': t('Target exceedance')})
    st.dataframe(segment_rows.style.format({t('Demand > target'): lambda value: f'{number(value * 100, 2, language)}%', t('Target exceedance'): lambda value: f'{number(value * 100, 2, language)}%'}), hide_index=True, width='stretch')
    st.caption(t('Coverage = share of complete item-review cycles with observed nine-day demand <= target stock. Demand > target is compared with 1 - the critical ratio. This checks target calibration, not fill rate or the probability of avoiding a stockout.'))
    st.caption(t('Windows 1-3 have 5,748 complete cycles each; their final reviews include two observed days after the window. Window 4 has 4,311 complete cycles and excludes 1,437 cycles from the May 16 review because May 23-24 sales are unknown. Overlapping protection periods are not independent.'))
    st.write(t('Next steps, not implemented: train directly on the nine-day demand sum with quantile loss at the critical ratio; use real stock and delivery data.'))
    st.subheader(t('Daily actuals and forecasts'))
    daily = rows.groupby('date', observed=True)[['sales', *MODELS]].sum().reset_index()
    figure = go.Figure()
    for field, label, color, dash in [('sales', t('Actual sales'), '#202a35', 'solid'), ('baseline', method_labels['baseline'], GRAY, 'dot'), ('moving_average', 'MA28', ORANGE, 'dash'), ('lightgbm', 'LightGBM', BLUE, 'solid')]:
        figure.add_trace(go.Scatter(x=daily.date, y=daily[field], name=label, line=dict(color=color, width=2, dash=dash)))
    figure.update_yaxes(title=t('Sales, units'))
    st.plotly_chart(style_chart(figure), width='stretch')
    st.caption(t('Daily WAPE uses item-days at h = 1. Weekly WAPE uses item-weeks: h = 1..7 forecasts issued together on each review day, then summed before scoring.'))

with order_tab:
    st.subheader(t('Order for {date}').format(date=summary['forecast_date']))
    st.caption(t('Historical example after M5 sales end. LightGBM was selected on the first validation split. Future actual sales are unknown. Enter stock on hand and stock already on order.'))
    department_column, stock_column, pipeline_column = st.columns(3)
    with department_column:
        department = translated_selectbox('Department', ['all'] + sorted(frames['future'].dept_id.astype(str).unique().tolist()), 'department', {'all': t('All')})
    with stock_column:
        stock = st.number_input(t('On hand per item'), min_value=0, max_value=10000, value=0, step=1, key='stock')
    with pipeline_column:
        on_order = st.number_input(t('On order per item'), min_value=0, max_value=10000, value=0, step=1, key='on_order')
    future_first, _, _, _, _ = validation_blocks(attach_levels(protection_totals(frames['future_calibration']), frames['safety_levels']))
    recommendations = apply_scaled_targets(attach_levels(protection_totals(frames['future']), frames['safety_levels']), fit_scaled_quantiles(future_first, business), business)
    recommendations['on_hand'] = stock
    recommendations['on_order'] = on_order
    recommendations['order_qty'] = (recommendations.target_lightgbm - stock - on_order).clip(lower=0)
    if department != 'all': recommendations = recommendations[recommendations.dept_id.eq(department)]
    st.write(t('Order: **{quantity} units** for **{items} items**').format(quantity=number(recommendations.order_qty.sum(), language=language), items=number(len(recommendations), language=language)))
    display = recommendations[['item_id', 'order_qty', 'lightgbm', 'safety_stock_lightgbm', 'target_lightgbm', 'on_hand', 'on_order', 'dept_id']].sort_values('order_qty', ascending=False).rename(columns={'item_id': t('Item'), 'order_qty': t('Order, units'), 'lightgbm': t('9-day forecast'), 'safety_stock_lightgbm': t('Safety stock'), 'target_lightgbm': t('Target stock'), 'on_hand': t('On hand'), 'on_order': t('On order'), 'dept_id': t('Department')})
    st.dataframe(display.style.format({t('9-day forecast'): decimal, t('Safety stock'): decimal}), hide_index=True, width='stretch', height=490)
    csv = display.to_csv(index=False, sep=';' if language == 'it' else ',', decimal=',' if language == 'it' else '.')
    st.download_button(t('Download orders CSV'), csv.encode('utf-8-sig'), f'orders-{language}.csv', 'text/csv')
    st.caption(t('Order = max(0, target stock - on hand - on order). The same entered stock is applied to each selected item.'))

with sensitivity_tab:
    st.subheader(t('Cost sensitivity'))
    st.caption(t('Default margin 30%. Rows vary shortage penalty; columns vary annual carrying rate on unit cost. Every cell recalibrates safety stock and reruns the complete stock simulation.'))
    grid = pd.read_csv(RESULTS / 'sensitivity.csv')
    minimum = min(0, grid[['savings', 'savings_moving_average']].to_numpy().min())
    maximum = grid[['savings', 'savings_moving_average']].to_numpy().max()
    for column, field, label in zip(st.columns(2), ['savings', 'savings_moving_average'], [t('vs seasonal naive'), t('vs MA28')]):
        with column:
            st.write(label)
            matrix = grid.pivot(index='shortage_cost_share', columns='annual_holding_rate', values=field)
            figure = go.Figure(go.Heatmap(z=matrix.to_numpy(), zmin=minimum, zmax=maximum, x=[percent(value) for value in matrix.columns], y=[percent(value) for value in matrix.index], text=matrix.map(money).to_numpy(), texttemplate='%{text}', colorscale='Blues', showscale=False))
            figure.update_xaxes(title=t('Annual carrying rate'), type='category')
            figure.update_yaxes(title=t('Shortage cost / price'), type='category')
            st.plotly_chart(style_chart(figure, 280), width='stretch')
    st.caption(t('The grid is a scenario analysis, not a confidence interval. Grid values retain the default margin and listed penalties when sidebar assumptions change.'))
    st.subheader(t('All methods and windows in the cost grid'))
    detail = pd.read_csv(RESULTS / 'sensitivity_windows.csv')
    carrying = st.selectbox(t('Annual rate in grid'), sorted(detail.annual_holding_rate.unique()), format_func=percent, key='grid_rate')
    penalty = st.selectbox(t('Penalty in grid'), sorted(detail.shortage_penalty_share.unique()), format_func=percent, key='grid_penalty')
    show_metrics_table(detail[detail.annual_holding_rate.eq(carrying) & detail.shortage_penalty_share.eq(penalty)][['fold', 'method', 'daily_wape', 'weekly_wape', 'bias', 'weekly_bias', 'loss', 'shortage_cost', 'holding_cost', 'fill_rate']])
    show_protection_tables(detail[detail.annual_holding_rate.eq(carrying) & detail.shortage_penalty_share.eq(penalty)][['fold', 'method', 'protection_wape', 'protection_bias', 'shortage_cost', 'holding_cost', 'loss', 'complete_cycles', 'excluded_cycles', 'exceeded_cycles', 'exceedance', 'coverage', 'target_coverage', 'target_exceedance']])
    st.subheader(t('Uncalibrated safety stock'))
    st.caption(t('Previous item-specific quantiles used all 56 overlapping validation origins, without scaling or pooling. These preserved default-cost results use exactly the same forecasts. Their lower costs came with much more under-protection.'))
    old = pd.read_csv(RESULTS / 'uncalibrated/protection_diagnostics.csv')
    show_protection_tables(old)
    st.subheader(t('Original h = 1 sensitivity'))
    st.caption(t('Previous daily policy: immediate delivery, review daily, day-end salvage, purchase 60%, salvage 20%, shortage penalty 20%. These USD results use a different stock and cost scenario, so their savings cannot be compared as a pure lead-time effect.'))
    legacy = pd.read_csv(RESULTS / 'daily_sensitivity.csv')
    show_metrics_table(legacy[legacy.fold.eq(0)].drop(columns=['fold', 'shortage_cost', 'excess_cost', 'weekly_aggregated_h1_wape', 'weekly_aggregated_h1_bias']))

with error_tab:
    st.subheader(t('Errors by segment'))
    labels = {'volume': t('Sales volume'), 'pattern': t('Demand pattern'), 'dept_id': t('Department')}
    dimension = translated_selectbox('Group by', list(labels), 'group', labels)
    segments = grouped_metrics(rows, dimension)
    segments[dimension] = segments[dimension].astype(str).map(t)
    chart = segments.melt(id_vars=dimension, value_vars=[f'wape_{model}' for model in MODELS], var_name=t('Method'), value_name='WAPE')
    chart[t('Method')] = chart[t('Method')].replace({f'wape_{model}': label for model, label in method_labels.items()})
    figure = px.bar(chart, x=dimension, y='WAPE', color=t('Method'), barmode='group', color_discrete_map=colors, labels={dimension: t('Segment')})
    figure.update_yaxes(tickformat='.0%')
    st.plotly_chart(style_chart(figure), width='stretch')
    st.caption(t('Segments use the 90 days before warm-up. Model weights and safety-stock quantiles stay fixed; observed sales update the next forecast each day.'))
    st.subheader(t('Effect distribution across items'))
    items = grouped_metrics(rows, 'item_id').sort_values('savings_moving_average')
    figure = px.histogram(items, x='savings_moving_average', nbins=60, log_y=True, labels={'savings_moving_average': t('Saved vs MA28, USD/item')}, color_discrete_sequence=[BLUE])
    figure.add_vline(x=0, line_color='#202a35', line_dash='dash')
    st.plotly_chart(style_chart(figure), width='stretch')
    st.write(t('Losses increased on {items} items versus MA28. Negative effects remain in the total.').format(items=int(items.savings_moving_average.lt(0).sum())))
    worst = items.head(10)[['item_id', 'loss_baseline', 'loss_moving_average', 'loss_lightgbm', 'savings_moving_average']].rename(columns={'item_id': t('Item'), 'loss_baseline': t('Naive loss, USD'), 'loss_moving_average': t('MA28 loss, USD'), 'loss_lightgbm': t('LightGBM loss, USD'), 'savings_moving_average': t('Saved, USD')})
    st.dataframe(worst.style.format(precision=2, decimal=',' if language == 'it' else '.', thousands='.' if language == 'it' else ','), hide_index=True, width='stretch')
    st.subheader(t('Censored demand check'))
    st.write(t('A possible stockout is a zero run of at least 7 days, following 90 days with mean sales at least 1 unit/day and positive sales on at least 80% of days, with a previously known price. Flags are retrospective and never enter forecasts or orders.'))
    censor = pd.read_csv(RESULTS / 'censoring_by_fold.csv')[['fold', 'evaluation_rows', 'flagged_rows', 'flagged_share', 'flagged_items']].rename(columns={'fold': t('Window'), 'evaluation_rows': t('Evaluated item-days'), 'flagged_rows': t('Flagged item-days'), 'flagged_share': t('Flagged share'), 'flagged_items': t('Items')})
    st.dataframe(censor.style.format({t('Flagged share'): percent}), hide_index=True, width='stretch')
    st.caption(t('Shortages use max(observed sales - available stock, 0). Hidden lost sales are not imputed. Flags remain in the main totals and can incur holding cost.'))
    st.subheader(t('Individual item history'))
    item = st.selectbox(t('Item'), sorted(rows.item_id.astype(str).unique()), key='item')
    selected = rows[rows.item_id.eq(item)].sort_values('date')
    figure = go.Figure()
    for field, label, color in [('sales', t('Actual sales'), '#202a35'), ('lightgbm', 'LightGBM h=1', BLUE), ('closing_lightgbm', t('Closing stock'), ORANGE)]:
        figure.add_trace(go.Scatter(x=selected.date, y=selected[field], name=label, line=dict(color=color, width=2)))
    figure.update_yaxes(title=t('Quantity, units'))
    st.plotly_chart(style_chart(figure), width='stretch')

with method_tab:
    st.subheader(t('Forecasts and validation'))
    st.write(t('I train nine direct LightGBM models per window, one for each h = 1..9. All demand and price observations end at t - h. Calendar fields are assumed known in advance. Seasonal naive uses lag 7 for h <= 7 and lag 14 for h = 8, 9. MA28 averages the 28 days known at the origin.'))
    st.write(t('Each window has 730 training targets, 56 validation origins, 28 warm-up days and 28 scored days. The first 28 validation origins fit tier-scaled residual quantiles; an eight-origin gap separates them from a later 20-origin block, checked at three weekly reviews. No second-split or test outcomes fit the quantiles.'))
    st.write(t('Inside the first block, I fit quantiles on 14 origins, leave eight origins for the full target gap, then select the forecast method by mean quantile pinball loss on six later origins. LightGBM is selected in all four windows and the future example. I then fit the correction on all 28 first-block origins and freeze it before the second validation check.'))
    st.dataframe(pd.read_csv(RESULTS / 'validation_check.csv').query('annual_holding_rate == 0.25 and shortage_penalty_share == 0.2')[['fold', 'method', 'complete_cycles', 'exceedance', 'target_exceedance']].assign(method=lambda frame: frame.method.map(method_labels)).rename(columns={'fold': t('Window'), 'method': t('Method'), 'complete_cycles': t('Complete cycles'), 'exceedance': t('Demand > target'), 'target_exceedance': t('Target exceedance')}).style.format({t('Demand > target'): lambda value: f'{number(value * 100, 2, language)}%', t('Target exceedance'): lambda value: f'{number(value * 100, 2, language)}%'}), hide_index=True, width='stretch')
    window = pd.DataFrame(summary['windows'])
    fields = ['fold', 'train_start_date', 'train_end_date', 'calibration_start_date', 'calibration_target_end_date', 'warmup_start_date', 'test_start_date', 'test_end_date']
    names = ['Window', 'Training start', 'Training end', 'Validation start', 'Validation target end', 'Warm-up start', 'Test start', 'Test end']
    st.dataframe(window[fields].rename(columns=dict(zip(fields, map(t, names)))), hide_index=True, width='stretch')
    st.subheader(t('Feature availability by horizon'))
    audit = pd.read_csv(RESULTS / 'feature_audit.csv')
    horizon = st.selectbox(t('Forecast horizon'), list(range(1, 10)), key='horizon')
    st.dataframe(audit[audit.forecast_horizon_days.eq(horizon)].rename(columns={'feature': t('Feature'), 'minimum_lag_days': t('Minimum lag, days'), 'maximum_lag_days': t('Maximum lag, days'), 'forecast_horizon_days': t('Forecast horizon, days')}), hide_index=True, width='stretch')
    st.caption(t('Lag names identify the h = 1 template. For horizon h, lag_1 / lag_7 / lag_14 / lag_28 actually use lags h / h+6 / h+13 / h+27. Rolling windows start at lag h. Guards reject missing horizons and validation targets that reach an order origin.'))
    st.subheader(t('Ordering and money'))
    st.code(t('unit_cost = price × (1 - gross_margin)\nholding_per_day = annual_rate × unit_cost / 365\nCu = price × (gross_margin + shortage_penalty)\nCo = holding_per_day × review_days\nq = Cu / (Cu + Co)\nsafety_stock = quantile_q(actual_9day_sum - forecast_9day_sum)\ntarget = ceil(max(0, forecast_9day_sum + safety_stock))\norder = max(0, target - on_hand - on_order)\nloss = shortage × Cu + closing_stock × holding_per_day'), language=None)
    st.write(t('Reviews happen before demand. An order on day d arrives before opening on d + 2. The nine-day forecast covers d through d + 8. Unsold stock carries forward and no terminal disposal charge is applied.'))
    st.write(t('For each method and window: scaled residual = (actual nine-day sum - forecast sum) / max(9 × MA28, 1 unit). I pool these residuals across items within low, medium and high recent-volume tiers. The tier quantile is multiplied by the current item scale. Overlapping outcomes and shared item shocks mean the pooled sample size does not imply independent observations.'))
    st.subheader(t('Limitations'))
    st.write(t('M5 contains observed sales, not unconstrained demand. Actual stock, procurement costs and delivery records are unavailable. Gross margin, carrying rate and shortage penalty are assumptions. Spoilage and shelf life are not modeled because M5 has no shelf-life data; this is a limitation for perishable FOODS items.'))
    st.write(t('Losses mean missed margin plus shortage penalty and inventory carrying cost, not measured accounting profit. The simulation keeps historical sales as its demand proxy even when its own policy runs out of stock.'))
    st.markdown(t('Data: [Kaggle M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), [Zenodo mirror](https://zenodo.org/records/10203108).'))
