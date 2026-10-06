import ast
import json
from pathlib import Path
from string import Formatter

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.i18n import ITALIAN, number, translate


ROOT = Path(__file__).resolve().parents[1]


def order_table(app, heading):
    return next(table.value for table in app.dataframe if heading in table.value.columns)


def test_every_interface_and_chart_text_has_an_italian_translation():
    texts = set()
    for path in [ROOT / 'app.py', ROOT / 'scripts/make_charts.py']:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ['t', 'translated_selectbox']:
                if node.args and isinstance(node.args[0], ast.Constant): texts.add(node.args[0].value)
    texts.update(['Up to 1 unit/day', '1-5 units/day', 'Over 5 units/day', 'Regular', 'Intermittent', 'Validation start', 'Validation target end', 'Warm-up start', 'low', 'medium', 'high'])
    assert texts <= ITALIAN.keys()
    for text in texts:
        original = {field for _, field, _, _ in Formatter().parse(text) if field}
        translated = {field for _, field, _, _ in Formatter().parse(ITALIAN[text]) if field}
        assert original == translated
    assert len(ITALIAN['Saved vs MA28']) <= 17
    assert len(ITALIAN['Fill rate']) <= 15


def test_numbers_use_the_selected_language():
    assert number(3583.75, 2, 'en') == '3,583.75'
    assert number(3583.75, 2, 'it') == '3.583,75'
    assert number(-1220.01, 2, 'it') == '-1.220,01'
    assert translate('Up to 1 unit/day', 'it') == 'Fino a 1 unità/giorno'
    assert translate('FOODS_1', 'it') == 'FOODS_1'


def test_language_switch_keeps_on_hand_pipeline_and_order_values():
    summary = json.loads((ROOT / 'results/summary.json').read_text())
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].value == '$ ' + number(summary['savings'])
    assert app.metric[1].value == '$ ' + number(summary['savings_moving_average'])
    assert any('Lowest simulated cost: LightGBM with tier-scaled safety stock.' in element.value for element in app.markdown)
    assert any('Main limitation:' in element.value for element in app.warning)
    app.selectbox(key='coverage_group_en').select('zero_share_band').run()
    coverage = order_table(app, 'Segment')
    expected = pd.read_csv(ROOT / 'results/coverage_segments.csv')
    expected = expected.query("fold == 0 and segment == '(75%, 100%]' and method == 'lightgbm'").exceedance.iloc[0]
    measured = coverage[(coverage.Segment == '(75%, 100%]') & (coverage.Method == 'LightGBM')]['Demand > target'].iloc[0]
    assert measured == pytest.approx(expected)
    app.selectbox(key='department_en').select('FOODS_3').run()
    app.number_input(key='stock').set_value(10).run()
    app.number_input(key='on_order').set_value(5).run()
    english = order_table(app, 'Order, units').copy()
    assert english['Order, units'].equals((english['Target stock'] - 15).clip(lower=0))
    app.selectbox(key='language').select('Italiano').run()
    assert not app.exception
    assert app.metric[0].label == 'Risparmio vs naive'
    assert app.metric[0].value == '$ ' + number(summary['savings'], language='it')
    assert app.metric[1].value == '$ ' + number(summary['savings_moving_average'], language='it')
    assert any('Costo simulato minimo: LightGBM con scorta scalata per fascia.' in element.value for element in app.markdown)
    assert app.number_input(key='stock').value == 10
    assert app.number_input(key='on_order').value == 5
    assert app.selectbox(key='department_it').value == 'FOODS_3'
    assert app.selectbox(key='coverage_group_it').value == 'zero_share_band'
    italian = order_table(app, 'Da ordinare, unità')
    assert italian['Da ordinare, unità'].tolist() == english['Order, units'].tolist()
    assert italian['Disponibile'].eq(10).all() and italian['In arrivo'].eq(5).all()
    app.selectbox(key='language').select('English').run()
    assert not app.exception
    assert app.selectbox(key='department_en').value == 'FOODS_3'
    assert order_table(app, 'Order, units')['Order, units'].tolist() == english['Order, units'].tolist()


def test_annual_carrying_control_replays_the_matching_grid_cell():
    grid = pd.read_csv(ROOT / 'results/sensitivity.csv')
    expected = grid[grid.shortage_cost_share.eq(0.5) & grid.annual_holding_rate.eq(0.4)].iloc[0]
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    app.slider(key='carrying').set_value(0.4).run()
    assert not app.exception
    assert app.metric[0].value == '$ ' + number(expected.savings)
    assert app.metric[1].value == '$ ' + number(expected.savings_moving_average)
    assert app.metric[2].value == '$ ' + number(min(expected.loss_baseline, expected.loss_moving_average, expected.loss_lightgbm))
    coverage = order_table(app, 'Demand > target')
    assert coverage['Target service'].tolist() == pytest.approx([expected['quantile']] * len(coverage))
    app.selectbox(key='horizon').select(9).run()
    assert not app.exception
    table = order_table(app, 'Minimum lag, days')
    assert table['Minimum lag, days'].ge(9).all()
