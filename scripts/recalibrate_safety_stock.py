import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import recent_levels
from src.data import load_data
from src.recalibrate import recalibrate


if __name__ == '__main__':
    import json
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--results', default=str(ROOT / 'results'))
    args = parser.parse_args()
    output = Path(args.results)
    config = json.loads((output / 'summary.json').read_text())['config']
    level_path = output / 'safety_levels.parquet'
    if level_path.exists():
        levels = pd.read_parquet(level_path)
    else:
        panel = load_data(args.raw_dir, config['store'], config['category'])
        levels = recent_levels(panel)
        first_day = pd.read_parquet(output / 'protection_calibration.parquet').origin_day.min()
        levels = levels[levels.origin_day.ge(first_day) & levels.origin_day.le(panel.loc[panel.sales.notna(), 'day'].max() + 1)]
    summary = recalibrate(output, levels)
    print({key: summary[key] for key in ['loss_baseline', 'loss_moving_average', 'loss_lightgbm']})
