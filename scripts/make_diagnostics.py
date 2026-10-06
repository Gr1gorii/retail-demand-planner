import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import protection_diagnostics, review_cycles


if __name__ == '__main__':
    results = ROOT / 'results'
    summary = json.loads((results / 'summary.json').read_text())
    targets = pd.read_parquet(results / 'targets.parquet')
    scored = pd.read_parquet(results / 'simulation.parquet')
    cycles = review_cycles(targets, summary['windows'], summary['config']['business']['review_period_days'])
    table = protection_diagnostics(cycles, scored, summary['critical_ratio'])
    cycles.to_parquet(results / 'review_cycles.parquet', index=False)
    table.to_csv(results / 'protection_diagnostics.csv', index=False)
    print(table.to_string(index=False))
