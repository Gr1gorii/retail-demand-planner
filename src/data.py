import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


FILES = ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"]


def load_data(raw_dir, store, category, future_days=9):
    raw_dir = Path(raw_dir)
    missing = [name for name in FILES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {', '.join(missing)}")

    parts = []
    for chunk in pd.read_csv(raw_dir / FILES[1], chunksize=2000):
        selected = chunk[(chunk.store_id == store) & (chunk.cat_id == category)]
        if len(selected):
            parts.append(selected)
    if not parts:
        raise ValueError("No items found for this store and category")
    sales = pd.concat(parts, ignore_index=True)
    days = [column for column in sales if column.startswith("d_")]
    panel = sales.melt(
        id_vars=["item_id", "dept_id"], value_vars=days,
        var_name="d", value_name="sales",
    )
    panel["day"] = panel.d.str[2:].astype("int16")
    panel["sales"] = panel.sales.astype("float32")

    calendar = pd.read_csv(raw_dir / FILES[0], parse_dates=["date"])
    calendar["day"] = calendar.d.str[2:].astype("int16")
    state = store.split("_")[0]
    calendar["snap"] = calendar[f"snap_{state}"]
    calendar["event"] = (
        calendar.event_name_1.fillna("none") + "|" +
        calendar.event_name_2.fillna("none")
    )
    calendar["is_event"] = (calendar.event_name_1.notna() | calendar.event_name_2.notna()).astype("int8")
    cols = ["day", "date", "wm_yr_wk", "wday", "month", "year", "snap", "event", "is_event"]
    panel = panel.drop(columns="d").merge(calendar[cols], on="day", validate="many_to_one")

    prices = []
    for chunk in pd.read_csv(raw_dir / FILES[2], chunksize=200000):
        prices.append(chunk[(chunk.store_id == store) & chunk.item_id.isin(sales.item_id)])
    prices = pd.concat(prices).drop(columns="store_id")
    panel = panel.merge(prices, on=["item_id", "wm_yr_wk"], how="left", validate="many_to_one")

    next_day = int(panel.day.max()) + 1
    future = sales[["item_id", "dept_id"]].merge(
        pd.DataFrame({"day": range(next_day, next_day + future_days)}), how="cross",
    )
    future["sales"] = np.nan
    future = future.merge(calendar[cols], on="day", validate="many_to_one")
    future["sell_price"] = np.nan
    panel = pd.concat([panel, future], ignore_index=True)
    panel = panel.sort_values(["item_id", "day"]).reset_index(drop=True)

    if panel.duplicated(["item_id", "day"]).any():
        raise ValueError("Duplicate item-day rows found")
    if panel.sales.dropna().lt(0).any():
        raise ValueError("Sales cannot be negative")
    if panel.groupby("item_id").day.diff().dropna().ne(1).any():
        raise ValueError("The calendar contains missing days")
    return panel


def source_hashes(raw_dir):
    hashes = {}
    for name in FILES:
        digest = hashlib.sha256()
        with open(Path(raw_dir) / name, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return hashes
