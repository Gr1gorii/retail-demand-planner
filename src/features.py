import pandas as pd
import numpy as np


CATEGORICAL = ["item_id", "dept_id", "event"]
FEATURES = CATEGORICAL + [
    "day", "wday", "month", "year", "snap", "is_event",
    "price_known", "price_change", "lag_1", "lag_7", "lag_14", "lag_28",
    "mean_7", "mean_28", "std_28", "zero_share_28",
]


def feature_availability(horizon=1):
    if not isinstance(horizon, int) or not 1 <= horizon <= 9:
        raise ValueError("Direct forecast horizons must be integers from 1 to 9")
    rows = []
    for lag in [1, 7, 14, 28]:
        shifted = lag + horizon - 1
        rows.append((f"lag_{lag}", shifted, shifted))
    for name, maximum in [("mean_7", 7), ("mean_28", 28), ("std_28", 28), ("zero_share_28", 28), ("moving_average_28", 28)]:
        rows.append((name, horizon, horizon + maximum - 1))
    rows.extend([("price_known", horizon, None), ("price_change", horizon, None)])
    seasonal_lag = 7 * int(np.ceil(horizon / 7))
    rows.append(("seasonal_naive", seasonal_lag, seasonal_lag))
    return pd.DataFrame(rows, columns=["feature", "minimum_lag_days", "maximum_lag_days"]).assign(forecast_horizon_days=horizon)


def make_features(panel, horizon=1):
    feature_availability(horizon)
    frame = panel.sort_values(["item_id", "day"]).copy()
    groups = frame.groupby("item_id", observed=True)
    for lag in [1, 7, 14, 28]:
        frame[f"lag_{lag}"] = groups.sales.shift(lag + horizon - 1).astype("float32")
    frame["seasonal_naive"] = groups.sales.shift(7 * int(np.ceil(horizon / 7)))
    for window in [7, 28]:
        mean = groups.sales.transform(
            lambda sales: sales.shift(horizon).rolling(window).mean()
        )
        frame[f"mean_{window}"] = mean.astype("float32")
        if window == 28:
            frame["moving_average_28"] = mean
    frame["std_28"] = groups.sales.transform(
        lambda sales: sales.shift(horizon).rolling(28).std()
    ).astype("float32")
    frame["zero_share_28"] = groups.sales.transform(
        lambda sales: sales.eq(0).astype(float).shift(horizon).rolling(28).mean()
    ).astype("float32")
    frame["price_known"] = groups.sell_price.transform(
        lambda prices: prices.ffill().shift(horizon)
    ).astype("float32")
    old_price = frame.groupby("item_id", observed=True).price_known.shift(7)
    frame["price_change"] = (frame.price_known / old_price - 1).astype("float32")
    for column in CATEGORICAL:
        frame[column] = frame[column].astype("category")
    return frame


def assign_segments(history):
    recent = history[history.day > history.day.max() - 90]
    stats = recent.groupby("item_id", observed=True).sales.agg(
        average="mean", zero_share=lambda values: values.eq(0).mean(),
    )
    stats["volume"] = pd.cut(
        stats.average, [-1, 1, 5, float("inf")],
        labels=["Up to 1 unit/day", "1-5 units/day", "Over 5 units/day"],
    ).astype(str)
    stats["pattern"] = "Regular"
    stats.loc[stats.zero_share >= 0.5, "pattern"] = "Intermittent"
    return stats[["volume", "pattern"]].reset_index()
