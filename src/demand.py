import pandas as pd


def flag_stockout_runs(frame, settings):
    rows = frame.sort_values(["item_id", "day"]).copy()
    groups = rows.groupby("item_id", observed=True)
    history = settings["history_days"]
    average = groups.sales.transform(lambda sales: sales.shift(1).rolling(history).mean())
    positive_share = groups.sales.transform(lambda sales: sales.gt(0).astype(float).shift(1).rolling(history).mean())
    zero = rows.sales.eq(0)
    run = (~zero).groupby(rows.item_id, observed=True).cumsum()
    keys = [rows.item_id, run]
    streak = zero.groupby(keys, observed=True).cumsum()
    normal = (average >= settings["minimum_mean_sales"]) & (positive_share >= settings["minimum_positive_share"])
    eligible_start = streak.eq(1) & normal & rows.price_known.notna()
    eligible_run = eligible_start.groupby(keys, observed=True).transform("max")
    length = zero.groupby(keys, observed=True).transform("sum")
    result = rows[["item_id", "day", "date"]].copy()
    result["likely_stockout"] = zero & eligible_run & length.ge(settings["minimum_zero_run"]) & rows.price_known.notna()
    result["zero_run_days"] = length.where(zero, 0).astype(int)
    result["pre_run_mean"] = average.where(streak.eq(1)).groupby(keys, observed=True).transform("max").where(zero)
    return result


def censoring_summary(rows):
    flagged = rows.likely_stockout
    return {
        "evaluation_rows": len(rows),
        "flagged_rows": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
        "flagged_items": int(rows.loc[flagged, "item_id"].nunique()),
        "flagged_observed_sales": float(rows.loc[flagged, "sales"].sum()),
        "imputed_lost_sales": 0,
    }
