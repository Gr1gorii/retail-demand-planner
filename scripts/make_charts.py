import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
BLUE = "#255885"
GRAY = "#a0abb5"
ORANGE = "#ba7638"
INK = "#202a35"


def plain_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#c9cfd6")
    axis.grid(axis="y", color="#e7eaee", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK)


def save_charts(language):
    output = ASSETS / language
    output.mkdir(parents=True, exist_ok=True)
    translations = json.loads((ROOT / "locales" / "it.json").read_text()) if language == "it" else {}

    def t(text):
        return translations.get(text, text)

    def number(value, position=None):
        text = f"{value:,.0f}"
        return text.replace(",", ".") if language == "it" else text

    def percent(value):
        text = f"{value:.1%}"
        return text.replace(".", ",") if language == "it" else text

    summary = json.loads((RESULTS / "summary.json").read_text())
    folds = pd.read_csv(RESULTS / "by_fold.csv")
    segments = pd.read_csv(RESULTS / "by_volume.csv")
    items = pd.read_csv(RESULTS / "by_item.csv")
    daily = pd.read_csv(RESULTS / "daily.csv", parse_dates=["date"])
    sensitivity = pd.read_csv(RESULTS / "sensitivity.csv")
    methods = ["baseline", "moving_average", "lightgbm"]
    method_labels = [t("Seasonal naive"), t("28-day moving average"), "LightGBM"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.1))
    figure.subplots_adjust(top=0.68, bottom=0.20, wspace=0.28, left=0.08, right=0.97)
    figure.text(0.06, 0.93, t("LightGBM wins with tier-scaled safety stock"), fontsize=22, weight="bold")
    figure.text(0.06, 0.86, t("CA_1 / FOODS · 1,437 items · {start} - {end} · simulation").format(start=summary['start'], end=summary['end']), fontsize=11, color="#5b6875")
    figure.text(0.06, 0.78, t("${value} saved vs naive · {share} less cost than MA28").format(value=number(summary['savings']), share=percent(summary['savings_share_moving_average'])), fontsize=18, color=BLUE, weight="bold")
    axis = axes[0]
    positions = np.arange(3)
    shortage = [summary[f"shortage_cost_{method}"] for method in methods]
    excess = [summary[f"holding_cost_{method}"] for method in methods]
    axis.bar(positions, shortage, width=0.62, color=BLUE, label=t("Shortage"))
    axis.bar(positions, excess, bottom=shortage, width=0.62, color=ORANGE, label=t("Holding"), hatch="//", edgecolor="white")
    totals = np.array(shortage) + np.array(excess)
    for position, total in zip(positions, totals):
        axis.text(position, total + totals.max() * 0.025, number(total), ha="center", fontsize=11)
    axis.set_xticks(positions, [t("Seasonal\nnaive"), t("28-day moving\naverage"), "LightGBM"])
    axis.set_ylabel(t("Ordering error cost, $"))
    axis.set_ylim(0, totals.max() * 1.25)
    axis.yaxis.set_major_formatter(FuncFormatter(number))
    axis.legend(frameon=False, loc="upper right", ncol=2)
    plain_axis(axis)

    axis = axes[1]
    positions = folds.fold.to_numpy()
    max_value = max(folds.savings.max(), folds.savings_moving_average.max())
    for offset, column, color, label in [
        (-0.18, "savings", GRAY, t("Vs seasonal naive")),
        (0.18, "savings_moving_average", BLUE, t("Vs moving average")),
    ]:
        bars = axis.bar(positions + offset, folds[column], width=0.33, color=color, label=label)
        for bar, value in zip(bars, folds[column]):
            label_value = number(value) if abs(value) < 1000 else f"{value / 1000:.1f}k"
            if language == "it":
                label_value = label_value.replace(".", ",")
            shift = max_value * 0.025 if value >= 0 else -max_value * 0.025
            axis.text(bar.get_x() + bar.get_width() / 2, value + shift, label_value, ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    labels = [t("Window {fold}\n{start} - {end}").format(fold=window['fold'], start=window['test_start_date'][5:], end=window['test_end_date'][5:]) for window in summary["windows"]]
    axis.set_xticks(positions, labels)
    axis.set_ylim(min(0, folds[['savings', 'savings_moving_average']].to_numpy().min() * 1.3), max_value * 1.50)
    axis.set_ylabel(t("Loss reduction, $"))
    axis.yaxis.set_major_formatter(FuncFormatter(number))
    axis.legend(frameon=False, loc="upper right")
    plain_axis(axis)
    figure.text(0.06, 0.08, t("Margin 30% · annual carrying 25% on unit cost · shortage penalty 20% · USD"), fontsize=10, color="#5b6875")
    figure.text(0.06, 0.035, t("Main limit: coverage still misses the target in windows 1 and 4 and for highly intermittent items."), fontsize=10, color="#5b6875")
    figure.savefig(output / "overview.png", dpi=160)
    plt.close(figure)

    diagnostics = pd.read_csv(RESULTS / "protection_diagnostics.csv")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    figure.subplots_adjust(top=0.75, bottom=0.22, wspace=0.28, left=0.07, right=0.97)
    figure.text(0.06, 0.93, t("Nine-day errors and target-stock coverage"), fontsize=22, weight="bold")
    figure.text(0.06, 0.85, t("Same-origin protection sums · complete item-review cycles · default USD cost assumptions"), fontsize=11, color="#5b6875")
    for axis, field, title in zip(axes, ["protection_wape", "exceedance"], [t("9-day WAPE"), t("Demand > target")]):
        for offset, model, label, color in zip([-0.24, 0, 0.24], methods, method_labels, [GRAY, ORANGE, BLUE]):
            values = diagnostics[diagnostics.method.eq(model) & diagnostics.fold.ne(0)].sort_values('fold')
            axis.bar(values.fold + offset, values[field], width=0.23, color=color, label=label)
        axis.set_xticks([1, 2, 3, 4], [t("Window") + f' {value}' for value in range(1, 5)])
        axis.set_ylabel(title)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: percent(value)))
        axis.set_ylim(0, diagnostics[field].max() * 1.35)
        plain_axis(axis)
    target = diagnostics.target_exceedance.iloc[0]
    axes[1].axhline(target, color=INK, linestyle='--', linewidth=1, label=t("Target exceedance") + f' {percent(target)}')
    axes[0].legend(frameon=False, fontsize=10)
    axes[1].legend(frameon=False, fontsize=10)
    figure.text(0.06, 0.09, t("Pooled LightGBM coverage meets the target; windows 1 and 4 still miss it."), fontsize=11, color="#5b6875")
    figure.text(0.06, 0.035, t("Complete cycles: 5,748 in windows 1-3; 4,311 in window 4. Its final 1,437 cycles lack two actual days."), fontsize=10, color="#5b6875")
    figure.savefig(output / "protection.png", dpi=160)
    plt.close(figure)

    old = pd.read_csv(RESULTS / 'uncalibrated/protection_diagnostics.csv').query('fold == 0').set_index('method')
    new = diagnostics.query('fold == 0').set_index('method')
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    figure.subplots_adjust(top=0.75, bottom=0.23, wspace=0.28, left=0.08, right=0.97)
    figure.text(0.06, 0.93, t("Same forecasts, different safety-stock calibration"), fontsize=22, weight='bold')
    figure.text(0.06, 0.85, t("Previous per-item correction vs scaled residuals pooled by recent-volume tier"), fontsize=11, color='#5b6875')
    positions = np.arange(3)
    for axis, field, title in zip(axes, ['loss', 'exceedance'], [t('Loss, USD'), t('Demand > target')]):
        axis.bar(positions - 0.18, old.loc[methods, field], width=0.33, color=GRAY, label=t('Uncalibrated safety stock'))
        axis.bar(positions + 0.18, new.loc[methods, field], width=0.33, color=BLUE, label=t('Tier-scaled safety stock'))
        axis.set_xticks(positions, [t('Seasonal\nnaive'), 'MA28', 'LightGBM'])
        axis.set_ylabel(title)
        axis.set_ylim(0, max(old[field].max(), new[field].max()) * 1.35)
        axis.legend(frameon=False, fontsize=10)
        axis.yaxis.set_major_formatter(FuncFormatter(number if field == 'loss' else lambda value, position: percent(value)))
        plain_axis(axis)
    target_label = f'{new.target_exceedance.iloc[0] * 100:.2f}%'
    if language == 'it': target_label = target_label.replace('.', ',')
    axes[1].axhline(new.target_exceedance.iloc[0], color=INK, linestyle='--', linewidth=1, label=t('Target exceedance') + ' ' + target_label)
    axes[1].legend(frameon=False, fontsize=10)
    figure.text(0.06, 0.09, t('Extra protection increases holding costs. Lower cost with under-protection does not establish better forecast quality.'), fontsize=10, color='#5b6875')
    figure.text(0.06, 0.035, t('USD · lead time 2 days · weekly review · no forecast retraining'), fontsize=10, color='#5b6875')
    figure.savefig(output / 'calibration.png', dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 4.8))
    figure.subplots_adjust(top=0.78, bottom=0.19, left=0.08, right=0.98)
    figure.text(0.06, 0.92, t("Category actuals and forecasts"), fontsize=21, weight="bold")
    figure.text(0.06, 0.84, t("CA_1 / FOODS · category total · one-day forecasts"), fontsize=11, color="#5b6875")
    axis.plot(daily.date, daily.sales, color=INK, linewidth=1.7, label=t("Actual sales"))
    for method, label, color, style in zip(methods, method_labels, [GRAY, ORANGE, BLUE], ["--", "-.", "-"]):
        axis.plot(daily.date, daily[method], color=color, linestyle=style, linewidth=1.5, label=label)
    axis.set_ylabel(t("Sales, units/day"))
    ticks = [daily.date.iloc[0]] + list(pd.date_range(daily.date.iloc[0], daily.date.iloc[-1], freq="MS")) + [daily.date.iloc[-1]]
    axis.set_xticks(ticks)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m" if language == "it" else "%m-%d"))
    axis.yaxis.set_major_formatter(FuncFormatter(number))
    axis.set_xlim(daily.date.min(), daily.date.max())
    axis.legend(frameon=False, ncol=4, loc="upper left", fontsize=10)
    plain_axis(axis)
    figure.text(0.06, 0.04, t("WAPE uses item-days. Individual errors can cancel each other out in this aggregate chart."), fontsize=10, color="#5b6875")
    figure.savefig(output / "forecast.png", dpi=160)
    plt.close(figure)

    order = ["Up to 1 unit/day", "1-5 units/day", "Over 5 units/day"]
    segments = segments.set_index("volume").reindex(order)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    figure.subplots_adjust(top=0.72, bottom=0.22, wspace=0.35, left=0.07, right=0.97)
    figure.text(0.06, 0.93, t("Where the model makes mistakes"), fontsize=22, weight="bold")
    losing = int(items.savings_moving_average.lt(0).sum())
    figure.text(0.06, 0.84, t("MAE by sales volume · losses increased vs moving average on {items} of 1,437 items").format(items=losing), fontsize=11, color="#5b6875")
    axis = axes[0]
    positions = np.arange(3)
    for offset, method, label, color in zip([-0.24, 0, 0.24], methods, method_labels, [GRAY, ORANGE, BLUE]):
        axis.bar(positions + offset, segments[f"mae_{method}"], width=0.23, color=color, label=label)
    axis.set_xticks(positions, [t("Up to 1"), "1-5", t("Over 5")])
    axis.set_xlabel(t("Pre-test average sales, units/day"))
    axis.set_ylabel(t("MAE, units/item-day"))
    axis.legend(frameon=False, fontsize=10)
    plain_axis(axis)

    axis = axes[1]
    axis.hist(items.savings_moving_average, bins=60, color=BLUE, edgecolor="white", linewidth=0.6)
    axis.axvline(0, color=INK, linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_ylim(0.8, 1400)
    axis.set_xlabel(t("Loss reduction vs moving average, $/item"))
    axis.set_ylabel(t("Item count, log scale"))
    axis.xaxis.set_major_formatter(FuncFormatter(number))
    plain_axis(axis)
    figure.text(0.06, 0.09, t("All items included. Segments use pre-test history. Negative effects are retained."), fontsize=10, color="#5b6875")
    figure.text(0.06, 0.035, t("All money in USD. Spoilage and shelf life are not modeled."), fontsize=10, color="#5b6875")
    figure.savefig(output / "errors.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    figure.subplots_adjust(top=0.76, bottom=0.21, wspace=0.32, left=0.11, right=0.94)
    figure.text(0.06, 0.93, t("Savings depend on shortage and carrying costs"), fontsize=21, weight="bold")
    figure.text(0.06, 0.86, t("Forecasts fixed · all three order policies recalibrated in every cell · 112 days"), fontsize=11, color="#5b6875")
    minimum = min(0, sensitivity[["savings", "savings_moving_average"]].to_numpy().min())
    maximum = sensitivity[["savings", "savings_moving_average"]].to_numpy().max()
    for axis, column, title in zip(axes, ["savings", "savings_moving_average"], [t("Vs seasonal naive"), t("Vs moving average")]):
        grid = sensitivity.pivot(index="shortage_cost_share", columns="annual_holding_rate", values=column).sort_index().sort_index(axis=1)
        axis.imshow(grid.to_numpy(), cmap="Blues", vmin=minimum, vmax=maximum, aspect="auto")
        axis.set_title(title)
        axis.set_xticks(range(len(grid.columns)), [f"{value:.0%}" for value in grid.columns])
        axis.set_yticks(range(len(grid.index)), [f"{value:.0%}" for value in grid.index])
        axis.set_xlabel(t("Annual carrying rate on unit cost"), labelpad=12)
        axis.set_ylabel(t("Shortage cost Cu / selling price"))
        for row in range(len(grid.index)):
            for column_index in range(len(grid.columns)):
                value = grid.iloc[row, column_index]
                color = "white" if (value - minimum) / (maximum - minimum) > 0.58 else INK
                axis.text(column_index, row, "$" + number(value), ha="center", va="center", color=color, fontsize=13, weight="bold")
        axis.spines[:].set_visible(False)
    figure.text(0.06, 0.10, t("Margin 30%; shortage cost = margin + penalty. Holding per day = rate × unit cost / 365."), fontsize=10, color="#5b6875")
    figure.text(0.06, 0.04, t("USD · lead time 2 days · weekly review · all cells recalibrate safety stock and replay inventory"), fontsize=10, color="#5b6875")
    figure.savefig(output / "sensitivity.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    for language in ["en", "it"]:
        save_charts(language)
