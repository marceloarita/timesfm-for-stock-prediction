"""
Script 03 — Baseline Models
============================
Establish reference benchmarks before applying TimesFM.
Two models: Naive and ARIMA, evaluated monthly over 2025.
Single horizon: 30 trading days.

Blocks:
  Block 1 — Expanding window rolling forecast (2025, monthly)
  Block 2 — Results table
  Block 3 — Per-month forecast charts
  Block 4 — Summary table visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils.data import load_stock
from utils.models import NaiveModel, ARIMAModel

CHARTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "charts")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

HORIZON    = 30
eval_dates = pd.date_range(start="2025-01-01", end="2025-12-01", freq="MS")  # Dec forecast → Jan 2026

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df    = load_stock("AAPL", start="2015-01-01", end="2026-03-31")
close = df["Close"]

# =============================================================================
# BLOCK 1 — Expanding window rolling forecast
# =============================================================================
results     = []   # one row per (eval_date, model)
predictions = []   # one row per (eval_date, model, step)

total = len(eval_dates)
for i, eval_date in enumerate(eval_dates):
    print(f"[{i+1}/{total}] eval_date={eval_date.date()}", end="  ")

    train_close = close[close.index < eval_date]

    future_idx = close[close.index >= eval_date].index[:HORIZON]
    if len(future_idx) < HORIZON:
        print("skip (not enough future data)")
        continue
    actual = close[future_idx].values

    naive = NaiveModel().fit(train_close)
    arima = ARIMAModel().fit(train_close)

    naive_pred = naive.predict(HORIZON)
    arima_pred = arima.predict(HORIZON)

    print(f"ARIMA order: {arima.order}")

    for model_name, pred in [("Naive", naive_pred), ("ARIMA", arima_pred)]:
        metrics = naive.evaluate(actual, pred)
        results.append({
            "eval_date":  eval_date,
            "model":      model_name,
            "ARIMA_order": arima.order if model_name == "ARIMA" else None,
            **metrics,
        })
        for step, (date, actual_val, pred_val) in enumerate(
            zip(future_idx, actual, pred), start=1
        ):
            predictions.append({
                "eval_date": eval_date,
                "model":     model_name,
                "step":      step,
                "date":      date,
                "actual":    round(float(actual_val), 2),
                "predicted": round(float(pred_val), 2),
                "abs_error": round(abs(float(actual_val) - float(pred_val)), 2),
            })

# =============================================================================
# BLOCK 2 — Results table
# =============================================================================
results_df = pd.DataFrame(results)

naive_df = results_df[results_df["model"] == "Naive"][
    ["eval_date", "MAE", "MAPE"]
].rename(columns={"MAE": "Naive_MAE", "MAPE": "Naive_MAPE"})

arima_df = results_df[results_df["model"] == "ARIMA"][
    ["eval_date", "MAE", "MAPE", "ARIMA_order"]
].rename(columns={"MAE": "ARIMA_MAE", "MAPE": "ARIMA_MAPE"})

summary = naive_df.merge(arima_df, on="eval_date")
summary["month"] = summary["eval_date"].dt.strftime("%Y-%m")
summary = summary[["month", "Naive_MAE", "Naive_MAPE", "ARIMA_MAE", "ARIMA_MAPE", "ARIMA_order"]]

avg_row = {
    "month":       "Average",
    "Naive_MAE":   round(summary["Naive_MAE"].mean(), 2),
    "Naive_MAPE":  round(summary["Naive_MAPE"].mean(), 2),
    "ARIMA_MAE":   round(summary["ARIMA_MAE"].mean(), 2),
    "ARIMA_MAPE":  round(summary["ARIMA_MAPE"].mean(), 2),
    "ARIMA_order": "",
}
display_df = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)

print("\n=== Baseline Results — Horizon 30d, 2025 ===")
print(display_df.to_string(index=False))

summary.to_csv(os.path.join(PROCESSED_DIR, "03_baseline_results.csv"), index=False)
print(f"\nSaved → data/processed/03_baseline_results.csv")

predictions_df = (
    pd.DataFrame(predictions)
    .sort_values(["eval_date", "model", "step"])
    .reset_index(drop=True)
)
predictions_df.to_csv(os.path.join(PROCESSED_DIR, "03_forecast_values.csv"), index=False)
print(f"Saved → data/processed/03_forecast_values.csv")

# =============================================================================
# BLOCK 3 — Per-month forecast charts
# =============================================================================
COLORS = {
    "Naive":   "#E8956D",
    "ARIMA":   "#7BAFD4",
    "Actual":  "#2D2D2D",
    "Context": "#999999",
}

# Fixed y-axis range across all charts
all_prices = close["2024-10-01":"2026-03-31"]
y_min = all_prices.min() * 0.97
y_max = all_prices.max() * 1.03

for eval_date in eval_dates:
    month_str = eval_date.strftime("%Y_%m")

    month_preds = predictions_df[predictions_df["eval_date"] == eval_date]
    if month_preds.empty:
        print(f"  WARNING: no predictions for {eval_date.date()}, skipping chart")
        continue

    actual_dates = pd.DatetimeIndex(
        month_preds[month_preds["model"] == "Naive"]["date"].values
    )
    actual_vals = month_preds[month_preds["model"] == "Naive"]["actual"].values

    context_start = close.index[close.index < eval_date][-60]
    context = close[context_start:close.index[close.index < eval_date][-1]]

    x_start = eval_date - pd.Timedelta(days=85)
    x_end   = actual_dates[-1] + pd.Timedelta(days=3)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(context.index, context.values,
            color=COLORS["Context"], linewidth=0.8, zorder=1)

    ax.axvline(context.index[-1], color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)
    ax.axvspan(context.index[-1], actual_dates[-1], alpha=0.10, color="#AAAAAA")

    # Bridge + actual
    ax.plot([context.index[-1], actual_dates[0]],
            [context.iloc[-1], actual_vals[0]],
            color=COLORS["Actual"], linewidth=1.0, zorder=5)
    ax.plot(actual_dates, actual_vals,
            color=COLORS["Actual"], linewidth=1.0, label="Actual", zorder=5)

    # Model forecasts
    for model_name in ["Naive", "ARIMA"]:
        m_preds = month_preds[month_preds["model"] == model_name]
        m_dates = pd.DatetimeIndex(m_preds["date"].values)
        m_vals  = m_preds["predicted"].values
        ax.plot([context.index[-1], m_dates[0]],
                [context.iloc[-1], m_vals[0]],
                linewidth=1.0, linestyle="--",
                color=COLORS[model_name], zorder=4)
        ax.plot(m_dates, m_vals,
                linewidth=1.0, linestyle="--",
                color=COLORS[model_name], alpha=0.85,
                label=model_name, zorder=4)

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_min, y_max)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.grid(False)
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.set_title(
        eval_date.strftime("%B %Y"),
        fontsize=14, fontweight="bold", color="#2D2D2D", pad=10, loc="left",
    )
    ax.set_ylabel("USD", fontsize=9, color="#888888")
    ax.set_xlabel("Date", fontsize=9, color="#888888")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f"03_forecast_{month_str}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → data/charts/03_forecast_{month_str}.png")

# =============================================================================
# BLOCK 4 — Summary table visualization
# =============================================================================
table_data = summary.copy()
months = table_data["month"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("white")
fig.suptitle("AAPL — Baseline Error Summary, Horizon 30d (2025)",
             fontsize=11, color="#2D2D2D", y=1.01)

metric_pairs = [
    ("MAE",  "Naive_MAE",  "ARIMA_MAE"),
    ("MAPE", "Naive_MAPE", "ARIMA_MAPE"),
]

for ax, (metric, naive_col, arima_col) in zip(axes, metric_pairs):
    naive_vals = table_data[naive_col].values.astype(float)
    arima_vals = table_data[arima_col].values.astype(float)

    all_vals   = np.concatenate([naive_vals, arima_vals])
    vmin, vmax = all_vals.min(), all_vals.max()
    cmap       = plt.get_cmap("YlOrRd")
    norm       = mcolors.Normalize(vmin=vmin, vmax=vmax)

    col_labels  = ["Month", "Naive", "ARIMA", "Δ (Naive−ARIMA)"]
    col_widths  = [0.22, 0.22, 0.22, 0.28]
    n_rows      = len(months)
    row_height  = 1.0 / (n_rows + 1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    x = 0
    for label, w in zip(col_labels, col_widths):
        ax.text(x + w / 2, 1 - row_height / 2, label,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#2D2D2D")
        x += w

    # Rows
    for r, (month, nv, av) in enumerate(zip(months, naive_vals, arima_vals)):
        y = 1 - (r + 1.5) * row_height
        delta = round(nv - av, 2)
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        delta_color = "#388E3C" if delta > 0 else "#C62828"

        row_vals = [month, f"{nv:.2f}", f"{av:.2f}", delta_str]
        row_colors = [None, cmap(norm(nv)), cmap(norm(av)), None]
        row_text_colors = ["#2D2D2D", "#2D2D2D", "#2D2D2D", delta_color]

        x = 0
        for val, w, bg, tc in zip(row_vals, col_widths, row_colors, row_text_colors):
            if bg is not None:
                rect = plt.Rectangle((x, y - row_height / 2), w, row_height,
                                     facecolor=bg, edgecolor="white", linewidth=0.5)
                ax.add_patch(rect)
            ax.text(x + w / 2, y, val,
                    ha="center", va="center", fontsize=8, color=tc)
            x += w

        # Horizontal separator
        ax.axhline(y - row_height / 2, color="#EEEEEE", linewidth=0.5)

    # Average row
    y = 1 - (n_rows + 1.5) * row_height
    avg_n = round(naive_vals.mean(), 2)
    avg_a = round(arima_vals.mean(), 2)
    avg_d = round(avg_n - avg_a, 2)
    avg_d_str = f"+{avg_d}" if avg_d > 0 else str(avg_d)
    avg_color = "#388E3C" if avg_d > 0 else "#C62828"

    for val, w, tc in zip(
        ["Average", f"{avg_n:.2f}", f"{avg_a:.2f}", avg_d_str],
        col_widths,
        ["#2D2D2D", "#2D2D2D", "#2D2D2D", avg_color],
    ):
        ax.text(x_pos := 0, y, "", fontsize=8)  # reset
        pass

    x = 0
    for val, w, tc in zip(
        ["Average", f"{avg_n:.2f}", f"{avg_a:.2f}", avg_d_str],
        col_widths,
        ["#2D2D2D", "#2D2D2D", "#2D2D2D", avg_color],
    ):
        ax.axhline(y + row_height / 2, color="#BBBBBB", linewidth=0.8)
        ax.text(x + w / 2, y, val,
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=tc)
        x += w

    ax.set_title(metric, fontsize=10, color="#2D2D2D", pad=8)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("error magnitude", fontsize=7, color="#888888")

plt.tight_layout()
path = os.path.join(CHARTS_DIR, "03_error_summary.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → data/charts/03_error_summary.png")

# =============================================================================
# BLOCK 5 — Two 3×2 grids: Jan–Jun and Jul–Dec
# =============================================================================
from PIL import Image, ImageDraw, ImageFont

months_saved = [
    eval_date.strftime("%Y_%m")
    for eval_date in eval_dates
    if not predictions_df[predictions_df["eval_date"] == eval_date].empty
]

imgs = [
    Image.open(os.path.join(CHARTS_DIR, f"03_forecast_{m}.png"))
    for m in months_saved
]

w, h    = imgs[0].size
NCOLS   = 2
NROWS   = 3
HEADER  = 110
PADDING = 6

def make_grid(img_list, subtitle):
    total_w = NCOLS * w + (NCOLS - 1) * PADDING
    total_h = HEADER + NROWS * h + (NROWS - 1) * PADDING
    canvas  = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw    = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 42)
    except Exception:
        font = ImageFont.load_default()
    title = f"AAPL — Baseline Forecasts 2025  ({subtitle})  |  Horizon: 30 days  |  Naive vs ARIMA"
    bbox  = draw.textbbox((0, 0), title, font=font)
    draw.text(
        ((total_w - (bbox[2] - bbox[0])) // 2, (HEADER - (bbox[3] - bbox[1])) // 2),
        title, fill=(45, 45, 45), font=font,
    )
    for idx, img in enumerate(img_list):
        row = idx // NCOLS
        col = idx  % NCOLS
        canvas.paste(img, (col * (w + PADDING), HEADER + row * (h + PADDING)))
    return canvas, total_w, total_h

grid_h1, w1, h1 = make_grid(imgs[:6],  "Jan – Jun")
grid_h2, w2, h2 = make_grid(imgs[6:], "Jul – Dec")

path_h1 = os.path.join(CHARTS_DIR, "03_forecast_grid_H1.png")
path_h2 = os.path.join(CHARTS_DIR, "03_forecast_grid_H2.png")
grid_h1.save(path_h1, dpi=(150, 150))
grid_h2.save(path_h2, dpi=(150, 150))
print(f"Saved → data/charts/03_forecast_grid_H1.png  ({w1}×{h1} px)")
print(f"Saved → data/charts/03_forecast_grid_H2.png  ({w2}×{h2} px)")


