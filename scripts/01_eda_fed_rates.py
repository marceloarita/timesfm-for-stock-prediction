"""
Script 01b — AAPL Price vs Federal Reserve Interest Rates
===========================================================
Hypothesis: When the Fed cuts rates to near-zero, investors move money
from fixed income into equities — pushing stocks like AAPL higher.
Conversely, aggressive rate hikes (2022) make safe bonds attractive again,
pulling money out of stocks.

What is the Federal Funds Rate?
  The interest rate at which U.S. banks lend money to each other overnight.
  It is set by the Federal Reserve (the U.S. central bank) and is the
  benchmark that drives all other interest rates in the economy.

  When rates are LOW  → borrowing is cheap → money flows into stocks
  When rates are HIGH → safe bonds yield well → money flows out of stocks

Data source: FRED (Federal Reserve Economic Data) — St. Louis Fed
  Series: FEDFUNDS — monthly effective federal funds rate
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils.data import get_close

# Pastel color palette
GRAY       = "#A8A8A8"
BLUE       = "#7BAFD4"
ORANGE     = "#E8956D"
LIGHT_GRAY = "#EFEFEF"
DARK       = "#404040"

CHARTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "charts")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

TECH_COLORS = {
    "AAPL":  "#A8A8A8",   # gray
    "GOOGL": "#7BAFD4",   # pastel blue
    "AMZN":  "#81C784",   # pastel green
    "MSFT":  "#9575CD",   # pastel purple
    "META":  "#F48FB1",   # pastel pink
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
close = get_close("AAPL", start="2015-01-01", end="2024-12-31")

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
fed = pd.read_csv(
    StringIO(requests.get(url, timeout=10).text),
    parse_dates=["observation_date"],
    index_col="observation_date",
)
fed = fed.loc["2015-01-01":"2024-12-31"].rename(columns={"FEDFUNDS": "rate"})

# ---------------------------------------------------------------------------
# Export dataset
# ---------------------------------------------------------------------------
# Align Fed rate (monthly) to daily AAPL index via forward fill
fed_daily = fed.reindex(close.index, method="ffill")

dataset = pd.DataFrame({
    "date":      close.index,
    "aapl_close": close.values,
    "fed_rate":   fed_daily["rate"].values,
})
dataset.to_csv(os.path.join(PROCESSED_DIR, "01b_aapl_vs_fed_rates.csv"), index=False)
print(f"Dataset saved: data/01b_aapl_vs_fed_rates.csv ({len(dataset)} rows)")
print(dataset.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

for spine in ["top", "right", "left"]:
    ax1.spines[spine].set_visible(False)
ax1.spines["bottom"].set_color(LIGHT_GRAY)
ax1.tick_params(axis="both", length=0)
ax1.yaxis.grid(True, color=LIGHT_GRAY, linewidth=0.6)
ax1.set_axisbelow(True)

# AAPL price — gray, fills the area for visual weight
ax1.fill_between(close.index, close.values, alpha=0.15, color=GRAY)
ax1.plot(close.index, close.values, color=GRAY, linewidth=1.0, label="AAPL Close Price")
ax1.set_ylabel("AAPL Price (USD)", color=GRAY, fontsize=8)
ax1.tick_params(axis="y", labelcolor=GRAY)
ax1.set_ylim(0, close.max() * 1.2)

# Fed rate — orange, secondary axis
ax2 = ax1.twinx()
for spine in ["top", "left", "bottom"]:
    ax2.spines[spine].set_visible(False)
ax2.spines["right"].set_color(LIGHT_GRAY)
ax2.tick_params(axis="both", length=0)
ax2.grid(False)

ax2.fill_between(fed.index, 0, fed["rate"], alpha=0.2, color=ORANGE)
ax2.plot(fed.index, fed["rate"], color=ORANGE, linewidth=1.0, label="Fed Funds Rate (%)")
ax2.set_ylabel("Fed Funds Rate (%)", color=ORANGE, fontsize=8)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax2.set_ylim(0, fed["rate"].max() * 2.5)

# Vertical dashed lines marking key Fed moments
KEY_EVENTS = [
    ("2020-03-15", "Rates cut\nto near-zero"),
    ("2022-03-16", "Rate hikes\nbegin"),
    ("2023-07-26", "Rate peaks\n(5.33%) — stabilizes"),
    ("2024-09-18", "First cut\n(recovery)"),
]
y_max = close.max() * 1.2
for date, label in KEY_EVENTS:
    ax1.axvline(pd.Timestamp(date), color="#707070", linewidth=0.8, linestyle="--", zorder=2)
    ax1.text(pd.Timestamp(date), y_max * 0.97, f" {label}",
             color="#707070", fontsize=7, va="top")

# X axis
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.set_xlim(close.index[0], close.index[-1])
ax1.set_xlabel("Date")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
           frameon=False, fontsize=8)

ax1.set_title("AAPL Price vs Federal Funds Rate (2015–2024)",
              fontsize=11, color=DARK, pad=14, loc="left")

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "01b_aapl_vs_fed_rates.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved: data/01b_aapl_vs_fed_rates.png")


# ---------------------------------------------------------------------------
# Chart 2 — Big Tech normalized vs Fed rate
# ---------------------------------------------------------------------------
# To compare stocks with very different price levels, we normalize all to 100
# at the start date. This shows relative performance — who grew more, who fell harder.
# If the Fed rate hypothesis holds, ALL big techs should move inversely to rates.

TICKERS = ["AAPL", "GOOGL", "AMZN", "MSFT", "META"]

print("\nDownloading big tech data...")
prices = {}
for ticker in TICKERS:
    prices[ticker] = get_close(ticker, start="2015-01-01", end="2024-12-31")
    print(f"  {ticker}: {len(prices[ticker])} trading days")

# Align all series to a common index (AAPL as reference)
common_index = prices["AAPL"].index
aligned = pd.DataFrame({t: prices[t].reindex(common_index, method="ffill") for t in TICKERS})

# Normalize to 100 at first trading day
normalized = (aligned / aligned.iloc[0]) * 100

# Export dataset
fed_daily2 = fed.reindex(common_index, method="ffill")
export2 = normalized.copy()
export2["fed_rate"] = fed_daily2["rate"].values
export2.index.name = "date"
export2.to_csv(os.path.join(PROCESSED_DIR, "01c_bigtech_normalized_vs_fed.csv"))
print(f"\nDataset saved: data/01c_bigtech_normalized_vs_fed.csv")

# Plot
fig, ax1 = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

for spine in ["top", "right", "left"]:
    ax1.spines[spine].set_visible(False)
ax1.spines["bottom"].set_color(LIGHT_GRAY)
ax1.tick_params(axis="both", length=0)
ax1.yaxis.grid(True, color=LIGHT_GRAY, linewidth=0.6)
ax1.set_axisbelow(True)

for ticker in TICKERS:
    ax1.plot(normalized.index, normalized[ticker],
             color=TECH_COLORS[ticker], linewidth=1.0, label=ticker)
    # Direct label — offset past the right edge to avoid overlap with right axis
    last_val = normalized[ticker].iloc[-1]
    ax1.text(normalized.index[-1] + pd.Timedelta(days=60), last_val, ticker,
             color=TECH_COLORS[ticker], fontsize=7.5, va="center")

ax1.set_ylabel("Normalized Price (Base = 100)", color=DARK, fontsize=8)
ax1.tick_params(axis="y", labelcolor=GRAY)
ax1.set_ylim(0, normalized.max().max() * 1.15)

# Extend x-axis to give room for the end labels
ax1.set_xlim(common_index[0], common_index[-1] + pd.Timedelta(days=200))
ax1.set_xlabel("Date")

# Fed rate — orange, right axis
ax2 = ax1.twinx()
for spine in ["top", "left", "bottom"]:
    ax2.spines[spine].set_visible(False)
ax2.spines["right"].set_color(LIGHT_GRAY)
ax2.tick_params(axis="both", length=0)
ax2.grid(False)

ax2.fill_between(fed.index, 0, fed["rate"], alpha=0.15, color=ORANGE)
ax2.plot(fed.index, fed["rate"], color=ORANGE, linewidth=1.0, label="Fed Funds Rate (%)")
ax2.set_ylabel("Fed Funds Rate (%)", color=ORANGE, fontsize=8)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax2.set_ylim(0, fed["rate"].max() * 2.5)

# X axis
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Legend — only Fed rate, stocks are directly labeled
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines2, labels2, loc="upper left", frameon=False, fontsize=8)

ax1.set_title("Big Tech Normalized Performance vs Federal Funds Rate (2015–2024)\nBase = 100 at Jan 2015",
              fontsize=11, color=DARK, pad=14, loc="left")

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "01c_bigtech_vs_fed.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved: data/01c_bigtech_vs_fed.png")


# ---------------------------------------------------------------------------
# Chart 3 — S&P 500 vs Federal Funds Rate
# ---------------------------------------------------------------------------
# The S&P 500 tracks the 500 largest U.S. companies — it IS the U.S. stock market.
# If the S&P 500 also moves inversely to Fed rates, the hypothesis becomes macro:
# it is not about Apple or big tech specifically — it is about capital allocation.
# When safe assets yield nothing, all risk assets rise. When they yield 5%, money leaves.

print("\nDownloading S&P 500 data...")
sp500 = get_close("^GSPC", start="2015-01-01", end="2024-12-31")
sp500_norm = (sp500 / sp500.iloc[0]) * 100

# Export dataset
fed_sp = fed.reindex(sp500.index, method="ffill")
export3 = pd.DataFrame({
    "date":       sp500.index,
    "sp500":      sp500.values,
    "sp500_norm": sp500_norm.values,
    "fed_rate":   fed_sp["rate"].values,
})
export3.to_csv(os.path.join(PROCESSED_DIR, "01d_sp500_vs_fed.csv"), index=False)
print(f"Dataset saved: data/01d_sp500_vs_fed.csv ({len(export3)} rows)")

# Plot
fig, ax1 = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

for spine in ["top", "right", "left"]:
    ax1.spines[spine].set_visible(False)
ax1.spines["bottom"].set_color(LIGHT_GRAY)
ax1.tick_params(axis="both", length=0)
ax1.yaxis.grid(True, color=LIGHT_GRAY, linewidth=0.6)
ax1.set_axisbelow(True)

# S&P 500 normalized — gray area
ax1.fill_between(sp500_norm.index, sp500_norm.values, alpha=0.12, color=GRAY)
ax1.plot(sp500_norm.index, sp500_norm.values, color=GRAY, linewidth=1.0, label="S&P 500 (normalized)")
ax1.set_ylabel("Normalized Price (Base = 100)", color=GRAY, fontsize=8)
ax1.tick_params(axis="y", labelcolor=GRAY)
ax1.set_ylim(0, sp500_norm.max() * 1.2)

# Fed rate — orange, right axis
ax2 = ax1.twinx()
for spine in ["top", "left", "bottom"]:
    ax2.spines[spine].set_visible(False)
ax2.spines["right"].set_color(LIGHT_GRAY)
ax2.tick_params(axis="both", length=0)
ax2.grid(False)

ax2.fill_between(fed.index, 0, fed["rate"], alpha=0.18, color=ORANGE)
ax2.plot(fed.index, fed["rate"], color=ORANGE, linewidth=1.0, label="Fed Funds Rate (%)")
ax2.set_ylabel("Fed Funds Rate (%)", color=ORANGE, fontsize=8)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax2.set_ylim(0, fed["rate"].max() * 2.5)

# Vertical dashed lines — same key moments
for date, label in KEY_EVENTS:
    ax1.axvline(pd.Timestamp(date), color="#707070", linewidth=0.8, linestyle="--", zorder=2)
    ax1.text(pd.Timestamp(date), sp500_norm.max() * 1.15, f" {label}",
             color="#707070", fontsize=7, va="top")

# X axis
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.set_xlim(sp500.index[0], sp500.index[-1])
ax1.set_xlabel("Date")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, fontsize=8)

ax1.set_title("S&P 500 vs Federal Funds Rate (2015–2024)\nThe entire U.S. market follows the same pattern",
              fontsize=11, color=DARK, pad=14, loc="left")

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "01d_sp500_vs_fed.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved: data/01d_sp500_vs_fed.png")
