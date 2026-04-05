"""
Script 01 — Exploratory Data Analysis (EDA)
=============================================
Before any modeling, we need to understand the data.
This script answers simple, direct questions about AAPL,
building intuition about the stock's behavior over time.

Blocks:
  Block 1 — Overview / Timeline
  Block 2 — Returns / Value Growth
  Block 3 — Behavioral Patterns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils.data import load_stock

CHARTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "charts")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

ticker = "AAPL"
df = load_stock(ticker, start="2015-01-01", end="2024-12-31")
close = df["Close"]

# =============================================================================
# BLOCK 1 — Overview / Timeline
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 How did the stock price evolve from 2015 to today?
# -----------------------------------------------------------------------------
# The most basic and most important chart: closing price over time.
# Closing price is the final price of the trading day — the most widely used
# reference in financial analysis.

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(close.index, close.values, linewidth=1, color="steelblue")
ax.set_title("AAPL — Closing Price (2015–2024)")
ax.set_ylabel("USD")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "01_1_closing_price.png"), dpi=150)
plt.show()


# -----------------------------------------------------------------------------
# 1.2 What were the major bull and bear periods?
# -----------------------------------------------------------------------------
# We highlight the most significant macroeconomic events visible in the chart.

# COVID had two distinct phases visible in the chart:
#   - Crash:    Feb 15 → Mar 23, 2020 — AAPL lost ~30% in 5 weeks
#   - Recovery: Mar 23 → Dec 31, 2020 — AAPL more than doubled from the bottom
COVID_PHASES = {
    "COVID Crash\n(Feb–Mar 2020)": ("2020-02-15", "2020-03-23", "tomato"),
    "Post-COVID Recovery\n(Mar–Dec 2020)": ("2020-03-23", "2020-12-31", "seagreen"),
}

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(close.index, close.values, linewidth=1, color="steelblue", zorder=3)

for label, (start, end, color) in COVID_PHASES.items():
    ax.axvspan(start, end, alpha=0.2, color=color, label=label)

ax.set_title("AAPL — COVID Impact: Crash and Recovery (2020)")
ax.set_ylabel("USD")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "01_2_covid_impact.png"), dpi=150)
plt.show()


# -----------------------------------------------------------------------------
# 1.3 Which year had the best and worst performance?
# -----------------------------------------------------------------------------
# Annual return: how much the stock gained or lost each calendar year.
# We resample to year-end closing prices and compute the percentage change.

annual_return = close.resample("YE").last().pct_change() * 100
annual_return.index = annual_return.index.year
annual_return = annual_return.dropna()

best_year = annual_return.idxmax()
worst_year = annual_return.idxmin()

print("\n--- 1.3 Annual Return ---")
for year, ret in annual_return.items():
    marker = " <- BEST" if year == best_year else (" <- WORST" if year == worst_year else "")
    print(f"  {year}: {ret:+.1f}%{marker}")

colors = ["seagreen" if r > 0 else "tomato" for r in annual_return.values]
fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(annual_return.index, annual_return.values, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("AAPL — Annual Return (%)")
ax.set_ylabel("%")
ax.set_xticks(annual_return.index)
for bar, val in zip(bars, annual_return.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (1 if val > 0 else -3),
            f"{val:+.0f}%", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "01_3_annual_return.png"), dpi=150)
plt.show()

print("\n=== Script 01 complete ===")
print("Next: scripts/01_eda_fed_rates.py")
