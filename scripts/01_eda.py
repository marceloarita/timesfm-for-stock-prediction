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
df = load_stock(ticker, start="2015-01-01", end="2025-12-31")
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
ax.set_title("AAPL — Closing Price (2015–2025)")
ax.set_ylabel("USD")
ax.set_xlabel("Date")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "01_1_closing_price.png"), dpi=150)
plt.show()


# -------------------------------------`----------------------------------------
# 1.2 What were the major bull and bear periods?
# -----------------------------------------------------------------------------
# We highlight the most significant macroeconomic events visible in the chart.

# COVID had two distinct phases visible in the chart:
#   - Crash:    Feb 15 → Mar 23, 2020 — AAPL lost ~30% in 5 weeks
#   - Recovery: Mar 23 → Dec 31, 2020 — AAPL more than doubled from the bottom
#
# Why did AAPL crash?
#   1. General panic — investors sold everything indiscriminately, not Apple-specific
#   2. Factories closed in China — Foxconn (Apple's main manufacturer) shut down in Zhengzhou.
#      Apple issued an official statement warning it would miss its quarterly revenue guidance —
#      one of the first major corporate COVID alerts.
#   3. Fear of demand collapse — the iPhone is a non-essential consumer good.
#      With economic uncertainty, the expectation was that people would stop buying.
#
# Why did it recover so fast — and go even higher?
#   4. Remote work accelerated tech demand — Mac, iPad, and AirPods sales surged.
#      People at home needed equipment, and Apple was best positioned to capture that.
#   5. Fed cut rates to near-zero — with fixed income yielding almost nothing,
#      money flooded into equities. Big tech was the preferred destination.
#      (explored in detail in 01_eda_fed_rates.py)
#   6. Apple became a "safe haven" within tech — USD 200B in cash, minimal debt.
#      In crises, investors concentrate in companies that will clearly survive.
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
ax.set_xlabel("Date")

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

# Cumulative return: if you invested USD 100 at the start of each year,
# how much would you have by the end of that year — compounding over time?
cumulative = (1 + annual_return / 100).cumprod() * 100

colors = ["seagreen" if r > 0 else "tomato" for r in annual_return.values]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.2]})
fig.suptitle("AAPL — Annual Performance (2016–2024)", fontsize=12, y=1.01)

# Top: cumulative return line
ax_top.plot(cumulative.index, cumulative.values, color="#7BAFD4", linewidth=1.5, marker="o",
            markersize=4, zorder=3)
ax_top.fill_between(cumulative.index, 100, cumulative.values,
                    where=cumulative.values >= 100, alpha=0.15, color="seagreen")
ax_top.fill_between(cumulative.index, 100, cumulative.values,
                    where=cumulative.values < 100, alpha=0.15, color="tomato")
ax_top.axhline(100, color="#A8A8A8", linewidth=0.8, linestyle="--")
ax_top.set_ylabel("Cumulative (Base = 100)")
ax_top.grid(axis="y", alpha=0.3)
for x, y in zip(cumulative.index, cumulative.values):
    ax_top.text(x, y + 8, f"{y:.0f}", ha="center", fontsize=7.5, color="#404040")

# Bottom: annual return bars
bars = ax_bot.bar(annual_return.index, annual_return.values, color=colors, edgecolor="white", width=0.6)
ax_bot.axhline(0, color="#404040", linewidth=0.8)
ax_bot.set_ylabel("Annual Return (%)")
ax_bot.set_xticks(annual_return.index)
ax_bot.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, annual_return.values):
    ax_bot.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (1 if val > 0 else -3.5),
                f"{val:+.0f}%", ha="center", fontsize=8, color="#404040")

ax_bot.set_xlabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "01_3_annual_return.png"), dpi=150)
plt.show()

print("\n=== Script 01 complete ===")
print("Next: scripts/01_eda_fed_rates.py")
