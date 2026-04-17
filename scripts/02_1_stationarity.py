"""
Script 02_1 — Stationarity
=========================
Before modeling, we need to understand the statistical nature of the series.
This script covers the concept of stationarity, shows visually why AAPL price
is non-stationary, and confirms that daily returns are stationary using the
Augmented Dickey-Fuller test.

Blocks:
  Block 1 — Intuition: what does "stationary" mean?
  Block 2 — Visual: prices vs returns
  Block 3 — Violations: rolling mean and variance on raw prices
  Block 4 — Formalization: ADF test
  Block 5 — Conclusion: what this means for the project
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from meteostat import Station, daily

from utils.data import load_stock

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "charts")

ticker = "AAPL"
df = load_stock(ticker, start="2015-01-01", end="2025-12-31")
close = df["Close"]
returns = close.pct_change().dropna()

# =============================================================================
# BLOCK 1 — Intuition: what does "stationary" mean?
# =============================================================================

# A stationary series behaves consistently over time.
# Formally, three conditions must hold:
#
#   1. Constant mean — the series fluctuates around a stable average.
#   2. Constant variance — the spread of fluctuations doesn't change.
#   3. No structural breaks — the statistical properties don't shift
#      suddenly from one regime to another.
#
# A good real-world example: daily temperature in Tokyo.
# Every year, temperatures follow the same seasonal cycle — cold winters (~5°C),
# hot summers (~28°C). For most of the period, mean and variance are stable.
# However, from 2022 onward both metrics shift noticeably — a reminder that
# stationarity is always an approximation, even in "well-behaved" series.
#
# We'll use this as a visual reference before showing AAPL breaking all three
# conditions far more dramatically.

# Tokyo station: JMA 47662 (Tokyo Observatory)
tokyo_raw = daily(Station("47662"), start=pd.Timestamp("2015-01-01"),
                  end=pd.Timestamp("2025-12-31")).fetch()
tokyo_temp = tokyo_raw["temp"].dropna()

rolling_mean_tokyo = tokyo_temp.rolling(365).mean()
rolling_std_tokyo  = tokyo_temp.rolling(365).std()

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Tokyo Daily Temperature (2015–2025) — An Almost Stationary Series", fontsize=12)

# Raw series
axes[0].plot(tokyo_temp.index, tokyo_temp.values, linewidth=0.6, color="#555555", alpha=0.7)
axes[0].set_ylabel("°C")
axes[0].set_title("Daily temperature — seasonal pattern repeats each year")
for spine in ["top", "right"]:
    axes[0].spines[spine].set_visible(False)

# Rolling mean — should be flat (seasonal cycle averages out)
mean_pre  = tokyo_temp[:"2021"].mean()
mean_post = tokyo_temp["2022":].mean()

axes[1].plot(rolling_mean_tokyo.index, rolling_mean_tokyo.values,
             linewidth=1.5, color="steelblue", label="Rolling Mean (365d)")
axes[1].hlines(mean_pre,  pd.Timestamp("2015-01-01"), pd.Timestamp("2021-12-31"),
               color="tomato",     linewidth=1.0, linestyle="--",
               label=f"Mean 2015–2021 ({mean_pre:.1f}°C)")
axes[1].hlines(mean_post, pd.Timestamp("2022-01-01"), pd.Timestamp("2025-12-31"),
               color="darkorange", linewidth=1.0, linestyle="--",
               label=f"Mean 2022–2025 ({mean_post:.1f}°C)")
axes[1].annotate(
    "", xy=(pd.Timestamp("2022-01-01"), mean_post),
    xytext=(pd.Timestamp("2022-01-01"), mean_pre),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5)
)
axes[1].set_ylabel("°C")
axes[1].set_title("Mean shifts upward from 2022")
axes[1].legend(fontsize=8, loc="upper left")
for spine in ["top", "right"]:
    axes[1].spines[spine].set_visible(False)

# Rolling std — should be flat
axes[2].plot(rolling_std_tokyo.index, rolling_std_tokyo.values,
             linewidth=1.5, color="seagreen", label="Rolling Std (365d)")
axes[2].set_ylabel("°C")
axes[2].set_xlabel("Date")
axes[2].set_title("Variance increases from 2022")
axes[2].legend(fontsize=8, loc="upper left")
axes[2].xaxis.set_major_locator(mdates.YearLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for spine in ["top", "right"]:
    axes[2].spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_0_tokyo_stationary.png"), dpi=150)
plt.show()

# Why does it matter for modeling?
#   Most classical models (ARIMA, linear regression) assume stationarity.
#   If the series has a trend or changing variance, the model parameters
#   estimated on one period won't generalize to another.


# =============================================================================
# BLOCK 2 — Visual: raw prices vs daily returns
# =============================================================================

# The difference is immediate:
#   - Price: clear upward trend, variance growing over time → non-stationary
#   - Returns: fluctuate around zero with no obvious drift → stationary

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(close.index, close.values, linewidth=0.8, color="steelblue")
ax.set_title("AAPL — Raw Price (2015–2025) — clear upward trend, growing variance")
ax.set_ylabel("Closing Price (USD)")
ax.set_xlabel("Date")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_1_raw_price.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(returns.index, returns.values, linewidth=0.6, color="#555555", alpha=0.8)
ax.axhline(0, color="tomato", linewidth=0.8, linestyle="--")
ax.set_title("AAPL — Daily Returns (2015–2025) — fluctuates around zero, no drift")
ax.set_ylabel("Daily Return")
ax.set_xlabel("Date")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_1_daily_returns.png"), dpi=150)
plt.show()


# =============================================================================
# BLOCK 3 — Violations: rolling mean and variance on raw prices
# =============================================================================

# We now show explicitly where each stationarity condition is violated
# in the raw price series.
#
# Rolling window of 252 trading days (~1 year):
#   - Rolling mean: if stationary, this line would be flat.
#     Instead it trends steadily upward — condition 1 violated.
#   - Rolling std: if stationary, this would be flat.
#     Instead it grows with the price level — condition 2 violated.
#
# The post-2020 regime shift is also visible: after the COVID recovery,
# the price settled at a structurally higher level with higher volatility —
# condition 3 (structural stability) violated.

window = 252

rolling_mean = close.rolling(window).mean()
rolling_std  = close.rolling(window).std()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle("AAPL — Stationarity Violations in Raw Price", fontsize=12)

# Mean
ax1.plot(close.index, close.values, linewidth=0.6, color="#AACCE8", label="Price")
ax1.plot(rolling_mean.index, rolling_mean.values, linewidth=1.5, color="steelblue",
         label=f"Rolling Mean ({window}d)")
ax1.axvspan("2020-03-23", "2025-12-31", alpha=0.07, color="orange",
            label="Post-COVID regime shift")
ax1.set_ylabel("USD")
ax1.set_title("Condition 1 violated: mean is not constant (growing trend)")
ax1.legend(fontsize=8, loc="upper left")
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

# Std
ax2.plot(rolling_std.index, rolling_std.values, linewidth=1.5, color="tomato",
         label=f"Rolling Std ({window}d)")
ax2.axvspan("2020-03-23", "2025-12-31", alpha=0.07, color="orange",
            label="Post-COVID regime shift")
ax2.set_ylabel("USD")
ax2.set_xlabel("Date")
ax2.set_title("Condition 2 violated: variance grows with price level")
ax2.legend(fontsize=8, loc="upper left")
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_2_rolling_stats.png"), dpi=150)
plt.show()


# =============================================================================
# BLOCK 4 — Formalization: Augmented Dickey-Fuller test
# =============================================================================

# The ADF test formalizes what we already saw visually.
# It tests whether a series has a "unit root" — a statistical signature
# of non-stationarity where shocks accumulate over time instead of fading.
#
# The test equation is:
#   Δy_t = α + β·t + φ·y_{t-1} + Σ γ_i·Δy_{t-i} + ε_t
#
# The key coefficient is φ:
#   - If φ = 0 (unit root), shocks are permanent → non-stationary
#   - If φ < 0, shocks decay → stationary
#
# H₀ (null hypothesis): φ = 0 → series has a unit root → non-stationary
# H₁ (alternative):     φ < 0 → series is stationary
#
# We reject H₀ when p-value < 0.05.

def run_adf(series, label):
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pvalue = result[0], result[1]
    critical = result[4]
    conclusion = "STATIONARY (reject H₀)" if pvalue < 0.05 else "NON-STATIONARY (fail to reject H₀)"

    print(f"\n--- ADF Test: {label} ---")
    print(f"  Test statistic : {stat:.4f}")
    print(f"  p-value        : {pvalue:.6f}")
    print(f"  Critical values:")
    for level, val in critical.items():
        print(f"    {level}: {val:.4f}")
    print(f"  Conclusion     : {conclusion}")
    return pvalue

print("\n=== Block 4 — ADF Test Results ===")
pvalue_price   = run_adf(close,   "Raw Price")
pvalue_returns = run_adf(returns, "Daily Returns")


# =============================================================================
# BLOCK 5 — Conclusion
# =============================================================================

# The ADF test confirms what the charts already showed:
#   - Raw prices: p-value >> 0.05 → non-stationary (unit root present)
#   - Daily returns: p-value << 0.05 → stationary
#
# For classical models (ARIMA and others in the next sections), we will work
# with daily returns, not raw prices.
#
# However, there is a catch relevant to this project:
#   TimesFM was designed to forecast the raw series — not returns.
#   When we apply it later, we will need to convert its price predictions
#   back into return space for fair comparison, or work with returns and
#   reconstruct prices by cumulative product.
#
# This tension — raw series vs stationary transformation — will be a recurring
# theme throughout the project.

print("\n=== Script 02 complete ===")
print("Next: scripts/03_autocorrelation.py")
