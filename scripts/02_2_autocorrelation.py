"""
Script 02_2 — Autocorrelation
==============================
Does today's return tell us anything about tomorrow's?
This script uses ACF and PACF plots to answer that question directly,
and interprets what the result means for lag selection in ARIMA models.

Blocks:
  Block 1 — Intuition: what is autocorrelation?
  Block 2 — ACF: correlation between return(t) and return(t-k)
  Block 3 — PACF: direct effect after removing intermediate lags
  Block 4 — Conclusion: what this means for ARIMA and the project
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from meteostat import Station, daily

from utils.data import load_stock

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "charts")

ticker = "AAPL"
df = load_stock(ticker, start="2015-01-01", end="2025-12-31")
close = df["Close"]
returns = close.pct_change().dropna()

tokyo_raw = daily(Station("47662"), start=pd.Timestamp("2015-01-01"),
                  end=pd.Timestamp("2025-12-31")).fetch()
tokyo_temp = tokyo_raw["temp"].dropna()

# =============================================================================
# BLOCK 1 — Intuition: what is autocorrelation?
# =============================================================================

# Autocorrelation measures how much a series correlates with its own past values.
# At lag k, it answers: "if return was high k days ago, is it likely high today?"
#
# A lag-1 autocorrelation of +0.3 would mean: a positive return today makes a
# positive return tomorrow slightly more likely. Traders could exploit this.
#
# In practice, for liquid stocks like AAPL, we expect very little autocorrelation
# in returns — this is the core claim of the Efficient Market Hypothesis (EMH):
# prices already reflect all available information, so past returns have no
# predictive power over future returns.
#
# Two tools to visualize this:
#
#   ACF (Autocorrelation Function):
#     Plots correlation between return(t) and return(t-k) for each lag k.
#     Includes both direct and indirect effects through intermediate lags.
#
#   PACF (Partial Autocorrelation Function):
#     Plots the direct correlation at lag k, after removing the contribution
#     of all intermediate lags. More useful for identifying the AR order in ARIMA.
#
# The blue shaded band is the 95% confidence interval. Bars outside the band
# are statistically significant — inside means "consistent with pure noise."
#
# To make this concrete before looking at AAPL, we first plot the ACF of
# Tokyo's daily temperature. Temperature has strong autocorrelation by nature —
# if it's cold today, it's almost certainly cold tomorrow. The ACF will show
# bars well outside the confidence band for many lags. Then we compare with
# AAPL returns, where the signal disappears almost immediately.


# =============================================================================
# BLOCK 2 — ACF
# =============================================================================

# First: Tokyo temperature — strong autocorrelation, bars persist for many lags
fig, ax = plt.subplots(figsize=(14, 4))
plot_acf(tokyo_temp, lags=40, ax=ax, color="steelblue", vlines_kwargs={"colors": "steelblue"})
ax.set_title("Tokyo Daily Temperature — ACF (40 lags) — strong, persistent autocorrelation")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Correlation")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_2_acf_tokyo.png"), dpi=150)
plt.show()

# Now: AAPL returns — almost no autocorrelation
fig, ax = plt.subplots(figsize=(14, 4))
plot_acf(returns, lags=40, ax=ax, color="steelblue", vlines_kwargs={"colors": "steelblue"})
ax.set_title("AAPL Daily Returns — Autocorrelation Function (ACF), 40 lags")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Correlation")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_2_acf.png"), dpi=150)
plt.show()


# =============================================================================
# BLOCK 3 — PACF
# =============================================================================

# First: Tokyo temperature — lag-1 dominates, then collapses.
# This reveals that the long ACF tail is a cascading effect (lag-1 → lag-2 → ...),
# not genuine direct dependence at higher lags.
fig, ax = plt.subplots(figsize=(14, 4))
plot_pacf(tokyo_temp, lags=40, ax=ax, color="steelblue", vlines_kwargs={"colors": "steelblue"})
ax.set_title("Tokyo Daily Temperature — PACF (40 lags) — direct effect collapses after lag-1")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Partial Correlation")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_2_pacf_tokyo.png"), dpi=150)
plt.show()

# Now: AAPL returns
fig, ax = plt.subplots(figsize=(14, 4))
plot_pacf(returns, lags=40, ax=ax, color="steelblue", vlines_kwargs={"colors": "steelblue"})
ax.set_title("AAPL Daily Returns — Partial Autocorrelation Function (PACF), 40 lags")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Partial Correlation")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_2_pacf.png"), dpi=150)
plt.show()


# =============================================================================
# BLOCK 4 — Conclusion
# =============================================================================

# Both ACF and PACF show almost no statistically significant lags.
# Nearly all bars fall within the 95% confidence band — consistent with white noise.
#
# This is the expected result for a liquid, heavily traded stock like AAPL.
# If strong autocorrelation existed, it would be a known, exploitable pattern —
# and arbitrage would quickly eliminate it.
#
# What this means for modeling:
#   - ARIMA: the AR component (p) can start at 0 or 1. There is no clear signal
#     of a lag structure worth modeling in returns.
#   - TimesFM: does not assume stationarity or a specific lag structure —
#     it may capture subtler patterns that ACF/PACF miss.
#
# The absence of autocorrelation in returns does NOT mean the series is
# unpredictable. Volatility (squared returns) often shows strong autocorrelation
# — that is the subject of the next section.

print("\n=== Script 02_2 complete ===")
print("Next: scripts/02_3_decomposition.py")
