"""
Script 02_3 — Decomposition
=============================
Time series decomposition separates a series into structural components:
trend, seasonality, and residual. For stock prices, this helps identify
whether any repeating patterns exist — and how much of the signal is
unexplained noise.

Blocks:
  Block 0 — Intuition: Tokyo monthly temperature (clean example)
  Block 1 — U.S. Retail Sales STL (log scale, monthly)
  Block 2 — AAPL STL (log scale, daily)
  Block 3 — Residual analysis (AAPL)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from meteostat import Station, daily

from utils.data import load_stock

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "charts")

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.titlesize": 14,
})

ticker = "AAPL"
df = load_stock(ticker, start="2015-01-01", end="2025-12-31")
close = df["Close"]

# =============================================================================
# BLOCK 0 — Intuition: decomposition on a clean dataset (Tokyo temperature)
# =============================================================================

# Before decomposing AAPL — where components are subtle — we illustrate the
# concept on Tokyo monthly temperature (Station 47662, same as 02_1).
# This dataset has:
#   - A mild upward trend (global warming, especially visible post-2022)
#   - Strong, regular seasonality (cold winters ~5°C, hot summers ~28°C)
#   - A residual that captures anomalous months relative to the expected cycle
#
# When the three components are this obvious, decomposition is easy to read.
# Keeping this reference in mind makes the AAPL result easier to interpret.

tokyo_raw     = daily(Station("47662"), start=pd.Timestamp("2015-01-01"),
                      end=pd.Timestamp("2025-12-31")).fetch()
tokyo_monthly = tokyo_raw["temp"].resample("ME").mean().dropna()

stl_tokyo       = STL(tokyo_monthly, seasonal=13, robust=True).fit()
tokyo_resid_pct = (stl_tokyo.resid / stl_tokyo.trend) * 100

# --- Pre-compute retail and AAPL STLs for shared residual y-axis ---
url    = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=RSXFS"
retail = pd.read_csv(StringIO(requests.get(url, timeout=10).text),
                     parse_dates=["observation_date"],
                     index_col="observation_date")
retail = retail.rename(columns={"RSXFS": "sales"})
log_retail          = np.log(retail["sales"].dropna())
stl_retail          = STL(log_retail, seasonal=13, robust=True).fit()
seasonal_pct_retail = (np.exp(stl_retail.seasonal) - 1) * 100
residual_pct_retail = (np.exp(stl_retail.resid)    - 1) * 100

log_close           = np.log(close)
stl_price           = STL(log_close, period=252, robust=True).fit()
seasonal_pct_price  = (np.exp(stl_price.seasonal) - 1) * 100
residual_pct_price  = (np.exp(stl_price.resid)    - 1) * 100

residuals_all = [
    tokyo_resid_pct.values,
    residual_pct_retail.values,
    residual_pct_price.values,
]
global_max    = max(np.nanmax(np.abs(r)) for r in residuals_all)
residual_ylim = (-global_max * 1.15, global_max * 1.15)
# -------------------------------------------------------------------

trend_x    = np.arange(len(stl_tokyo.trend.dropna()))
trend_y    = stl_tokyo.trend.dropna().values
trend_fit  = np.poly1d(np.polyfit(trend_x, trend_y, 1))
trend_line = pd.Series(trend_fit(trend_x), index=stl_tokyo.trend.dropna().index)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Tokyo Monthly Temperature — STL Decomposition (2015–2025)")

tokyo_components = [
    (tokyo_monthly,      "°C",          "Original — monthly average temperature (cold winters, hot summers)", "steelblue"),
    (stl_tokyo.trend,    "°C",          "Trend — slow-moving direction; subtle upward shift post-2022",       "steelblue"),
    (stl_tokyo.seasonal, "°C",          "Seasonal — repeating annual cycle (°C deviation from trend)",        "seagreen"),
    (tokyo_resid_pct,    "Residual (%)", "Residual — unexplained variation after removing trend and seasonality", "#555555"),
]

for ax, (data, ylabel, title, color) in zip(axes, tokyo_components):
    ax.plot(data.index, data.values, linewidth=0.8, color=color)
    if "Trend" in title:
        ax.plot(trend_line.index, trend_line.values, linewidth=1.0,
                color="#AAAAAA", linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if "Residual" in title:
        ax.axhline(0, color="tomato", linewidth=0.6, linestyle="--")
        ax.set_ylim(residual_ylim)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_3_stl_tokyo.png"), dpi=150)
plt.show()


# =============================================================================
# BLOCK 1 — U.S. Retail Sales STL (log scale, monthly)
# =============================================================================

# A second reference dataset before looking at AAPL.
# U.S. retail sales (FRED: RSXFS) has all three components in sharp relief:
#   - Trend: steady consumer spending growth over decades
#   - Seasonal: December holiday spike every year without exception
#   - Residual: COVID lockdown in April 2020 appears as a dramatic negative spike
#
# Decomposing on log scale makes the seasonal component additive and stable —
# equivalent to expressing it as a % deviation from trend.

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("U.S. Retail Sales (RSXFS, monthly log scale) — STL Decomposition")

retail_components = [
    (log_retail,            "Log Sales",           "Original — long-term growth with recurring annual spikes",           "steelblue"),
    (stl_retail.trend,      "Log Sales",           "Trend — slow-moving direction, unaffected by seasonality",           "steelblue"),
    (seasonal_pct_retail,   "Seasonal Effect (%)", "Seasonal — repeating December holiday spike every year",             "seagreen"),
    (residual_pct_retail,   "Residual (%)",        "Residual — unexplained noise; COVID crash visible Apr 2020",         "#555555"),
]

for ax, (data, ylabel, title, color) in zip(axes, retail_components):
    ax.plot(data.index, data.values, linewidth=0.8, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if "Residual" in title:
        ax.axhline(0, color="tomato", linewidth=0.6, linestyle="--")
        ax.axvline(pd.Timestamp("2020-04-01"), color="tomato", linewidth=1.0,
                   linestyle="--", label="COVID crash (Apr 2020)")
        ax.legend(loc="lower left")
        ax.set_ylim(residual_ylim)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_3_stl_retail.png"), dpi=150)
plt.show()


# =============================================================================
# BLOCK 2 — AAPL STL (log scale, daily)
# =============================================================================

# Now the same decomposition applied to AAPL daily closing price.
# The contrast with the previous two datasets is instructive:
#   - Trend: dominates, as expected for a growth stock
#   - Seasonal: present but much weaker than retail sales — and inconsistent
#   - Residual: spiky, driven by earnings, macro events, and product launches
#
# Period of 252 = approximate number of trading days per year.

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("AAPL Closing Price (daily, log scale) — STL Decomposition (period=252)")

price_components = [
    (log_close,            "Log Price",           "Original — daily closing price (log scale)",                                    "steelblue"),
    (stl_price.trend,      "Log Price",           "Trend — slow-moving direction extracted by STL",                                "steelblue"),
    (seasonal_pct_price,   "Seasonal Effect (%)", "Seasonal — repeating annual cycle — if any pattern exists, it appears here",   "seagreen"),
    (residual_pct_price,   "Residual (%)",        "Residual — unexplained variation after removing trend and seasonality",         "#555555"),
]

for ax, (data, ylabel, title, color) in zip(axes, price_components):
    ax.plot(data.index, data.values, linewidth=0.8, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if "%" in ylabel:
        ax.axhline(0, color="tomato", linewidth=0.6, linestyle="--")
    if "Residual" in title:
        ax.set_ylim(residual_ylim)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_3_stl_price.png"), dpi=150)
plt.show()

dates_of_interest = [
    "2018-10-01",
    "2019-01-03",
    "2020-08-21",
    "2022-12-28",
    "2023-01-03",
]

print("\n=== STL Component Breakdown for Key Dates ===")
for date in dates_of_interest:
    ts = pd.Timestamp(date)
    if ts not in close.index:
        ts = close.index[close.index.get_indexer([ts], method='nearest')[0]]

    price    = close[ts]
    trend    = stl_price.trend[ts]
    seasonal = stl_price.seasonal[ts]
    resid_v  = stl_price.resid[ts]

    price_from_trend = np.exp(trend)
    seasonal_pct     = (np.exp(seasonal) - 1) * 100
    resid_pct        = (np.exp(resid_v)  - 1) * 100

    print(f"\n{ts.date()}")
    print(f"  Actual price:        ${price:.2f}")
    print(f"  Trend (expected):    ${price_from_trend:.2f}")
    print(f"  Seasonal effect:     {seasonal_pct:+.1f}%")
    print(f"  Residual (surprise): {resid_pct:+.1f}%")


# =============================================================================
# BLOCK 3 — Residual analysis (AAPL)
# =============================================================================

# The residual from Block 2 captures what STL cannot explain with trend or
# seasonality. In AAPL, large spikes correspond to surprise events —
# earnings beats/misses, macro shocks, product announcements.
#
# Annotating the top 3 positive and negative dates connects the statistical
# output back to real-world events, and calibrates expectations: these are
# the moments that will always be hard to predict.

resid = residual_pct_price.dropna()

n       = 3
top_pos = resid.nlargest(n)
top_neg = resid.nsmallest(n)
spikes  = pd.concat([top_pos, top_neg]).sort_index()

# Compute staggered offsets to avoid overlapping labels.
# For each spike, base offset is proportional to its value magnitude.
# When two spikes are within 30 trading days of each other, alternate
# the text offset up/down to avoid collision.
dates  = list(spikes.index)
offsets = []
base   = 1.2
for i, (date, val) in enumerate(spikes.items()):
    direction = 1 if val > 0 else -1
    # Check if any previous spike is within 30 calendar days
    too_close = any(abs((date - dates[j]).days) < 45 for j in range(i))
    if too_close:
        direction *= -1.8  # push further in opposite direction for separation
    offsets.append(direction * base)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(resid.index, resid.values, linewidth=0.6, color="#555555", alpha=0.7)
ax.axhline(0, color="tomato", linewidth=0.6, linestyle="--")

for (date, val), offset in zip(spikes.items(), offsets):
    color  = "seagreen" if val > 0 else "tomato"
    valign = "bottom" if offset > 0 else "top"
    ax.scatter(date, val, color=color, s=40, zorder=5)
    ax.annotate(date.strftime("%Y-%m-%d"), xy=(date, val),
                xytext=(date, val + offset),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
                fontsize=10, color=color, ha="center", va=valign)

ax.set_title("AAPL — STL Residuals: largest unexplained price movements")
ax.set_ylabel("Residual (%)")
ax.set_xlabel("Date")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "02_3_residuals.png"), dpi=150)
plt.show()

print("\n=== Script 02_3 complete ===")
print("Next: scripts/02_4_volatility.py")
