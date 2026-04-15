# Stock Price Forecasting with TimesFM

A hands-on study of [TimesFM](https://github.com/google-research/timesfm) — Google's foundation model for time series forecasting — applied to financial data. The goal is not to build a production-ready stock predictor, but to understand how foundation models approach time series problems, where they perform well, and where they fall short.

The focus asset is AAPL (Apple), using daily closing prices from 2015 to 2025.

---

## Sections

1. Exploratory Data Analysis
2. Time Series Fundamentals
3. Baseline Models
4. Foundation Models for Time Series
5. TimesFM — Zero-shot
6. TimesFM — With Covariates
7. TimesFM — Fine-tuning
8. Final Evaluation

---

## Repository Structure

```
├── data/
│   ├── raw/          # Raw price data downloaded via yfinance (AAPL, GOOGL, AMZN, MSFT, META, ^GSPC)
│   ├── processed/    # Processed datasets (e.g. AAPL vs Fed rate aligned series)
│   └── charts/       # Generated charts from EDA scripts
├── scripts/
│   ├── 01_eda.py              # EDA — price evolution, COVID impact, annual performance
│   └── 01_eda_fed_rates.py    # EDA — AAPL vs Federal Funds Rate, Big Tech, S&P 500
├── utils/
│   ├── data.py       # Data loading and caching utilities
│   ├── metrics.py    # Evaluation metrics
│   └── style.py      # Chart style helpers
├── pyproject.toml
└── uv.lock
```

---

## Stage 1 — Exploratory Data Analysis

Before any modeling, the goal was to understand AAPL's historical behavior and identify macroeconomic drivers that could inform feature engineering.

**Key findings:**

- AAPL grew ~10x from 2015 to 2025, but the growth was non-linear — years of flat regime followed by a structural shift post-2020
- The COVID crash (Feb–Mar 2020) was driven by general market panic, supply chain disruption (Foxconn), and fear of demand collapse — external shocks with no signal in historical price data
- The recovery was partially explained by the Fed's rate cut to near-zero in March 2020, which drove capital into equities
- The inverse relationship between Fed rates and AAPL price holds consistently across four key monetary policy events (2020–2024) and extends to all Big Tech stocks and the S&P 500 — confirming it is a macro capital allocation effect, not company-specific

**Implication for modeling:** Fed rate will be incorporated as an exogenous covariate in later stages.

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Run EDA
uv run scripts/01_eda.py
uv run scripts/01_eda_fed_rates.py
```

---

## Notes

- Raw data is downloaded via `yfinance` and cached locally in `data/raw/` to avoid redundant API calls
- TimesFM was trained on data up to approximately 2023. To avoid data contamination, the test period for all TimesFM evaluations is restricted to **2024–2025**