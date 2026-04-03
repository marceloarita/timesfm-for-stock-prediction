"""
Script 01 — Exploração dos Dados (AAPL)
========================================
Objetivo: entender a estrutura dos dados antes de modelar qualquer coisa.

Conceitos de Time Series introduzidos aqui:
- O que é uma série temporal?
- Componentes: tendência, sazonalidade, ruído
- Stationarity (estacionaridade) — pré-requisito para ARIMA
- Autocorrelação — a série está correlacionada com o passado?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from utils.data import load_stock, get_close

# ---------------------------------------------------------------------------
# 1. Download dos dados
# ---------------------------------------------------------------------------
ticker = "AAPL"
df = load_stock(ticker, start="2015-01-01", end="2024-12-31")

print("\n--- Informações básicas ---")
print(df.info())
print("\n--- Primeiros 5 registros ---")
print(df.head())
print("\n--- Estatísticas descritivas (Close) ---")
print(df["Close"].describe())


# ---------------------------------------------------------------------------
# 2. Visualização do preço de fechamento
# ---------------------------------------------------------------------------
# Uma série temporal é simplesmente uma sequência de valores ordenados no tempo.
# No nosso caso: preço de fechamento diário da AAPL.

close = df["Close"]

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# --- Preço bruto ---
ax = axes[0]
ax.plot(close.index, close.values, linewidth=0.8, color="steelblue")
ax.set_title(f"{ticker} — Preço de Fechamento (2015–2024)")
ax.set_ylabel("USD")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3)

# --- Retorno diário (log-return) ---
# Em finanças, trabalhamos frequentemente com retornos e não com preço absoluto.
# Log-return: r_t = log(P_t / P_{t-1})
# Vantagem: log-returns são mais estacionários que o preço bruto.
log_return = np.log(close / close.shift(1)).dropna()

ax = axes[1]
ax.plot(log_return.index, log_return.values, linewidth=0.5, color="dimgray")
ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
ax.set_title(f"{ticker} — Log-Return Diário")
ax.set_ylabel("log-return")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3)

# --- Volatilidade realizada (rolling std dos log-returns) ---
# Volatilidade = desvio padrão dos retornos numa janela deslizante.
# Alta volatilidade = maior incerteza no período.
rolling_vol = log_return.rolling(30).std() * np.sqrt(252)  # anualizada

ax = axes[2]
ax.fill_between(rolling_vol.index, rolling_vol.values, alpha=0.5, color="tomato")
ax.set_title(f"{ticker} — Volatilidade Realizada (janela 30d, anualizada)")
ax.set_ylabel("volatilidade")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../data/01_price_overview.png"), dpi=150)
plt.show()
print("[01] Gráfico salvo em data/01_price_overview.png")


# ---------------------------------------------------------------------------
# 3. Decomposição da série — Tendência, Sazonalidade, Ruído
# ---------------------------------------------------------------------------
# Toda série temporal pode ser decomposta em:
#   - Tendência (trend):      movimento de longo prazo (ex: AAPL crescendo ao longo dos anos)
#   - Sazonalidade (seasonal):padrões que se repetem em ciclos fixos (ex: todo dezembro sobe)
#   - Ruído (residual):       o que sobra — variações aleatórias inexplicadas
#
# Decomposição ADITIVA:  Y = Trend + Seasonal + Residual
# Decomposição MULT.:    Y = Trend × Seasonal × Residual (use quando a amplitude sazonal cresce com o nível)

from statsmodels.tsa.seasonal import seasonal_decompose

# Reamostrar para dados semanais para a decomposição ficar mais legível
close_weekly = close.resample("W").last()
decomp = seasonal_decompose(close_weekly, model="multiplicative", period=52)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
decomp.observed.plot(ax=axes[0], title="Observado", color="steelblue")
decomp.trend.plot(ax=axes[1], title="Tendência", color="darkorange")
decomp.seasonal.plot(ax=axes[2], title="Sazonalidade", color="seagreen")
decomp.resid.plot(ax=axes[3], title="Ruído (Resíduo)", color="gray")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../data/01_decomposition.png"), dpi=150)
plt.show()
print("[01] Decomposição salva em data/01_decomposition.png")


# ---------------------------------------------------------------------------
# 4. Teste de Estacionaridade — ADF (Augmented Dickey-Fuller)
# ---------------------------------------------------------------------------
# CONCEITO CHAVE: Estacionaridade
# Uma série é ESTACIONÁRIA se sua média, variância e autocorrelação NÃO mudam com o tempo.
# A maioria dos modelos clássicos (como ARIMA) exige estacionaridade.
# Preço de ações tipicamente NÃO é estacionário (tem tendência).
# Log-returns geralmente SÃO (ou são próximos de) estacionários.
#
# Teste ADF (Augmented Dickey-Fuller):
#   H0: a série NÃO é estacionária (tem raiz unitária)
#   H1: a série É estacionária
#   Se p-value < 0.05 → rejeita H0 → série é estacionária

def adf_test(series: pd.Series, name: str):
    result = adfuller(series.dropna())
    print(f"\n--- ADF Test: {name} ---")
    print(f"  ADF statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.6f}")
    print(f"  Lags usados   : {result[2]}")
    conclusion = "ESTACIONÁRIA" if result[1] < 0.05 else "NÃO estacionária"
    print(f"  Conclusão     : {conclusion} (p {'< 0.05' if result[1] < 0.05 else '>= 0.05'})")

adf_test(close, "Preço de fechamento (Close)")
adf_test(log_return, "Log-return diário")


# ---------------------------------------------------------------------------
# 5. Autocorrelação (ACF e PACF)
# ---------------------------------------------------------------------------
# ACF (Autocorrelation Function): correlação da série com ela mesma em diferentes lags.
#   lag 1 = correlação com o valor de ontem
#   lag 5 = correlação com o valor de 5 dias atrás
#
# PACF (Partial ACF): correlação com lag k removendo o efeito dos lags intermediários.
#
# Esses gráficos são usados para escolher os parâmetros p e q do ARIMA (próximo script).

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(close, lags=40, ax=axes[0, 0], title="ACF — Preço de Fechamento")
plot_pacf(close, lags=40, ax=axes[0, 1], title="PACF — Preço de Fechamento", method="ywm")
plot_acf(log_return.dropna(), lags=40, ax=axes[1, 0], title="ACF — Log-Return")
plot_pacf(log_return.dropna(), lags=40, ax=axes[1, 1], title="PACF — Log-Return", method="ywm")

for ax in axes.flat:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../data/01_acf_pacf.png"), dpi=150)
plt.show()
print("[01] ACF/PACF salvo em data/01_acf_pacf.png")

print("\n=== Script 01 concluído ===")
print("Próximo: scripts/02_baseline_models.py")
