"""
Script 03 — ARIMA
==================
ARIMA = AutoRegressive Integrated Moving Average
É o modelo clássico de Time Series, baseline obrigatório antes de partir para ML/DL.

Componentes:
  AR(p) — AutoRegressive:     y_t depende de y_{t-1}, ..., y_{t-p}
  I(d)  — Integrated:         diferenciação para tornar a série estacionária
  MA(q) — Moving Average:     y_t depende dos erros passados e_{t-1}, ..., e_{t-q}

ARIMA(p, d, q):
  p = lags autoregressivos (lido no PACF)
  d = grau de diferenciação (quantas vezes diferenciar para estacionarizar)
  q = lags do erro (lido no ACF após diferenciação)

Neste script:
1. Escolha dos hiperparâmetros p, d, q
2. Ajuste do modelo em dados de treino
3. Previsão e avaliação no conjunto de teste
4. Comparação com os baselines do script 02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils.data import get_close
from utils.metrics import summary

# ---------------------------------------------------------------------------
# 1. Dados
# ---------------------------------------------------------------------------
close = get_close("AAPL", start="2015-01-01", end="2024-12-31")

TEST_SIZE = 252
train = close.iloc[:-TEST_SIZE]
test  = close.iloc[-TEST_SIZE:]

# ---------------------------------------------------------------------------
# 2. Escolha do grau de diferenciação (d)
# ---------------------------------------------------------------------------
# d=1 significa: em vez de modelar y_t, modelamos Δy_t = y_t - y_{t-1}
# Isso remove tendência e geralmente deixa a série estacionária.

diff1 = train.diff().dropna()

def run_adf(series, label):
    p = adfuller(series)[1]
    status = "✓ estacionária" if p < 0.05 else "✗ não estacionária"
    print(f"  {label}: p={p:.6f} → {status}")

print("--- Teste ADF ---")
run_adf(train,  "Close bruto")
run_adf(diff1,  "Close 1ª diferença")

# ---------------------------------------------------------------------------
# 3. ACF / PACF da série diferenciada → escolha de p e q
# ---------------------------------------------------------------------------
# Após d=1, analisamos:
#   PACF: se corta no lag p → AR(p)
#   ACF:  se corta no lag q → MA(q)
# Um corte abrupto em lag k significa que colocar k como parâmetro é suficiente.

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(diff1, lags=30, ax=axes[0], title="ACF — Close diferenciado (d=1)")
plot_pacf(diff1, lags=30, ax=axes[1], title="PACF — Close diferenciado (d=1)", method="ywm")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../data/03_arima_acf.png"), dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 4. ARIMA estático — prevê todo o horizonte de teste de uma vez
# ---------------------------------------------------------------------------
# Parâmetros escolhidos: ARIMA(2, 1, 2)
# d=1: série precisa de 1 diferenciação
# p=2, q=2: valores conservadores lidos nos gráficos ACF/PACF
# Na prática, usa-se auto_arima (pmdarima) para busca automática.

print("\n--- Ajustando ARIMA(2,1,2) ---")
model_static = ARIMA(train, order=(2, 1, 2))
fit_static   = model_static.fit()

print(fit_static.summary().tables[0])

# Previsão para os próximos TEST_SIZE passos
forecast_static = fit_static.forecast(steps=TEST_SIZE)
forecast_static.index = test.index

metrics_static = summary(test, forecast_static, "ARIMA(2,1,2) estático")
print("\n", metrics_static.to_frame().T.to_string())


# ---------------------------------------------------------------------------
# 5. ARIMA Walk-Forward — re-treina a cada passo (mais realista)
# ---------------------------------------------------------------------------
# Em produção, você re-treinaria o modelo com novos dados periodicamente.
# Walk-forward com step=1 é a simulação mais rigorosa, mas lenta.
# Aqui usamos step=5 (semanal) para equilibrar velocidade e realismo.

STEP = 5
print(f"\n--- ARIMA Walk-Forward (step={STEP} dias) --- [pode demorar alguns segundos]")

history = list(train.values)
wf_preds = []

for i in range(0, len(test), STEP):
    model = ARIMA(history, order=(2, 1, 2))
    fit   = model.fit()
    steps_ahead = min(STEP, len(test) - i)
    fc = fit.forecast(steps=steps_ahead)
    wf_preds.extend(fc.tolist())
    # Incorpora os valores reais ao histórico
    history.extend(test.iloc[i:i+steps_ahead].values)

wf_series = pd.Series(wf_preds[:len(test)], index=test.index)
metrics_wf = summary(test, wf_series, f"ARIMA(2,1,2) walk-forward step={STEP}")
print(metrics_wf.to_frame().T.to_string())


# ---------------------------------------------------------------------------
# 6. Comparação final
# ---------------------------------------------------------------------------
# Baselines do script 02 (recalculados aqui para comparação)
naive_pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)
ma20_pred  = pd.Series([train.rolling(20).mean().iloc[-1]] * len(test), index=test.index)

results = pd.DataFrame([
    summary(test, naive_pred,      "Naive"),
    summary(test, ma20_pred,       "Moving Avg (20d)"),
    metrics_static,
    metrics_wf,
]).set_index(pd.Index(["Naive", "Moving Avg (20d)", "ARIMA estático", f"ARIMA walk-forward"]))

print("\n--- Comparação Geral ---")
print(results.sort_values("MAPE").to_string())


# ---------------------------------------------------------------------------
# 7. Visualização
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

context = train.iloc[-60:]
ax.plot(context.index, context.values, color="steelblue", linewidth=1.2, label="Treino (contexto)")
ax.plot(test.index, test.values, color="black", linewidth=1.5, label="Real")
ax.plot(forecast_static.index, forecast_static.values, linestyle="--", color="tomato", label="ARIMA estático")
ax.plot(wf_series.index, wf_series.values, linestyle="--", color="seagreen", label=f"ARIMA walk-forward (step={STEP})")
ax.plot(naive_pred.index, naive_pred.values, linestyle=":", color="gray", alpha=0.6, label="Naive")

ax.axvline(test.index[0], color="gray", linestyle=":", linewidth=1)
ax.set_title("AAPL — ARIMA vs Baselines")
ax.set_ylabel("USD")
ax.legend()
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../data/03_arima.png"), dpi=150)
plt.show()
print("[03] Gráfico salvo em data/03_arima.png")

print("\n=== Script 03 concluído ===")
print("Próximo: scripts/04_timesfm.py")
