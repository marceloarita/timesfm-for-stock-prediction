"""
Script 05 — Comparação Final
==============================
Consolida todos os modelos num painel único de avaliação.

Métricas usadas: MAE, RMSE, MAPE, SMAPE
Modelos comparados:
  1. Naive
  2. Moving Average (20d, 60d)
  3. ARIMA(2,1,2) estático
  4. ARIMA(2,1,2) walk-forward
  5. TimesFM zero-shot
  6. TimesFM walk-forward

Conceito: o objetivo de um modelo de forecasting não é prever o futuro perfeitamente,
mas sim ser MELHOR DO QUE O BASELINE de forma consistente.
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
import timesfm

from utils.data import get_close
from utils.metrics import summary

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
close = get_close("AAPL", start="2015-01-01", end="2024-12-31")
TEST_SIZE = 252
train = close.iloc[:-TEST_SIZE]
test  = close.iloc[-TEST_SIZE:]

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
naive_pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)
ma20_pred  = pd.Series([train.rolling(20).mean().iloc[-1]] * len(test), index=test.index)
ma60_pred  = pd.Series([train.rolling(60).mean().iloc[-1]] * len(test), index=test.index)

# ---------------------------------------------------------------------------
# ARIMA estático
# ---------------------------------------------------------------------------
print("Ajustando ARIMA estático...")
arima_static_pred = ARIMA(train, order=(2, 1, 2)).fit().forecast(steps=len(test))
arima_static_pred.index = test.index

# ---------------------------------------------------------------------------
# ARIMA walk-forward
# ---------------------------------------------------------------------------
print("ARIMA walk-forward...")
ARIMA_STEP = 5
history = list(train.values)
arima_wf_preds = []
for i in range(0, len(test), ARIMA_STEP):
    fit = ARIMA(history, order=(2, 1, 2)).fit()
    steps = min(ARIMA_STEP, len(test) - i)
    arima_wf_preds.extend(fit.forecast(steps=steps).tolist())
    history.extend(test.iloc[i:i+steps].values)
arima_wf_pred = pd.Series(arima_wf_preds[:len(test)], index=test.index)

# ---------------------------------------------------------------------------
# TimesFM zero-shot
# ---------------------------------------------------------------------------
print("TimesFM zero-shot...")
CONTEXT_LEN = 512
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=32, horizon_len=len(test)),
    checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
)
ctx = train.values[-CONTEXT_LEN:]
out = tfm.forecast(inputs=[ctx], freq=[0])
tfm_zero_pred = pd.Series(out[0][0][:len(test)], index=test.index)

# ---------------------------------------------------------------------------
# TimesFM walk-forward
# ---------------------------------------------------------------------------
print("TimesFM walk-forward...")
TFM_STEP = 21
all_vals = list(train.values)
tfm_wf_preds = []
for i in range(0, len(test), TFM_STEP):
    steps = min(TFM_STEP, len(test) - i)
    tfm_step = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=32, horizon_len=steps),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )
    ctx = np.array(all_vals[-CONTEXT_LEN:])
    out = tfm_step.forecast(inputs=[ctx], freq=[0])
    tfm_wf_preds.extend(out[0][0][:steps].tolist())
    all_vals.extend(test.iloc[i:i+steps].values.tolist())
tfm_wf_pred = pd.Series(tfm_wf_preds[:len(test)], index=test.index)

# ---------------------------------------------------------------------------
# Tabela de resultados
# ---------------------------------------------------------------------------
all_preds = {
    "Naive":                naive_pred,
    "Moving Avg 20d":       ma20_pred,
    "Moving Avg 60d":       ma60_pred,
    "ARIMA estático":       arima_static_pred,
    f"ARIMA WF step=5":     arima_wf_pred,
    "TimesFM zero-shot":    tfm_zero_pred,
    f"TimesFM WF step=21":  tfm_wf_pred,
}

results = pd.DataFrame({name: summary(test, pred, name) for name, pred in all_preds.items()}).T
results = results.sort_values("MAPE")

print("\n========== COMPARAÇÃO FINAL ==========")
print(results.to_string())
print("="*42)

# ---------------------------------------------------------------------------
# Gráfico 1 — Comparação de previsões
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 6))
context = train.iloc[-60:]
ax.plot(context.index, context.values, color="steelblue", linewidth=1.2, label="Treino")
ax.plot(test.index, test.values, color="black", linewidth=2.0, label="Real", zorder=10)

styles = [
    ("Naive",               naive_pred,        "gray",        ":"),
    ("Moving Avg 20d",      ma20_pred,         "lightcoral",  "--"),
    ("ARIMA estático",      arima_static_pred, "tomato",      "--"),
    ("ARIMA WF",            arima_wf_pred,     "darkorange",  "-"),
    ("TimesFM zero-shot",   tfm_zero_pred,     "mediumpurple","--"),
    ("TimesFM WF",          tfm_wf_pred,       "seagreen",    "-"),
]
for label, pred, color, ls in styles:
    ax.plot(pred.index, pred.values, linestyle=ls, color=color, linewidth=1, label=label, alpha=0.85)

ax.axvline(test.index[0], color="gray", linestyle=":", linewidth=1)
ax.set_title("AAPL — Todos os Modelos vs Real (2024)")
ax.set_ylabel("USD")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../data/05_comparison_forecasts.png"), dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Gráfico 2 — Bar chart das métricas
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metrics_to_plot = ["MAPE", "MAE"]

for ax, metric in zip(axes, metrics_to_plot):
    values = results[metric]
    colors = ["seagreen" if i == 0 else "steelblue" for i in range(len(values))]
    bars = ax.barh(values.index, values.values, color=colors)
    ax.set_title(f"{metric} por Modelo (menor = melhor)")
    ax.set_xlabel(metric)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values.values):
        ax.text(bar.get_width() + bar.get_width() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../data/05_comparison_metrics.png"), dpi=150)
plt.show()
print("[05] Gráficos salvos em data/05_comparison_*.png")

print("\n=== Script 05 concluído ===")
print("Próximos passos: explorar previsão de direção (alta/baixa) em scripts/06_direction.py")
