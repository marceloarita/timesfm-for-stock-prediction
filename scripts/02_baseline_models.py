"""
Script 02 — Modelos Baseline
==============================
Objetivo: estabelecer benchmarks simples ANTES de usar modelos sofisticados.

REGRA DE OURO em forecasting: sempre compare com um baseline.
Se seu modelo complexo não bate o modelo ingênuo, há algo errado.

Baselines implementados:
1. Naive (persistência):  ŷ_{t+h} = y_t  — "o futuro = último valor"
2. Seasonal Naive:        ŷ_{t+h} = y_{t-s+h}  — repete o mesmo período do ciclo anterior
3. Moving Average:        média dos últimos k valores

Conceito de Train/Test Split em Time Series:
- NÃO podemos fazer split aleatório (vaza informação do futuro para o treino)
- Sempre cortamos cronologicamente: treino = passado, teste = futuro
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils.data import get_close
from utils.metrics import summary

# ---------------------------------------------------------------------------
# 1. Dados e split treino/teste
# ---------------------------------------------------------------------------
close = get_close("AAPL", start="2015-01-01", end="2024-12-31")

# Usaremos os últimos 252 dias úteis (~1 ano) como conjunto de teste.
# Isso é uma convenção comum em backtesting de ações.
TEST_SIZE = 252
train = close.iloc[:-TEST_SIZE]
test  = close.iloc[-TEST_SIZE:]

print(f"Treino: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} dias)")
print(f"Teste:  {test.index[0].date()}  → {test.index[-1].date()}  ({len(test)} dias)")


# ---------------------------------------------------------------------------
# 2. Definição dos modelos baseline
# ---------------------------------------------------------------------------

def naive_forecast(train: pd.Series, horizon: int) -> pd.Series:
    """
    Naive (persistência): prevê que o próximo valor = último valor observado.
    É o modelo mais simples possível e surpreendentemente difícil de bater
    para séries com forte componente de tendência.
    """
    last_value = train.iloc[-1]
    return pd.Series([last_value] * horizon, index=test.index)


def seasonal_naive_forecast(train: pd.Series, horizon: int, season: int = 5) -> pd.Series:
    """
    Seasonal Naive: repete os últimos `season` valores observados ciclicamente.
    season=5 para dias úteis (semana de trabalho).
    """
    tail = train.iloc[-season:].values
    preds = [tail[i % season] for i in range(horizon)]
    return pd.Series(preds, index=test.index)


def moving_average_forecast(train: pd.Series, horizon: int, window: int = 20) -> pd.Series:
    """
    Moving Average: prevê com a média dos últimos `window` valores.
    window=20 ≈ 1 mês de dias úteis (convenção comum em análise técnica).
    Simples mas suaviza ruído e captura tendência de curto prazo.
    """
    ma = train.rolling(window).mean().iloc[-1]
    return pd.Series([ma] * horizon, index=test.index)


# ---------------------------------------------------------------------------
# 3. Previsões
# ---------------------------------------------------------------------------
horizon = len(test)

preds = {
    "Naive":           naive_forecast(train, horizon),
    "Seasonal Naive":  seasonal_naive_forecast(train, horizon, season=5),
    "Moving Avg (20d)": moving_average_forecast(train, horizon, window=20),
    "Moving Avg (60d)": moving_average_forecast(train, horizon, window=60),
}


# ---------------------------------------------------------------------------
# 4. Métricas
# ---------------------------------------------------------------------------
print("\n--- Métricas de Avaliação (conjunto de TESTE) ---")
results = pd.DataFrame({name: summary(test, pred, name) for name, pred in preds.items()}).T
print(results.to_string())

# Interpretação rápida das métricas:
print("\nInterpretação:")
print("  MAE   = erro médio em USD (quanto erramos em média)")
print("  RMSE  = erro médio penalizando erros grandes")
print("  MAPE  = erro relativo em % do valor real")
print("  SMAPE = MAPE simétrico, mais estável")


# ---------------------------------------------------------------------------
# 5. Visualização
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

# Contexto: mostra os últimos 60 dias de treino + todo o período de teste
context = train.iloc[-60:]
ax.plot(context.index, context.values, color="steelblue", linewidth=1.2, label="Treino (contexto)")
ax.plot(test.index, test.values, color="black", linewidth=1.5, label="Real (teste)")

colors = ["tomato", "seagreen", "darkorange", "purple"]
for (name, pred), color in zip(preds.items(), colors):
    ax.plot(pred.index, pred.values, linewidth=1, linestyle="--", color=color, label=name)

ax.axvline(test.index[0], color="gray", linestyle=":", linewidth=1)
ax.set_title("AAPL — Baselines vs Real (último ano)")
ax.set_ylabel("USD")
ax.legend()
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../data/02_baselines.png"), dpi=150)
plt.show()
print("[02] Gráfico salvo em data/02_baselines.png")


# ---------------------------------------------------------------------------
# 6. Walk-Forward Validation (conceito avançado de baseline)
# ---------------------------------------------------------------------------
# O que fizemos acima é uma previsão estática: treinamos UMA vez e prevemos TUDO.
# Mas na prática, modelos são re-treinados conforme novos dados chegam.
# Walk-Forward (ou expanding window) simula isso: prevemos 1 passo de cada vez,
# sempre incorporando o valor real ao treino antes da próxima previsão.

print("\n--- Walk-Forward: Naive 1-step ahead ---")
wf_preds = []
for i in range(len(test)):
    # A previsão para o dia i é o último valor conhecido (dia i-1 do teste, ou fim do treino)
    if i == 0:
        wf_preds.append(train.iloc[-1])
    else:
        wf_preds.append(test.iloc[i - 1])

wf_series = pd.Series(wf_preds, index=test.index)
wf_metrics = summary(test, wf_series, "Naive Walk-Forward 1d")
print(wf_metrics.to_frame().T.to_string())

print("\n=== Script 02 concluído ===")
print("Próximo: scripts/03_arima.py")
