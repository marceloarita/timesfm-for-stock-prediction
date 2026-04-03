"""
Script 04 — TimesFM (Google)
==============================
TimesFM é um modelo fundacional (Foundation Model) para previsão de séries temporais.
Desenvolvido pelo Google Research, publicado em 2024.

Diferença fundamental de ARIMA:
  - ARIMA:    modelo estatístico, ajustado para CADA série individualmente
  - TimesFM:  pré-treinado em ~100 bilhões de pontos reais e sintéticos
              funciona ZERO-SHOT: prevê sem precisar de re-treino na sua série

O que é Zero-Shot Forecasting?
  Você passa uma janela de contexto (histórico recente) e o modelo prevê
  os próximos h passos sem ter visto essa série durante o treinamento.
  É análogo ao GPT respondendo perguntas sem fine-tuning.

Parâmetros importantes do TimesFM:
  - context_len:   tamanho da janela de entrada (quantos pontos passados o modelo recebe)
  - horizon_len:   quantos passos à frente prever
  - input_patch_len / output_patch_len: tamanho dos "patches" internos (como tokens em LLMs)
  - backend:       "cpu" ou "gpu" (para rodar localmente, usamos "cpu")

Referência: https://github.com/google-research/timesfm
Paper: "A decoder-only foundation model for time-series forecasting" (Das et al., 2024)
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

import timesfm

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
# 2. Inicializar o TimesFM
# ---------------------------------------------------------------------------
# TimesFM usa pesos pré-treinados que são baixados automaticamente na primeira execução.
# O modelo base tem 200M parâmetros — será baixado para ~/.cache/huggingface/
#
# backend="cpu" é suficiente para experimentos; use "gpu" se tiver GPU disponível.
# input_patch_len=32, output_patch_len=128 são os valores do modelo padrão.

print("Inicializando TimesFM (pode baixar pesos na primeira vez ~1GB)...")

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=32,
        horizon_len=128,          # máximo de passos que o modelo consegue prever de uma vez
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
    ),
)

print("TimesFM carregado.")


# ---------------------------------------------------------------------------
# 3. Previsão Zero-Shot estática
# ---------------------------------------------------------------------------
# Passamos uma janela de contexto (últimos N pontos do treino) e pedimos
# horizon_len passos à frente.
#
# context_len recomendado: pelo menos 4× o horizon que você quer prever.
# TimesFM internamente divide o contexto em patches de 32 pontos.

CONTEXT_LEN  = 512   # ~2 anos de dados diários
FORECAST_LEN = 252   # 1 ano à frente (nosso teste)

# O modelo aceita FORECAST_LEN ≤ horizon_len (128 por padrão).
# Para prever 252 dias, precisamos fazer em chunks de 128 ou usar horizon_len=252.
# Vamos recriar o modelo com horizon_len=252 para simplificar.

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=32,
        horizon_len=FORECAST_LEN,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
    ),
)

context_values = train.values[-CONTEXT_LEN:]

# A API do TimesFM recebe uma lista de arrays (suporta batch de séries)
forecast_output = tfm.forecast(
    inputs=[context_values],
    freq=[0],          # 0 = alta frequência (intra-dia ou diário); 1 = semanal; 2 = mensal
)

# forecast_output é uma lista de arrays; pegamos o primeiro (nossa única série)
# Retorna shape (1, horizon_len) — pegamos o ponto médio da distribuição
point_forecast = forecast_output[0][0][:FORECAST_LEN]

forecast_series = pd.Series(point_forecast, index=test.index)
metrics_zero_shot = summary(test, forecast_series, "TimesFM zero-shot")
print("\n--- TimesFM Zero-Shot ---")
print(metrics_zero_shot.to_frame().T.to_string())


# ---------------------------------------------------------------------------
# 4. Walk-Forward com TimesFM
# ---------------------------------------------------------------------------
# Assim como no ARIMA, podemos re-aplicar o modelo deslizando a janela de contexto.
# Aqui usamos step=21 (~1 mês) para equilibrar velocidade e qualidade.

STEP = 21
print(f"\n--- TimesFM Walk-Forward (step={STEP} dias) ---")

wf_preds = []
all_train_values = list(train.values)

for i in range(0, len(test), STEP):
    steps_ahead = min(STEP, len(test) - i)

    # Recria modelo com horizon_len = steps_ahead
    tfm_wf = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=steps_ahead,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
        ),
    )

    ctx = np.array(all_train_values[-CONTEXT_LEN:])
    out = tfm_wf.forecast(inputs=[ctx], freq=[0])
    preds = out[0][0][:steps_ahead]
    wf_preds.extend(preds.tolist())

    # Incorpora valores reais
    all_train_values.extend(test.iloc[i:i+steps_ahead].values.tolist())
    print(f"  [{i+steps_ahead}/{len(test)}] ✓")

wf_series = pd.Series(wf_preds[:len(test)], index=test.index)
metrics_wf = summary(test, wf_series, f"TimesFM walk-forward step={STEP}")
print(metrics_wf.to_frame().T.to_string())


# ---------------------------------------------------------------------------
# 5. Comparação com baselines e ARIMA
# ---------------------------------------------------------------------------
naive_pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)
ma20_pred  = pd.Series([train.rolling(20).mean().iloc[-1]] * len(test), index=test.index)

results = pd.DataFrame([
    summary(test, naive_pred,     "Naive"),
    summary(test, ma20_pred,      "Moving Avg (20d)"),
    metrics_zero_shot,
    metrics_wf,
]).set_index(pd.Index(["Naive", "Moving Avg (20d)", "TimesFM zero-shot", f"TimesFM walk-forward"]))

print("\n--- Comparação: TimesFM vs Baselines ---")
print(results.sort_values("MAPE").to_string())


# ---------------------------------------------------------------------------
# 6. Visualização
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

context = train.iloc[-60:]

for ax, (pred, label, color) in zip(
    axes,
    [
        (forecast_series, "TimesFM zero-shot", "tomato"),
        (wf_series,       f"TimesFM walk-forward (step={STEP})", "seagreen"),
    ]
):
    ax.plot(context.index, context.values, color="steelblue", linewidth=1.2, label="Treino (contexto)")
    ax.plot(test.index, test.values, color="black", linewidth=1.5, label="Real")
    ax.plot(pred.index, pred.values, linestyle="--", color=color, linewidth=1.2, label=label)
    ax.plot(naive_pred.index, naive_pred.values, linestyle=":", color="gray", alpha=0.6, label="Naive")
    ax.axvline(test.index[0], color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"AAPL — {label}")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../data/04_timesfm.png"), dpi=150)
plt.show()
print("[04] Gráfico salvo em data/04_timesfm.png")

print("\n=== Script 04 concluído ===")
print("Próximo: scripts/05_comparison.py")
