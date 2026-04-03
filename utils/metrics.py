"""
Métricas de avaliação para modelos de Time Series.

As mais usadas em forecasting:
- MAE  (Mean Absolute Error):      erro médio em unidades da série — fácil de interpretar
- RMSE (Root Mean Squared Error):  penaliza erros grandes mais que o MAE
- MAPE (Mean Absolute Pct Error):  erro relativo em % — útil para comparar séries diferentes
- SMAPE (Symmetric MAPE):          variante do MAPE mais estável quando os valores são próximos de zero
"""
import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def rmse(y_true, y_pred) -> float:
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def summary(y_true, y_pred, model_name: str = "model") -> pd.Series:
    """Retorna todas as métricas num pd.Series."""
    return pd.Series({
        "MAE":   round(mae(y_true, y_pred), 4),
        "RMSE":  round(rmse(y_true, y_pred), 4),
        "MAPE":  round(mape(y_true, y_pred), 4),
        "SMAPE": round(smape(y_true, y_pred), 4),
    }, name=model_name)
