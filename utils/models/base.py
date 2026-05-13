from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, close: pd.Series, **kwargs) -> "BaseModel":
        """Fit model on training data. Returns self for chaining."""
        ...

    @abstractmethod
    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        """Predict horizon steps ahead. Returns array of predicted prices."""
        ...

    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Compute MAE, RMSE, MAPE. Can be overridden."""
        from utils.models.metrics import mae, rmse, mape
        return {
            "MAE":  round(mae(actual, predicted), 2),
            "RMSE": round(rmse(actual, predicted), 2),
            "MAPE": round(mape(actual, predicted), 2),
        }

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} ({status})"
