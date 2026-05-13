import numpy as np
import pandas as pd
from utils.models.base import BaseModel


class NaiveModel(BaseModel):
    """
    Naive forecast: predict the last known price for all future steps.
    Serves as the minimum baseline — any useful model must beat this.
    """

    def __init__(self):
        super().__init__(name="Naive")
        self.last_price = None

    def fit(self, close: pd.Series, **kwargs) -> "NaiveModel":
        self.last_price = close.iloc[-1]
        self.is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        assert self.is_fitted, "Model must be fitted before predicting."
        return np.full(horizon, self.last_price)
