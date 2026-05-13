import numpy as np
import pandas as pd
from pmdarima import auto_arima
from utils.models.base import BaseModel


class ARIMAModel(BaseModel):
    """
    ARIMA on daily returns. Uses auto_arima to select p, d, q.
    Reconstructs prices via cumulative product of predicted returns.
    """

    def __init__(self, seasonal: bool = False, stepwise: bool = True,
                 information_criterion: str = "aic", **autoarima_kwargs):
        super().__init__(name="ARIMA")
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.information_criterion = information_criterion
        self.autoarima_kwargs = autoarima_kwargs
        self.model = None
        self.last_price = None
        self.order = None

    def fit(self, close: pd.Series, **kwargs) -> "ARIMAModel":
        returns = close.pct_change().dropna()
        self.model = auto_arima(
            returns,
            seasonal=self.seasonal,
            stepwise=self.stepwise,
            information_criterion=self.information_criterion,
            suppress_warnings=True,
            **self.autoarima_kwargs,
        )
        self.last_price = close.iloc[-1]
        self.order = self.model.order
        self.is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        assert self.is_fitted, "Model must be fitted before predicting."
        predicted_returns = np.asarray(self.model.predict(n_periods=horizon))
        return self.last_price * np.cumprod(1 + predicted_returns)
