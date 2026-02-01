import numpy as np

class ProphetModel:
    """
    Prophet model:
    y(t) = trend(t) + seasonality(t) + noise

    Features:
    - Linear or piecewise-linear trend with changepoints
    - Fourier seasonality (yearly / custom period)
    - L2-regularized least squares (MAP-like)

    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        seasonality_period: int = 365,
        fourier_order: int = 10,
        changepoints=None,
        trend_prior_scale: float = 10.0,
        seasonality_prior_scale: float = 10.0,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.seasonality_period = seasonality_period
        self.fourier_order = fourier_order
        self.changepoints = changepoints
        self.trend_prior_scale = trend_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale

        # learned quantities
        self.coef_ = None
        self.trend_ = None
        self.seasonality_ = None

    # --------------------------------------------------
    # Design matrices
    # --------------------------------------------------
    def _trend_matrix(self, t):
        X = [np.ones_like(t), t]

        if self.changepoints is not None:
            for cp in self.changepoints:
                X.append(np.maximum(0, t - cp))

        return np.column_stack(X)

    def _seasonality_matrix(self, t):
        if not self.yearly_seasonality:
            return None
        X = []
        for k in range(1, self.fourier_order + 1):
            X.append(np.sin(2 * np.pi * k * t / self.seasonality_period))
            X.append(np.cos(2 * np.pi * k * t / self.seasonality_period))
        return np.column_stack(X)

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(self, t, y):
        t = np.asarray(t)
        y = np.asarray(y)

        x_trend = self._trend_matrix(t)
        x_season = self._seasonality_matrix(t)

        if x_season is not None:
            x = np.column_stack([x_trend, x_season])
        else:
            x = x_trend

        # ----- Ridge regularization (MAP (Maximum A Posteriori) estimate) -----
        n_params = x.shape[1]
        reg = np.zeros((n_params, n_params))

        # Trend regularization
        reg[: x_trend.shape[1], : x_trend.shape[1]] = np.eye(x_trend.shape[1]) / self.trend_prior_scale**2

        # Seasonality regularization
        if x_season is not None:
            s = x_trend.shape[1]
            reg[s:, s:] = np.eye(x_season.shape[1]) / self.seasonality_prior_scale**2

        # Closed-form ridge solution
        xtx = x.T @ x + reg
        xty = x.T @ y
        self.coef_ = np.linalg.solve(xtx, xty)

        # Decompose fitted components
        self.trend_ = x_trend @ self.coef_[: x_trend.shape[1]]
        if x_season is not None:
            self.seasonality_ = x_season @ self.coef_[x_trend.shape[1] :]
        else:
            self.seasonality_ = np.zeros_like(self.trend_)

        return self

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    def predict(self, t):
        t = np.asarray(t)

        X_trend = self._trend_matrix(t)
        X_season = self._seasonality_matrix(t)

        if X_season is not None:
            X = np.column_stack([X_trend, X_season])
        else:
            X = X_trend

        return X @ self.coef_
