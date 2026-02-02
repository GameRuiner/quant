from models.arima.sarima_mle import SARIMAModel


class ARIMAModel(SARIMAModel):
    """
    ARIMA(p, d, q) model implemented as a special case of SARIMA.
    """

    def __init__(self, y, order=(1, 0, 0), trend=None):
        super().__init__(
            y=y,
            order=order,
            seasonal_order=(0, 0, 0, 0),
            trend=trend,
        )