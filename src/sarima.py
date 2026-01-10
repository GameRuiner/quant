from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAModel:
    def __init__(self, params, param_names):
        self.params = params
        self.param_names = param_names


def sarima_model(y, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None):
    """
    Fit an SARIMA(p,d,q)(P,D,Q)s model
    """
    integrated = order[1] > 0
    if trend is None:
        trend = "c" if not integrated else "n"
    spec_arima = SARIMAXSpecification(y, order=order, seasonal_order=seasonal_order, trend=trend)
    exog = spec_arima._model.data.orig_exog
    model = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order).fit()
    return SARIMAModel(model.params, model.param_names)