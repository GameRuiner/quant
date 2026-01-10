from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAModel:
    def __init__(self, params, param_names):
        self.params = params
        self.param_names = param_names


def arima_model(y, p=1, d=0, q=0):
    """
    Fit an ARIMA(p, d, q) model
    """
    order = (p, d, q)
    integrated = order[1] > 0
    trend = "c" if not integrated else "n"
    spec_arima = SARIMAXSpecification(y, order=order, trend=trend)
    exog = spec_arima._model.data.orig_exog
    model = SARIMAX(y, exog=exog, order=order).fit()
    return ARIMAModel(model.params, model.param_names)