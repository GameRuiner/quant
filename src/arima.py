from sarima import sarima_model

class ARIMAModel:
    def __init__(self, params, param_names):
        self.params = params
        self.param_names = param_names


def arima_model(y, p=1, d=0, q=0):
    """
    Fit an ARIMA(p, d, q) model
    """
    return sarima_model(y, (p, d, q))