import numpy as np
import statsmodels.api as sm
from ma import MovingAverageMLE

class ARIMAModel:
    def __init__(self, params, param_names):
        self.params = params
        self.param_names = param_names

def difference(y, d=1):
    """Apply differencing to the time series data y, d times."""
    for _ in range(d):
        y = np.diff(y, n=1)
    return y

def arima_model(y, p=1, d=0, q=0):
    """Fit an ARIMA(p, d, q) model to the time series data y."""

    y_diff = difference(y, d)
    n = len(y_diff)
    if n <= max(p, q):
        raise ValueError("Not enough data after differencing for given p and q.")

    params = []
    param_names = []

    if p > 0:
        Y = y_diff[p:]
        X = np.column_stack([y_diff[p - i - 1: n - i - 1] for i in range(p)])
        if d == 0:
            X = sm.add_constant(X)
            ar_param_names = ['const'] + [f'ar.L{i+1}' for i in range(p)]
        else:
            ar_param_names = [f'ar.L{i+1}' for i in range(p)]
        ar_model = sm.OLS(Y, X).fit()
        params.extend(ar_model.params)
        param_names.extend(ar_param_names)
        resid = ar_model.resid
    else:
        resid = y_diff

    if q > 0:
        fit_intercept = (d == 0)
        ma_model = MovingAverageMLE(resid, q=q, fit_intercept=fit_intercept).fit(disp=False)
        params.extend(ma_model.params)
        param_names.extend(ma_model.param_names)
    return ARIMAModel(np.array(params), param_names)