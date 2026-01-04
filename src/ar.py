import statsmodels.api as sm
import numpy as np

def ar_model(y, p = 1):
    """
    Fit an AR(p) model using OLS.
    Returns the fitted model.
    """
    n = len(y)
    if n <= p:
        raise ValueError("Length of y must be greater than p")
    Y = y[p:]
    X = np.column_stack([y[p - i - 1: n - i - 1] for i in range(p)])
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model