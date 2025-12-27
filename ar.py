import statsmodels.api as sm

def ar_model(y):
    y_lag = y[:-1]
    y_now = y[1:]
    X = sm.add_constant(y_lag)
    model = sm.OLS(y_now, X).fit()
    return model