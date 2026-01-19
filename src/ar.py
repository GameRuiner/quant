import statsmodels.api as sm
import numpy as np

class AR:
    def __init__(self, y, p):
        n = len(y)
        if n <= p:
            raise ValueError("Length of y must be greater than p")
        self.Y = y[p:]
        self.X = np.column_stack([y[p - i - 1: n - i - 1] for i in range(p)])
        self.X = sm.add_constant(self.X)

    def fit(self):
        return sm.OLS(self.Y, self.X).fit()