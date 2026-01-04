from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
import pandas as pd

class MovingAverageMLE(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, q=1, **kwargs):
        self.q = q
        super().__init__(endog, exog, **kwargs)
    
    def initialize(self):
        super().initialize()
        extra_params_names = [f'beta{i+1}' for i in range(self.q)] + ['std']
        self._set_extra_params_names(extra_params_names)
        self.start_params = np.array([0.1] * (1 + self.q) + [0.1])
    
    def calc_conditional_et(self, intercept, betas):
        df = pd.DataFrame({"xt": self.endog})
        ets = [0.0] * self.q
        for i in range(1, len(df)):
            error_term = intercept
            for j in range(self.q):
                error_term += betas[j] * ets[-(j+1)]
            et = df.iloc[i]["xt"] - error_term
            ets.append(et)
        return ets[self.q:]
    
    def loglike(self, params):
        intercept = params[0]
        betas = params[1:1+self.q]
        std = params[-1]
        ets = self.calc_conditional_et(intercept, betas)
        return stats.norm.logpdf(
            ets,
            scale=std,
        ).sum()