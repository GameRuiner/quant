from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
import pandas as pd

class MovingAverageMLE(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, q=1, fit_intercept=True, **kwargs):
        self.q = q
        self.fit_intercept = fit_intercept
        super().__init__(endog, exog, **kwargs)
    
    def initialize(self):
        super().initialize()
        names = []
        if self.fit_intercept:
            names.append('const')
        names += [f'ma.L{i+1}' for i in range(self.q)] + ['sigma2']
        self._param_names = names
        self.start_params = np.array([0.1] * len(names))
    
    def fit(self, **kwargs):
        result = super().fit(**kwargs)
        result.param_names = self._param_names
        return result
    
    def calc_conditional_et(self, *params):
        idx = 0
        if self.fit_intercept:
            intercept = params[0]
            idx = 1
        else:
            intercept = 0.0
        betas = params[idx:idx+self.q]
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
        ets = self.calc_conditional_et(*params[:-1])
        std = params[-1]
        return stats.norm.logpdf(ets, scale=std).sum()
        
    @property
    def param_names(self):
        return self._param_names