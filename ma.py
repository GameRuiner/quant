from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
import pandas as pd

class MovingAverageMLE(GenericLikelihoodModel):
    def initialize(self):
        super().initialize()
        extra_params_names = ['beta', 'std']
        self._set_extra_params_names(extra_params_names)
        self.start_params = np.array([0.1, 0.1, 0.1])
    
    def calc_conditional_et(self, intercept, beta):
        df = pd.DataFrame({"xt": self.endog})
        ets = [0.0]
        for i in range(1, len(df)):
            ets.append(df.iloc[i]["xt"] - intercept - (beta * ets[i-1]))
        return ets
    
    def loglike(self, params):
        ets = self.calc_conditional_et(params[0], params[1])
        return stats.norm.logpdf(
            ets,
            scale=params[2],
        ).sum()