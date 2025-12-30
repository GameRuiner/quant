import sys
import os
import numpy as np
import pytest
import logging
from statsmodels.tsa.arima.model import ARIMA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ma import MovingAverageMLE

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def sample_data():
    # Generate stationary MA(1) process for reproducibility
    np.random.seed(42)
    n = 1000
    theta = 0.6
    intercept = 1.2
    noise = np.random.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = intercept + noise[t] + theta * noise[t-1]
    return y

def test_ma_model_vs_arima(sample_data, caplog):
    y = sample_data

    # Fit custom MA(1)
    model = MovingAverageMLE(y)
    fit = model.fit(disp=False)
    my_intercept, my_theta, my_std = fit.params

    # Fit statsmodels ARIMA MA(1)
    sm_ma = ARIMA(y, order=(0, 0, 1)).fit()
    sm_intercept, sm_theta, sm_std = sm_ma.params

    with caplog.at_level(logging.INFO):
        logger.info(f"My MA(1) params: intercept={my_intercept}, theta={my_theta}, std={my_std}")
        logger.info(f"ARIMA MA(1) params: intercept={sm_intercept}, theta={sm_theta}, std={sm_std}")

    np.testing.assert_allclose(my_intercept, sm_intercept, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(my_theta, sm_theta, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(my_std, sm_std, rtol=1e-2, atol=1e-2)