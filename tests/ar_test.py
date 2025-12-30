import sys
import os
import numpy as np
import pytest
import logging
from statsmodels.tsa.ar_model import AutoReg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ar import ar_model

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def sample_data():
    # Generate stationary AR(1) process for reproducibility
    np.random.seed(42)
    n = 1000
    phi = 0.7
    intercept = 1.5
    noise = np.random.normal(0, 1, n)
    y = np.zeros(n)
    y[0] = intercept / (1 - phi)
    for t in range(1, n):
        y[t] = intercept + phi * y[t-1] + noise[t]
    return y

def test_ar_model_vs_autoreg(sample_data, caplog):
    y = sample_data

    my_ar = ar_model(y)
    my_intercept, my_phi = my_ar.params

    sm_ar = AutoReg(y, lags=1, old_names=False).fit()
    sm_intercept, sm_phi = sm_ar.params

    with caplog.at_level(logging.INFO):
        logger.info(f"My AR(1) params: intercept={my_intercept}, phi={my_phi}")
        logger.info(f"AutoReg AR(1) params: intercept={sm_intercept}, phi={sm_phi}")
    np.testing.assert_equal(my_intercept, sm_intercept)
    np.testing.assert_equal(my_phi, sm_phi)