import sys
import os
import numpy as np
import pytest
import logging
from statsmodels.tsa.ar_model import AutoReg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ar import ar_model

logger = logging.getLogger(__name__)

@pytest.fixture
def generate_ar_data():
    def _gen(coeffs, intercept=0.0, n=1000, seed=42):
        np.random.seed(seed)
        p = len(coeffs)
        noise = np.random.normal(0, 1, n)
        y = np.zeros(n)
        for i in range(p):
            y[i] = intercept / (1 - sum(coeffs)) if sum(coeffs) < 1 else 0
        for t in range(p, n):
            y[t] = intercept + sum(coeffs[j] * y[t-j-1] for j in range(p)) + noise[t]
        return y
    return _gen

def test_ar_model_vs_autoreg_p1(generate_ar_data, caplog):
    coeffs = [0.7]
    intercept = 0
    y = generate_ar_data(coeffs, intercept=intercept)

    my_ar = ar_model(y, p=1)
    my_intercept, my_phi = my_ar.params

    sm_ar = AutoReg(y, lags=1, old_names=False).fit()
    sm_intercept, sm_phi = sm_ar.params

    with caplog.at_level(logging.INFO):
        logger.info(f"My AR(1) params: intercept={my_intercept}, phi={my_phi}")
        logger.info(f"AutoReg AR(1) params: intercept={sm_intercept}, phi={sm_phi}")
    np.testing.assert_allclose(my_intercept, sm_intercept, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(my_phi, sm_phi, rtol=1e-5, atol=1e-5)

def test_ar_model_vs_autoreg_p2(generate_ar_data, caplog):
    coeffs = [0.7, -0.3]
    intercept = 1.5
    y = generate_ar_data(coeffs, intercept=intercept)

    my_ar = ar_model(y, p=2)
    my_intercept, my_phi1, my_phi2 = my_ar.params

    sm_ar = AutoReg(y, lags=2, old_names=False).fit()
    sm_intercept, sm_phi1, sm_phi2 = sm_ar.params

    with caplog.at_level(logging.INFO):
        logger.info(f"My AR(2) params: intercept={my_intercept}, phi1={my_phi1}, phi2={my_phi2}")
        logger.info(f"AutoReg AR(2) params: intercept={sm_intercept}, phi1={sm_phi1}, phi2={sm_phi2}")
    np.testing.assert_allclose(my_intercept, sm_intercept, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(my_phi1, sm_phi1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(my_phi2, sm_phi2, rtol=1e-5, atol=1e-5)
    
    
def test_ar_model_vs_autoreg_p3(generate_ar_data, caplog):
    coeffs = [0.7, -0.3, 0.5]
    intercept = 0.5
    y = generate_ar_data(coeffs, intercept=intercept)

    my_ar = ar_model(y, p=3)
    my_intercept, my_phi1, my_phi2, my_phi3 = my_ar.params

    sm_ar = AutoReg(y, lags=3, old_names=False).fit()
    sm_intercept, sm_phi1, sm_phi2, sm_phi3 = sm_ar.params

    with caplog.at_level(logging.INFO):
        logger.info(f"My AR(2) params: intercept={my_intercept}, phi1={my_phi1}, phi2={my_phi2}, phi3={my_phi3}")
        logger.info(f"AutoReg AR(2) params: intercept={sm_intercept}, phi1={sm_phi1}, phi2={sm_phi2}, phi3={sm_phi3}")
    np.testing.assert_allclose(my_intercept, sm_intercept, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(my_phi1, sm_phi1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(my_phi2, sm_phi2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(my_phi3, sm_phi3, rtol=1e-5, atol=1e-5)