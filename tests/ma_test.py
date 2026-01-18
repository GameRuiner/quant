import sys
import os
import numpy as np
import pytest
import logging
from statsmodels.tsa.arima.model import ARIMA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ma import MovingAverageMLE
from tests.simulate import generate_ma_data

logger = logging.getLogger(__name__)

def test_ma_model_vs_arima_q1(caplog):
    coeffs = [0.6]
    y = generate_ma_data(coeffs)

    # Fit custom MA(1)
    model = MovingAverageMLE(y, q=1)
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
    np.testing.assert_allclose(my_std, sm_std, rtol=1e-1, atol=1e-1)

def test_ma_model_vs_arima_q2(caplog):
    coeffs = [0.5, -0.3]
    y = generate_ma_data(coeffs)

    # Fit custom MA(2)
    model = MovingAverageMLE(y, q=2)
    fit = model.fit(disp=False)
    my_intercept, my_theta1, my_theta2, my_std = fit.params

    # Fit statsmodels ARIMA MA(2)
    sm_ma = ARIMA(y, order=(0, 0, 2)).fit()
    sm_intercept, sm_theta1, sm_theta2, sigma2 = sm_ma.params
    sm_std = sigma2 ** 0.5

    with caplog.at_level(logging.INFO):
        logger.info(f"My MA(2) params: intercept={my_intercept}, theta1={my_theta1}, theta2={my_theta2}, std={my_std}")
        logger.info(f"ARIMA MA(2) params: intercept={sm_intercept}, theta1={sm_theta1}, theta2={sm_theta2}, std={sm_std}")

    np.testing.assert_allclose(my_intercept, sm_intercept, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(my_theta1, sm_theta1, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(my_theta2, sm_theta2, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(my_std, sm_std, rtol=1e-1, atol=1e-1)

def test_ma_model_vs_arima_q3(caplog):
    coeffs = [0.4, -0.2, 0.1]
    y = generate_ma_data(coeffs)

    # Fit custom MA(3)
    model = MovingAverageMLE(y, q=3)
    fit = model.fit(disp=False)
    my_intercept, my_theta1, my_theta2, my_theta3, my_std = fit.params

    # Fit statsmodels ARIMA MA(3)
    sm_ma = ARIMA(y, order=(0, 0, 3)).fit()
    sm_intercept, sm_theta1, sm_theta2, sm_theta3, sigma2 = sm_ma.params
    sm_std = sigma2 ** 0.5

    with caplog.at_level(logging.INFO):
        logger.info(f"My MA(3) params: intercept={my_intercept}, theta1={my_theta1}, theta2={my_theta2}, theta3={my_theta3}, std={my_std}")
        logger.info(f"ARIMA MA(3) params: intercept={sm_intercept}, theta1={sm_theta1}, theta2={sm_theta2}, theta3={sm_theta3}, std={sm_std}")

    np.testing.assert_allclose(my_intercept, sm_intercept, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(my_theta1, sm_theta1, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(my_theta2, sm_theta2, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(my_theta3, sm_theta3, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(my_std, sm_std, rtol=1e-1, atol=1e-1)