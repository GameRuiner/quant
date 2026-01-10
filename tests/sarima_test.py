import sys
import os
import numpy as np
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tests.simulate import generate_sarima_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from sarima import sarima_model

logger = logging.getLogger(__name__)

def test_sarima_100_100_12_vs_statsmodels(caplog):
    y = generate_sarima_data(
        ar_coeffs=[0.5],
        seasonal_ar_coeffs=[0.4],
        intercept=1.0,
        s=12,
        n=2500,
    )

    model = sarima_model(
        y,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 12),
        trend="c"
    )
    my_params = model.params
    my_names = model.param_names

    sm_model = SARIMAX(
        y,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()

    sm_params = sm_model.params
    sm_names = sm_model.param_names

    with caplog.at_level(logging.INFO):
        logger.info(f"My SARIMA params: {dict(zip(my_names, my_params))}")
        logger.info(f"Statsmodels SARIMA params: {dict(zip(sm_names, sm_params))}")

    for name in set(my_names) & set(sm_names):
        np.testing.assert_allclose(
            my_params[my_names.index(name)],
            sm_params[sm_names.index(name)],
            rtol=0.1,
            atol=0.1,
        )

def test_sarima_200_100_12_vs_statsmodels(caplog):
    y = generate_sarima_data(
        ar_coeffs=[0.6, -0.2],
        seasonal_ar_coeffs=[0.5],
        intercept=0.5,
        s=12,
        n=3000,
    )

    model = sarima_model(
        y,
        order=(2, 0, 0),
        seasonal_order=(1, 0, 0, 12),
        trend="c",
    )

    my_params = model.params
    my_names = model.param_names

    sm_model = SARIMAX(
        y,
        order=(2, 0, 0),
        seasonal_order=(1, 0, 0, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()

    sm_params = sm_model.params
    sm_names = sm_model.param_names

    for name in set(my_names) & set(sm_names):
        np.testing.assert_allclose(
            my_params[my_names.index(name)],
            sm_params[sm_names.index(name)],
            rtol=0.15,
            atol=0.15,
        )

def test_sarima_101_100_12_vs_statsmodels(caplog):
    y = generate_sarima_data(
        ar_coeffs=[0.5],
        seasonal_ar_coeffs=[0.3],
        intercept=0.0,
        s=12,
        n=3000,
    )

    model = sarima_model(
        y,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 0, 12),
        trend="n",
    )

    my_params = model.params
    my_names = model.param_names

    sm_model = SARIMAX(
        y,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 0, 12),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()

    sm_params = sm_model.params
    sm_names = sm_model.param_names

    for name in set(my_names) & set(sm_names):
        np.testing.assert_allclose(
            my_params[my_names.index(name)],
            sm_params[sm_names.index(name)],
            rtol=0.2,
            atol=0.2,
        )