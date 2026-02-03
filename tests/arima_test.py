import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from arima import ARIMAModel
from tests.data.linear import ARGenerator

logger = logging.getLogger(__name__)


def test_arima_model_vs_statsmodels_110(caplog):
    coeffs = [0.7]
    intercept = 0.5
    y = ARGenerator(coeffs, intercept=intercept).generate()

    # Fit custom ARIMA(1,1,0)
    model = ARIMAModel(y, order=(1, 1, 0)).fit()
    my_params = model.params
    my_names = model.param_names

    # Fit statsmodels ARIMA(1,1,0)
    sm_model = ARIMA(y, order=(1, 1, 0)).fit()
    sm_params = sm_model.params
    sm_names = sm_model.param_names

    with caplog.at_level(logging.INFO):
        logger.info(f"My ARIMA(1,1,0) params: {dict(zip(my_names, my_params))}")
        logger.info(f"Statsmodels ARIMA(1,1,0) params: {dict(zip(sm_names, sm_params))}")

    for name in set(my_names) & set(sm_names):
        np.testing.assert_allclose(
            my_params[my_names.index(name)],
            sm_params[sm_names.index(name)],
            rtol=1e-2, atol=1e-2
        )

def test_arima_model_vs_statsmodels_210(caplog):
    coeffs = [0.7, -0.3]
    intercept = 1.0
    y = ARGenerator(coeffs, intercept=intercept).generate()

    # Fit custom ARIMA(2,1,0)
    model = ARIMAModel(y, order=(2, 1, 0)).fit()
    my_params = model.params
    my_names = model.param_names

    # Fit statsmodels ARIMA(2,1,0)
    sm_model = ARIMA(y, order=(2, 1, 0)).fit()
    sm_params = sm_model.params
    sm_names = sm_model.param_names

    with caplog.at_level(logging.INFO):
        logger.info(f"My ARIMA(2,1,0) params: {dict(zip(my_names, my_params))}")
        logger.info(f"Statsmodels ARIMA(2,1,0) params: {dict(zip(sm_names, sm_params))}")

    for name in set(my_names) & set(sm_names):
        np.testing.assert_allclose(
            my_params[my_names.index(name)],
            sm_params[sm_names.index(name)],
            rtol=1e-2, atol=1e-2
        )


def test_arima_model_vs_statsmodels_001(caplog):
    coeffs = [0.7, -0.3]
    intercept = 1.0
    y = ARGenerator(coeffs, intercept=intercept).generate()

    # Fit custom ARIMA(0,0,1)
    model = ARIMAModel(y, order=(0, 0, 1)).fit()
    my_params = model.params
    my_names = model.param_names

    # Fit statsmodels ARIMA(0,0,1)
    sm_model = ARIMA(y, order=(0, 0, 1)).fit()
    sm_params = sm_model.params
    sm_names = sm_model.param_names

    with caplog.at_level(logging.INFO):
        logger.info(f"My ARIMA(0,0,1) params: {dict(zip(my_names, my_params))}")
        logger.info(f"Statsmodels ARIMA(0,0,1) params: {dict(zip(sm_names, sm_params))}")

    assert len(my_params) == len(sm_params)
    for name in set(my_names) & set(sm_names):
        np.testing.assert_allclose(
            my_params[my_names.index(name)],
            sm_params[sm_names.index(name)],
            rtol=1e-1, atol=1e-1
        )
        
def test_arima_model_vs_statsmodels_101(caplog):
    coeffs = [0.5]
    intercept = 0.0
    y = ARGenerator(coeffs, intercept=intercept).generate()

    # Fit custom ARIMA(1,0,1)
    model = ARIMAModel(y, order=(1, 0, 1)).fit()
    my_params = model.params
    my_names = model.param_names

    # Fit statsmodels ARIMA(1,0,1)
    sm_model = ARIMA(y, order=(1, 0, 1)).fit()
    sm_params = sm_model.params
    sm_names = sm_model.param_names

    with caplog.at_level(logging.INFO):
        logger.info(f"My ARIMA(1,0,1) params: {dict(zip(my_names, my_params))}")
        logger.info(f"Statsmodels ARIMA(1,0,1) params: {dict(zip(sm_names, sm_params))}")

    for name in set(my_names) & set(sm_names):
        np.testing.assert_allclose(
            my_params[my_names.index(name)],
            sm_params[sm_names.index(name)],
            rtol=0.05, atol=0.05
        )
        
def test_arima_model_vs_statsmodels_202(caplog):
    coeffs = [0.5, -0.2]
    intercept = 1.0
    y = ARGenerator(coeffs, intercept=intercept, n=2000).generate()

    model = ARIMAModel(y, order=(2, 0, 2)).fit()
    my_params = model.params
    my_names = model.param_names

    sm_model = ARIMA(y, order=(2, 0, 2)).fit()
    sm_params = sm_model.params
    sm_names = sm_model.param_names

    with caplog.at_level(logging.INFO):
        logger.info(f"My ARIMA(1,2,1) params: {dict(zip(my_names, my_params))}")
        logger.info(f"Statsmodels ARIMA(1,2,1) params: {dict(zip(sm_names, sm_params))}")

    for name in set(my_names) & set(sm_names):
        np.testing.assert_allclose(
            my_params[my_names.index(name)],
            sm_params[sm_names.index(name)],
            rtol=0.1, atol=0.1
        )