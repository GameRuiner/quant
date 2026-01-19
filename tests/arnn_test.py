import numpy as np
from statsmodels.tsa.ar_model import AutoReg

from tests.simulate import generate_ar_data
import logging
from src.arnn import ARNN

logger = logging.getLogger(__name__)

def test_arnn_vs_autoreg_p1_linear(caplog):
    """
    ARNN with linear structure should match AutoReg for AR(1)
    in terms of one-step predictions.
    """

    coeffs = [0.7]
    intercept = 0.0
    y = generate_ar_data(coeffs, intercept=intercept, n=2000)

    # ----- Statsmodels AutoReg -----
    sm_ar = AutoReg(y, lags=1, trend="n").fit()
    sm_pred = sm_ar.predict(start=1, end=len(y) - 1)

    # ----- Autoregressive Neural Network -----
    arnn = ARNN(
        y,
        p=1,
        hidden_size=0,
        include_intercept=False,
        random_state=42,
    ).fit()

    my_pred = arnn.predict(start=1, end=len(y) - 1)

    with caplog.at_level(logging.INFO):
        logger.info(f"AutoReg AR(1) prediction head: {sm_pred[:5]}")
        logger.info(f"ARNN AR(1) prediction head: {my_pred[:5]}")

    # ----- Functional equivalence -----
    np.testing.assert_allclose(
        my_pred,
        sm_pred,
        rtol=1e-2,
        atol=1e-2,
    )