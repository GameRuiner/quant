import numpy as np
import torch
from statsmodels.tsa.ar_model import AutoReg

from tests.data.linear import ARGenerator
from tests.data.nonlinear import NonlinearARGenerator
import logging
from models.neural.arnn import ARNN

logger = logging.getLogger(__name__)

def test_arnn_vs_autoreg_p1_linear(caplog):
    """
    ARNN with linear structure should match AutoReg for AR(1)
    in terms of one-step predictions.
    """

    coeffs = [0.7]
    intercept = 0.0
    y = ARGenerator(coeffs, intercept=intercept, n=2000).generate()

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

def test_arnn_hidden_beats_linear_ar(caplog):
    """
    Nonlinear ARNN should outperform linear AR
    on nonlinear autoregressive data.
    """

    y = NonlinearARGenerator(n=10000).generate()

    # ----- Linear AR (baseline) -----
    arnn_linear = ARNN(
        y,
        p=1,
        hidden_size=0,
        include_intercept=True,
        random_state=42,
    ).fit()

    pred_linear = arnn_linear.predict(start=1, end=len(y) - 1)
    y_true = y[1:]
    mse_linear = np.mean((y_true - pred_linear) ** 2)

    # ----- Nonlinear ARNN -----
    arnn_nonlinear = ARNN(
        y,
        p=1,
        hidden_size=16,
        include_intercept=True,
        random_state=42,
    ).fit()

    pred_nonlinear = arnn_nonlinear.predict(start=1, end=len(y) - 1)
    mse_nonlinear = np.mean((y_true - pred_nonlinear) ** 2)

    with caplog.at_level(logging.INFO):
        logger.info(f"Linear AR MSE: {mse_linear:.6f}")
        logger.info(f"Nonlinear ARNN MSE: {mse_nonlinear:.6f}")

    # ----- Expect nonlinear to improve -----
    assert mse_nonlinear < mse_linear

def test_arnn_hidden_training_reduces_loss():
    y = NonlinearARGenerator(n=2000).generate()

    model = ARNN(
        y,
        p=1,
        hidden_size=10,
        include_intercept=True,
        random_state=42,
    )

    # Loss before training
    X, y_target = model._make_lag_matrix()
    with torch.no_grad():
        y_hat = model.model(torch.tensor(X))
        loss_before = torch.mean((y_hat.squeeze() - torch.tensor(y_target)) ** 2).item()

    model.fit()

    loss_after = model.loss_

    assert loss_after < loss_before

def test_arnn_hidden_is_reproducible():
    y = NonlinearARGenerator(n=2000).generate()

    model1 = ARNN(y, p=1, hidden_size=5, random_state=123).fit()
    model2 = ARNN(y, p=1, hidden_size=5, random_state=123).fit()

    pred1 = model1.predict(start=1, end=100)
    pred2 = model2.predict(start=1, end=100)

    np.testing.assert_equal(pred1, pred2)