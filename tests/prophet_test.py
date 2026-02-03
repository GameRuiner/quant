import numpy as np

from prophet import ProphetModel

from tests.data.prophet import ProphetLikeGenerator, ChangepointGenerator


def test_prophet_beats_naive_forecast():
    t, y, trend, seasonality = ProphetLikeGenerator(n=1000).generate()

    # ----- Naive baseline -----
    naive_pred = np.full_like(y[1:], y[:-1].mean())
    naive_mse = np.mean((y[1:] - naive_pred) ** 2)

    # ----- Your Prophet -----
    model = ProphetModel(
        yearly_seasonality=True,
        changepoints=None,
    ).fit(t, y)

    pred = model.predict(t[1:])
    prophet_mse = np.mean((y[1:] - pred) ** 2)

    print('\n', prophet_mse)
    print(naive_mse)

    assert prophet_mse < 0.5 * naive_mse


def test_prophet_recovers_trend_shape():
    t, y, true_trend, _ = ProphetLikeGenerator(n=1000).generate()

    model = ProphetModel(
        yearly_seasonality=False,
    ).fit(t, y)

    trend_hat = model.trend_

    corr = np.corrcoef(true_trend, trend_hat)[0, 1]
    assert corr > 0.98

def test_prophet_recovers_seasonality():
    t, y, _, true_seasonality = ProphetLikeGenerator(n=1000).generate()

    model = ProphetModel(
        yearly_seasonality=True,
    ).fit(t, y)

    seasonal_hat = model.seasonality_

    corr = np.corrcoef(true_seasonality, seasonal_hat)[0, 1]
    assert corr > 0.95

def test_prophet_forecast_accuracy():
    t, y, _, _ = ProphetLikeGenerator(n=1200).generate()

    train_t, test_t = t[:1000], t[1000:]
    train_y, test_y = y[:1000], y[1000:]

    model = ProphetModel(
        yearly_seasonality=True,
    ).fit(train_t, train_y)

    forecast = model.predict(test_t)

    mse = np.mean((test_y - forecast) ** 2)
    assert mse < 2.0

def test_prophet_detects_changepoint():
    changepoint = 500
    t, y, true_trend = ChangepointGenerator(changepoint).generate()

    model = ProphetModel(
        changepoints=[changepoint],
        yearly_seasonality=False,
    ).fit(t, y)

    trend_hat = model.trend_

    corr = np.corrcoef(true_trend, trend_hat)[0, 1]
    assert corr > 0.95