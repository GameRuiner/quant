import numpy as np

def generate_ar_data(coeffs, intercept=0.0, n=1000, seed=42):
    np.random.seed(seed)
    p = len(coeffs)
    noise = np.random.normal(0, 1, n)
    y = np.zeros(n)
    for i in range(p):
        y[i] = intercept / (1 - sum(coeffs)) if sum(coeffs) < 1 else 0
    for t in range(p, n):
        y[t] = intercept + sum(coeffs[j] * y[t-j-1] for j in range(p)) + noise[t]
    return y

def generate_ma_data(thetas, intercept=0.0, n=1000, seed=42):
    np.random.seed(seed)
    q = len(thetas)
    noise = np.random.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(n):
        ma_sum = sum(thetas[j] * noise[t-j-1] for j in range(q) if t-j-1 >= 0)
        y[t] = intercept + noise[t] + ma_sum
    return y

def generate_sarima_data(
    ar_coeffs,
    seasonal_ar_coeffs,
    intercept=0.0,
    s=12,
    n=2000,
    noise_std=1.0,
    seed=42,
):
    """
    Simple SARIMA data generator:
    (p,0,0) x (P,0,0,s)
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    eps = rng.normal(scale=noise_std, size=n)

    for t in range(max(len(ar_coeffs), s * len(seasonal_ar_coeffs)), n):
        val = intercept + eps[t]

        # non-seasonal AR
        for i, phi in enumerate(ar_coeffs, start=1):
            val += phi * y[t - i]

        # seasonal AR
        for j, Phi in enumerate(seasonal_ar_coeffs, start=1):
            val += Phi * y[t - j * s]

        y[t] = val

    return y

def generate_nonlinear_ar_data(n=2000, noise_std=0.1, seed=42):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float32)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 1] ** 2 + rng.normal(0, noise_std)
    return y

def generate_prophet_data(
    n=1000,
    trend_slope=0.05,
    trend_intercept=10.0,
    seasonal_amplitude=5.0,
    seasonal_period=365,
    noise_std=0.5,
    seed=42,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    trend = trend_intercept + trend_slope * t
    seasonality = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)

    y = trend + seasonality + rng.normal(0, noise_std, size=n)

    return t, y, trend, seasonality

def generate_changepoint_data(changepoint=500):
    t = np.arange(1000)
    trend = np.where(t < changepoint, 0.05 * t, 0.05 * changepoint + 0.15 * (t - changepoint))
    y = trend + np.random.normal(0, 0.3, size=1000)
    return t, y, trend