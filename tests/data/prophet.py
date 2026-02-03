import numpy as np

from tests.data.base import TimeSeriesGenerator


class ProphetLikeGenerator(TimeSeriesGenerator):
    def __init__(
        self,
        n=1000,
        trend_slope=0.05,
        trend_intercept=10.0,
        seasonal_amplitude=5.0,
        seasonal_period=365,
        noise_std=0.5,
        seed=42,
    ):
        super().__init__(n, seed)
        self.trend_slope = trend_slope
        self.trend_intercept = trend_intercept
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_period = seasonal_period
        self.noise_std = noise_std

    def generate(self):
        t = np.arange(self.n)
        trend = self.trend_intercept + self.trend_slope * t
        seasonality = (
            self.seasonal_amplitude
            * np.sin(2 * np.pi * t / self.seasonal_period)
        )
        y = trend + seasonality + self.rng.normal(
            0, self.noise_std, size=self.n
        )

        return t, y, trend, seasonality

class ChangepointGenerator(TimeSeriesGenerator):
    def __init__(self, changepoint=500, n=1000, seed=42):
        super().__init__(n, seed)
        self.changepoint = changepoint

    def generate(self):
        t = np.arange(self.n)
        trend = np.where(
            t < self.changepoint,
            0.05 * t,
            0.05 * self.changepoint + 0.15 * (t - self.changepoint),
        )
        y = trend + self.rng.normal(0, 0.3, size=self.n)
        return t, y, trend