import numpy as np

from tests.data.base import TimeSeriesGenerator


class ARGenerator(TimeSeriesGenerator):
    def __init__(self, coeffs, intercept=0.0, n=1000, seed=42):
        super().__init__(n, seed)
        self.coeffs = np.asarray(coeffs)
        self.intercept = intercept

    def generate(self):
        p = len(self.coeffs)
        y = np.zeros(self.n)
        noise = self.rng.normal(size=self.n)

        if np.sum(self.coeffs) < 1:
            y[:p] = self.intercept / (1 - np.sum(self.coeffs))

        for t in range(p, self.n):
            y[t] = (
                self.intercept
                + np.dot(self.coeffs, y[t - np.arange(1, p + 1)])
                + noise[t]
            )
        return y


class MAGenerator(TimeSeriesGenerator):
    def __init__(self, thetas, intercept=0.0, n=1000, seed=42):
        super().__init__(n, seed)
        self.thetas = np.asarray(thetas)
        self.intercept = intercept

    def generate(self):
        q = len(self.thetas)
        y = np.zeros(self.n)
        eps = self.rng.normal(size=self.n)

        for t in range(self.n):
            ma_sum = np.dot(
                self.thetas[: min(q, t)],
                eps[t - np.arange(1, min(q, t) + 1)],
            )
            y[t] = self.intercept + eps[t] + ma_sum

        return y

class SARIMAGenerator(TimeSeriesGenerator):
    """
    (p,0,0) x (P,0,0,s) generator
    """

    def __init__(
        self,
        ar_coeffs,
        seasonal_ar_coeffs,
        s=12,
        intercept=0.0,
        noise_std=1.0,
        n=2000,
        seed=42,
    ):
        super().__init__(n, seed)
        self.ar = np.asarray(ar_coeffs)
        self.seasonal_ar = np.asarray(seasonal_ar_coeffs)
        self.s = s
        self.intercept = intercept
        self.noise_std = noise_std

    def generate(self):
        y = np.zeros(self.n)
        eps = self.rng.normal(scale=self.noise_std, size=self.n)

        start = max(len(self.ar), self.s * len(self.seasonal_ar))
        for t in range(start, self.n):
            val = self.intercept + eps[t]

            for i, phi in enumerate(self.ar, start=1):
                val += phi * y[t - i]

            for j, Phi in enumerate(self.seasonal_ar, start=1):
                val += Phi * y[t - j * self.s]

            y[t] = val

        return y