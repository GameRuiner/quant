import numpy as np

from tests.data.base import TimeSeriesGenerator


class NonlinearARGenerator(TimeSeriesGenerator):
    def __init__(self, n=2000, noise_std=0.1, seed=42):
        super().__init__(n, seed)
        self.noise_std = noise_std

    def generate(self):
        y = np.zeros(self.n, dtype=np.float32)

        for t in range(1, self.n):
            y[t] = (
                0.5 * y[t - 1]
                + 0.3 * y[t - 1] ** 2
                + self.rng.normal(0, self.noise_std)
            )

        return y