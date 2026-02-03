import numpy as np

class TimeSeriesGenerator:
    def __init__(self, n=1000, seed=42):
        self.n = n
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self):
        raise NotImplementedError