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