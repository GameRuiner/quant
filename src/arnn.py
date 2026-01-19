import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ARNN:
    def __init__(
        self,
        y,
        p: int,
        hidden_size: int = 0,
        include_intercept: bool = True,
        random_state: int = 42,
        device: str = "cpu",
    ):
        self.loss_ = None
        self.y = np.asarray(y, dtype=np.float32)
        self.p = p
        self.hidden_size = hidden_size
        self.include_intercept = include_intercept
        self.device = device

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = self._build_model().to(device)

    # -------------------------
    # Model architecture
    # -------------------------
    def _build_model(self):
        if self.hidden_size == 0:
            # EXACT AR(p)
            layer = nn.Linear(self.p, 1, bias=self.include_intercept)
            return layer

        else:
            # ARNN(p, hidden_size)
            layers = [nn.Linear(self.p, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, 1)]
            if not self.include_intercept:
                layers[-1].bias = None
            return nn.Sequential(*layers)

    # -------------------------
    # Design matrix
    # -------------------------
    def _make_lag_matrix(self):
        T = len(self.y)
        X = np.zeros((T - self.p, self.p), dtype=np.float32)
        for i in range(self.p):
            X[:, i] = self.y[self.p - i - 1 : T - i - 1]
        y_target = self.y[self.p :]
        return X, y_target

    # -------------------------
    # Fit
    # -------------------------
    def fit(self, max_iter: int = 500):
        X, y_target = self._make_lag_matrix()

        X = torch.tensor(X, device=self.device)
        y_target = torch.tensor(y_target, device=self.device).unsqueeze(1)

        criterion = nn.MSELoss()

        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y_target)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.loss_ = closure().item()
        return self

    # -------------------------
    # Predict (one-step-ahead)
    # -------------------------
    def predict(self, start=None, end=None):
        """
        One-step-ahead predictions using true past values
        (AutoReg-compatible behavior).
        """
        T = len(self.y)

        if start is None:
            start = self.p
        if end is None:
            end = T - 1

        if start < self.p:
            raise ValueError("start must be >= p")

        preds = []

        self.model.eval()
        with torch.no_grad():
            for t in range(start, end + 1):
                x = self.y[t - self.p : t][::-1].copy()
                x = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
                y_hat = self.model(x).item()
                preds.append(y_hat)

        return np.array(preds, dtype=np.float32)

    # -------------------------
    # Parameters (AR-compatible)
    # -------------------------
    @property
    def params(self):
        with torch.no_grad():
            if self.hidden_size == 0:
                w = self.model.weight.cpu().numpy().ravel()
                if self.include_intercept:
                    b = self.model.bias.cpu().numpy()
                    return np.concatenate([b, w])
                return w
            else:
                return np.concatenate(
                    [p.detach().cpu().numpy().ravel() for p in self.model.parameters()]
                )

    @property
    def param_names(self):
        names = []
        if self.include_intercept:
            names.append("const")
        for i in range(1, self.p + 1):
            names.append(f"ar.L{i}")
        return names