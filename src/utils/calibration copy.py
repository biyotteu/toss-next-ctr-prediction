# src/utils/calibration.py
from __future__ import annotations
import numpy as np, torch
from sklearn.isotonic import IsotonicRegression

class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = torch.nn.Parameter(torch.zeros(1))  # T = exp(log_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_temp)

def fit_temperature(logits: np.ndarray, y: np.ndarray, lr=0.05, iters=200):
    z = torch.tensor(logits, dtype=torch.float32)
    t = torch.tensor(y.astype(np.float32))
    scaler = TemperatureScaler()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=iters)

    def _nll():
        opt.zero_grad()
        zt = scaler(z)
        p = torch.sigmoid(zt).clamp(1e-7, 1-1e-7)
        # 50:50 가중 로스(평가식과 일치)
        w_pos = (t==1).float().mean().clamp(min=1e-6)
        w_neg = 1.0 - w_pos
        loss = - ( (t*torch.log(p)).mean()/w_pos + ((1-t)*torch.log(1-p)).mean()/w_neg ) * 0.5
        loss.backward()
        return loss

    opt.step(_nll)
    return scaler

class Calibrator:
    """
    method: 'temperature', 'isotonic', 'temperature+isotonic'
    """
    def __init__(self, method: str = "temperature", lr=0.05, iters=200):
        self.method = method
        self.lr = lr; self.iters = iters
        self.temp_scaler = None
        self.iso = None

    def fit(self, logits: np.ndarray, y: np.ndarray):
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)

        if self.method in ("temperature", "temperature+isotonic"):
            self.temp_scaler = fit_temperature(z, y, lr=self.lr, iters=self.iters)

        if self.method in ("isotonic", "temperature+isotonic"):
            if self.temp_scaler is not None:
                with torch.no_grad():
                    zt = self.temp_scaler(torch.tensor(z, dtype=torch.float32)).numpy()
                p = 1.0 / (1.0 + np.exp(-zt))
            else:
                p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), 1e-7, 1-1e-7)
            self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.iso.fit(p, y)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64)
        if self.temp_scaler is not None:
            with torch.no_grad():
                z = self.temp_scaler(torch.tensor(z, dtype=torch.float32)).numpy()
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), 1e-7, 1-1e-7)
        if self.iso is not None:
            p = self.iso.predict(p)
            p = np.clip(p, 1e-7, 1-1e-7)
        return p
