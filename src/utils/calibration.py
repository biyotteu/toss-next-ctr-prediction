# src/utils/calibration.py
from __future__ import annotations
import numpy as np, torch
from sklearn.isotonic import IsotonicRegression

# --- 안정 시그모이드 (오버플로우 방지) ---
def _sigmoid_stable_numpy(z: np.ndarray) -> np.ndarray:
    # clip 후 exp
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

class TemperatureScaler(torch.nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_temp = torch.nn.Parameter(torch.tensor(np.log(init_T), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, clamp_T: tuple[float,float] | None = (0.2, 5.0)) -> torch.Tensor:
        T = torch.exp(self.log_temp)
        if clamp_T is not None:
            T = torch.clamp(T, clamp_T[0], clamp_T[1])
        return logits / T

def fit_temperature(logits: np.ndarray, y: np.ndarray, lr=0.05, iters=200,
                    clamp_T=(0.2, 5.0), l2_reg=1e-3):
    z = torch.tensor(logits, dtype=torch.float32)
    t = torch.tensor(y.astype(np.float32))

    scaler = TemperatureScaler(init_T=1.0)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=iters, line_search_fn="strong_wolfe")

    # 50:50 WLL
    with torch.no_grad():
        w_pos = (t == 1).float().mean().clamp(min=1e-6)
        w_neg = 1.0 - w_pos

    def _closure():
        opt.zero_grad()
        zt = scaler(z, clamp_T=clamp_T)
        p  = torch.sigmoid(zt).clamp(1e-7, 1-1e-7)
        loss_pos = -(t * torch.log(p)).mean() / w_pos
        loss_neg = -((1 - t) * torch.log(1 - p)).mean() / w_neg
        # T 정규화: (T-1)^2
        T = torch.exp(scaler.log_temp)
        if clamp_T is not None:
            T = torch.clamp(T, clamp_T[0], clamp_T[1])
        reg = l2_reg * (T - 1.0) ** 2
        loss = 0.5 * (loss_pos + loss_neg) + reg
        loss.backward()
        return loss

    opt.step(_closure)
    return scaler

class Calibrator:
    """
    method: 'temperature', 'isotonic', 'temperature+isotonic'
    """
    def __init__(self, method: str = "temperature", lr=0.05, iters=200,
                 clamp_T=(0.2, 5.0), l2_reg=1e-3, min_iso_nodes: int = 8):
        self.method = method
        self.lr = lr; self.iters = iters
        self.clamp_T = clamp_T
        self.l2_reg = l2_reg
        self.min_iso_nodes = min_iso_nodes
        self.temp_scaler: TemperatureScaler | None = None
        self.iso: IsotonicRegression | None = None

    def fit(self, logits: np.ndarray, y: np.ndarray):
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)

        # 1) Temperature (logit 도메인)
        if self.method in ("temperature", "temperature+isotonic"):
            self.temp_scaler = fit_temperature(
                z, y, lr=self.lr, iters=self.iters, clamp_T=self.clamp_T, l2_reg=self.l2_reg
            )

        # 2) Isotonic (확률 도메인) + 표본 가중치(50:50)
        if self.method in ("isotonic", "temperature+isotonic"):
            if self.temp_scaler is not None:
                with torch.no_grad():
                    zt = self.temp_scaler(torch.tensor(z, dtype=torch.float32), clamp_T=self.clamp_T).numpy()
                p = _sigmoid_stable_numpy(zt)
            else:
                p = _sigmoid_stable_numpy(z)

            # 클래스 균형 가중치로 50:50 맞춤
            n_pos = max(1, int(y.sum()))
            n_neg = max(1, int((y == 0).sum()))
            w_pos = 0.5 / n_pos
            w_neg = 0.5 / n_neg
            sw = np.where(y == 1, w_pos, w_neg)

            # 노드 수 너무 작으면 아이소토닉은 포기(온도만 사용)
            # 정량적으론 서로 다른 p의 개수 사용
            if np.unique(p).size < self.min_iso_nodes:
                self.iso = None
            else:
                self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                self.iso.fit(p, y, sample_weight=sw)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64)
        if self.temp_scaler is not None:
            with torch.no_grad():
                z = self.temp_scaler(torch.tensor(z, dtype=torch.float32), clamp_T=self.clamp_T).numpy()
        p = _sigmoid_stable_numpy(z)
        if self.iso is not None:
            p = self.iso.predict(np.clip(p, 1e-7, 1-1e-7))
        return np.clip(p, 1e-7, 1-1e-7)
