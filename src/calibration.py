import numpy as np
import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    """Learn a single scalar T>0 such that calibrated logits = logits / T"""
    def __init__(self, lr: float = 0.05, max_iter: int = 1000, use_wll_weights: bool = False):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T=1
        self.lr = lr
        self.max_iter = max_iter
        self.use_wll_weights = use_wll_weights

    def _loss(self, logits, labels):
        # BCE with optional class-balanced weights (0.5 for each class)
        if self.use_wll_weights:                  # ⬅️ WLL 방식일 때
            probs = torch.sigmoid(logits)
            pos = (labels == 1)
            neg = ~pos
            n_pos = torch.clamp(pos.sum(), min=1)
            n_neg = torch.clamp(neg.sum(), min=1)
            w = torch.zeros_like(labels, dtype=logits.dtype)
            w[pos] = 0.5 / n_pos
            w[neg] = 0.5 / n_neg
            bce = nn.BCEWithLogitsLoss(weight=w, reduction='sum')
            return bce(logits, labels.float())
        else:                                     # ⬅️ 일반 BCE
            bce = nn.BCEWithLogitsLoss()
            return bce(logits, labels.float())

    def forward(self, logits):
        T = torch.exp(self.log_T)
        T = T.to(logits.device)
        return logits / T

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        optim = torch.optim.LBFGS([self.log_T], lr=1.0, max_iter=self.max_iter,
                                  tolerance_grad=1e-9, tolerance_change=1e-9)
        def closure():
            optim.zero_grad()
            loss = self._loss(self.forward(logits), labels)  # ⬅️ 공통 경로에서 _loss 사용
            loss.backward()
            return loss
        optim.step(closure)
        with torch.no_grad():
            self.log_T.clamp_(min=np.log(1e-2), max=np.log(100.0))  # ⬅️ T 안정 범위
        return torch.exp(self.log_T).detach().cpu().item()

    def fit_from_arrays(self, val_logits_np: np.ndarray, val_labels_np: np.ndarray):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logits = torch.tensor(val_logits_np, dtype=torch.float32, device=device).flatten()
        labels = torch.tensor(val_labels_np, dtype=torch.float32, device=device).flatten()
        self.to(device)
        T = self.fit(logits, labels)
        return T