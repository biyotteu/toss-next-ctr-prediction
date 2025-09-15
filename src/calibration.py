import numpy as np
import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    """Learn a single scalar T>0 such that calibrated logits = logits / T"""
    def __init__(self, lr: float = 0.05, max_iter: int = 1000):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T=1
        self.lr = lr
        self.max_iter = max_iter

    def forward(self, logits):
        T = torch.exp(self.log_T)
        T = T.to(logits.device)
        return logits / T

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        self.to(logits.device)
        optim = torch.optim.LBFGS([self.log_T], lr=1.0, max_iter=self.max_iter)
        bce = nn.BCEWithLogitsLoss()
        def closure():
            optim.zero_grad()
            loss = bce(self.forward(logits), labels.float())
            loss.backward()
            return loss
        optim.step(closure)
        return torch.exp(self.log_T).detach().cpu().item()

    def fit_from_arrays(self, val_logits_np: np.ndarray, val_labels_np: np.ndarray):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logits = torch.tensor(val_logits_np, dtype=torch.float32, device=device).flatten()
        labels = torch.tensor(val_labels_np, dtype=torch.float32, device=device).flatten()
        self.to(device)
        T = self.fit(logits, labels)
        return T