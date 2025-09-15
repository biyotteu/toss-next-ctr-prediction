import torch
import torch.nn as nn

class QNNBlock(nn.Module):
    """A simple quadratic interaction block (multi-head)."""
    def __init__(self, in_dim: int, hidden: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.U = nn.Linear(in_dim, hidden * heads, bias=False)
        self.V = nn.Linear(in_dim, hidden * heads, bias=False)
        self.bn = nn.BatchNorm1d(hidden * heads)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden * heads, in_dim)

    def forward(self, x):
        q = self.U(x) * self.V(x)  # elementwise (Khatri–Rao style)
        q = self.bn(q)
        q = self.act(q)
        q = self.drop(q)
        out = self.proj(q) + x  # residual
        return out

class QuadraticLayer(nn.Module):
    def __init__(self, input_dim, num_row=2, net_dropout=0.1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim * num_row),
            nn.Dropout(net_dropout)
        )
        self.num_row = num_row
        self.input_dim = input_dim
    def forward(self, x):  # Khatri-Rao 스타일
        h = self.linear(x).view(-1, self.num_row, self.input_dim)  # B×R×D
        x = torch.einsum("bd,brd->bd", x, h)
        return x

class QuadraticNeuralNetworks(nn.Module):
    def __init__(self, input_dim, num_layers=3, net_dropout=0.1, num_row=2, batch_norm=False):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([QuadraticLayer(input_dim, num_row=num_row, net_dropout=net_dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(num_layers)]) if batch_norm else None
        self.dropout = nn.ModuleList([nn.Dropout(net_dropout) for _ in range(num_layers)]) if net_dropout > 0 else None
        self.activation = nn.ModuleList([nn.PReLU() for _ in range(num_layers)])
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        for i in range(self.num_layers):
            residual = x
            x = self.layers[i](x)
            if self.norm is not None:
                x = self.norm[i](x)
            x = self.activation[i](x)
            if self.dropout is not None:
                x = self.dropout[i](x)
            x = x + residual
        return self.fc(x)
