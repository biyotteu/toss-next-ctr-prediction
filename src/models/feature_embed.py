from __future__ import annotations
import torch
import torch.nn as nn

class NumericFeatureEmbedding(nn.Module):
    """
    (B, F_num) -> (B, F_num, D)
    각 연속 피처 j에 대해 길이 f_embed의 기울기/바이어스 벡터를 학습하고,
    마지막에 공용 선형(out_proj)으로 D로 정렬.
    """
    def __init__(self, n_features: int, f_embed: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.n_features = n_features
        self.f_embed = f_embed
        self.weight = nn.Parameter(torch.randn(n_features, f_embed) * 0.02)  # slope
        self.bias = nn.Parameter(torch.zeros(n_features, f_embed)) if bias else None
        self.out_proj = nn.Linear(f_embed, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        B, F = x.shape
        assert F == self.n_features
        e = x.unsqueeze(-1) * self.weight  # (B,F,f_embed)
        if self.bias is not None:
            e = e + self.bias
        # nn.Linear는 마지막 축에 적용되므로 (B,F,f_embed) -> (B,F,D)
        return self.out_proj(e)

class BinaryFeatureEmbedding(nn.Module):
    """
    (B, F_mask) -> (B, F_mask, D)
    마스크(0/1)를 스케일링하는 피처별 임베딩. 바이어스는 기본 끔(원하면 True).
    """
    def __init__(self, n_features: int, f_embed: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.n_features = n_features
        self.f_embed = f_embed
        self.weight = nn.Parameter(torch.randn(n_features, f_embed) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, f_embed)) if bias else None
        self.out_proj = nn.Linear(f_embed, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F = x.shape
        assert F == self.n_features
        e = x.unsqueeze(-1) * self.weight
        if self.bias is not None:
            e = e + self.bias
        return self.out_proj(e)
