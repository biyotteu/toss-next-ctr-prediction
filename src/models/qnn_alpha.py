from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

def make_norm(name: str, d: int):
    return RMSNorm(d) if name.lower()=="rms" else nn.LayerNorm(d)

class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r), nn.ReLU(inplace=True),
            nn.Linear(c // r, c), nn.Sigmoid()
        )
    def forward(self, x):  # x: (B, C)
        w = self.fc(x.mean(dim=0, keepdim=True))  # 채널 통합 통계 기반 (간단형)
        return x * w

class QNNAlphaDetailed(nn.Module):
    """
    상세 QNN-α 추상화:
    - 입력: (B, F, D)  (F: 피처 수, D: 피처 임베딩 차원)
    - 멀티헤드 저랭크 이차 상호작용 (FM-trick + Khatri-Rao 유사)
    - pair_grouping: all | block
      * all: 모든 피처 쌍 집약(저랭크 근사)
      * block: (연속/마스크/카테고리/시퀀스표현) 블록별 쌍대 상호작용만
    - SE 채널 리웨이트 옵션
    - 잔차 + 정규화 + MLP 헤드
    """
    def __init__(self, *,
                 in_feat: int,
                 emb_dim: int,
                 heads: int = 8,
                 rank: int = 32,
                 proj_dim: int = 256,
                 mlp_hidden = (512,256),
                 dropout: float = 0.1,
                 use_se: bool = True,
                 se_reduction: int = 8,
                 use_residual: bool = True,
                 norm: str = "rms",
                 pair_grouping: str = "block",
                 block_slices: dict | None = None):
        super().__init__()
        self.in_feat = in_feat
        self.emb_dim = emb_dim
        self.heads = heads
        self.rank = rank
        self.proj_dim = proj_dim
        self.use_se = use_se
        self.use_residual = use_residual
        self.pair_grouping = pair_grouping
        self.block_slices = block_slices or {}  # {"num":(s,e), "mask":(s,e), "cat":(s,e), "seq":(s,e)}

        self.pre_norm = make_norm(norm, in_feat * emb_dim)
        # 멀티헤드 저랭크 투사 파라미터
        self.U = nn.Parameter(torch.randn(heads, emb_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(heads, rank, proj_dim) * 0.02)

        self.dropout = nn.Dropout(dropout)
        out_dim = heads * proj_dim

        if use_se:
            self.se = SEBlock(out_dim, r=se_reduction)
        else:
            self.se = nn.Identity()

        # 최종 MLP
        mlp_layers = []
        in_dim = out_dim + in_feat * emb_dim  # 원본 + 상호작용 결합
        for h in mlp_hidden:
            mlp_layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def _pair_interaction_all(self, z):  # z: (B,F,D)
        # FM trick on low-rank projections per head
        outs = []
        for h in range(self.heads):
            U = self.U[h]     # (D,r)
            V = self.V[h]     # (r,proj)
            A = torch.matmul(z, U)        # (B,F,r)
            s = A.sum(dim=1)              # (B,r)
            quad = s*s - (A*A).sum(dim=1) # (B,r)
            out = torch.matmul(quad, V)   # (B,proj)
            outs.append(out)
        return torch.cat(outs, dim=1)     # (B, heads*proj)

    def _pair_interaction_block(self, z):  # z: (B,F,D)
        # 블록내 상호작용만 집약하여 과도한 혼합을 줄임
        outs = []
        for name, (s, e) in self.block_slices.items():
            if e - s <= 1:  # 단일 피처면 건너뜀
                continue
            z_blk = z[:, s:e, :]  # (B,Fb,D)
            outs.append(self._pair_interaction_all(z_blk))
        return torch.stack(outs, dim=0).sum(dim=0) if outs else self._pair_interaction_all(z)

    def forward(self, feats):  # feats: (B,F,D)
        B, F, D = feats.shape
        z = feats.reshape(B, F*D)
        z = self.pre_norm(z)
        z = z.reshape(B, F, D)

        if self.pair_grouping == "block" and self.block_slices:
            inter = self._pair_interaction_block(z)
        else:
            inter = self._pair_interaction_all(z)

        inter = self.se(inter)
        inter = self.dropout(inter)

        base = z.reshape(B, F*D)
        if self.use_residual:
            out = torch.cat([base, inter], dim=1)
        else:
            out = torch.cat([base.detach(), inter], dim=1)  # 잔차 미사용 시 정보 중복 방지

        logit = self.mlp(out).squeeze(1)
        return logit
