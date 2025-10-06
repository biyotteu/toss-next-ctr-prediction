from __future__ import annotations
import torch, math
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        # x: (..., d)
        return self.w * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

def make_norm(name: str, d: int):
    if name.lower() == "rms":
        return RMSNorm(d)
    return nn.LayerNorm(d)

class PositionalBias(nn.Module):
    """
    간단한 상대적 위치 바이어스 (K 길이 내에서 거리 임베딩)
    """
    def __init__(self, max_len=512, n_heads=4, dim=32):
        super().__init__()
        self.rel = nn.Embedding(2*max_len+1, n_heads)
        self.max_len = max_len
        self.n_heads = n_heads
    def forward(self, qlen, klen):
        # distance: (qlen, klen)
        i = torch.arange(qlen, device=self.rel.weight.device).unsqueeze(1)
        j = torch.arange(klen, device=self.rel.weight.device).unsqueeze(0)
        d = j - i   # -qlen..klen
        d = d.clamp(-self.max_len, self.max_len) + self.max_len
        # (qlen, klen, n_heads)
        bias = self.rel(d)  # last dim n_heads
        return bias.permute(2, 0, 1)  # (n_heads, qlen, klen)

class DAREEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, mha_dropout, ffn_hidden, ffn_dropout, norm="rms", add_pos_bias=True, max_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=mha_dropout, batch_first=True)
        self.norm1 = make_norm(norm, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden), nn.GELU(), nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden, d_model)
        )
        self.norm2 = make_norm(norm, d_model)
        self.add_pos_bias = add_pos_bias
        self.pbias = PositionalBias(max_len=max_len, n_heads=n_heads) if add_pos_bias else None

    def forward(self, x, key_padding_mask=None):
        # x: (B,K,D)
        B, K, D = x.shape
        if self.add_pos_bias:
            bias = self.pbias(K, K)  # (H,K,K)
            # MultiheadAttention의 attn_mask는 (B*num_heads, K, K) 또는 (K,K)
            # 여기서는 (K,K)로 평균치 적용
            attn_mask = bias.mean(0)  # (K,K)
        else:
            attn_mask = None

        h, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + h
        x = self.norm1(x)
        h2 = self.ffn(x)
        x = x + h2
        x = self.norm2(x)
        return x

class DARE(nn.Module):
    """
    상세 DARE:
    - seq att/rep 임베딩 분리
    - 쿼리(query_mode=S1/S2/concat)
    - recency decay + dot-score로 Top-K 후보 선택
    - 후보 위에 Transformer N층 (상대좌표 바이어스 옵션)
    - 가중집약(gating=softmax|relu)
    """
    def __init__(self, seq_vocab: int, emb_dim: int, dropout: float,
                 query_mode: str="S1",
                 transformer_block: bool=True,
                 top_k: int=80,
                 recency_tau: int=256,
                 pad_id: int=0,
                 tfm_cfg: dict | None = None):
        super().__init__()
        self.emb_att = nn.Embedding(seq_vocab, emb_dim, padding_idx=pad_id)
        self.emb_rep = nn.Embedding(seq_vocab, emb_dim, padding_idx=pad_id)
        self.query_mode = query_mode
        self.top_k = top_k
        self.tau = recency_tau
        self.pad_id = pad_id
        self.dropout = nn.Dropout(dropout)

        self.transformer_block = transformer_block
        if transformer_block:
            n_layers = tfm_cfg.get("n_layers", 2)
            n_heads  = tfm_cfg.get("n_heads", 4)
            mha_dropout = tfm_cfg.get("mha_dropout", 0.1)
            ffn_hidden  = tfm_cfg.get("ffn_hidden", 256)
            ffn_dropout = tfm_cfg.get("ffn_dropout", 0.1)
            norm        = tfm_cfg.get("norm", "rms")
            add_pos     = tfm_cfg.get("add_positional_bias", True)
            self.layers = nn.ModuleList([
                DAREEncoderLayer(emb_dim, n_heads, mha_dropout, ffn_hidden, ffn_dropout, norm, add_pos, max_len=top_k)
                for _ in range(n_layers)
            ])
        gating = tfm_cfg.get("gating", "softmax") if tfm_cfg else "softmax"
        self.gating = gating

        # DARE-only 보조헤드(자기엔상블용)
        self.aux_head = nn.Linear(emb_dim, 1)

    def topk_select(self, seq_ids, query_vec):
        """
        seq_ids: (B,L) long
        query_vec: (B,D) float
        return: selected rep vectors (B,K,D) and weights logits (B,K)
        """
        att = self.emb_att(seq_ids)  # (B,L,D)
        rep = self.emb_rep(seq_ids)  # (B,L,D)

        B, L, D = att.shape
        pos = torch.arange(L, device=seq_ids.device).view(1, L).expand(B, L)
        decay = torch.exp(-(L - 1 - pos).float() / max(1.0, float(self.tau))).unsqueeze(-1)  # (B,L,1)

        scores = (att * query_vec.unsqueeze(1)).sum(-1, keepdim=True)  # (B,L,1)
        scores = scores + torch.log(decay + 1e-8)

        pad_mask = (seq_ids == self.pad_id).unsqueeze(-1)
        scores = scores.masked_fill(pad_mask, -1e9)

        K = min(self.top_k, L)
        vals, idx = scores.squeeze(-1).topk(k=K, dim=1)  # (B,K)
        sel_rep = torch.gather(rep, 1, idx.unsqueeze(-1).expand(-1, -1, D))  # (B,K,D)
        return sel_rep, vals  # vals는 집약 가중 로짓

    def forward(self, seq_ids, query_vec):
        sel_rep, vals = self.topk_select(seq_ids, query_vec)  # (B,K,D), (B,K)
        if self.transformer_block:
            # 모든 토큰 유효(Top-K에서 pad 제거): key_padding_mask 없음
            x = sel_rep
            for layer in self.layers:
                x = layer(x, key_padding_mask=None)
        else:
            x = sel_rep

        # 게이팅
        if self.gating == "relu":
            w = F.relu(vals)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        else:
            w = torch.softmax(vals, dim=1)  # 안정

        u_seq = (x * w.unsqueeze(-1)).sum(dim=1)  # (B,D)
        u_seq = self.dropout(u_seq)

        # 보조헤드 로그잇 (자기엔상블용)
        aux_logit = self.aux_head(u_seq).squeeze(1)  # (B,)
        return u_seq, aux_logit
