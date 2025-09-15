import torch
import torch.nn as nn
import torch.nn.functional as F
from .qnn import QNNBlock

class QINLike(nn.Module):
    def __init__(self, n_cat: int, n_num: int, cat_buckets: int, seq_vocab_size: int,
                 emb_dim: int = 32, hidden_dim: int = 256, qnn_hidden: int = 256,
                 qnn_heads: int = 4, dropout: float = 0.1, attn_dim: int = 128, attn_topk: int = 30):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_cat = n_cat
        self.attn_topk = attn_topk

        # categorical embeddings (hash buckets share one table)
        self.cat_emb = nn.Embedding(cat_buckets + 1, emb_dim)
        # sequence token embedding
        self.seq_emb = nn.Embedding(seq_vocab_size + 1, emb_dim)

        # numeric projection
        self.num_proj = nn.Linear(n_num, emb_dim) if n_num > 0 else None

        # target/ad embedding (reuses cat table)
        self.tgt_emb = self.cat_emb

        # target-conditioned sparse attention over sequence (Q, K, V)
        self.q = nn.Linear(emb_dim, attn_dim, bias=False)
        self.k = nn.Linear(emb_dim, attn_dim, bias=False)
        self.v = nn.Linear(emb_dim, attn_dim, bias=False)
        self.attn_out = nn.Linear(attn_dim, emb_dim)

        # fusion
        fused_in = emb_dim * (3 + (1 if n_num > 0 else 0))  # tgt + interest + [num] + pooled cat

        self.cat_pool = nn.Linear(n_cat * emb_dim, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.qnn1 = QNNBlock(hidden_dim, qnn_hidden, qnn_heads, dropout)
        self.qnn2 = QNNBlock(hidden_dim, qnn_hidden, qnn_heads, dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, cats, nums, seqs, tgt_ids):
        # embeddings
        cat_e = self.cat_emb(cats)              # [B, n_cat, D]
        cat_pool = self.cat_pool(cat_e.reshape(cat_e.size(0), -1))  # [B, D]

        if self.num_proj is not None and nums.numel() > 0:
            num_e = self.num_proj(nums)
        else:
            num_e = torch.zeros(cat_pool.size(0), self.emb_dim, device=cat_pool.device)

        tgt_e = self.tgt_emb(tgt_ids)           # [B, D]

        # seq embeddings
        B, L = seqs.size(0), seqs.size(1)
        seq_e = self.seq_emb(seqs)              # [B, L, D]
        # attention scores: Q(target) vs K(seq)
        q = self.q(tgt_e).unsqueeze(1)          # [B, 1, A]
        k = self.k(seq_e)                       # [B, L, A]
        v = self.v(seq_e)                       # [B, L, A]
        scores = torch.matmul(q, k.transpose(1,2)).squeeze(1) / (k.size(-1) ** 0.5)  # [B, L]
        # top-k sparse attention
        topk = min(self.attn_topk, scores.size(1))
        vals, idx = torch.topk(scores, topk, dim=1)
        mask = torch.zeros_like(scores).scatter_(1, idx, 1.0)
        attn_weights = F.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=1)
        interest = torch.bmm(attn_weights.unsqueeze(1), v).squeeze(1)  # [B, A]
        interest = self.attn_out(interest)  # [B, D]

        fused = torch.cat([tgt_e, interest, cat_pool, num_e], dim=1)
        h = self.mlp(fused)
        h = self.qnn1(h)
        h = self.qnn2(h)
        logit = self.out(h).squeeze(1)
        return logit