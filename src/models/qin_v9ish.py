# src/models/qin_v9ish.py
import torch
import torch.nn as nn
from .qnn import QuadraticNeuralNetworks

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    def forward(self, Q, K, V, scale=None, mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores * mask  # 0=mask
        attention = self.relu(scores)             # QIN 계열: softmax 대신 ReLU 게이팅
        output = torch.matmul(attention, V)
        return output

class MultiHeadTargetAttention(nn.Module):
    def __init__(self, input_dim=64, attention_dim=128, use_scale=True):
        super().__init__()
        self.attention_dim = attention_dim
        self.scale = (self.attention_dim ** 0.5) if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_o = nn.Linear(attention_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()
    def forward(self, target_item, history_sequence, mask=None):
        B = target_item.size(0)
        Q = self.W_q(target_item).view(B, 1, self.attention_dim)
        K = self.W_k(history_sequence)
        V = self.W_v(history_sequence)
        out = self.dot_attention(Q, K, V, scale=self.scale, mask=mask)
        out = out.view(B, -1)
        out = self.W_o(out) + target_item  # residual
        return out

class QINV9ish(nn.Module):
    """QIN_v9에 가까운 구현: ReLU 게이팅 어텐션 + QuadraticNeuralNetworks"""
    def __init__(self, n_cat, n_num, cat_buckets, seq_vocab_size,
                 emb_dim=32, dropout=0.1, attn_dim=None,
                 qnn_num_layers=3, qnn_num_row=2, qnn_net_dropout=0.1, qnn_batch_norm=False,
                 attn_use_scale=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.cat_emb = nn.Embedding(cat_buckets + 1, emb_dim)
        self.seq_emb = nn.Embedding(seq_vocab_size + 1, emb_dim)
        self.num_proj = nn.Linear(n_num, emb_dim) if n_num > 0 else None
        self.tgt_emb = self.cat_emb

        item_info_dim = emb_dim
        A = attn_dim or (item_info_dim * 2)
        self.attn = MultiHeadTargetAttention(input_dim=item_info_dim, attention_dim=A, use_scale=attn_use_scale)
        self.cat_pool = nn.Linear(n_cat * emb_dim, emb_dim)

        fused_in = emb_dim * (3 + (1 if n_num > 0 else 0))  # target + pooled_attn + cat_pool + [num]
        self.qnn = QuadraticNeuralNetworks(
            input_dim=fused_in,
            num_layers=qnn_num_layers,
            net_dropout=qnn_net_dropout,
            num_row=qnn_num_row,
            batch_norm=qnn_batch_norm,
        )

    def forward(self, cats, nums, seqs, tgt_ids):
        cat_e = self.cat_emb(cats)                    # [B, n_cat, D]
        cat_pool = self.cat_pool(cat_e.reshape(cat_e.size(0), -1))
        if self.num_proj is not None and nums.numel() > 0:
            num_e = self.num_proj(nums)
        else:
            num_e = torch.zeros(cat_pool.size(0), self.emb_dim, device=cat_pool.device)
        tgt_e = self.tgt_emb(tgt_ids)
        seq_e = self.seq_emb(seqs)                    # [B, L, D]
        mask = (seqs > 0).float().unsqueeze(1)        # [B,1,L]
        pooled = self.attn(tgt_e, seq_e, mask=mask)   # [B, D]
        fused = torch.cat([tgt_e, pooled, cat_pool, num_e], dim=1)
        logit = self.qnn(fused).squeeze(1)
        return logit
