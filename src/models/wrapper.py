from __future__ import annotations
import torch, torch.nn as nn
from .dare import DARE
from .qnn_alpha import QNNAlphaDetailed
from .feature_embed import NumericFeatureEmbedding, BinaryFeatureEmbedding

class CTRModel(nn.Module):
    def __init__(self, cfg, seq_vocab: int, num_feat_dim: int, mask_feat_dim: int,
                 cat_cardinals: dict, cat_cols_order: list):
        super().__init__()
        D = cfg["model"]["emb_dim"]
        f_embed = int(cfg["model"]["qnn_alpha"].get("feature_embed_dim", max(8, D//4)))

        self.cat_cols_order = cat_cols_order
        self.num_dim = num_feat_dim
        self.mask_dim = mask_feat_dim
        self.D = D

        # (A) 임베딩 드롭아웃 추가
        p = float(cfg["model"].get("embedding_dropout", 0.0))
        self.emb_dropout = nn.Dropout(p)

        # (a) 연속/마스크 임베딩 모듈
        self.num_embed = NumericFeatureEmbedding(num_feat_dim, f_embed, D) if num_feat_dim>0 else None
        self.mask_embed = BinaryFeatureEmbedding(mask_feat_dim, f_embed, D) if mask_feat_dim>0 else None

        # 카테고리 임베딩
        self.cat_embs = nn.ModuleDict()
        self.cat_proj = nn.ModuleDict()
        cat_dims_map = (cfg["model"].get("cat_embedding_dims", {}) if "model" in cfg else {})

        for c, card in cat_cardinals.items():
            d_in = int(cat_dims_map.get(c, D))
            self.cat_embs[c] = nn.Embedding(card, d_in)
            self.cat_proj[c] = nn.Linear(d_in, D, bias=False)

        # 컨텍스트 압축(S2 쿼리용)
        # 입력은: [num_ctx(D?), mask_ctx(D?), cat_mean(D)] 합성 후 MLP
        ctx_in = 0
        if num_feat_dim > 0:  ctx_in += D
        if mask_feat_dim > 0: ctx_in += D
        ctx_in += D  # cat_mean
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_in, D), nn.ReLU(inplace=True))

        # DARE (b: n_layers/n_heads는 cfg.sequence.tfm.*로 제어)
        tfm_cfg = cfg["sequence"].get("tfm", {})
        self.dare = DARE(seq_vocab=seq_vocab,
                         emb_dim=D,
                         dropout=cfg["model"]["dare_dropout"],
                         query_mode=cfg["sequence"]["query_mode"],
                         transformer_block=cfg["sequence"]["transformer_block"],
                         top_k=cfg["sequence"]["top_k"],
                         recency_tau=cfg["sequence"]["recency_tau"],
                         pad_id=cfg["sequence"]["pad_id"],
                         tfm_cfg=tfm_cfg)

        # QNN-α 상세 (c: pair_grouping 'all' 지원은 모듈 내부에서 이미 처리)
        self.use_qnn = cfg["model"]["qnn_alpha"]["enabled"]
        if self.use_qnn:
            # 피처 개수 F 계산: [u_seq(1) + num(Fn) + mask(Fm) + cats(Fc)]
            F_seq = 1
            F_num = num_feat_dim if num_feat_dim>0 else 0
            F_mask= mask_feat_dim if mask_feat_dim>0 else 0
            F_cat = len(cat_cardinals)
            F_all = F_seq + F_num + F_mask + F_cat

            # 블록 슬라이스(블록 모드에서 사용; 'all'일 땐 무시)
            block_slices = {}
            ofs = 0
            block_slices["seq"]  = (ofs, ofs+F_seq);  ofs += F_seq
            if F_num>0:
                block_slices["num"]  = (ofs, ofs+F_num); ofs += F_num
            if F_mask>0:
                block_slices["mask"] = (ofs, ofs+F_mask); ofs += F_mask
            block_slices["cat"]  = (ofs, ofs+F_cat);  ofs += F_cat
            assert ofs == F_all

            qa = cfg["model"]["qnn_alpha"]
            self.qnn = QNNAlphaDetailed(
                in_feat = F_all,
                emb_dim = D,
                heads   = qa["heads"],
                rank    = qa["rank"],
                proj_dim= qa["proj_dim"],
                mlp_hidden = qa["mlp_hidden"],
                dropout    = qa["dropout"],
                use_se     = qa["use_se"],
                se_reduction = qa["se_reduction"],
                use_residual = qa["use_residual"],
                norm         = qa["norm"],
                pair_grouping= qa["pair_grouping"],  # "all" / "block"
                block_slices = block_slices
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(D * (1 + (num_feat_dim>0) + (mask_feat_dim>0) + len(cat_cardinals)), 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 1)
            )

        self.sigmoid = nn.Sigmoid()
        self.query_key = cfg["sequence"]["query_key"]
        self.aux_weight = float(cfg["model"]["qnn_alpha"].get("aux_head_weight", 0.0))

    def _embed_cats(self, X_cat):
        embs = []
        for i, name in enumerate(self.cat_cols_order):
            e = self.cat_embs[name](X_cat[:, i])     # (B, d_in)
            e = self.cat_proj[name](e)               # (B, D)
            embs.append(e)
        return embs  # list of (B,D)

    def _context_vector(self, num_e, mask_e, cat_embs):
        """
        S2/concat용 컨텍스트: 연속·마스크는 피처 평균, 카테고리는 평균.
        """
        ctx_parts = []
        if num_e is not None:   ctx_parts.append(num_e.mean(dim=1))   # (B,D)
        if mask_e is not None:  ctx_parts.append(mask_e.mean(dim=1))  # (B,D)
        cat_mean = torch.stack(cat_embs, dim=1).mean(dim=1) if len(cat_embs)>0 else 0
        if isinstance(cat_mean, int):  # no cats
            cat_mean = torch.zeros((ctx_parts[0].shape[0], self.D), device=ctx_parts[0].device) if ctx_parts else None
        if cat_mean is not None: ctx_parts.append(cat_mean)
        ctx = torch.cat(ctx_parts, dim=1) if len(ctx_parts)>0 else None
        return ctx

    def _make_query(self, qmode, query_key, feats_ctx, cat_embs):
        if qmode == "S1":
            q_idx = self.cat_cols_order.index(query_key)
            return cat_embs[q_idx]
        elif qmode == "S2":
            return self.ctx_mlp(feats_ctx)
        else:
            q_idx = self.cat_cols_order.index(query_key)
            return 0.5 * (cat_embs[q_idx] + self.ctx_mlp(feats_ctx))

    def forward(self, batch):
        device = next(self.parameters()).device
        X_num = batch["X_num"].to(device, non_blocking=True).float()
        X_mask= batch["X_mask"].to(device, non_blocking=True).float()
        X_cat = batch["X_cat"].to(device, non_blocking=True).long()
        seq   = batch["seq"].to(device, non_blocking=True).long()

        # 임베딩
        num_e  = self.num_embed(X_num)  if (self.num_embed is not None and X_num.shape[1]>0) else None   # (B,Fn,D)
        mask_e = self.mask_embed(X_mask) if (self.mask_embed is not None and X_mask.shape[1]>0) else None# (B,Fm,D)
        cat_embs = self._embed_cats(X_cat)  # list of (B,D)
        cat_stack = torch.stack(cat_embs, dim=1) if len(cat_embs)>0 else None  # (B,Fc,D)
        cat_stack = self.emb_dropout(cat_stack) if cat_stack is not None else None

        # 쿼리
        feats_ctx = self._context_vector(num_e, mask_e, cat_embs)  # (B,*,D) -> (B,k*D)
        query_vec = self._make_query(self.dare.query_mode, self.query_key, feats_ctx, cat_embs)

        # DARE
        u_seq, aux_logit = self.dare(seq, query_vec)  # (B,D), (B,)

        # 헤드
        if self.use_qnn:
            feat_list = [u_seq.unsqueeze(1)]
            if num_e is not None:   feat_list.append(num_e)
            if mask_e is not None:  feat_list.append(mask_e)
            if cat_stack is not None: feat_list.append(cat_stack)
            xF = torch.cat(feat_list, dim=1)  # (B,F,D)
            logits = self.qnn(xF)
        else:
            feats_fc = [u_seq]
            if num_e is not None:   feats_fc.append(num_e.mean(dim=1))
            if mask_e is not None:  feats_fc.append(mask_e.mean(dim=1))
            feats_fc += cat_embs
            all_feats = torch.cat(feats_fc, dim=1)
            logits = self.fc(all_feats).squeeze(1)

        prob = self.sigmoid(logits)
        return logits, prob, aux_logit
