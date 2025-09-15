import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from .config import Cfg
from .dataset import CTRFrame, infer_feature_types, CTRDataset, collate_fn
from .models.qin_like import QINLike
from .models.qin_v9ish import QINV9ish

@torch.no_grad()
def predict_main(cfg_path: str):
    cfg = Cfg.load(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    ckpt = os.path.join(cfg.output_dir, 'model_best.pt')
    ck = torch.load(ckpt, map_location=device)
    cfg_loaded = Cfg(ck.get('cfg', cfg.d))

    # optional temperature
    T = 1.0
    cal_path = os.path.join(cfg.artifacts_dir, 'temperature.json')
    if os.path.exists(cal_path):
        with open(cal_path, 'r', encoding='utf-8') as f:
            T = float(json.load(f).get('T', 1.0))
        print(f"[CAL] apply temperature T={T:.4f}")

    test_frame = CTRFrame(cfg.paths.test_parquet, cfg.force_drop_cols)
    # Use train header to infer feature sets
    from .dataset import CTRFrame as _CF
    train_frame = _CF(cfg.paths.train_parquet, cfg.force_drop_cols)
    train_df_head = train_frame.read_row_group(0)
    cats, nums = infer_feature_types(train_df_head, cfg.label_col, cfg.seq_col)
    cats = [c for c in cats if c not in set(cfg.force_drop_cols)]
    nums = [c for c in nums if c not in set(cfg.force_drop_cols)]

    # build model
    model_name = cfg.model.get('name', 'qin_like')
    if model_name == 'qin_v9ish':
        model = QINV9ish(
            n_cat=len(cats), n_num=len(nums),
            cat_buckets=cfg.category_hash_buckets, seq_vocab_size=cfg.seq_vocab_size,
            emb_dim=cfg.embedding_dim, dropout=cfg.model.dropout, attn_dim=cfg.model.attn_dim,
            qnn_num_layers=cfg.model.qnn.num_layers, qnn_num_row=cfg.model.qnn.num_row,
            qnn_net_dropout=cfg.model.qnn.net_dropout, qnn_batch_norm=cfg.model.qnn.batch_norm,
            attn_use_scale=cfg.model.attn_use_scale,
        ).to(device)
    else:
        model = QINLike(
            n_cat=len(cats), n_num=len(nums),
            cat_buckets=cfg.category_hash_buckets,
            seq_vocab_size=cfg.seq_vocab_size,
            emb_dim=cfg.embedding_dim,
            hidden_dim=cfg.model.hidden_dim,
            qnn_hidden=cfg.model.qnn_hidden,
            qnn_heads=cfg.model.qnn_heads,
            dropout=cfg.model.dropout,
            attn_dim=cfg.model.attn_dim,
            attn_topk=cfg.model.attn_topk,
        ).to(device)

    model.load_state_dict(ck['model_state'])
    model.eval()

    # numeric stats from training
    cfg_loaded.d.setdefault('num_stats', {})

    pf = test_frame.pf
    all_cols = [c for c in test_frame.all_cols if c not in set(cfg.force_drop_cols)]
    id_col = cfg.id_col
    ids_all, probs_all = [], []

    # Row group 처리 진행상황 표시
    rg_pbar = tqdm(range(pf.num_row_groups), desc="Processing row groups", leave=True)
    
    for rg in rg_pbar:
        df = test_frame.read_row_group(rg, columns=all_cols)
        ids = df[id_col].values
        ds = CTRDataset(df, cfg_loaded, cats, nums, is_train=False)
        bs = cfg.batch_size
        
        # 현재 row group 내의 배치 처리 진행상황 표시
        num_batches = (len(ds) + bs - 1) // bs
        batch_pbar = tqdm(range(0, len(ds), bs), 
                         desc=f"RG {rg+1}/{pf.num_row_groups} batches", 
                         leave=False,
                         total=num_batches)
        
        for i in batch_pbar:
            batch = [ds[j] for j in range(i, min(i+bs, len(ds)))]
            cats_b, nums_b, seq_b, tgt_b, _ = collate_fn(batch, cfg_loaded)
            cats_b = cats_b.to(device)
            nums_b = nums_b.to(device)
            seq_b = seq_b.to(device)
            tgt_b = tgt_b.to(device)
            logits = model(cats_b, nums_b, seq_b, tgt_b)
            logits = logits / T
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            
            # 배치 진행상황 업데이트
            batch_pbar.set_postfix({
                'samples': f'{min(i+bs, len(ds))}/{len(ds)}',
                'batch_size': len(batch)
            })
        
        ids_all.append(ids)
        
        # Row group 진행상황 업데이트
        rg_pbar.set_postfix({
            'samples_processed': len(ids),
            'total_batches': num_batches
        })

    probs_all = np.concatenate(probs_all)
    ids_all = np.concatenate(ids_all)

    sub = pd.DataFrame({cfg.id_col: ids_all, 'clicked': probs_all})
    sub = sub[[cfg.id_col, 'clicked']].sort_values(cfg.id_col)
    os.makedirs(os.path.dirname(cfg.pred_output_csv), exist_ok=True)
    sub.to_csv(cfg.pred_output_csv, index=False)
    print(f"[SUBMISSION] saved to {cfg.pred_output_csv}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default='./configs/config.yaml')
    args = p.parse_args()
    predict_main(args.cfg)