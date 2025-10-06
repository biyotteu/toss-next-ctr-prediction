# src/data/sampler.py
from __future__ import annotations
import math, numpy as np, torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler[list[int]]):
    """
    y 기준으로 배치마다 pos:neg = pos_fraction:(1-pos_fraction)로 맞춰서 인덱스 배출.
    - replacement=True 로 샘플링 → class 수가 적어도 항상 배치 구성 안정
    - set_epoch(e) 지원 → 재현성 보장
    """
    def __init__(self, y: np.ndarray, batch_size: int, pos_fraction: float = 0.5,
                 num_batches: int | None = None, replacement: bool = True,
                 seed: int = 777, drop_last: bool = True):
        assert y.ndim == 1
        self.y = y.astype(np.int8, copy=False)
        self.batch_size = int(batch_size)
        self.pos_fraction = float(pos_fraction)
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self.pos_idx = np.where(self.y == 1)[0]
        self.neg_idx = np.where(self.y == 0)[0]
        assert len(self.pos_idx) > 0 and len(self.neg_idx) > 0, "BalancedBatchSampler requires both classes."

        N = len(y)
        if num_batches is None:
            nb = N // self.batch_size if drop_last else math.ceil(N / self.batch_size)
        else:
            nb = int(num_batches)
        self.num_batches = max(1, nb)

        self.n_pos = int(round(self.batch_size * self.pos_fraction))
        self.n_neg = self.batch_size - self.n_pos

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        P, N = len(self.pos_idx), len(self.neg_idx)

        for _ in range(self.num_batches):
            if self.replacement:
                p_sel = self.pos_idx[g.integers(0, P, size=self.n_pos)]
                n_sel = self.neg_idx[g.integers(0, N, size=self.n_neg)]
            else:
                # 무교체 시 pool 소진 대비: 필요하면 재셔플/순환
                if self.n_pos > P or self.n_neg > N:
                    raise ValueError("replacement=False but class pool smaller than per-batch requirement")
                p_sel = self.pos_idx[g.permutation(P)[:self.n_pos]]
                n_sel = self.neg_idx[g.permutation(N)[:self.n_neg]]
            batch = np.concatenate([p_sel, n_sel])
            g.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches
