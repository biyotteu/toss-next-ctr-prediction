from __future__ import annotations
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class ShardedNPY:
    """
    샤드 하나의 6개 배열을 mmap으로 열어 보관
    """
    def __init__(self, shard_meta: dict):
        self.paths = {k: shard_meta[k]["path"] for k in ["X_num","X_mask","X_cat","seq","y","groups","ids"]}
        self.arrs = {}
        self.rows = shard_meta["rows"]

    def open(self):
        # lazy open
        for k, p in self.paths.items():
            # test셋에는 y가 0으로 저장되어 있을 수 있음
            if os.path.exists(p):
                if k == "ids":
                    self.arrs[k] = np.load(p, allow_pickle=False)
                else:
                    self.arrs[k] = np.load(p, mmap_mode="r")
            else:
                self.arrs[k] = None

    def get_row(self, i: int, train: bool):
        if not self.arrs:
            self.open()
        out = {
            "X_num": self.arrs["X_num"][i],
            "X_mask": self.arrs["X_mask"][i],
            "X_cat": self.arrs["X_cat"][i],
            "seq": self.arrs["seq"][i],
            "groups": self.arrs["groups"][i],
        }
        if not train and "ids" in self.arrs:
            out["ids"] = self.arrs["ids"][i]
        if train:
            out["y"] = self.arrs["y"][i]
        return out

class ShardedDataset(Dataset):
    """
    - manifest.json을 읽어 전체 전역 인덱스 공간(0..N-1)을 제공
    - __getitem__은 전역 idx → (shard, local)로 변환해서 해당 샤드에서 로우 로드
    """
    def __init__(self, manifest_path: str, index: np.ndarray, train: bool, cat_cols: list):
        with open(manifest_path, "r") as f:
            man = json.load(f)
        self.manifest = man
        self.index = index.astype(np.int64)
        self.train = train
        self.cat_cols = cat_cols

        self.shards = []
        self.bounds = []
        for meta in man["shards"]:
            self.shards.append(ShardedNPY(meta))
            self.bounds.append((meta["start"], meta["end"]))
        self.bounds = np.array(self.bounds, dtype=np.int64)  # (S,2)
        self.starts = self.bounds[:,0]
        self.ends   = self.bounds[:,1]

    def __len__(self):
        return self.index.shape[0]

    def _locate(self, gidx: int):
        # 전역 gidx가 속하는 shard 찾기 (starts<=gidx<ends)
        s = np.searchsorted(self.ends, gidx, side="right")
        shard_id = s
        start = self.starts[shard_id]
        return shard_id, int(gidx - start)

    def __getitem__(self, i):
        gidx = int(self.index[i])
        sid, li = self._locate(gidx)
        return self.shards[sid].get_row(li, self.train)

def load_labels_groups_for_split(manifest_path: str):
    """
    KFold 분할용으로 y, groups만 메모리에 이어붙여 반환(용량 manageable)
    """
    with open(manifest_path, "r") as f:
        man = json.load(f)
    ys, gs = [], []
    for meta in tqdm(man["shards"], desc="join y/groups from shards", unit="shard"):
        y = np.load(meta["y"]["path"], mmap_mode="r")
        g = np.load(meta["groups"]["path"], mmap_mode="r")
        ys.append(np.array(y, copy=False))
        gs.append(np.array(g, copy=False))
    y = np.concatenate(ys)
    groups = np.concatenate(gs)
    return y, groups

def collate_sharded(batch):
    B = len(batch)
    keys = batch[0].keys()

    def stack_to_tensor(name, dtype=None):
        arrs = [b[name] for b in batch]
        # np.stack -> 새 연속 메모리(쓰기 가능) 확보 → from_numpy 시 경고 없음
        x = np.ascontiguousarray(np.stack(arrs, axis=0))
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return torch.from_numpy(x)

    out = {
        "X_num": stack_to_tensor("X_num", np.float32),
        "X_mask": stack_to_tensor("X_mask", np.float32),   # 마스크도 float로 처리(모델에서 float 기대)
        "X_cat": stack_to_tensor("X_cat", np.int64),
        "seq": stack_to_tensor("seq", np.int64),
    }
    if "y" in keys:
        y = np.asarray([b["y"] for b in batch], dtype=np.float32)
        out["y"] = torch.from_numpy(y)
    if "ids" in keys:
        # ids는 문자열/정수 혼재 가능 → numpy object/str 배열로 유지
        out["ids"] = np.asarray([b["ids"] for b in batch])
    if "groups" in keys:
        out["groups"] = torch.from_numpy(np.asarray([b["groups"] for b in batch], dtype=np.int64))
    return out
