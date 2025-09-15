import numpy as np
import torch, random, os

DEF_EPS = 1e-8

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# fast 64-bit hashing → bucket id
# we avoid external deps; this is a simple mix-hash (not cryptographic)
def hash64(x: str, buckets: int) -> int:
    h = 1469598103934665603  # FNV offset basis
    for ch in x:
        h ^= ord(ch)
        h *= 1099511628211
        h &= 0xFFFFFFFFFFFFFFFF
    return int(h % buckets)