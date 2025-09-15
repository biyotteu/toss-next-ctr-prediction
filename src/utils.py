import numpy as np
import torch, random, os
import glob, re

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

# ADD
def _extract_epoch_from_path(p: str) -> int:
    m = re.search(r'model_epoch(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else -1

def _pick_resume_path(output_dir: str, explicit: str = None) -> str | None:
    # 우선순위: explicit → model_epoch*.pt 중 가장 최신 → model_latest.pt → model_best.pt → model_final.pt
    if explicit and os.path.exists(explicit):
        return explicit
    candidates = glob.glob(os.path.join(output_dir, "model_epoch*.pt"))
    if candidates:
        candidates.sort(key=_extract_epoch_from_path)
        return candidates[-1]
    for name in ("model_latest.pt", "model_best.pt", "model_final.pt"):
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return p
    return None
