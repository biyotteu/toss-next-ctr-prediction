import os, random, numpy as np, torch

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 완전 재현 모드
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS 재현성 (CUDA 11+/12+ 공통)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    else:
        torch.backends.cudnn.benchmark = True
