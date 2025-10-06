python -c "import yaml,json; from src.data.build_cache_v1 import build_train_and_test; cfg=yaml.safe_load(open('cfgs/v4_k120_s1.yaml')); mp_tr, mp_te = build_train_and_test(cfg); print(mp_tr, mp_te)"
