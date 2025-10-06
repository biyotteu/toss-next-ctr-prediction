# toss_v6 — CTR 예측 파이프라인 (DARE + QNN-α)

본 리포지토리는 대규모 클릭 예측(CTR) 문제를 위한 엔드투엔드 파이프라인입니다. Parquet 원천 데이터 → NPY 샤드 캐시화 → 시퀀스 인코더(DARE) + 피처 상호작용(QNN-α) 학습 → 앙상블/캘리브레이션 → 제출 파일 생성

- 데이터 전처리: Polars/PyArrow 기반 Parquet 스트리밍, 수치 결측 처리, 카테고리 해싱, 시퀀스 토큰화, 대용량 NPY 샤드 캐싱(manifest.json 관리)
- 모델 학습: DARE 시퀀스 인코더 + QNN-α 헤드, AMP/compile, EMA, K-Fold, balanced sampler(옵션)
- 평가/로깅: AP/WLL/Score(0.5*AP+0.5*WLL), TensorBoard, CSV 로그
- 추론/제출: 폴드 체크포인트 앙상블(logit mean 등), 제출 CSV 저장(ID, clicked)
- 유틸/도구: 공출현(covis) 피처 빌드, 제출물 블렌딩, 캘리브레이션 등

---

## 1) 빠른 시작

### 환경 준비
- Python, CUDA 환경을 준비합니다(가능하면 NVIDIA GPU 권장).
- 의존성 설치:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> CUDA가 필요한 경우 PyTorch 공식 가이드에 따라 CUDA 버전에 맞춰 설치하세요.

### 데이터 경로 설정
- 기본 설정 파일은 `cfgs/dare_qnn_next.yaml` 또는 변형(`cfgs/v3_base.yaml` 등)을 사용합니다.
- 아래 항목을 실제 데이터 경로로 수정하세요:
  - `data.train_path`, `data.test_path` (Parquet)
  - `data.cache_dir` (샤드 캐시 저장 경로)

### 캐시 빌드 (선택: 최초 1회)
- 캐시가 없거나 `data.use_cache: false` 인 경우 캐시를 빌드합니다.
- 제공 스크립트를 사용하거나, 한 줄 명령으로 실행할 수 있습니다.

```bash
# 예시: shard.sh 의 동작과 동일
python -c "import yaml,json; from src.data.build_cache_v1 import build_train_and_test; cfg=yaml.safe_load(open('cfgs/dare_qnn_next.yaml')); mp_tr, mp_te = build_train_and_test(cfg); print(mp_tr, mp_te)"
```

### 학습
- 기본 학습 스크립트:

```bash
bash train.sh
# 또는
python -m src.train --cfg cfgs/dare_qnn_next.yaml
```

- 결과물(체크포인트/로그)은 `runs/<exp_name>/` 아래에 저장됩니다.
  - 체크포인트: `ckpt_folds_*.pt`
  - 로그: `train_log.csv`, TensorBoard(`runs/<exp_name>/` 디렉토리)

### 추론/제출 생성
- 학습 완료 후 추론을 수행해 제출 파일을 만듭니다.

```bash
bash infer.sh
# 또는
python -m src.infer --cfg cfgs/dare_qnn_next.yaml
```

- 출력: `runs/<exp_name>/submission.csv` (헤더: `ID,clicked`)

### 제출물 블렌딩(옵션)
- 두 제출 파일을 다양한 방식으로 블렌딩할 수 있습니다.

```bash
python src/tools/blend_submissions.py \
  --sub1 runs/v3_k132_s1/submission.csv \
  --sub2 runs/dare_qnn_next_k100_s1/submission.csv \
  --out runs/blend_logit50_50.csv \
  --method logit_mean --w 0.5
```

---

## 2) 설정(Configurations)
`cfgs/*.yaml` 의 공통 키를 요약합니다. 실제 사용은 각 YAML을 참고하세요.

- `exp_name`: 실험명(결과 디렉토리 이름)
- `seed`, `device`, `deterministic`, `amp`, `use_compile`
- `data`
  - `train_path`, `test_path`: Parquet 경로
  - `cache_dir`, `use_cache`, `chunked_build`, `shard_rows`
  - `add_isna_mask`, `impute_strategy`
  - `cat_cols`, `num_cols_explicit` 또는 `num_patterns`
  - `hash_buckets`(+`hash_buckets_margin`), `manifest_train`, `manifest_test`
- `sequence`
  - `col`, `max_len`, `pad_id`
  - `top_k`, `recency_tau`
  - `query_mode`(S1|S2|concat), `query_key`
  - `transformer_block`, `tfm.{n_layers,n_heads,dropout,ffn_hidden,...}`
- `model`
  - `emb_dim`, `dare_dropout`, `cat_embedding_dims`
  - `qnn_alpha.{enabled, feature_embed_dim, heads, rank, proj_dim, mlp_hidden, dropout, use_se, se_reduction, use_residual, norm, pair_grouping, aux_head_weight}`
- `train`
  - `batch_size`, `epochs`, `lr`, `weight_decay`, `warmup_epochs`, `grad_clip_norm`, `early_stop_patience`
- `cv`
  - `n_splits`, `group_key`, `stratify_target`, `time_key`, `composite_group`
- `calibration`
  - `enabled`, `method`(temperature), `lr`, `iters`
- `logging`
  - `log_dir`, `tb`, `csv_log`, `verbose_steps`
- `ensemble`
  - `method`(logit_mean|mean|geom_mean|median|trim_mean|rank_avg|weighted)
  - `trim_ratio`, `weights`, `val_weight_temperature`
- `sampler`
  - `type`(balanced 사용 시), `pos_fraction`
- `ema`
  - `enabled`, `decay`, `eval_with_ema` 등
- `features.covis`
  - 공출현(covis) 피처 관련 파라미터(선택 기능)

---

## 3) 프로젝트 구조
- `cfgs/`: 실험 구성 YAML 모음
- `src/`
  - `data/`: Parquet → 샤드 캐시 빌드, Dataset/Collate (`build_cache_v1.py`, `dataset.py`)
  - `features/`: 예시 공출현 피처(`covis.py`)
  - `models/`: DARE, QNN-α, 래퍼(`dare.py`, `qnn_alpha.py`, `wrapper.py`)
  - `utils/`: 로깅/메트릭/시드/스케줄러/EMA 등
  - `tools/`: 블렌딩/캘리브레이션/진단 스크립트
  - `train.py`, `infer.py`: 학습/추론 엔트리포인트
- 최상위 스크립트: `train.sh`, `infer.sh`, `blend.sh`, `shard.sh`
- 결과 디렉토리: `runs/<exp_name>/`

---

## 4) 지표/로깅
- 검증 시 `AP`, `WLL(50:50)`, `Score=0.5*AP+0.5*WLL`를 기록합니다.
- 로깅: 콘솔 + CSV(`train_log.csv`) + TensorBoard(`runs/<exp_name>/`)

---

## 5) 자주 묻는 질문(FAQ) / 트러블슈팅
- 추론 시 "No checkpoints found"
  - 학습 완료 후 `runs/<exp_name>/ckpt_folds_*.pt`가 존재하는지 확인하세요(설정의 `exp_name` 일치 필요).
- OOM(메모리 부족)
  - `train.batch_size`를 줄이거나 `sequence.max_len`, `top_k`를 축소하세요. 필요 시 AMP(`amp: bf16|fp16`)를 고려하세요.
- 데이터 경로 오류
  - 설정 파일의 `data.train_path`, `data.test_path`가 실제 Parquet 경로인지 재확인하세요.

---

## 6) 라이선스
- 별도 라이선스 파일이 없다면 내부/과제 용도로 사용하세요. 외부 배포가 필요하면 프로젝트 소유자와 상의하세요.

---

## 7) 참고
- DARE, QNN-α 등 구현/사용법은 각 모듈 및 설정 파일을 참고하세요(`src/models/*`, `cfgs/*`).
