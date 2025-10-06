python src/tools/blend_submissions.py \
  --sub1 runs/v3_k132_s1/submission.csv \
  --sub2 runs/dare_qnn_next_k100_s1/submission.csv \
  --out runs/blend_logit50_50.csv \
  --method logit_mean --w 0.5