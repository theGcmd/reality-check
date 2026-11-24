# TimeDistill audit — pre-registration (DRAFT, not frozen; nothing scored)

Paper: "TimeDistill: Efficient Long-Term Time Series Forecasting with MLP via Cross-Architecture Distillation",
KDD 2026, arXiv 2502.15016. Code: github.com/LingFengGold/TimeDistill (shallow clone, 2026-11-24).

## Claim under test
ETTh1, input 720, average over H ∈ {96,192,336,720} (Table 1): TimeDistill MSE 0.429; best baseline TimeMixer 0.459;
PatchTST 0.498. Paper: teachers/baselines use "default configurations from their original papers"; drop_last=False.

## What the released code does (verified 2026-11-24)
`scripts/long_term_forecast/ETTh1/PatchTST.sh` runs PatchTST with e_layers 1, n_heads 16 and run.py defaults:
d_model 512, d_ff 2048, dropout 0.1, 10 epochs, patience 5, batch 32, lr 1e-4.
PatchTST's own ETTh1 setting (official repo / paper) is e_layers 3, n_heads 4, d_model 16, d_ff 128, dropout 0.3,
up to 100 epochs. PatchTST's paper reports ETTh1 average MSE 0.413 at input 512 (0.370/0.413/0.422/0.447) and
0.417 at input 336.

## Question
Does TimeDistill's ETTh1 average (0.429) beat PatchTST run in TimeDistill's own harness, input 720, every test
window kept, with PatchTST's own published ETTh1 hyperparameters?

## Design (to be reviewed before freezing)
- TimeDistill's repo and data pipeline, ETTh1, input 720, H ∈ {96,192,336,720}, seeds {2021, 42, 123}.
- Arm P-official: PatchTST with its official ETTh1 hyperparameters (verified in yuqinie98/PatchTST
  `PatchTST_supervised/scripts/PatchTST/etth1.sh`: e_layers 3, n_heads 4, d_model 16, d_ff 128, dropout 0.3,
  fc_dropout 0.3, head_dropout 0, batch 128, lr 1e-4, 100 epochs; the official script uses seq_len 336),
  early stopping on validation with the official repo's patience default (to confirm before freezing).
- Arm P-released: TimeDistill's own PatchTST script unchanged (anchor: should land near 0.498). Cost note: 752 s per
  epoch on this CPU, so 1 seed only here, or all seeds on a rented GPU.
- TimeDistill itself is not re-run in round 1 (needs a trained teacher per horizon); its reported 0.429 ± 0.004
  (5 seeds, Table 8) is the comparison value. Re-running it is round 2 if round 1 is close.
## Pre-registered verdicts (on the 4-horizon average MSE)
- ANCHOR OK if P-released mean is within 0.015 of 0.498. If not, stop: harness differs from the paper's.
- BASELINE UNDER-CONFIGURED if P-official mean ≤ 0.429 + 0.004 (TimeDistill's reported mean + its own std).
- HEADLINE STANDS if P-official mean > 0.429 + 2 × 0.004.
- Otherwise INCONCLUSIVE (re-run TimeDistill in round 2).
