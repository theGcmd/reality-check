# GTR-SOLAR-1 — pre-registration (DRAFT r1, not frozen; nothing scored)

Paper: "Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval" (GTR), Cao, Dai, Han, Xiong,
ICLR 2026, arXiv 2602.10847 (read as HTML, https://arxiv.org/html/2602.10847, 2026-11-25; Table 7 per-horizon
baseline cells did not extract cleanly and are NOT used here). Code: github.com/macovaseas/gtr at
d94161906151bdc0c94f4d21c2b0ca356aeb3135 (`harness/repos.lock`).

## Claim under test
- Sec. 4.2.1: "it exceeds the CycleNet by 8.2% in MSE and 6.5% in MAE" (Solar-Energy). Table 1 (T = 96, average
  over H ∈ {96,192,336,720}): GTR 0.194 / 0.245, CycleNet 0.210 / 0.261, TQNet 0.198 / 0.256 (second best by MSE).
  8.2% = (0.210 − 0.194) / 0.194 (our arithmetic; the paper does not state the formula).
- README: "outperforms the second-best model by 8.2% in MSE". By Table 1 the second best by MSE is TQNet (0.198),
  not CycleNet. This is a wording observation about the README, not tested by any run.

## What the paper and released code say about the protocol (verified 2026-11-25)
- Sec. 4.1: "learning rate selected from {1e-3, 3e-3, 5e-4}"; selection procedure not described.
- App. A.3: "for all baseline methods, we adopted the hyperparameter settings from their respective original
  publications." Not stated whether baseline numbers were rerun or copied.
- Table 10 (read via summary, not verbatim): GTR std over 5 runs, seeds {2025,…,2029}; Solar avg 0.194 ± 0.004.
  Only GTR has std. Released `scripts/GTR/solar.sh` uses seed 2024 only.
- Table 6 / `scripts/GTR/solar.sh`: seq_len 96, cycle 144, lr 3e-3, batch 64, 30 epochs, patience 5, use_revin 0,
  d_model 512 (run.py default), dropout 0 (run.py:74 default), lradj type3 (run.py:90 default: lr constant for
  epochs < 3, then × 0.8 per epoch, utils/tools.py:18-19).
- Data: `Dataset_Solar` (data_loader.py:420-460) 70/10/20 split by time, train-set StandardScaler; test loader
  drop_last = False (data_factory.py:20-21), so every test window is scored. Early stopping on val loss only
  (exp_main.py:225-231); test loss is printed each epoch but not used for selection.
- The harness contains its own CycleNet (models/CycleNet.py), called with the cycle index like GTR
  (exp_main.py:176-177). The official CycleNet repo is NOT pinned in `harness/repos.lock`.

## Question
Run in GTR's own harness with every test window kept, does GTR's Solar 4-horizon average MSE gain over CycleNet
(claimed 8.2%) survive when CycleNet gets the same learning-rate search GTR got ({1e-3, 3e-3, 5e-4}, selected on
validation) and both models get the same seeds?

## Arms (all: GTR repo, Solar, features M, enc_in 137, seq_len 96, cycle 144, batch 64, 30 epochs, patience 5,
## lradj type3, use_revin 0, CPU)
- **G-grid**: GTR, released Solar flags, lr ∈ {1e-3, 3e-3, 5e-4}.
- **C-grid**: CycleNet in GTR's harness, model_type mlp, d_model 512, use_revin 0, cycle 144, same lr grid.
  (CycleNet's own published Solar flags — to be confirmed in review; the CycleNet paper's Solar T=96 MLP row is
  believed to be 0.210/0.261, matching GTR Table 1. If CycleNet's published lr is outside the grid, it is added
  as a 4th lr for C only, as C-pub.)
- Anchor **G-rel** is the G-grid cell lr = 3e-3 (released config).
- Seeds: selection phase {2025, 2026, 2027} for every (model, lr, H); confirmation phase adds {2028, 2029} at the
  selected lr only. Scored numbers use all 5 seeds at the selected lr.
- Selection: per (model, H), the lr with the lowest mean best-checkpoint validation MSE over the 3 selection seeds.
  Ties (< 1e-5) go to the released lr for G and the published lr for C. Test is never looked at for selection.

Runs: 3 lr × 3 seeds × 4 H × 2 models = 72, plus 2 seeds × 4 H × 2 models = 16 → 88 runs (cost to be measured in
step c; if > 25 CPU h, DECISION NEEDED before running).

## Metrics (both code paths)
- Path 1: MSE/MAE as `exp_main.test()` prints them (normalised scale, as in the paper).
- Path 2: recompute from `per_window_mse.npy` / `per_window_mae.npy` saved by the wrapper (one float32 per test
  window; mean over windows, channels and steps). Paths must agree to 1e-6 relative or the run is invalid.
- Per (model, H): mean over 5 seeds. Headline quantity: A_m = mean over H of the per-H means.
- Gain g = (A_C − A_G) / A_G, same formula as the paper's 8.2%. 95% CI by bootstrap over seeds (10,000
  resamples, seeds resampled independently per model, horizons kept together within a seed).

## Pre-registered verdicts (thresholds fixed now; never moved after results)
- **ANCHOR**: G-rel 5-seed A within 0.194 ± 0.012 (3 × reported std). If outside, the run reports
  ANCHOR FAILED, no gain verdict is issued, and environment drift is investigated first.
- Gain verdict, on the CI [lo, hi] of g:
  - **SURVIVES**: lo ≥ 4.1% (at least half the claimed 8.2%, with confidence).
  - **SHRINKS**: lo > 0 and hi < 4.1%.
  - **VANISHES**: hi < 1.0% (no gain distinguishable from zero, or reversed).
  - **INCONCLUSIVE**: anything else.
- Secondary (reported, no verdict): g with C at its published lr only (does the lr search change C?);
  per-horizon g; MAE gain; whether the selected lr for G equals 3e-3.

## Not tested (state in any candidate)
TQNet (second best by MSE; not in this harness), all other baselines, other datasets, input lengths 336/720,
the official CycleNet repo, GPU nondeterminism, the paper's own (unpublished) selection procedure.

## Patches (wrapper `audits/gtr/run_gtr.py`, to be built in step c)
- Save per-window errors from `exp_main.test()` (no change to training or scoring).
- CPU device redirect only if the repo hard-codes CUDA (to check); stated in provenance.
- No other changes to GTR's code.
