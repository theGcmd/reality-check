# TFPS audit — pre-registration r2 (DRAFT, not frozen; nothing below has been scored)

Revision history: r1 (commit 3e62bbb) reviewed 2026-11-24 → NOT_LICENSED (FOREGONE), `_design_review.json`.
r2 (this text) rewrites the design per OWNER DECISIONS 1-8 (kept verbatim at the end) and every r1 finding.

Paper: "Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift" (TFPS),
NeurIPS 2025, arXiv 2410.09836. Code audited: `tfps` at 83a11827 (`harness/repos.lock`). Baseline code:
`patchtst_official` at 204c21ef, subdir `PatchTST_supervised/`. Data: `ETDataset` at 1d16c8f4, `ETTh1.csv`.

**Paper text status.** Quotes below come from r1's summarised fetch of arXiv HTML v2. This session's fetch of the
arXiv HTML/PDF was not permitted (permission prompt timed out). **Before FREEZE every quote marked [Q] must be
re-checked against the PDF**; if that is impossible, FREEZE must say so and the fairness claims are weakened to
"per the summarised HTML".

## 1. What the paper claims (ETTh1, input L=96, Table 1, single runs) [Q]
TFPS MSE 0.423 / 0.484 at H = 192 / 336; best baseline 0.434 (DLinear) / 0.491 (TSLANet, iTransformer).
Claimed margins used below: **m_192 = 0.011, m_336 = 0.007**. Paper says D is "random Gaussian
distribution-initialized" (Alg. 1) [Q]; no mean/std is stated in the text r1 saw [Q]; EDESC / k-means+SVD
initialisation is not described [Q]. Paper: lr grid {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}, "best parameters are selected
based on the lowest validation loss", baselines rerun "with codes provided by their official implementation" [Q].

## 2. What the released code does (verified by reading the code at 83a11827)
- `layers/Cluster.py:30` allocates `D = Parameter(torch.Tensor(n·d, n·d))` (n = experts, d = c_out·d_model/n; here
  112×112) without initialising it. `layers/InitializeD.py` / `Initialization_D` is imported (`exp/exp_main.py:24`)
  and never called. On this machine 3 fresh processes gave 4586-4635 NaN entries of 12544 (Q1 below).
- Affinity (`Cluster.py:61-69`): s_i = (‖z·D_i‖² + η·d) / ((η+1)·d), then each row normalised to sum 1; η = 5
  (`run_longExp.py:84`). With D = 0, s is exactly uniform (1/n) and ∂loss/∂D = 0 through the KL term and both
  constraints (`layers/Constraint.py`), so D stays 0 (Adam, no weight decay).
- Router (`layers/SparseMoE.py:57-66`, `NoisyTopkRouter_Cluster`): noisy_logits = s + randn·softplus(s), top-1,
  **at train and at eval** (no `self.training` guard). With top_k = 1 the gate weight after softmax over one
  finite logit is exactly 1, so the forecast loss sends no gradient into s through the gate; s is trained only by
  the auxiliary KL + constraint losses. With s ≈ 1/n = 0.25, noise sd ≈ softplus(0.25) ≈ 0.83, larger than the
  whole range [0,1] of s, so routing may be close to random whatever D is — this is what arms 3-4 measure.
- The cluster input is reshaped without a permute (`layers/PatchTST_MoE_backbone_cluster_time.py:77` vs its later
  permute at :83-87; frequency file :79 vs :83-87), so row j of s is not patch j of one series. Q2 is therefore
  scoped as "does D's initialisation matter in the released code", not "does patch-pattern routing work".
- Validation loss used for early stopping is MSE + α·KL/constraint loss (`exp/exp_main.py:130`, `:257-262`).
- Test loader `drop_last=True`, batch 128 (`data_provider/data_factory.py:17-20`, inherited unchanged from official
  PatchTST: same sha256). At audited horizons: H=192 drops 1 of 2689 windows; H=336 drops 113 of 2545.
  Val loader is shuffle + drop_last (released, kept for all arms; H=336 drops 113 of 2545 val windows).
- `test()` stacks per-batch arrays with `np.array` (`exp/exp_main.py:361-365`): ragged with drop_last off → crash.
- Checkpoint dir is `./checkpoints/<setting>` relative to cwd; `setting` (`run_longExp.py:134`) contains neither
  seed nor D-init → collisions across seeds/arms.
- Released per-horizon config (`scripts/etth1.sh`): common: seq_len 96, e_layers 3, n_heads 4, d_model 16, d_ff 128,
  dropout 0.3, fc_dropout 0.3, head_dropout 0, patch_len 16, stride 8, β 0.1, train_epochs 100, batch 128,
  itr 1, seed 2021; defaults from `run_longExp.py`: patience 20, lradj type3, η 5, α = γ = 1, revin 1.
  **H=192 (:77-89): T/F experts 4/4, top-1/top-1, lr 5e-4. H=336 (:133-145): T/F experts 4/4, top-1/top-1, lr 5e-3.**

## 3. Byte-identity of baseline code (sha256 prefix, tfps vs patchtst_official/PatchTST_supervised)
Identical: `models/PatchTST.py` 49d8bb86, `models/DLinear.py` 0893b53c, `layers/PatchTST_layers.py` 21c06c70,
`layers/RevIN.py` e64c0ccd, `data_provider/data_factory.py` 98f83b72, `data_provider/data_loader.py` 1d49e220,
`utils/tools.py` 80c2f839, `utils/metrics.py` 9d61cbf6, `utils/timefeatures.py` 2ad031a4.
Differ: `layers/PatchTST_backbone.py` (7f0b0830 vs df671731: diff is two commented-out `print` lines and a blank
line only — functionally identical), `exp/exp_main.py`, `run_longExp.py` (TFPS-specific; baselines use the official
ones). So baselines see the same data pipeline, scaler (fit on train only), splits and metric code as TFPS.

## 4. Questions (neutral wording)
- **Q1 (descriptive, already measured, not a verdict):** on this machine (CPU, torch 2.4.1), what fraction of fresh
  processes give a finite D with the released code? Extend to N = 10 processes in step (c); report as-is.
- **Q2:** In the released code, does a Gaussian initialisation of D (paper Alg. 1) give lower ETTh1 test MSE than
  D = 0 (affinity-blind routing)? Is the router exercising the affinity at all?
- **Q3:** With every test window kept, how does TFPS (D_gauss, released per-horizon config) compare with PatchTST
  and DLinear trained in the official PatchTST code on the same split, seeds and batch, with a per-seed validation
  lr search?

## 5. Arms (per horizon H ∈ {192, 336}; seeds S = {2021, 42, 123, 456, 7})
| # | arm | training | eval | role |
|---|-----|----------|------|------|
| 1 | `tfps_gauss` | released config, D ~ N(0, 1) i.i.d., drawn from torch's global RNG right after `torch.manual_seed(seed)` inside EDESC construction (both instances) | released noisy router, eval-noise RNG seeded `10000+seed` before the scoring pass | **primary TFPS arm** (fixed in advance) |
| 2 | `tfps_zero` | released config, D = 0 | as arm 1 | control: routing ignores affinity (uniformly random expert) |
| 3 | `tfps_gauss_nonoise` | **no training** — the arm-1 checkpoint | router noise removed at eval only (`SparseMoE.py:59-60` bypassed: noisy_logits = s) | does deterministic affinity routing change test MSE? |
| 4 | routing measurement | no training — arm-1 checkpoint | on the test set, for each of the two routers (time, frequency): A = fraction of rows with noisy top-1 = argmax s; H_r = entropy of the empirical expert-usage distribution (nats); also for arm 2 | manipulation check |
| 5 | `patchtst` | official `run_longExp.py`, model PatchTST, official ETTh1 arch (`scripts/PatchTST/etth1.sh`: e_layers 3, n_heads 4, d_model 16, d_ff 128, dropout 0.3, fc_dropout 0.3, head_dropout 0, patch 16, stride 8) with **seq_len 96** | official, deterministic | baseline |
| 6 | `dlinear` | official `run_longExp.py`, model DLinear, individual 0, **seq_len 96** | official | baseline |

**D_gauss scale (owner decision 1 asks for a justified choice).** The text r1 saw states no parameters [Q]. We use
standard N(0,1) (`torch.nn.init.normal_(D, 0, 1)`): "random Gaussian" with no stated parameters most literally reads
as the standard normal, and it is the default of `torch.randn`. Alternative not run: N(0, 1/(n·d)), which matches
the unit-norm column target of D_constraint1 at init. Scale matters here because η·d does not scale with D:
with N(0,1), ‖z·D_i‖² ≈ d·‖z‖² dominates η·d whenever ‖z‖² ≫ 5. The reviewers are asked to check this choice.
If the PDF states parameters, they replace N(0,1) before FREEZE (that is a prereg fix, not a new arm).

Shared settings, all arms: ETTh1, features M, enc_in 7, batch 128, train_epochs 100, **patience 20, lradj type3**,
train/val/test loaders as released (drop_last as released during training, including the per-epoch test pass both
repos run, which affects nothing but logs and RNG), num_workers 0, CPU.
- TFPS lr: the released per-horizon lr (5e-4 at H=192, 5e-3 at H=336), no search (owner decision 6). This was
  tuned by the authors under an unknown D; whether it favours TFPS is not established, so we do not call it an
  advantage.
- Baseline lr: per seed, train one run for each lr ∈ {1e-4, 5e-4, 1e-3, 5e-3}; select the lr whose run has the
  lowest best-epoch validation loss (the value `EarlyStopping` saved), ties to the smaller lr. Only the selected
  run's test score enters verdicts; all runs are kept and reported.
- Deviations from the baselines' own published configs, stated plainly: PatchTST published seq_len 336, lr 1e-4,
  patience 100 (official default); DLinear published seq_len 336, batch 32, lr 5e-3, patience 100. We use seq_len 96
  (the paper's Table 1 setting), batch 128 (owner decision 4), the lr grid (contains both published lrs), and
  patience 20 (TFPS's released value, the same for every arm).

## 6. Scoring (one code path, then an independent recompute in step f)
- After training, the wrapper loads the checkpoint `EarlyStopping` selected and runs **one eval-only pass over the
  full test set (drop_last off)**, saving `per_window_mse.npy` (float32, one value per test window: mean over
  pred_len × 7 of squared error in the normalised space, as `utils/metrics.py` does) and `per_window_mae.npy`.
- **Primary score:** mean of all N windows. **Released-loader score:** mean of the first ⌊N/128⌋·128 windows of the
  same pass (identical to drop_last=True with shuffle off). Both from one pass, so they differ only by the windows.
- TFPS arms 1-2 get a second eval-noise draw (seed `20000+seed`), reported, not used in verdicts.
- The repo's own printed test metric (its `test()` with the released loader) is also recorded in `metrics.json` for
  comparison; for TFPS it uses a different noise draw, so it is not expected to match exactly.
- Sensitivity (reported, not gating): TFPS arms re-selected on pure val MSE — the wrapper also saves the checkpoint
  at the epoch of lowest val MSE (computed in the same val pass) and scores it the same way.
- A run is **failed** if any training loss or the primary score is non-finite. Failed runs are reported, not
  replaced. An arm with ≥ 2 failed seeds at a horizon is BLOCKED there; with 1 failure, verdicts use n = 4.

## 7. Pre-registered verdicts (per horizon; first matching rule wins; every label is suffixed "— ETTh1, L=96, H=…, released code, CPU")
Welch 90% CI on a difference of means across seeds (two one-sided 5% tests); for the paired arm-3 comparison, a
paired t 90% CI over seeds (same checkpoint). m_H as in §1.

**Q2 — D initialisation (Δ = mean tfps_gauss − mean tfps_zero, primary score).**
Premise check first: in every tfps_zero run, max|D| = 0 at the end and max−min of s < 1e-6 on the test pass, for both
EDESC instances; else `Q2 PREMISE FAILED` and no Q2 verdict.
1. `INIT MATTERS (gauss better)`: CI upper < 0.
2. `INIT REVERSED (zero better)`: CI lower > 0.
3. `INIT EQUIVALENT`: CI inside (−m_H, +m_H).
4. `INCONCLUSIVE`.
(When 1 holds and the CI is also inside (−m_H, 0), report "INIT MATTERS, smaller than m_H".)
Qualifier from arm 4: if, for both routers, mean A over seeds < 1/n + 0.05 = 0.30 (n = 4 experts), every Q2 label
gets the suffix `— ROUTING NOT FOLLOWING AFFINITY (A = …)`. A and H_r are reported for arms 1 and 2 regardless.

**Q2b — eval noise (Δ_nn = tfps_gauss_nonoise − tfps_gauss, paired):** same four labels with NOISE-OFF BETTER /
NOISE-OFF WORSE / NOISE-OFF EQUIVALENT / INCONCLUSIVE. Descriptive; does not feed Q3.

**Q3 — TFPS vs baselines.** For each baseline B ∈ {patchtst, dlinear}: d_B = mean(B) − mean(tfps_gauss), 90% CI
[L_B, U_B], primary score. Intersection-union: "TFPS better" requires it against both B.
1. `REVERSED`: U_B < 0 for some B.
2. `VANISHES`: for some B, L_B ≤ 0 and U_B < m_H (no detectable gain, and a gain as large as claimed is excluded).
3. `SHRINKS`: L_B > 0 for both B, and U_B < m_H for at least one B (gain real but smaller than claimed).
4. `SURVIVES`: L_B > 0 and U_B ≥ m_H for both B.
5. `INCONCLUSIVE`.
Rules 1-5 are exhaustive. The same table is reported on the released-loader score, not gating.
Item level: if both horizons get the same Q3 label, that label; otherwise `MIXED (192: …, 336: …)`.

**Reachability / power.** With n = 5 per arm, Welch df ≈ 8, t₀.₉₅ ≈ 1.86; CI half-width ≈ 1.86·√2·σ/√5 ≈ 1.18σ for
equal seed sd σ. SURVIVES / SHRINKS / VANISHES need half-width < m_H, i.e. σ < 0.0093 at H=192 and σ < 0.0059 at
H=336; INIT EQUIVALENT needs the same. Seed sd for these models on ETTh1 is unmeasured here. **Variance-triggered
extension, once:** after all 5 seeds of every arm at a horizon are scored, if any Q2/Q3 rule is unreachable because
the half-width exceeds m_H (judged on half-width only, not on which verdict it would give), add seeds
{1, 2, 3, 4, 5} to every arm at that horizon, then apply the rules once at n = 10. No further extension.

**Anchor (not gating):** report every TFPS arm's mean (released-loader score) minus the paper's value; flag
`ANCHOR DRIFT` if the paper value lies outside that arm's 90% CI by more than 0.02. Baseline anchors: the paper's
Table 1 PatchTST / DLinear numbers at L=96 [Q — to be copied from the PDF before FREEZE].

## 8. Engineering required in step (c), with self-tests, before FREEZE
Wrappers under `audits/tfps/` only; nothing under `work/repos/` is edited. Each run executes with cwd = its own run
directory (checkpoints in `$RC_WORK/ckpt/tfps/<arm>/<H>/seed<k>[/lr<x>]`, never committed; sha256 of the selected
checkpoint in provenance), absolute `--root_path`, unique `--model_id <arm>_<H>_s<seed>[_lr<x>]`.
Patches (file:line overridden), each with a self-test:
- `layers.Cluster.EDESC.__init__` (`Cluster.py:30`): D init gauss / zero. Test: D finite, mean ≈ 0, sd ≈ 1 (gauss);
  D ≡ 0 and still ≡ 0 after 3 optimiser steps (zero).
- `layers.SparseMoE.NoisyTopkRouter_Cluster.forward` (`SparseMoE.py:57-66`): eval-only noise switch + recording of
  (indices, argmax s). Test: with switch off, indices == argmax s on every row; with it on, outputs equal the
  released forward bit-for-bit under the same RNG state.
- Eval-only scoring pass (replaces the crashing `np.array` stack at `exp_main.py:361-365` by not calling it for the
  keep-all score; concatenation instead). Test: N rows equal the dataset length; first ⌊N/128⌋·128 rows reproduce
  the released-loader MSE from the same pass to 1e-6.
- Pure-val-MSE checkpoint for the sensitivity analysis (hook on `Exp_Main.vali`, `exp_main.py:74-135`). Test: the
  recorded val MSE equals an independent recompute on that checkpoint.
- Official PatchTST harness wrapper (same eval pass, no model patches). Test: its released-loader score equals
  the repo's printed test MSE (deterministic models) to 1e-5.
Wall time: step (c) smoke test measures epochs-to-stop and s/epoch per arm. If a single TFPS run cannot finish
inside one ~45-min chunk, step (c) must add per-epoch resumable state (model, optimiser, scheduler, EarlyStopping,
all RNG states) with a self-test that a run interrupted after epoch 2 and resumed is bitwise identical to an
uninterrupted one at epoch 4; otherwise runs are never resumed mid-run (killed runs restart from scratch).

Budget estimate (to be replaced by step (c) measurements): TFPS 2 arms × 2 H × 5 seeds = 20 runs at ~25-30 s/epoch,
likely 30-60 epochs → ~5-10 CPU-h; baselines 2 × 2 × 5 × 4 lr = 80 runs, small models at L=96 → ~2-4 CPU-h. Total
≈ 7-14 CPU-h at n = 5; up to ~2× if the extension fires.

## 9. Not claimed / not tested
Other datasets, H = 96 / 720, input lengths ≠ 96; the paper's best baselines at H=336 (TSLANet, iTransformer) and
at H=192 other than DLinear; D_svd / `Initialization_D`; the PI→Linear ablation; an lr search for TFPS; GPU
numerics; anything about what the authors ran or whether their D was finite. Statements are about the released code
as run here.

## 10. Paper-vs-code mismatches noted by r1 (from the summarised fetch; re-check against the PDF) [Q]
K = 16 experts at H=96 lies outside the paper's stated grid {1, 2, 4, 8}; η = 5 in code vs η = d in the method text;
placement of α. None changes the arms above; listed for completeness.

## OWNER DECISIONS (Gustav, 2026-11-24 20:00 London, on design review r1)

1. **Primary mechanism arm: D_gauss**, the paper's Alg.1 "random Gaussian distribution-initialized" D. Use the
   paper's parameters if it states any; otherwise state the choice and its justification in the rewrite for the
   review to check. D_zero stays as a control: under the released noisy router it is uniformly random routing, so
   it is the "routing ignores the affinity" reference, not a "constant expert" arm. D_orth is dropped.
2. **Add an eval-time noise-off arm** for D_gauss: training unchanged; at evaluation only, the router's added noise
   is disabled so top-1 follows the affinity (`layers/SparseMoE.py:55-63`, by runtime patch only).
3. **Add a direct routing measurement** on every trained D_gauss model, on the test set: fraction of patches where
   the noisy router's top-1 equals argmax of the affinity, plus routing entropy, per seed. No extra training.
4. **Baselines in the official PatchTST repo** (`patchtst_official` in `harness/repos.lock`): PatchTST and DLinear,
   same split, batch 128, per-seed lr chosen on validation from {1e-4, 5e-4, 1e-3, 5e-3}. Confirm in the rewrite, by
   sha256, which model/data/util files are byte-identical to TFPS's.
5. **Horizons H=192 and H=336 only**, each with the released per-horizon TFPS config (`scripts/etth1.sh`:
   H=192 4/4 experts lr 5e-4; H=336 4/4 experts lr 5e-3). Seeds as before.
6. **Not now:** D_svd (k-means/SVD init), the PI→Linear ablation and H=96, an lr search for TFPS (TFPS keeps its
   authors' tuned per-horizon config, which favours TFPS; a search would cost ~30 CPU-h), H=720. Any of these may be
   proposed after round-1 results.
7. Fix every other round-1 finding in the rewrite: keep-all test scoring crash, checkpoint-path collisions across
   arms/seeds, verdict rules (Welch CI / TOST text as supplied by the statistics lens, an explicit INCONCLUSIVE, no
   test-selected "best arm").
8. **Timing (owner):** TFPS work starts at 03:00 London on 2026-11-25. Until then the queue row stays BLOCKED and the
   loop works TIMEDISTILL-1; the reviewing chat flips the row to READY at 03:00. Then step (b): rewrite + review r2.
   Expected budget for this scope: about 10 CPU-h.
