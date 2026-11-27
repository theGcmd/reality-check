# TFPS audit — pre-registration r3 (DRAFT, not frozen; nothing below has been scored)

Revision history: r1 (commit 3e62bbb) → NOT_LICENSED (FOREGONE), `_design_review.json`. r2 (commit 32c41b2) →
NOT_LICENSED (1 blocking: D scale), `_design_review_r2.json`. r3 (this text) applies OWNER DECISIONS 9-11 and every
r2 fix. Round 3 is the last allowed (protocol 5b); if it is not licensed the item goes BLOCKED.

Paper: "Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift" (TFPS),
arXiv 2410.09836 v2. Code audited: `tfps` at 83a11827 (`harness/repos.lock`). Baseline code: `patchtst_official`
at 204c21ef, subdir `PatchTST_supervised/`. Data: `ETDataset` at 1d16c8f4, `ETTh1.csv`.

**Paper text status.** Quotes and numbers marked [E] come from `audits/tfps/PAPER_EXTRACT.md` (WebFetch reading
of the v2 PDF by the reviewing chat; quotes returned as exact; Table 1 cells "reconstructed from garbled layout").
**FREEZE precondition:** the eight Table 1 cells used below (ETTh1 MSE at H=192/336 for TFPS, PatchTST, DLinear,
and the best other baseline) are checked against the PDF by Gustav or the reviewing chat and the check is recorded
in `PAPER_EXTRACT.md`. If any cell differs, the margins in §1 are replaced by the checked values before FREEZE
(a transcription fix, not a design change). If the check cannot be done, every Q3 label carries "margins per
unverified extract".

## 1. What the paper claims (ETTh1, Table 1, single runs, no error bars) [E]
- Table 1 caption: "The input lengths are L = 104 for ILI and L = 96 for others." The appendix also says input
  lengths were searched over {96, 192, 336, 512} with the best reported per horizon [E]; the two statements are not
  reconciled in the extract. The released `scripts/etth1.sh` uses seq_len 96 at every horizon; we use L = 96 for
  every arm and say so in every label.
- ETTh1 MSE [E]: TFPS 0.423 / 0.484 (H = 192 / 336); PatchTST 0.460 / 0.497; DLinear 0.434 / 0.499; best of all
  Table 1 baselines 0.434 (DLinear) at H=192 and 0.489 (FITS) at H=336.
- **Per-baseline claimed margins (Q3 gates):** m_{H,B} = paper(B) − paper(TFPS):
  m_{192,PatchTST} = 0.037, m_{192,DLinear} = 0.011, m_{336,PatchTST} = 0.013, m_{336,DLinear} = 0.015.
  All are > 0, so the paper claims a gain over both baselines at both horizons. (The r2 fairness lens recalled
  DLinear ≈ 0.481 at H=336 from other papers' tables; the extract gives 0.499; the FREEZE check settles it. If the
  checked m_{H,B} ≤ 0 for some (H, B), Q3 at that (H, B) becomes a non-gating descriptive comparison.)
- **Q2 equivalence margin:** m_H = paper's margin over its best Table 1 baseline: m_192 = 0.011, m_336 = 0.005.
  Used only as "an effect as large as the paper's smallest claimed win"; it is not a claim about D.
- D: Algorithm 1 input is a "random Gaussian distribution-initialized subspace" [E]; no mean or standard deviation
  is given; no k-means / SVD initialisation is described [E].
- Settings [E]: Adam; lr grid {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}; experts ∈ {1, 2, 4, 8}; "The best parameters are
  selected based on the lowest validation loss."; baselines: "We rerun all the experiments with codes provided by
  their official implementation."; α_t = α_f = 1e-3. Batch, epochs, patience, hardware not stated.

## 2. What the released code does (verified by reading the code at 83a11827; re-verified by r2 lenses)
- `layers/Cluster.py:30` allocates `D = Parameter(torch.Tensor(n·d, n·d))` (n = 4 experts, d = 28; 112×112) without
  initialising it. `layers/InitializeD.py` / `Initialization_D` (SVD left singular vectors: orthonormal columns,
  entries ≈ 1/√112) is imported (`exp/exp_main.py:24`) and never called. `Cluster.py` follows EDESC closely (its
  commented-out `self.D = Parameter(torch.Tensor(n_z, n_clusters))` at :27; "Eq 9/12/13/15/16" comments). On this
  machine 3 fresh processes gave 4586-4635 NaN entries of 12544 (Q1; N = 10 log produced in step (c)).
- Affinity (`Cluster.py:61-69`): s_i = (‖z·D_i‖² + η·d) / ((η+1)·d), each row normalised to sum 1; η = 5
  (`run_longExp.py:84`), η·d = 140. With D = 0, s is exactly 1/n and ∂loss/∂D = 0 through KL and both constraints
  (measured by the r2 foregone lens: grad exactly 0), so D stays 0 (Adam, no weight decay, `exp_main.py:55`).
- Constraints (`layers/Constraint.py:14-17, 30-39`): loss_d1 = 1e-3·‖diag(DᵀD) − I‖_F (unit-norm columns),
  loss_d2 = 1e-3·‖off-block(DᵀD)‖_F. Validation loss used for early stopping = MSE + α·aux_time + γ·aux_freq
  (`exp_main.py:130`, `:257-262`), α = γ = 1. **Scale matters for selection** (r2, measured): per EDESC instance
  d1 + d2 ≈ 2.23 under D ~ N(0,1) (≈ 4.4 for both, vs val MSE ≈ 0.45), ≈ 0.011 under N(0, 1/112), 0.0106 (constant)
  under D = 0. Under N(0,1) Adam can move each entry by at most ≈ 0.4 over a run at H=192, so D stays far from the
  constraint set and the aux term likely dominates checkpoint selection for that arm; this is why arm 2 exists.
- Router (`layers/SparseMoE.py:57-66`, `NoisyTopkRouter_Cluster`): noisy_logits = s + randn·softplus(s), top-1, at
  train and at eval (no `self.training` guard). With top_k = 1 the gate weight is exactly 1, so the forecast loss
  sends no gradient into s through the gate; s is trained only by the auxiliary losses. The noise sd (≈ 0.83 at
  s = 0.25) is large relative to the range of s; arm 5 measures how far routing follows the affinity. r2 forward
  probe at init (D ~ N(0,1)): mean row range of s ≈ 0.11, A ≈ 0.283 (chance 0.25). Router-only simulation: A
  ceiling ≈ 0.61 for a near-one-hot s at n = 4.
- The cluster input is reshaped without a permute (time file `layers/PatchTST_MoE_backbone_cluster_time.py:77` vs
  its permutes at :81-85; frequency file :79 vs :83-87), so row j of s is not patch j of one series, but rows stay
  inside one sample (no cross-window mixing). Q2 is scoped as "does D's initialisation matter in the released code",
  not "does patch-pattern routing work".
- Test loader `drop_last=True`, batch 128 (`data_provider/data_factory.py:17-20`, same sha256 as official
  PatchTST). N = 2881 − H test windows: H=192 drops 1 of 2689; H=336 drops 113 of 2545. Val loader shuffle +
  drop_last (released, kept for all arms).
- `test()` stacks per-batch arrays with `np.array` (`exp/exp_main.py:361-365`): ragged with drop_last off → crash.
  `test(test=1)` hard-codes `./checkpoints/`+setting (`exp_main.py:282`; official `:224`).
- `setting` (`run_longExp.py:134`) contains neither seed nor D-init → checkpoint collisions across seeds/arms.
- Both `run_longExp.py` default `itr = 2` (tfps :91, official :77) and loop (tfps :132, official :118).
- Both repos build OneCycleLR (tfps `exp_main.py:160-164`, official `:120-124`), which sets lr to max_lr/25 at
  construction and is never stepped under type3: epoch 1 trains at lr/25, epochs 2-3 at lr, then lr·0.9^(k−3)
  (`utils/tools.py:18-19`, identical in both repos). Symmetric across arms; stated, not changed.
- Released per-horizon config (`scripts/etth1.sh`): common: seq_len 96, e_layers 3, n_heads 4, d_model 16, d_ff 128,
  dropout 0.3, fc_dropout 0.3, head_dropout 0, patch_len 16, stride 8, β 0.1, train_epochs 100, batch 128, itr 1,
  seed 2021; defaults from `run_longExp.py`: patience 20 (:94), lradj type3 (:98), η 5 (:84), α = γ = 1 (:86-87),
  revin 1 (:46), affine 0 (:47), subtract_last 0 (:48), padding_patch 'end' (:45), decomposition 0 (:49),
  **individual 1 (:51)** (official default 0), num_workers 10 (:90); expert dropout 0.1 hard-coded
  (`SparseMoE.py:16`). **H=192 (:77-89): T/F experts 4/4, top-1/top-1, lr 5e-4. H=336 (:133-145): 4/4, top-1, lr 5e-3.**

## 3. Byte-identity of baseline code (sha256 prefix, tfps vs patchtst_official/PatchTST_supervised; r2 recomputed)
Identical: `models/PatchTST.py` 49d8bb86, `models/DLinear.py` 0893b53c, `layers/PatchTST_layers.py` 21c06c70,
`layers/RevIN.py` e64c0ccd, `data_provider/data_factory.py` 98f83b72, `data_provider/data_loader.py` 1d49e220,
`utils/tools.py` 80c2f839, `utils/metrics.py` 9d61cbf6, `utils/timefeatures.py` 2ad031a4, `exp/exp_basic.py`
6c69bada; also `models/Linear.py`, `models/NLinear.py`. Differ: `layers/PatchTST_backbone.py` (7f0b0830 vs
df671731: two commented-out `print` lines and a blank line only), `exp/exp_main.py` (9ba1303f vs de76924d) and
`run_longExp.py` (9c1728eb vs 44345cb6) (TFPS-specific; baselines use the official ones). Baselines see the same
data pipeline, scaler (fit on train only), splits and metric code as TFPS.

## 4. Questions (neutral wording)
- **Q1 (descriptive, not a verdict):** on this machine (CPU, torch 2.4.1), what fraction of fresh processes give a
  finite D with the released code? N = 10 processes in step (c); reported as-is. Says nothing about the authors' D.
- **Q2:** In the released code with an audit-supplied D init, does a Gaussian D (paper Alg. 1; two scales) give a
  different ETTh1 test MSE than D = 0 (affinity-blind routing)? How far does the router follow the affinity?
- **Q2b (descriptive):** does removing the router noise at evaluation only change test MSE of a model trained with
  the released noisy router?
- **Q3:** With every test window kept, how do the TFPS arms (released per-horizon config) compare with PatchTST and
  DLinear trained in the official PatchTST code on the same split, seeds and batch?

## 5. Arms (per horizon H ∈ {192, 336}; seeds S = {2021, 42, 123, 456, 7})
| # | arm | training | eval | role |
|---|-----|----------|------|------|
| 1 | `tfps_gauss` | released config, D = G (see below), sd 1 | released noisy router | TFPS arm, literal standard normal (owner decision 1) |
| 2 | `tfps_gauss_unit` | released config, D = G/√(n·d) = G/√112, i.e. N(0, 1/112): unit-norm columns at init, matching D_constraint1 and the unused `Initialization_D` scale | as arm 1 | TFPS arm, constraint-consistent scale (owner decision 9) |
| 3 | `tfps_zero` | released config, D = 0 | as arm 1 | control: routing ignores affinity (uniformly random expert) |
| 4 | `*_nonoise` | **no training** — the arm-1 and arm-2 checkpoints | router noise removed at eval only (`SparseMoE.py:59-60` bypassed: noisy_logits = s) | Q2b |
| 5 | routing measurement | no training — arms 1-3 checkpoints | test set, each router (time, freq): A = fraction of rows with noisy top-1 = argmax s; H_r = entropy (nats) of expert usage; mean row max and row range of s | manipulation check |
| 6 | `patchtst` | official `run_longExp.py`, model PatchTST, official ETTh1 arch (`scripts/PatchTST/etth1.sh:30-42`: e_layers 3, n_heads 4, d_model 16, d_ff 128, dropout 0.3, fc_dropout 0.3, head_dropout 0, patch 16, stride 8), individual 0, **seq_len 96** | official, deterministic | baseline |
| 7 | `dlinear` | official `run_longExp.py`, model DLinear, individual 0, **seq_len 96** | official | baseline |

**D draws.** G is a 112×112 standard-normal matrix drawn from a dedicated `torch.Generator` seeded 30000+seed for
the time EDESC and 40000+seed for the frequency EDESC; the global RNG is not touched. Arms 1-3 at a given seed
therefore share every non-D initial weight, the data order and the training noise stream up to the first point
where D changes a computation; arms 1 and 2 use the same G (arm 2 is arm 1's D scaled by 1/√112). Neither scale is
claimed to be "the paper's": the paper states none. N(0,1) is the default of `torch.randn`; N(0, 1/(n·d)) is the
scale the repo's constraint and unused initialiser point to. Expected consequence, stated in advance: under arm 1
the aux term is ≈ 10× val MSE at init and likely drives checkpoint selection; under arm 2 it is ≈ 0.02× val MSE but
s is nearly uniform at init (r2: row range ≈ 0.02, A ≈ 0.26).

**Expected by construction (sanity checks, not evidence):** arm 3: max|D| = 0 throughout, s ≡ 1/n, argmax s =
expert 0 on every row (torch tie-break), so A ≈ 0.25 and H_r ≈ ln 4 ≈ 1.386; flag if A ∉ 0.25 ± 0.02.

**Shared settings, all arms:** ETTh1, features M, enc_in 7, batch 128, train_epochs 100, patience 20, lradj type3,
**`--itr 1`**, train/val/test loaders as released during training (including the per-epoch test pass both repos
print; it runs under eval/no_grad and changes nothing but logs and the RNG stream, by shape only), num_workers 0,
CPU. Deviations for every arm, stated: num_workers 0 (released 10), CPU (authors: GPU, unstated). `individual`
is kept as each repo released it (TFPS 1, baselines 0).
- **TFPS lr:** the released per-horizon lr (5e-4 at H=192, 5e-3 at H=336), no search (owner decision 6). Its
  provenance is unknown (the paper says val-selected; the code cannot show it); Q3 cannot separate method gain from
  that choice.
- **Baseline lr (verdict tables):** per seed, one run for each lr ∈ {1e-4, 5e-4, 1e-3, 5e-3}; select the lr with
  the lowest `val_loss_min` as written (full float) by the EarlyStopping subclass (§8), ties to the smaller lr; a
  run with any non-finite val loss cannot be selected. Only the selected run enters verdict tables; all are kept.
- **Baseline lr (published-lr table, owner decision 11, no extra runs):** the grid runs at PatchTST lr 1e-4 and
  DLinear lr 5e-3 (the lrs in `scripts/PatchTST/etth1.sh` and `scripts/Linear/etth1.sh`).
- Deviations from the baselines' configs as released in the PatchTST repo: PatchTST seq_len 336, lr 1e-4,
  patience 100; DLinear (as configured in `scripts/Linear/etth1.sh`) seq_len 336, batch 32, lr 5e-3, patience 100.
  We use seq_len 96 (Table 1 setting), batch 128 (owner decision 4), the lr grid (contains both released lrs) and
  patience 20 (TFPS's released value, same for every arm).

## 6. Scoring (one code path, then an independent recompute in step f)
- After training, the wrapper loads the checkpoint EarlyStopping selected **by absolute path** (sha256 in
  provenance; never via `test(test=1)`) and runs eval-only passes over the **full test set (drop_last off)**,
  saving per-window MSE and MAE (float32 on disk, accumulated in float64; one value per window = mean over
  pred_len × 7 of the error in the normalised space, as `utils/metrics.py`).
- **TFPS eval noise:** K = 5 eval passes per TFPS checkpoint with the noise RNG seeded 10000·k + seed, k = 1..5.
  **Per-run TFPS score = mean over the 5 draws** (chosen now, before any score is seen). Each draw's score is saved;
  the mean |draw − mean| per run is reported next to seed sd, and flagged beside every TFPS label if it exceeds
  m_H/4. Baselines are deterministic: one pass.
- **Primary score:** mean over all N windows. **Released-loader score:** mean of the first ⌊N/128⌋·128 windows of the
  same passes (reported, not gating).
- The repo's own printed test metric is also recorded (for TFPS it uses another noise draw; not expected to match).
- **Pure-val-MSE selection (co-reported, attaches suffixes):** for every TFPS run the wrapper also saves
  `checkpoint_valmse.pth` at the epoch of lowest val MSE (forecast part only, from the same val pass, keyed on the
  val loader; §8) and scores it the same way.
- **Per-epoch log (every TFPS run):** val MSE, each aux component per router, ‖D‖_F, mean column norm of D, the
  saved-checkpoint epoch; per run: epochs-to-stop, aux share of val loss at the saved epoch, ‖D_final − D_init‖_F /
  ‖D_init‖_F (arms 1-2). Descriptive.
- **Failed run:** any non-finite training loss, val loss or primary score. Failed runs are reported, never replaced.
  A baseline seed fails only if all 4 lr runs fail.
- **BLOCKED:** an arm is BLOCKED at H if ≥ 2 of 5 seeds fail (≥ 3 of 10 after extension). Q2 contrast (a vs zero)
  is BLOCKED if either arm is; Q3 for TFPS arm a is BLOCKED if a or either baseline is. With one failure the arm
  uses n = 4 and every label resting on it gets "— 1 FAILED RUN IN <arm>", plus a sensitivity where the failed
  seed is scored as that arm's worst observed seed.

## 7. Pre-registered verdicts (per horizon; first matching rule wins)
**Label suffix (every label):** "— ETTh1, L=96, H=…, released TFPS code with audit-supplied D init (<arm>), vs
PatchTST and DLinear rerun at L=96 in the official PatchTST code (baselines lr-searched per seed, TFPS not), CPU".
Every label is printed with its point estimate and CI.

**Statistics.** Welch with Welch-Satterthwaite df from the observed per-seed sds; one-sided bounds as stated per
rule. Paired comparisons (Q2b) use a paired t over seeds. CIs cover seed variance only, conditional on the ETTh1
test split; no window-level test is run (windows overlap and are autocorrelated). For Q2, the paired difference
over seeds (arms share non-D init) is reported next to the Welch CI, not gating.

**Q2 — D initialisation.** For each a ∈ {tfps_gauss, tfps_gauss_unit}: Δ_a = mean(a) − mean(tfps_zero), 90% CI.
0. `BLOCKED` (§6) or `Q2 PREMISE FAILED` (in any tfps_zero run, max|D| ≠ 0 at the end, or max−min of s ≥ 1e-6 on
   the test pass, for either EDESC).
1. `GAUSS(<sd>) BETTER THAN ZERO`: CI upper < 0 (add ", smaller than m_H" when the CI is also inside (−m_H, 0)).
2. `ZERO BETTER THAN GAUSS(<sd>)`: CI lower > 0 (add ", smaller than m_H" when inside (0, m_H)).
3. `GAUSS(<sd>) EQUIVALENT TO ZERO`: CI inside (−m_H, +m_H).
4. `INCONCLUSIVE`.
Selection: the label is computed on the released selection and on pure-val-MSE selection; if they differ, the
released label gets "— SELECTION-SENSITIVE (pure-val-MSE: …)".
Routing suffix, per router and per arm a ∈ {1, 2}: if mean A over seeds < 0.30 for that router, add
"— ROUTER NOISE EXCEEDS AFFINITY SPREAD (<TIME|FREQ>: A = …, ceiling = …, mean row range of s = …)". The cut 0.30 is
expected to fire given the code (§2); the ceiling (A for a one-hot s under this router, simulated in step (c)) is
printed so A is read against what is attainable. A's distribution across seeds is reported. A and H_r for arm 3
are a reference (expected 0.25 / ln 4).
Item level for each a: same label at both horizons → that label; else `MIXED (192: …, 336: …)`.

**Q2b — eval noise (descriptive, does not feed Q3).** For a ∈ {1, 2}: Δ_nn = a_nonoise − a, paired 90% CI; rules 1-4
as Q2 with labels NOISE-OFF BETTER / NOISE-OFF WORSE / NOISE-OFF EQUIVALENT / INCONCLUSIVE, margin m_H. Concerns
evaluation only of a model trained with the noisy router. Item level as Q2.

**Q3 — TFPS vs baselines.** For each TFPS arm a ∈ {tfps_gauss, tfps_gauss_unit, tfps_zero} and each baseline
B ∈ {patchtst, dlinear}: d_{a,B} = mean(B) − mean(a) (positive = TFPS better); L_{a,B} = one-sided 95% lower bound,
U_{a,B} = one-sided 97.5% upper bound (Bonferroni over the two baselines for the "some B" conditions; L needs no
correction under intersection-union). Margin m_{H,B} from §1.
Per-arm label for a (first match):
0. `BLOCKED`.
1. `REVERSED` (a baseline beats TFPS): U_{a,B} < 0 for some B.
2. `VANISHES` (gain over some baseline not detected, and the claimed margin over it excluded): for some B,
   L_{a,B} ≤ 0 and U_{a,B} < m_{H,B}.
3. `SHRINKS` (gain detected over both, but the claimed margin excluded for at least one): L_{a,B} > 0 for both B,
   and U_{a,B} < m_{H,B} for at least one B.
4. `SURVIVES` (gain detected over both; claimed margins not excluded): L_{a,B} > 0 and U_{a,B} ≥ m_{H,B} for both B.
5. `INCONCLUSIVE`.
Rules 0-5 are exhaustive.
**Horizon label (owner decision 9):** if all three per-arm labels are equal → that label. Else, if every per-arm
label is one of REVERSED / VANISHES / SHRINKS → the least negative of them (order REVERSED < VANISHES < SHRINKS),
with the per-arm table. Otherwise → `DEPENDS ON D INIT (gauss: …, unit: …, zero: …)`. A negative horizon label
therefore needs every TFPS arm to be at least that negative.
**Qualifiers on a negative horizon label (REVERSED / VANISHES / SHRINKS):** (i) recompute it with the baselines at
their published lr (§5); if it does not hold there, add "— ONLY WITH PER-SEED LR SEARCH FOR BASELINES (published-lr
label: …)"; (ii) recompute it with pure-val-MSE selection for the TFPS arms; if it differs, add
"— SELECTION-SENSITIVE (pure-val-MSE: …)". Both tables are printed in full regardless.
Item level: same horizon label at both H → that label; else `MIXED (192: …, 336: …)`.
Each baseline's paper margin and the released-loader-score table are printed beside the CIs, not gating.

**Reachability (stated per label; equal seed sd σ unless noted).** n = 5: Welch df 8, t₀.₉₅ = 1.860,
t₀.₉₇₅ = 2.306, so hw_L = 1.176σ, hw_U = 1.458σ; with sd ratio 0.1 (e.g. DLinear nearly deterministic) df ≈ 4.1,
hw_L = 0.953σ_T, hw_U = 1.238σ_T. n = 10: df 18, hw_L = 0.775σ, hw_U = 0.940σ.
- REVERSED, VANISHES, SURVIVES, Q2 rules 1-2: no width floor (can fire from the point estimate at any σ).
- Q2 EQUIVALENT and NOISE-OFF EQUIVALENT: 90% hw < m_H → σ < 0.0094 (H=192) / 0.0043 (H=336) at n = 5;
  0.0142 / 0.0064 at n = 10.
- SHRINKS, and the ", smaller than m_H" sub-labels: hw_L + hw_U < m_{H,B} → at n = 5, σ < m/2.635:
  PatchTST 0.0140 / 0.0049, DLinear 0.0042 / 0.0057 (H = 192 / 336); at n = 10, σ < m/1.715.
- Plausible seed sds (r2 lenses, unmeasured here): PatchTST ≈ 0.002-0.005, DLinear ≈ 0.0005-0.002, TFPS unknown and
  possibly larger (lr 5e-3 at H=336, router noise partly averaged by K = 5). SHRINKS against DLinear at H=192 and Q2
  EQUIVALENT at H=336 may stay unreachable even at n = 10; this is stated, not fixed.
- Monte Carlo (r2 statistics lens) for the r2 rules: with the true gain exactly m, VANISHES fires 4-9%; with a true
  gain of m/2, VANISHES fires 35-52% and SURVIVES 16-32%: VANISHES means "claimed margin excluded", not "gain = 0".

**Variance-triggered extension, once.** After all 5 seeds of every arm at a horizon are scored, a committed script
(`audits/tfps/extension_trigger.py`, written and self-tested in step (c)) reads only per-seed primary scores and
computes, at that H: T = [any Q2 contrast with 90% hw ≥ m_H] OR [any (a, B) with hw_L + hw_U ≥ m_{H,B}]. It is
judged on widths only (n = 4 for an arm with one failure). Its output and the n = 5 per-run files are committed
**before** any verdict table or label is printed. If T, add seeds {1, 2, 3, 4, 5} to every arm at that H (baselines
rerun the full lr grid, lr re-selected per new seed by the same rule), then apply the rules once at n = 10 and
report the n = 5 table beside it. No further extension. The extension's type-I inflation (≈ 0.5-1 pp per one-sided
bound, r2 simulation) is accepted. Q2b is excluded from T.

**Anchor (not gating):** for each TFPS arm, a 90% prediction interval for a single run, mean ± t·s·√(1 + 1/n), on
the released-loader score; flag `ANCHOR DRIFT` if the paper value [E] lies outside it by more than 0.02. Same for
PatchTST and DLinear at their published lr against Table 1 [E].

## 8. Engineering required in step (c), with self-tests, before FREEZE
Wrappers under `audits/tfps/` only; nothing under `work/repos/` is edited. Each run executes with cwd = its own run
directory (checkpoints in `$RC_WORK/ckpt/tfps/<arm>/<H>/seed<k>[/lr<x>]`, never committed), absolute
`--root_path`, unique `--model_id <arm>_<H>_s<seed>[_lr<x>]`, `--itr 1`; the wrapper asserts exactly one setting
directory and one `checkpoint.pth` per run. Patches (file:line overridden), each with a self-test:
- `layers.Cluster.EDESC.__init__` (`Cluster.py:30`): D init gauss / gauss_unit / zero from the dedicated generator.
  Tests: gauss finite, mean ≈ 0, sd ≈ 1; gauss_unit = gauss/√112 exactly, mean column norm ≈ 1; time-D ≠ freq-D;
  zero ≡ 0 and still ≡ 0 after 3 optimiser steps; **all non-D parameters bitwise equal across the three arms at
  construction** and the global RNG state after model construction equal across arms.
- `layers.SparseMoE.NoisyTopkRouter_Cluster.forward` (`SparseMoE.py:57-66`): eval-only noise switch + recording of
  (indices, argmax s, row max/range of s). Tests: switch off → indices == argmax s on every row; switch on → output
  bit-identical to the released forward under the same RNG state. Router-only simulation of the A ceiling (one-hot
  s, n = 4, ≥ 20k rows) committed with its number.
- Eval-only scoring pass (concatenation instead of the crashing `np.array` stack at `exp_main.py:361-365`). Tests:
  N rows = dataset length; first ⌊N/128⌋·128 rows reproduce a drop_last=True pass's MSE from the same RNG state to
  1e-6; mean of per-window MSE = `utils/metrics.MSE` on the concatenated arrays to 1e-6.
- `EarlyStopping` subclass replacing `exp.exp_main.EarlyStopping` (imported by name at
  `exp_main.py:4` in both repos; built at tfps `:152`, official `:112`): writes full-precision `val_loss_min` and best epoch to the run's metrics file at each save;
  for TFPS also the pure-val-MSE checkpoint `checkpoint_valmse.pth` (never overwriting `checkpoint.pth`). Tests:
  value = min over per-epoch val losses; saved checkpoint epoch matches.
- Val-pass hook on `Exp_Main.vali` (`exp_main.py:74-135`), keyed on the val loader object identity (the loader
  built at `:139`), saving torch/numpy/python RNG state before each val pass. Tests: fires exactly once per epoch,
  only on a dataset with `set_type == 1`, never on the test loader; restoring the saved RNG state on the saved
  checkpoint reproduces the recorded val MSE to 1e-6.
- Official PatchTST harness wrapper (same eval pass, EarlyStopping subclass, no model patches). Test: its
  released-loader score equals the repo's printed test MSE to 1e-5.
- `extension_trigger.py` (§7) with a self-test on synthetic per-seed scores.
Wall time: step (c) measures epochs-to-stop and s/epoch per arm. If a single TFPS run cannot finish inside one
~45-min chunk, step (c) adds per-epoch resumable state (model, optimiser, scheduler, EarlyStopping, all RNG states)
with a self-test that a run interrupted after epoch 2 and resumed is bitwise identical to an uninterrupted one at
epoch 4; otherwise runs are never resumed mid-run (killed runs restart from scratch).

Budget estimate (replaced by step (c) measurements): TFPS 3 arms × 2 H × 5 seeds = 30 runs at ~25-30 s/epoch,
likely 30-60 epochs → ~8-15 CPU-h; eval passes (5 draws + noise-off + val-MSE checkpoint) ≈ +0.5 CPU-h; baselines
2 × 2 × 5 × 4 lr = 80 runs, small models at L=96 → ~2-4 CPU-h. Total ≈ 10-19 CPU-h at n = 5; up to ~2× if the
extension fires.

## 9. Not claimed / not tested
Other datasets; H = 96 / 720 (owner decision 6: cost, and the released H=96 config uses 16 experts, outside the
paper's grid); input lengths ≠ 96 (the paper's input-length search is not reproduced); the paper's other baselines
(FITS, TSLANet, iTransformer, …); D_svd / `Initialization_D`; the PI→Linear ablation; an lr search for TFPS (TFPS
lr provenance unknown); **whether training with noise-free affinity routing improves accuracy** (NOISE-OFF labels
concern evaluation only); GPU numerics; anything about what the authors ran, which D scale they had, or whether
their D was finite. Statements are about the released code as run here.

## 10. Paper-vs-code mismatches (to re-check against the PDF; not used by any rule)
K = 16 experts at H=96 lies outside the paper's stated grid {1, 2, 4, 8} [E]; η = 5 in code vs η = d in the method
text (r1 summarised fetch, not in the extract); input lengths searched (appendix) vs L = 96 (Table 1 caption) [E].

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

## OWNER DECISIONS, round 2 (Gustav, 2026-11-27 00:25 London, on design review r2)

9. **Add `tfps_gauss_unit`:** D ~ N(0, 1/(n·d)) (unit-scale columns, matching the repo's constraints), alongside
   D_gauss N(0,1). A negative Q3 verdict (SHRINKS / VANISHES) may be stated only if it holds for every TFPS arm.
10. **Paper access:** the reviewing chat commits the paper text (Alg. 1, the init sentences, Table 1 ETTh1 rows) to
    `audits/tfps/PAPER_EXTRACT.md` once it can fetch arXiv 2410.09836. Round 3 (the last allowed) starts only after
    that file exists, so the review can verify quotes and per-baseline margins.
11. **Also report Q3 with the baselines at their published lr** (no extra runs; reporting only, not a verdict arm).
