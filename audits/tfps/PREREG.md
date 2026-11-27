# TFPS audit — pre-registration r3.1 (DRAFT, not frozen; nothing below has been scored)

Revision history: r1 (commit 3e62bbb) → NOT_LICENSED (FOREGONE), `_design_review.json`. r2 (commit 32c41b2) →
NOT_LICENSED (1 blocking: D scale), `_design_review_r2.json`. r3 (commit bc7055d) applied OWNER DECISIONS 9-11 and
every r2 fix → LICENSED_WITH_FIXES (5/5 lenses, 0 blocking), `_design_review_r3.json`. **r3.1 (this text)** applies
the r3 review's `fixes_before_freeze` (13 items) mechanically; no new review round (round 3 was the last allowed,
protocol 5b). Where r3 lenses proposed different wordings for one fix, the consolidated `fixes_before_freeze` text
decides; owner decision 9's logic (a negative Q3 label needs every TFPS arm) is unchanged. Each r3.1 change is
listed in `audits/tfps/STATE.md` (2026-11-27 ~06:15-07:00 UTC entry).

Paper: "Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift" (TFPS),
arXiv 2410.09836 v2. Code audited: `tfps` at 83a11827 (`harness/repos.lock`). Baseline code: `patchtst_official`
at 204c21ef, subdir `PatchTST_supervised/`. Data: `ETDataset` at 1d16c8f4, `ETTh1.csv`.

**Paper text status.** Quotes and numbers marked [E] come from `audits/tfps/PAPER_EXTRACT.md` (WebFetch reading
of the v2 PDF by the reviewing chat; quotes returned as exact; Table 1 cells "reconstructed from garbled layout").
**FREEZE precondition (PDF check):** Gustav or the reviewing chat checks against the PDF, and records in
`PAPER_EXTRACT.md`: (a) the Table 1 column headers (model names and order); (b) the full ETTh1 MSE rows at H=192 and
H=336 for all 9 models (TFPS and 8 baselines); (c) the PDF context of the input-length sentence — whether "searched
input lengths among 96, 192, 336, and 512" applies to TFPS's Table 1 numbers or to a separate study, versus the
caption's "L = 96 for others". After the check, m_{H,B}, m_H and the anchor values in §1 / §7 are recomputed from
the checked cells (a transcription fix, not a design change). The agent does not FREEZE without this record.
**If the check cannot be done and Gustav nonetheless orders FREEZE**, no margin-dependent rule is applied: Q3 rules
2-4 and Q2/Q2b rule 3 and the ", smaller than m_H" sub-labels are not applied; Q3 per-arm labels are only BLOCKED,
REVERSED (U < 0 for some B), GAIN DETECTED OVER BOTH (L > 0 for both B) or INCONCLUSIVE (r3 Fable lens wording);
every baseline is treated as gating (all extract margins are > 0), labelled "unverified"; Q2 only rules 0, 1, 2,
4 and Q2b only rules 1, 2, 4; the anchor is not computed; the extension trigger is not applied (n = 5 only). (The r3 fallback "margins per unverified extract" is dropped.)

## 1. What the paper claims (ETTh1, Table 1, single runs, no error bars) [E]
- Table 1 caption: "The input lengths are L = 104 for ILI and L = 96 for others." The appendix also says input
  lengths were searched over {96, 192, 336, 512} with the best reported per horizon [E]; the two statements are not
  reconciled in the extract. The released `scripts/etth1.sh` uses seq_len 96 at every horizon; we use L = 96 for
  every arm and say so in every label. **Input-length qualifier:** if the PDF check (c) finds that the TFPS Table 1
  values come from the input-length search, or cannot settle it, every negative Q3 label (REVERSED / VANISHES /
  SHRINKS, per arm and horizon) and every TFPS `ANCHOR DRIFT` flag gets "— paper's TFPS values may come from an
  input-length search (L ∈ {96,192,336,512}); m_{H,B} may include an input-length gain not tested here". If the
  check finds the TFPS values are at L = 96, no qualifier. No change to the arms either way.
- ETTh1 MSE [E]: TFPS 0.423 / 0.484 (H = 192 / 336); PatchTST 0.460 / 0.497; DLinear 0.434 / 0.499; best of all
  Table 1 baselines 0.434 (DLinear) at H=192 and 0.489 (FITS) at H=336.
- **Per-baseline claimed margins (Q3 gates):** m_{H,B} = paper(B) − paper(TFPS):
  m_{192,PatchTST} = 0.037, m_{192,DLinear} = 0.011, m_{336,PatchTST} = 0.013, m_{336,DLinear} = 0.015.
  All are > 0, so the paper claims a gain over both baselines at both horizons. (The r2 fairness lens recalled
  DLinear ≈ 0.481 at H=336 from other papers' tables; the extract gives 0.499; the FREEZE check settles it.)
- **Non-gating baselines (checked m_{H,B} ≤ 0).** A baseline B with checked m_{H,B} ≤ 0 at horizon H is a baseline
  the paper does not claim to beat there. At that H, B is **removed** from Q3 rules 1-4, from the horizon label and
  its qualifiers, and from the extension trigger; "some B" and "both B" range over the remaining (gating)
  baselines; U uses the one-sided (1 − 0.05/k) bound with k = number of gating baselines (k = 1: one-sided 95%, no
  Bonferroni). d_{a,B} and its CI for a removed B are printed as descriptive. If no baseline is gating at H, Q3 at H
  is `DESCRIPTIVE ONLY`.
- **Q2 equivalence margin:** m_H = paper's margin over its best Table 1 baseline: m_192 = 0.011, m_336 = 0.005.
  Used only as "an effect as large as the paper's smallest claimed win"; it is not a claim about D. If the checked
  m_H ≤ 0 (TFPS not best in Table 1 at H), Q2 and Q2b at H use m_H := min over gating B of m_{H,B}; if no B is
  gating, m_H := 0.005, stated as audit-chosen in every label that uses it.
- D: Algorithm 1 input is a "random Gaussian distribution-initialized subspace" [E]; no mean or standard deviation
  is given; no sentence on k-means / SVD initialisation was found in the extract [E].
- Settings [E]: Adam; lr grid as printed {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.005} (0.005 listed twice);
  experts ∈ {1, 2, 4, 8}; "The best parameters are
  selected based on the lowest validation loss."; baselines: "We rerun all the experiments with codes provided by
  their official implementation."; α_t = α_f = 1e-3. Batch, epochs, patience, hardware not stated.

## 2. What the released code does (verified by reading the code at 83a11827; re-verified by r2 lenses)
- `layers/Cluster.py:30` allocates `D = Parameter(torch.Tensor(n·d, n·d))` (n = 4 experts, d = 28; 112×112) without
  initialising it. `layers/InitializeD.py` / `Initialization_D` (SVD left singular vectors: orthonormal columns,
  entries ≈ 1/√112) is imported (`exp/exp_main.py:24`) and never called; `sklearn.cluster.KMeans` is also imported
  (`exp/exp_main.py:12`) and unused (the EDESC k-means + SVD init pipeline is present only as unused imports). `Cluster.py` follows EDESC closely (its
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
| 5 | routing measurement | no training — arms 1-3 checkpoints | test set, each router (time, freq): A = fraction of rows with noisy top-1 = argmax s; H_r = entropy (nats) of expert usage; mean row max and row range of s; each computed on every one of the K = 5 eval-noise draws (§6) and averaged per run (same draws as the score; no draw is chosen) | manipulation check |
| 6 | `patchtst` | official `run_longExp.py`, model PatchTST, official ETTh1 arch (`scripts/PatchTST/etth1.sh:30-42`: e_layers 3, n_heads 4, d_model 16, d_ff 128, dropout 0.3, fc_dropout 0.3, head_dropout 0, patch 16, stride 8), individual 0, **seq_len 96** | official, deterministic | baseline |
| 7 | `dlinear` | official `run_longExp.py`, model DLinear, individual 0, **seq_len 96** | official | baseline |

**D draws.** G is a 112×112 standard-normal matrix drawn from a dedicated `torch.Generator` seeded 900000+seed for
the time EDESC and 910000+seed for the frequency EDESC (r3 used 30000/40000+seed, which collided with eval-noise
draws k = 3, 4; moved in r3.1); the global RNG is not touched. **Seed sets are pairwise disjoint** (self-test, §8):
training seeds S ∪ {1..5} ∪ {999}; D seeds {900000+s, 910000+s}; eval-noise seeds {10000·k + s, k = 1..5}, for every
s in S, the extension seeds {1..5} and the timing seed 999. Arms 1-3 at a given seed
therefore share every non-D initial weight, the data order and the training noise stream up to the first point
where D changes a computation; arms 1 and 2 use the same G (arm 2 is arm 1's D scaled by 1/√112). Neither scale is
claimed to be "the paper's": the paper states none. N(0,1) is the default of `torch.randn`; N(0, 1/(n·d)) is the
scale the repo's constraint and unused initialiser point to; for a 112×112 matrix it is also exactly the scale of
`torch.nn.init.xavier_normal_` (std = √(2/224)). Expected consequence, stated in advance: under arm 1 the aux term
is ≈ 10× val MSE at init and likely drives checkpoint selection (arm 1 may run up to the full 100 epochs); under
arm 2 it is ≈ 0.02× val MSE but s is nearly uniform at init (r3 forward probe, H=192 config, seed 2021, first 4
test batches: mean row range of s ≈ 0.005 (time) / 0.008 (freq), max ≈ 0.02-0.035; A ≈ 0.251 / 0.250, within one
draw-sd of the zero arm's 0.250). Unit-arm affinity equals the N(0,1) arm's affinity with z shrunk by √112
(s_unit(z) = s_N01(z/√112), `Cluster.py:61-68`), and argmax s is scale-invariant, so arms 1 and 2 share argmax s at
init. **At init, arm 2 routing is indistinguishable from arm 3; Q2(gauss_unit) EQUIVALENT or INCONCLUSIVE is the
expected outcome unless training sharpens s** (expected, not guaranteed; "no" is a legitimate answer to Q2).

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
  the lowest `val_loss_min` as returned by the repo's `vali` (float32 for the official repo) and written with
  `repr()` by the EarlyStopping subclass (§8), ties to the smaller lr; a run with any non-finite val loss cannot be
  selected. Only the selected run enters verdict tables; all are kept.
- **Baseline lr (published-lr table, owner decision 11, no extra runs):** the grid runs at PatchTST lr 1e-4 and
  DLinear lr 5e-3 (the lrs in `scripts/PatchTST/etth1.sh` and `scripts/Linear/etth1.sh`).
- Deviations from the baselines' configs as released in the PatchTST repo: PatchTST seq_len 336, lr 1e-4,
  patience 100; DLinear (as configured in `scripts/Linear/etth1.sh`) seq_len 336, batch 32, lr 5e-3, patience 100.
  We use seq_len 96 (Table 1 setting), batch 128 (owner decision 4), the lr grid (contains both released lrs) and
  patience 20 (TFPS's released value, same for every arm). DLinear runs from the PatchTST repo's copy
  (`models/DLinear.py` 0893b53c), not from the LTSF-Linear repo; its training defaults (epochs, patience) are
  PatchTST's harness defaults as above, not LTSF-Linear's.

## 6. Scoring (one code path, then an independent recompute in step f)
- After training, the wrapper loads the checkpoint EarlyStopping selected **by absolute path** (sha256 in
  provenance; never via `test(test=1)`) and runs eval-only passes over the **full test set (drop_last off)**,
  saving per-window MSE and MAE (float32 on disk, accumulated in float64; one value per window = mean over
  pred_len × 7 of the error in the normalised space, as `utils/metrics.py`).
- **TFPS eval noise:** K = 5 eval passes per TFPS checkpoint with the noise RNG seeded 10000·k + seed, k = 1..5.
  **Per-run TFPS score = mean of the 5 per-draw MSEs** (chosen now, before any score is seen; forecasts are never
  averaged across draws, which would be an ensemble). Each draw's score is saved;
  the mean |draw − mean| per run is reported next to seed sd, and flagged beside every TFPS label if it exceeds
  m_H/4. Baselines are deterministic: one pass.
- **Primary score:** mean over all N windows. **Released-loader score:** mean of the first ⌊N/128⌋·128 windows of the
  same passes (reported, not gating).
- The repo's own printed test metric is also recorded (for TFPS it uses another noise draw; not expected to match).
- **Pure-val-MSE selection (co-reported, attaches suffixes):** for every TFPS run the wrapper also saves
  `checkpoint_valmse.pth` at the epoch of lowest val MSE (forecast part only, from the same val pass, keyed on the
  val loader; §8) and scores it the same way. Candidate set: epochs 1..stop epoch of the released criterion (it is
  a min over the epochs the released run reaches, not an early stop on val MSE); ties to the earliest epoch. The
  per-arm mean number of candidate epochs is printed beside the pure-val-MSE table.
- **Per-epoch log (every TFPS run):** val MSE, each aux component per router, ‖D‖_F, mean column norm of D, mean row
  range of s per router (val pass), the saved-checkpoint epoch; per run: epochs-to-stop, aux share of val loss at the saved epoch, ‖D_final − D_init‖_F /
  ‖D_init‖_F (arms 1-2). Descriptive.
- **Failed run:** any non-finite training loss, val loss or primary score. Failed runs are reported, never replaced.
  A baseline seed fails only if all 4 lr runs fail. In the published-lr table (owner decision 11) a non-finite run
  at the published lr is a failed run for that table, with the same BLOCKED / reduced-n rules.
- **BLOCKED:** an arm is BLOCKED at H if ≥ 2 of 5 seeds fail (≥ 3 of 10 after extension). Q2 contrast (a vs zero)
  is BLOCKED if either arm is; Q3 for TFPS arm a is BLOCKED if a or either gating baseline is. With k allowed
  failures (k = 1 at n = 5; k ≤ 2 at n = 10) the arm uses its remaining seeds and every label resting on it gets
  "— k FAILED RUN(S) IN <arm>", plus a sensitivity where every failed seed is scored as that arm's worst observed
  seed.

## 7. Pre-registered verdicts (per horizon; first matching rule wins)
**Label suffix (every label):** "— ETTh1, L=96, H=…, released TFPS code with audit-supplied D init (<arm>), vs
PatchTST and DLinear rerun at L=96 in the official PatchTST code (baselines lr-searched per seed, TFPS not), CPU".
`<arm>` is the arm name of a per-arm label (Q2 contrast: both arm names); for Q3 horizon and item labels it reads
"audit-supplied D init (three inits: N(0,1), N(0,1/112), 0)". Every label is printed with its point estimate and CI.
**Glosses, printed with the label every time:** "VANISHES (gain over <B> not detected; claimed margin <m> excluded)"
— it does not mean the gain is zero; "SHRINKS (gain detected over both; claimed margin over <B> excluded)".
**Base label** = the label token before any comma, dash-suffix or parenthesis; every "same label" comparison at item
level compares base labels only, with suffixes, sub-labels and per-arm contents printed per horizon.

**Statistics.** Welch with Welch-Satterthwaite df from the observed per-seed sds; one-sided bounds as stated per
rule. Paired comparisons (Q2b) use a paired t over seeds. CIs cover seed variance only, conditional on the ETTh1
test split; no window-level test is run (windows overlap and are autocorrelated). For Q2, the paired difference
over seeds (arms share non-D init) is reported next to the Welch CI, not gating; Welch is conservative under
this shared-seed pairing (both for BETTER rules and for EQUIVALENT), which is accepted.

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
are a reference (expected 0.25 / ln 4). Whether or not the suffix fires, A (per router), the ceiling and the mean
row range of s are printed next to every Q2 label; Q2 is about D's initialisation under the released router, not
about whether patch-pattern routing works.
Item level for each a: same base label at both horizons → that label; else `MIXED (192: …, 336: …)`.

**Q2b — eval noise (descriptive, does not feed Q3).** For a ∈ {1, 2}: Δ_nn = a_nonoise − a, paired 90% CI; rules 1-4
as Q2 with labels NOISE-OFF BETTER / NOISE-OFF WORSE / NOISE-OFF EQUIVALENT / INCONCLUSIVE, margin m_H. Concerns
evaluation only of a model trained with the noisy router. Q2b uses every seed the arms have (n = 10 if the
extension fires, same seeds), with the n = 5 table beside it. Item level as Q2.

**Q3 — TFPS vs baselines.** For each TFPS arm a ∈ {tfps_gauss, tfps_gauss_unit, tfps_zero} and each baseline
B ∈ {patchtst, dlinear}: d_{a,B} = mean(B) − mean(a) (positive = TFPS better); L_{a,B} = one-sided 95% lower bound,
U_{a,B} = one-sided 97.5% upper bound (Bonferroni over the two baselines for the "some B" conditions; L needs no
correction under intersection-union). Margin m_{H,B} from §1 (FREEZE-checked values). B ranges over the gating
baselines at H only (§1, non-gating baselines); with k gating baselines U is one-sided (1 − 0.05/k).
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
**Horizon label (owner decision 9), first match:**
0. If any gating baseline is BLOCKED at H → `BLOCKED`. If no baseline is gating → `DESCRIPTIVE ONLY`. If any TFPS
   arm is BLOCKED at H → `INCOMPLETE (<arm> BLOCKED; others: …)` — neither negative nor positive.
1. All three per-arm labels equal → that label.
2. Every per-arm label is one of REVERSED / VANISHES / SHRINKS → the least negative of them (order REVERSED <
   VANISHES < SHRINKS), with the per-arm table.
3. Otherwise → `NOT CONSISTENT ACROSS TFPS ARMS (gauss: …, unit: …, zero: …)` (neutral wording; r3 called it
   DEPENDS ON D INIT, renamed in r3.1 because it fires often when the arms are identical — r3 Monte Carlo,
   identical arms, within-seed correlation ρ = 0 unless noted: statistics lens 0.09-0.13 at true gain m and σ_T 0.002
   (0.04-0.06 at ρ = 0.9), 0.11-0.22 at true gain m and σ_T 0.005-0.01, 0.73-0.77 at true gain m and σ_T 0.02
   (0.25-0.35 at ρ = 0.9), 0.55 / 0.78 (H = 192 / 336) at true gain m/2 and σ_T 0.01 (0.26 / 0.29 at ρ = 0.9);
   foregone lens 0.60 / 0.75 at true gain m/2 and σ_T 0.006 / 0.012). The per-arm triple is part of the label and is never dropped or shortened. Direction of the
   split is stated: when the two Gaussian arms share a label X and tfps_zero has Y ≠ X, the label is printed as
   `NOT CONSISTENT ACROSS TFPS ARMS — <X> AT BOTH GAUSSIAN SCALES, <Y> WITH D = 0 (affinity-blind control)`, and
   the report carries the fixed sentence: "Gaussian arms SURVIVE while the D = 0 control does not is the pattern
   the paper's mechanism predicts; the reverse is not." If both Q2 contrasts at this H are EQUIVALENT or
   INCONCLUSIVE, append "— Q2 detects no D effect; arm disagreement may be seed noise".
A negative horizon label therefore needs every TFPS arm to be at least that negative.
**Qualifiers on every horizon label** (including SURVIVES, INCONCLUSIVE and NOT CONSISTENT ACROSS TFPS ARMS):
(i) recompute the full three-arm horizon label with the baselines at their published lr (§5; "published-lr (lr from
the PatchTST repo scripts; L=96, batch 128, patience 20 as §5)"). If it is the identical label (base label and
per-arm contents), nothing is added. If the searched label is negative (REVERSED / VANISHES / SHRINKS) and the
published-lr label is the same or more negative (REVERSED < VANISHES < SHRINKS), the searched label holds; when
more negative, add "— PUBLISHED-LR LABEL: …". If the searched label is negative and the published-lr label is not
negative or is less negative, add "— ONLY WITH PER-SEED LR SEARCH FOR BASELINES (published-lr label: …)". In any
other case of difference, add "— PUBLISHED-LR LABEL: …". (ii) Recompute the full three-arm horizon label with
pure-val-MSE selection for the TFPS arms; if it is not the identical label (base label and per-arm contents), add
"— SELECTION-SENSITIVE (pure-val-MSE: …)". Both tables are printed in full regardless.
Item level: same base horizon label at both H → that label; else `MIXED (192: …, 336: …)`.
Each baseline's paper margin and the released-loader-score table are printed beside the CIs, not gating.

**Reachability (stated per label; equal seed sd σ unless noted).** n = 5: Welch df 8, t₀.₉₅ = 1.860,
t₀.₉₇₅ = 2.306, so hw_L = 1.176σ, hw_U = 1.458σ; with sd ratio 0.1 (e.g. DLinear nearly deterministic) df ≈ 4.1,
hw_L = 0.953σ_T, hw_U = 1.238σ_T. n = 10: df 18, hw_L = 0.775σ, hw_U = 0.940σ. One arm at n = 4 vs n = 5 (one
failure): df 6.57, hw_L = 1.284σ, hw_U = 1.608σ.
- REVERSED, VANISHES, SURVIVES, Q2 rules 1-2: no width floor (can fire from the point estimate at any σ).
- Q2 EQUIVALENT: 90% hw < m_H → σ < 0.0094 (H=192) / 0.0043 (H=336) at n = 5; 0.0142 / 0.0064 at n = 10.
- Q2 ", smaller than m_H" sub-labels: 2·hw_90 < m_H → σ < m_H/2.352: 0.0047 / 0.0021 at n = 5; 0.0071 / 0.0032 at
  n = 10.
- NOISE-OFF EQUIVALENT (paired t, df n − 1): t_{0.95,n−1}·sd_diff/√n < m_H → sd_diff < 0.0115 / 0.0052 at n = 5;
  0.0190 / 0.0086 at n = 10.
- SHRINKS: hw_L + hw_U < m_{H,B} → at n = 5, σ < m/2.635: PatchTST 0.0140 / 0.0049, DLinear 0.0042 / 0.0057
  (H = 192 / 336); at n = 10, σ < m/1.715 (0.0216 / 0.0076, 0.0064 / 0.0087).
- Plausible seed sds (r2 lenses, unmeasured here): PatchTST ≈ 0.002-0.005, DLinear ≈ 0.0005-0.002, TFPS unknown and
  possibly larger (lr 5e-3 at H=336, router noise partly averaged by K = 5). SHRINKS against DLinear at H=192 and Q2
  EQUIVALENT at H=336 may stay unreachable even at n = 10; this is stated, not fixed.
- Monte Carlo (r3 statistics lens, r3 rules, `_r3/statistics.json` → simulations): with the true gain exactly m
  for both B, P(U_B < m for either B) = 0.031-0.039; with true gain 0, P(L > 0) = 0.048-0.054 per baseline; with a
  true gain of m/2, per-arm VANISHES fires 0.07-0.34 and horizon-level VANISHES 0.00-0.23: VANISHES means "claimed
  margin excluded", not "gain = 0".

**Variance-triggered extension, once.** After all 5 seeds of every arm at a horizon are scored, a committed script
(`audits/tfps/extension_trigger.py`, written and self-tested in step (c)) reads only per-seed primary scores and
computes, at that H:
T(H) = OR over a ∈ {gauss, unit}, both non-BLOCKED, of [t_{.95,ν}·se(a, zero) ≥ m_H]
       OR over non-BLOCKED a ∈ {gauss, unit, zero} and gating B of [(t_{.95,ν} + t_{.975,ν})·se(a, B) ≥ m_{H,B}],
where se and ν are Welch-Satterthwaite from **released-selection** primary scores (TFPS) and **lr-searched**
baseline scores, with the seeds each arm has (n = 4 for an arm with one failure); margins are the FREEZE-checked
ones (t_{.975} is replaced by the (1 − 0.05/k) quantile when k ≠ 2 gating baselines). If se = 0 the width is 0 (and
ν is not needed). BLOCKED arms and non-gating pairs are excluded. It is judged on widths only. Expected
P(extend) (r3 statistics lens): 0.55-0.85 at σ_T 0.005, ≈ 1 at σ_T ≥ 0.01, driven mostly by DLinear at H=192 and Q2
at H=336; the budget in §8 allows for it. Its output and the n = 5 per-run files are committed
**before** any verdict table or label is printed. If T, add seeds {1, 2, 3, 4, 5} to every arm at that H (baselines
rerun the full lr grid, lr re-selected per new seed by the same rule), then apply the rules once at n = 10 and
report the n = 5 table beside it. No further extension. The extension's type-I inflation (≈ 0.5-1 pp per one-sided
bound, r2 simulation) is accepted. Q2b is excluded from T.

**Anchor (not gating):** for each TFPS arm, a 90% prediction interval for a single run, mean ± t·s·√(1 + 1/n), on
the released-loader score; for TFPS arms s² is replaced by s² + v̄_draw (v̄_draw = mean over runs of the within-run
variance of the K = 5 per-draw scores), because the paper's value is a single noisy draw and our per-run score is
a 5-draw mean; flag `ANCHOR DRIFT` if the paper value (FREEZE-checked) lies outside it by more than 0.02 (with the
§1 input-length qualifier when it applies). Same for
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
- Seed-set disjointness (§5): self-test asserting training seeds, D-generator seeds and eval-noise seeds are
  pairwise disjoint for S, the extension seeds and the timing seed 999.
- Instrumentation leaves training unchanged: a 2-epoch TFPS run (arm 1, train_epochs 2) and a 2-epoch PatchTST run,
  each with and without the instrumentation (val-pass hook, EarlyStopping subclass, extra checkpoint save, router
  recording off at train), give bitwise-equal state_dicts, val losses and global RNG state at the end. These
  self-test runs use seed 999 and the test-line filter below.
**Timing / feasibility runs before FREEZE (leakage guard).** Every pre-FREEZE run at H = 192 / 336 (timing,
self-tests) uses training seed 999 (outside S and the extension seeds), is never scored, and its stdout passes
through a filter that drops lines containing "Test Loss", "test", "mse:" or "mae:" (case-insensitive) before
anything is written or displayed; the wrapper deletes `./result_*.txt`, `result*.txt` and `test_results/` after
the run. Epochs-to-stop and s/epoch are read from the filtered log. STATE.md records that no H = 192 / 336 test
number was seen before FREEZE. Disclosure: the step-(a)-era feasibility logs `audits/tfps/feas_zero.log` and
`feas_orth.log` (H = 96, 3 epochs, seed 2021, lr 5e-4, D_zero / D_orth — not a scored configuration or horizon)
contain per-epoch test losses at H = 96 that were seen in r1.
Wall time: step (c) measures epochs-to-stop and s/epoch per arm. If a single TFPS run cannot finish inside one
~45-min chunk, step (c) adds per-epoch resumable state (model, optimiser, scheduler, EarlyStopping, all RNG states)
with a self-test that a run interrupted after epoch 2 and resumed is bitwise identical to an uninterrupted one at
epoch 4; otherwise runs are never resumed mid-run (killed runs restart from scratch).

Budget estimate (replaced by step (c) measurements): TFPS 3 arms × 2 H × 5 seeds = 30 runs at ~25-30 s/epoch,
likely 30-60 epochs for arms 2-3 and up to the full 100 epochs for arm 1 (its val loss keeps falling with the
constraint term) → ~11-19 CPU-h; eval passes (5 draws + noise-off + val-MSE checkpoint) ≈ +0.5 CPU-h; baselines
2 × 2 × 5 × 4 lr = 80 runs, small models at L=96 → ~2-4 CPU-h. Total ≈ 14-24 CPU-h at n = 5; up to ~2× if the
extension fires (likely: P(extend) ≈ 0.55-1 at σ_T ≥ 0.005, §7).

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
