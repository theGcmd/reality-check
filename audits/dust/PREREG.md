# DUST-1 — pre-registration r1 (DRAFT, not frozen; nothing below has been run or scored)

Queue row 2 (owner, 2026-11-27): **prereg drafting + design review r1 only. NO runs** — the FineWeb data must come
from Gustav, and Dust arms need a GPU. This text is the r1 draft that design review r1 reviews.

Paper: "Dust: Pretraining Transformers Without Backpropagation", Q Labs (Dahal, Mandal, Gülbahar, Vegesna), October
2026, https://qlabs.sh/research/dust. Code audited: `dust` at b20f7c03 (`harness/repos.lock`). Data: FineWeb
`sample-10BT` via the repo's `prepare_data.py` (not reachable from this workspace).

**Paper text status.** Every paper-side number below comes from `audits/dust/SOURCE_NOTES.md`, a WebFetch
paraphrase; the page itself, its Table 1 and its Appendix F were NOT readable. Every such number is marked [U]
(unverified). Code-side facts are marked with file:line at the pinned commit and were read directly.
**FREEZE precondition P1 (paper check):** Gustav or the reviewing chat records in `audits/dust/PAPER_EXTRACT.md`,
from the page itself: (a) the measured 1M-token test losses (mean and spread, seed count) for Dust at every
population and for every backprop variant (SGD and Adam), at the 8×512 model; (b) the exact sentence(s) in which the
paper states what Dust beats or matches at 1M tokens, and whether that statement refers to a measured cell or to the
fitted population limit; (c) the backprop baselines' tuning grids (lr, momentum, Adam betas/eps/wd, schedule,
batch), and whether batch size was ever varied for backprop; (d) how val is used (checkpoint selection only, or also
the lr/momentum grid). No FREEZE without this record.
**FREEZE precondition P2 (data):** the four files `train_1M.pt`, `val.pt`, `test.pt` (and `manifest.json`) produced
by the pinned `prepare_data.py --tokens 1M`, committed under `audits/dust/data/` or provided at a path the harness
reads, with sha256 recorded. (P2 blocks runs, not the FREEZE text.)

## 1. What the paper claims (all [U])
- Method: zeroth-order per-token output-noise estimator; one SGD-with-momentum step per batch.
- Setup: 8 layers × 512 wide GPT, FineWeb, BPE-4096, seq 2048, batch 8 sequences (16,384 tokens), **one epoch**,
  budgets 100k / 1M / 10M / 20M tokens; held-out val and test 544 sequences each; test loss at the best-val
  checkpoint; 3 seeds per cell (Adam comparison: 5 seeds Dust/backprop).
- Backprop baseline (main): SGD + momentum, constant lr, same (lr, momentum) grid as Dust, tuned per budget. An Adam
  variant at 1M re-tunes backprop's lr.
- 1M tokens, Adam comparison: Dust **fitted population limit** 5.248 (95% 5.17-5.30) vs backprop 5.361. This is a
  power-law extrapolation over population, not a measured run. The measured Dust cells at 1M are NOT in our notes.
- 10M tokens, measured (test loss, 2M / 7.3M / 38M / 243M params): backprop 5.180 / 5.066 / 5.015 / 5.048; Dust at
  16k draws 5.171 / 5.065 / 5.036 / 5.086.
- Authors' own limits: compute efficiency not the focus; gains beyond 1M rely on extrapolation; gap to backprop grows
  slightly with model size.

## 2. Code facts (read directly, pinned commit)
- `baselines/backprop.py:16-19`: 1M → SGD lr 0.175, momentum 0.98, 61 steps, eval every 2 steps. `:22-28`: param
  groups = {all except wte: lr}, {wte: lr 1000.0}; weight_decay 0; no schedule; no grad clipping.
- `dust.py:453-467` (Dust's optimizer): wte 1000.0, **value_embeds 0.3, resid/x0 lambdas 0.03**, rest = recipe lr.
  The released backprop baseline does NOT have the value_embeds / lambdas groups; they get the base lr 0.175. Not a
  claim about the paper's runs (P1 c may say otherwise).
- `prepare_data.py:11`: 1M → 488 training sequences (= 61 × 8); `:65-69` val/test follow a ~20.5M-token pool.
- `backprop.py:84-88`, `dust.py:520-524`: training order = `randperm(seed+1)`; one pass; `make_model(seed)`.
- `backprop.py:112-144`: val evaluated at step 0, 1, every 2 steps and last; checkpoint with lowest val loss kept;
  test evaluated once on that checkpoint. Val is the selection set; test is touched once per run.
- `backprop.py:59-60`, `dust.py:490-491`: refuse to run without CUDA. CUDA path uses bf16 autocast and bf16
  embeddings (`model.py:162-166`); CPU path is fp32.

## 3. Question
**Q1 (primary).** At 1M tokens, one epoch, same model / init scheme / data / val / test, is backprop configured the
way small transformers are normally trained (AdamW, warmup + cosine decay, weight decay on matrices, grad clipping,
batch size and peak lr chosen on val) still worse than **Dust's measured 1M test loss** (P1 a, best population the
paper measures, with its reported spread)?
**Q2 (reported, not a verdict on Dust).** The batch-matched framing the paper uses: the tuned AdamW arm restricted
to batch 8 (61 steps).
**Q0 gate (decides whether Q1 is a question at all).** If P1 shows that no measured Dust cell at 1M beats the
paper's own 1M backprop numbers (i.e. the paper's 1M advantage exists only for the fitted limit), Q1 is **MOOT**:
we report `NO MEASURED 1M ADVANTAGE IN THE PAPER TO TEST` and run only the anchor and the descriptive Q3. This is
not a finding against Dust; the paper's limit claim is labelled as an extrapolation by its authors.
**Q3 (descriptive only).** Tuned backprop vs the fitted limit 5.248 [U], reported with the words "an extrapolation,
per the authors".

## 4. Arms (backprop only; all run here on CPU; no Dust arm is run)
All arms: `GPTConfig()` 8×512, `make_model(config, cpu, seed)` unchanged; the same 488 training sequences in the
order `randperm(seed+1)`, each seen once; tokens = 488 × 2048 ≈ 1.0M in every arm.
- **B0 `bp_released`** (anchor): `backprop.py` 1M defaults (SGD 0.175, m 0.98, batch 8, 61 steps, wte 1000, no
  wd/schedule/clip), released eval schedule (every 2 steps, full val). Run through a wrapper that only removes the
  CUDA refusal and adds gradient accumulation (see §6).
- **B1 `bp_adamw`** (primary): AdamW β = (0.9, 0.95), ε = 1e-8; weight decay 0.1 on 2-D matrices (attn/MLP/lm_head),
  0 on wte, value_embeds, lambdas; grad clip 1.0 (global norm); linear warmup over the first 10% of steps, then
  cosine to 10% of peak. Tuned grid, chosen on val only: batch b ∈ {1, 2, 8} sequences (488 / 244 / 61 steps) ×
  peak lr ∈ {3e-4, 1e-3, 3e-3, 1e-2} for matrices; embeddings (wte, value_embeds) at 10 × peak (rms-normed inputs,
  a common practice; declared choice), lambdas at peak. 12 cells.
- **B2 `bp_sgd_dustgroups`** (OPTIONAL, owner decides; exploratory, no verdict): B0 with Dust's own param groups
  (value_embeds 0.3, lambdas 0.03), to see whether the released baseline's missing groups matter.

## 5. Selection, seeds, budget fairness
- **Tuning:** each B1 cell run at tuning seed 1000 only. Selection score = val loss of the cell's selected
  checkpoint (§6 eval schedule), on the full 544 val sequences. Ties within 0.005 → larger batch (cheaper). Test is
  never evaluated during tuning (wrapper refuses test for tuning runs).
- **Scored:** the selected B1 cell and B0 at seeds S = {0, 1, 2, 3, 4} (disjoint from 1000); test evaluated once per
  run at its best-val checkpoint, as the repo does.
- **Fairness of budget:** Dust's (lr, momentum) were tuned per budget and population on val [U]; B1 gets a 12-cell
  grid on val at one seed. Both sides: one epoch, same tokens. B1 gets more optimizer steps at b < 8; that is the
  point of the question and is labelled in every verdict ("matched tokens, not matched steps or batch").
  Compute is NOT matched (Dust's forward-only cost per step is far higher [U]); not tested, not claimed.

## 6. Eval schedule and engineering (CPU cost measured, `audits/dust/timing_cpu.log`)
- Measured here (2 cores, fp32, random tokens, seed 999): fwd+bwd of one 2048-token sequence ≈ 2.9-3.2 s → one
  8-sequence step ≈ 23.5 s via accumulation; **full val pass (544 seq) ≈ 9.0 min**. A full 8-sequence backward OOMs
  in 7 GB RAM, so every arm uses micro-batch 1 with gradient accumulation over b sequences (sum of per-token-mean
  losses / b; same gradient as the batched mean up to fp order; self-test required).
- Training compute per run ≈ 24 min at any b (same tokens). The released eval schedule (32 full val passes) costs
  ≈ 4.8 h per run: B0 ≈ 5.3 h/run.
- **B1 eval schedule (declared):** full val at 5 points (60/70/80/90/100% of steps), best-of-5 checkpoint. With a
  decaying schedule the best checkpoint is expected near the end; the step-0 / early checkpoints the released
  schedule also scores are dropped (declared deviation; it can only hurt B1).
- **B0 eval schedule:** DECISION NEEDED (a) released schedule, all 5 seeds (≈ 26 h); or (b) released schedule at
  seed 0 only (anchor), 5-point schedule for seeds 0-4 (labelled).
- **Budget:** tuning 12 × (24 + 45 min) ≈ 14 h; B1 scored 5 × 69 min ≈ 5.8 h; B0 26 h (a) or ≈ 11 h (b).
  Total ≈ 31-46 CPU-h. Cheaper options for Gustav: GPU; tuning evals on the first 136 val sequences (≈ 2.3 min/pass,
  full val for the scored runs only); or drop b = 1.
- Precision: paper ran bf16 on GPU [U]; we run fp32 on CPU. B0 is the anchor for this drift.

## 7. Verdict rules (fixed before any run)
Let R = the paper's measured best Dust 1M test loss (P1 a) with its reported SD s_R over n_R seeds; if no spread is
reported, s_R := our B1 seed SD (declared). Let D = mean(B1) − R; CI = 90% Welch interval from (B1 seeds, s_R/n_R).
- **ANCHOR:** |mean(B0) − paper's 1M SGD backprop test loss (P1 a)| ≤ 0.03 → ANCHOR OK; else ANCHOR DRIFT, and every
  Q1 label gets "in our fp32/CPU environment; released baseline did not reproduce the paper's value".
- **Q1 labels** (only if Q0 passes):
  - `SURVIVES`: CI lower bound > 0 (tuned backprop still worse than Dust's measured value).
  - `REVERSED`: CI upper bound < 0 (tuned backprop better than Dust's measured value).
  - `SHRINKS`: SURVIVES and D < 0.5 × (paper's backprop − R) (more than half the paper's measured gap closed).
  - `INCONCLUSIVE`: otherwise (CI contains 0).
  Every Q1 label carries: "at 1M tokens, 8×512, one epoch, matched tokens (not steps/batch/compute); Dust not
  re-run here, compared to the paper's reported value [U until P1]".
- **Q2:** same rules with B1 restricted to b = 8 (selected lr at b = 8, its own 5 seeds only if Gustav approves the
  extra ≈ 5.8 h; otherwise reported from the tuning seed only, no label).
- **Q3:** D_lim = mean(B1) − 5.248 reported with CI; no label.
- Nothing about Dust's population limit, scaling, or larger budgets is tested.

## 8. What would make this unfair or uninformative (for the reviewers)
- If the paper's Adam backprop at 1M already used a schedule / wd / batch search, B1 adds little (P1 c).
- If R is from a population we cannot match (16k draws on 8 GPUs), that is fine: Dust is not re-run.
- If Q0 fails, Q1 is moot by construction — the review should say whether that is what P1 will most likely show.

## 9. Not tested
Dust itself (no GPU); other budgets (100k, 10M, 20M); model sizes; compute-matched comparison; the population-limit
fit; bf16 effects beyond the B0 anchor.
