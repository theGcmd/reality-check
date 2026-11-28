# TFPS-1 — state

## Done
- 2026-11-24, step (a) design review round 1 → **NOT_LICENSED (FOREGONE)**. `_design_review.json` (5 lenses; Fable ran the
  answer-built-in lens). Foregone: MECHANISM-INERT is unreachable as written — the released router
  (`layers/SparseMoE.py:55-63`) adds `randn*softplus(s)` noise before top-1, also at eval, so D_zero gives uniform
  affinity and *uniformly random* routing, not constant routing; and the same noise nearly randomises D_orth's routing,
  so a zero-vs-orth null is close to built in. Other lenses: fairness / leakage / statistics LICENSED_WITH_FIXES, Fable
  NOT_LICENSED.
- Nothing scored. No runs. PREREG.md unchanged (still DRAFT).

## Prereg facts the review found wrong (must be fixed in step b regardless of decisions)
- TFPS config at the audited horizons is not "16 experts, lr 5e-4": `scripts/etth1.sh:77-89` H=192 → 4/4 experts, lr
  5e-4; `:133-145` H=336 → 4/4 experts, lr 5e-3 (16/16 is H=96).
- D_zero routing premise (above). D does stay exactly 0 under D_zero (grad 0 through KL and both constraints).
- Baselines cannot run in TFPS's `exp/exp_main.py` (3-way unpack at :99, :203, :324; `model.model_time.cluster`).
- Keep-all test scoring crashes in `test()` (ragged `np.array`, exp_main.py:361-365).
- Checkpoint setting string lacks seed/arm → runs can overwrite each other's checkpoints.
- Verdict rules: undefined SE, test-selected "best arm", no INCONCLUSIVE, INERT by failure-to-reject. Statistics lens
  supplied concrete Welch-CI / TOST rule text (in `_design_review.json`).
- Paper (arXiv HTML v2, via summarised fetch; re-check against PDF): Alg.1 says D is "random Gaussian
  distribution-initialized"; EDESC / k-means+SVD not described.

## Next
- (superseded, see 2026-11-25 entry below)

## Patches in force (wrapper `run_tfps_dinit.py`, not yet used for scoring)
- `layers.Cluster.EDESC.__init__` (layers/Cluster.py:30 allocation) — D init asis/zero/orth.
- `data_provider.data_factory.DataLoader` (data_factory.py:17-20) — test drop_last off when TFPS_TEST_DROP_LAST=0
  (currently crashes downstream in test(); see above).

## 2026-11-24 20:00 London — owner decisions recorded
- PREREG.md now has OWNER DECISIONS 1-8. Row BLOCKED until 03:00 London 2026-11-25 (owner timing); then step (b).

## 2026-11-25 ~03:10-03:40 UTC — step (b): prereg r2 + design review round 2 → NOT_LICENSED (not foregone)
- PREREG.md rewritten (r2, commit 32c41b2): arms tfps_gauss (primary, N(0,1)), tfps_zero (affinity-blind control),
  tfps_gauss_nonoise (eval-only, same checkpoint), routing measurement A/H_r, patchtst + dlinear in official PatchTST
  repo with per-seed val lr grid; Welch/paired 90% CI rules with INCONCLUSIVE; byte-identity table; engineering list.
- Review r2 (`_design_review_r2.json`, raw lenses in `_r2/`): foregone / fairness / leakage / statistics
  LICENSED_WITH_FIXES; Fable answer-built-in LICENSED_WITH_FIXES with 1 blocking issue → round not licensed.
  Foregone lens: every verdict reachable; forward-only probe A ≈ 0.283 under N(0,1) (chance 0.25), s near-uniform.
- Shared major finding (4 lenses, re-checked): under N(0,1) the D constraints add ~4.4 to TFPS's early-stopping val
  loss (val MSE ~0.45), vs ~0.01 under D=0 or N(0,1/(n·d)) → arm-1 checkpoint selected mostly on D constraint decay.
- Fixes to apply in r3 regardless of decision: dedicated torch.Generator for D; pin --itr 1; EarlyStopping subclass
  writing full-precision val_loss_min; NaN val loss = failed; sensitivity hook keyed on val loader with RNG restore;
  load checkpoints by absolute path; correct power text (SHRINKS needs half-width < m_H/2; SURVIVES/VANISHES none;
  Welch df can be ~4-6); exact extension formula committed by script before labels; Bonferroni (97.5% one-sided)
  over the two baselines or state the 8.2% rate; failed-run suffix + sensitivity; BLOCKED/PREMISE first in order;
  Q2 item-level rule; A qualifier per router, arm 1 only, with stated ceiling (~0.61); anchor via prediction
  interval; state type3 runs epoch 1 at lr/25 (both repos); list individual/num_workers deviations; label scope
  "released code with audit-supplied D init"; permute cite :81-85; per-baseline margins m_{H,B} from PDF Table 1.
- Paper still unread (arXiv fetch permission prompt timed out, 2 agents). [Q] items unverified.

## Next
- BLOCKED on Gustav: D scale decision + paper access (DIGEST 2026-11-25). Then step (b) round 3 of max 3: r3 rewrite +
  review. If r3 is not licensed, item goes BLOCKED with reasons per protocol.

## 2026-11-27 ~03:12-04:00 UTC — step (b) r3: prereg r3 + design review round 3 → LICENSED_WITH_FIXES
- PREREG.md r3 (commit bc7055d): arms tfps_gauss N(0,1), tfps_gauss_unit N(0,1/112) (owner 9), tfps_zero, eval-only
  nonoise for both Gaussian arms, routing measurement; D from dedicated generators; K=5 eval-noise draws averaged;
  per-baseline margins m_{H,B} from PAPER_EXTRACT [E] (192: P 0.037, D 0.011; 336: P 0.013, D 0.015); Bonferroni
  97.5% upper bounds; three-arm horizon rule (owner 9); published-lr table + qualifier (owner 11); pure-val-MSE
  SELECTION-SENSITIVE suffix; exact extension trigger script; failed/BLOCKED rules; all r2 engineering fixes.
- Review r3 (`_design_review_r3.json`, raw lenses `_r3/`): foregone, fairness, leakage, statistics, answer_built_in
  (Fable) all LICENSED_WITH_FIXES, 0 blocking. Foregone probe: tfps_gauss_unit routes like zero at init
  (A 0.2516/0.2505, s row range 0.005/0.008) → its Q2 likely EQUIVALENT/INCONCLUSIVE (not certain).
- 0 runs scored. Nothing frozen.

## Next
1. Apply the r3 fixes in `_design_review_r3.json` → `fixes_before_freeze` mechanically (no new review round).
   Note: the DEPENDS ON D INIT label is renamed (logic of owner decision 9 unchanged).
2. Step (c): wrappers + self-tests + timing smoke on a seed outside S and the extension seeds, test lines filtered.
3. Step (d) FREEZE only after the Table 1 PDF check (headers + full H=192/336 rows) is recorded in PAPER_EXTRACT.md.

## Patches in force
- None used for scoring. Planned patches listed in PREREG.md §8.

## 2026-11-27 ~06:15-07:00 UTC — step (b) close-out: r3 fixes applied → PREREG r3.1 (no new review round)
- All 13 `fixes_before_freeze` items of `_design_review_r3.json` applied to PREREG.md (r3.1). Main changes:
  - Q3 horizon fallback renamed `NOT CONSISTENT ACROSS TFPS ARMS (gauss, unit, zero)`; direction printed ("<X> AT BOTH
    GAUSSIAN SCALES, <Y> WITH D = 0"); "Q2 detects no D effect" note; new rule 0: baseline BLOCKED → BLOCKED, TFPS arm
    BLOCKED → `INCOMPLETE`. Owner decision 9 logic unchanged (rule 2 untouched).
  - Published-lr and SELECTION-SENSITIVE qualifiers now on every horizon label; "holds" defined. Choice made here:
    "identical label" compares base label AND per-arm contents (stricter than "base label"; flags more often).
  - Non-gating baselines (checked m_{H,B} <= 0) removed from rules/qualifiers/trigger, Bonferroni k adapts,
    DESCRIPTIVE ONLY if none; m_H fallback.
  - FREEZE precondition widened: column headers + full ETTh1 H=192/336 rows (9 models) + input-length context.
    "margins per unverified extract" dropped; if Gustav orders FREEZE without the check, only margin-free rules
    (Fable wording incl. GAIN DETECTED OVER BOTH), all baselines gating "unverified", no anchor, no extension.
  - Input-length qualifier on negative Q3 labels and TFPS ANCHOR DRIFT if the PDF shows (or can't settle) a searched L.
  - D generator seeds 30000/40000+seed → 900000/910000+seed; seed-set disjointness self-test.
  - Step (c) timing/self-test runs: seed 999, test-line filter, result files deleted; feas_*.log H=96 exposure disclosed.
  - Unit-arm init numbers corrected (A 0.251/0.250, row range 0.005/0.008); s_unit = s_N01(z/sqrt 112); xavier note.
  - Reachability numbers (Q2 sub-label, paired Q2b, n=4, n=10), r3 Monte Carlo figures, exact extension formula,
    Q2b n after extension, k-failure suffix, glosses, base-label item rule, draw-averaged routing measurement,
    pure-val-MSE candidate set + tie rule, instrumentation-unchanged self-test, DLinear-copy and float32 notes,
    anchor variance + draw variance, arm-1 up to 100 epochs (budget now ~14-24 CPU-h at n = 5).
- Also taken from lens text beyond the 13 items: anchor s^2 + mean draw variance; "Welch conservative under pairing";
  lr grid as printed.
- Transcription check by one subagent (not a review round): all 13 items APPLIED/PARTLY; its 5 inconsistency notes
  (fallback trigger/gating, seed-set wording, Q2b rule 0, MC figures' rho, this STATE reference) fixed in the same step.
- 0 runs. Nothing frozen. No test number seen.

## Next
1. Step (c): wrappers + self-tests (§8 list) + timing smoke at seed 999 with the test-line filter.
2. Step (d) FREEZE only after the PDF check (headers, full H=192/336 rows, input-length context) is in PAPER_EXTRACT.md.

## Patches in force
- None used for scoring. Planned patches listed in PREREG.md §8.

## 2026-11-27 ~09:12-10:00 UTC — step (c) part 1: wrappers + self-tests + timing (seed 999, test lines filtered)
- New files: `rc_tfps.py` (TFPS wrapper, patches P1-P7), `rc_patchtst.py` (official-PatchTST baseline wrapper, B1-B2),
  `extension_trigger.py` (+ `--selftest`), `selftest_tfps.py` (fast / train / resume), `timing_tfps.py`. Results in
  `audits/tfps/selftest/*.json|log`. **35/35 self-tests PASS** (fast 20 incl. extension_trigger 7/7 as one, train 7, resume 2,
  baselines 6; 0 `"pass": false` in any file).
  - D init: time-D = Generator(900000+seed), freq-D = Generator(910000+seed); gauss sd 1.008; unit = gauss/√112
    exactly (col norm 1.005); zero stays 0 after 3 Adam steps (grad exactly 0); 177 non-D tensors bitwise equal across
    the 3 arms; global RNG equal after construction. Seed sets disjoint.
  - Router: noise-off ⇒ indices == argmax s; noise-on eval path bit-identical to released forward; train path ignores
    the switch; recording eval-only. **A ceiling (one-hot s, n = 4, 100k rows) = 0.616**; uniform s → 0.250.
  - Eval pass: N = 2689 rows at H=192 (released loader 2688); prefix reproduces drop_last pass exactly (diff 0);
    per-window mean == utils.metrics MSE/MAE (diff ~1e-7).
  - Instrumentation on vs off (2 epochs, TFPS and both baselines): checkpoint state_dict + torch/numpy RNG bitwise equal
    (val losses compared only on instrumented runs). EarlyStopping record, val-hook RNG replay (val loss + val MSE to
    1e-6), pure-val-MSE checkpoint, pre-FREEZE cleanup: PASS.
  - Baselines: released-loader score == repo's printed test MSE (diff 1.1e-7 PatchTST, 4.1e-8 DLinear; only the diff
    was recorded/seen).
  - **Resume (P7, `resumable_train`):** each tool call is capped at 10 min, so a full TFPS run (~27 min) cannot run
    in one foreground call → per-epoch resumable state was added (PREREG §8 provision). Tests: resumable train() ==
    released train() bitwise (4 epochs: checkpoints, val losses, RNG); run interrupted after EVERY epoch (4 processes)
    == uninterrupted, bitwise. Every pre-FREEZE run used resumable or released train at seed 999.
- Timing (seed 999, filtered; only val numbers seen): TFPS ≈ 16 s/epoch at H=192 and ≈ 17 s/epoch at H=336 incl. val +
  test passes; PatchTST ≈ 7 s/epoch, DLinear < 1 s/epoch (H=192). **tfps_gauss H=192 ran all 100 epochs** (released
  criterion best at epoch 100; val loss 5.80 → 3.28, aux share of val loss ≈ 0.67 throughout; mean column norm of D
  10.6 → 8.1; val MSE minimum ≈ epoch 11) — as PREREG §5 predicted for arm 1. ~27 min per arm-1 run.
- Q1 (bare EDESC construction in 10 fresh processes, outside the training pipeline): 5 of 12544 entries non-finite
  every time. Not the PREREG Q1 measurement (which is in the released run context, earlier 4586-4635); Q1 proper
  still to do.
- No H = 192 / 336 test number was seen. Test lines dropped by the filter; no test pass run in TFPS pre-FREEZE mode.
- Patch sha256 (12): rc_tfps af7746404ad4, rc_patchtst 963d25b03b89, extension_trigger a63941f9b45c (rc_tfps:
  one comment line changed after the train/resume tests; fast tests re-run after it, 20/20).

## Patches in force (none used for scoring yet)
- P1 `layers/Cluster.py:30` EDESC.__init__ D init; P2 `layers/SparseMoE.py:57-66` router eval-only noise switch +
  recording; P3 `utils/tools.py:40-69` (via `exp/exp_main.py:4,152`) EarlyStopping subclass; P4 `exp/exp_main.py:73-135`
  vali hook (val loader only); P5 `exp/exp_main.py:50-52` _get_data remembers val loader; P6 `exp/exp_main.py:277-397`
  test → audit scoring pass; P7 `exp/exp_main.py:137-275` train → resumable_train (bitwise-tested equal).
  Baselines: B1 EarlyStopping subclass, B2 test → scoring pass (official repo). num_workers 0 via argv for all.

## Next (step c, part 2)
1. Per-epoch "mean row range of s per router (val pass)" (PREREG §6) is not yet logged; add + self-test.
2. Battery driver: per-run dirs `runs/<arm>/<H>/seed<k>/` with provenance.json, metrics.json (repo vs recomputed),
   per_window_mse.npy, draws, routing summaries (A, H_r), nonoise and valmse scores; segment-resumable.
3. Q1 proper (10 fresh processes, released pipeline, D non-finite count).
4. Epochs-to-stop for tfps_gauss_unit / tfps_zero at H=192 and H=336 (seed 999), then the budget line.
5. Then step (d) FREEZE — still needs the PDF check in PAPER_EXTRACT.md.

## 2026-11-27 ~12:13-13:00 UTC — step (c) part 2: battery driver, s row-range log, Q1 proper, epochs-to-stop
- `rc_tfps.py` (sha256 5f5237c52240) changes: (1) P4 val hook now also logs the mean row range of s per router per val
  pass (eval-only router recording; numpy copies, no RNG use) — `s_rowrange_time|freq` in epochs.json; (2) new P8
  `exp/exp_main.py` Exp_Main.__init__ wrapper names the two routers time/freq (plain attribute, not in state_dict).
  **Bug found and fixed by the driver self-test:** before P8 the routers were unnamed in the real run pipeline (only the
  self-test builder named them), so scoring-pass routing records of time and freq would have been merged under "?";
  (3) P6 now runs the repo's released `test()` first (RNG as training left it, as run_longExp.py does) and captures
  its printed MSE/MAE via `exp_main.py:5 metric` (§6 "repo's own printed metric"); then the audit scoring pass.
  Repo's results/ (pred.npy) and test_results/ are deleted from the run dir.
- New `run_battery.py` (a7fbf797d049): one (arm, H, seed[, lr]) per call; `runs/<arm>/<H>/seed<k>[/lr<x>]/` with
  provenance.json (all §7 fields, per-segment start/end UTC + pid, file sha256s), metrics.json (per-draw, mean,
  released-loader score, nonoise, pure-val-MSE, routing A/H_r/row max/range averaged over the K=5 draws, repo-printed
  metric, training summary incl. D_maxabs_final and aux share), per_window_mse.npy (float32; per window mean of the 5
  draws), per-draw / nonoise / valmse arrays. TFPS runs resume by segment; baselines restart from scratch. Refuses any
  seed ≠ 999 unless a `FREEZE tfps` commit exists and PREREG.md is unchanged since; refuses to overwrite a finished run.
- New `selftest_battery.py` (d6e07b3978ed): **24/24 PASS** (seed 999, H=192, 2 epochs, scores the VAL split only —
  set_type 1 asserted; no test number computed): refusals (scored seed pre-FREEZE ×2, overwrite), segment+resume
  through the driver, provenance fields, float32 per-window files K×N, score recomputed from saved arrays < 1e-6,
  pure-val-MSE files, s row-range logged, no .pth/prediction arrays in run dirs, routing per named router, tfps_zero
  premise (D stays 0, s row range exactly 0 in val log and scoring pass, A 0.249/0.251, H_r ≈ ln 4), noise-off A == 1,
  baselines per-lr dirs + recompute.
- Re-run after the rc_tfps change: `selftest_tfps.py` fast 20/20, train 7/7 (instrumentation on/off still bitwise),
  resume 2/2 (bitwise). Baseline wrapper unchanged (6/6 from part 1 stands).
- **Q1 proper** (`q1_released.py`, b10891078429; released run_longExp.py via runpy, H=192 argv, seed 2021, no audit
  patch, stopped at Exp_Main.train entry, no data): **10/10 fresh processes give a non-finite D**; time-D 616-736 of
  12544 entries non-finite every time, freq-D 0 in 9 processes and 683 in 1. (Earlier figures differ: r1 4586-4635
  in another context; part-1 bare EDESC 5.) Descriptive only; says nothing about the authors' D. `selftest/Q1_proper.json`.
- Epochs-to-stop at seed 999 (test lines filtered, no test number seen; only val quantities read):
  tfps_zero 192: stop 23 (best 3), ~6 min; tfps_gauss_unit 192: stop 23 (best 3), ~6.3 min; tfps_zero 336: stop 22
  (best 2), ~7.4 min; tfps_gauss_unit 336: stop 23 (best 3), ~7.8 min. Val-pass mean row range of s, unit arm:
  H=192 rises 0.005→0.017 (time) / 0.009→0.031 (freq); H=336 falls to ~0.001-0.004. tfps_zero: exactly 0 every epoch.
  tfps_gauss 336 still measured on 2 epochs only (assume ~100 epochs × 17-20 s ≈ 30 min).
- **Budget line (measured):** per seed per H ≈ 27-30 min (gauss) + ~7 (unit) + ~7 (zero) ≈ 42-45 min TFPS → 30 TFPS
  runs ≈ 7-7.5 CPU-h; baselines 80 runs ≈ 2-3.5 CPU-h (PatchTST ~7 s/epoch, DLinear < 1 s/epoch, stop epochs not yet
  measured); eval ≈ 0.3 CPU-h → **≈ 10-11 CPU-h at n = 5**, up to ~2× with the extension. At ~45 min of runs per
  session that is ~14 sessions at n = 5.
- 0 scored runs. Nothing frozen. No H = 192 / 336 test number seen.

## Patches in force (none used for scoring yet)
- P1-P7 as in part 1; P4 extended (s row range); P6 extended (released test() first, metric captured);
  **P8** `exp/exp_main.py` Exp_Main.__init__ (wrapper after the released init): router naming only.
  Baselines B1-B2 unchanged. num_workers 0 via argv for all.

## Next
1. Step (c) is complete. Step (d) FREEZE — blocked on the PDF check (Table 1 headers + full ETTh1 H=192/336 rows
   for all 9 models + input-length context) recorded in PAPER_EXTRACT.md, then recompute m_{H,B}, m_H, anchors.
   FREEZE should also add the measured budget line to PREREG §8 (transcription, not design).
2. Then step (e) chunks via `run_battery.py` (first chunk suggestion: seed 2021 H=192, zero + unit + baselines).

## 2026-11-28 ~03:12-03:30 UTC — step (d): FREEZE
- Precondition met: Table 1 check recorded in `PAPER_EXTRACT.md` by the reviewing chat (00:55 London, Gustav present),
  against the arXiv HTML v2 (disclosed in PREREG: not a cell-by-cell PDF comparison). Queue said next step (d).
- PREREG.md r3.1 frozen in the commit `FREEZE tfps — ...`. Transcription-only changes: FREEZE record + disclosure at
  the top; precondition status DONE (§ head); input-length qualifier resolved as NOT applied (Table 1 at L = 96);
  m_{H,B} / m_H / anchors recomputed by script from the checked cells → unchanged (192: P 0.037, D 0.011, m_H 0.011
  vs DLinear; 336: P 0.013, D 0.015, m_H 0.005 vs FITS); all four pairs gating; Q1-proper numbers in §2; measured
  budget (~10-11 CPU-h at n = 5) in §8. No rule, threshold, arm, seed or selection changed.
- New `frozen_margins.json` (margins, anchors, gating list, input-length flag) for the trigger / aggregation.
- 0 scored runs. No H = 192 / 336 test number has been seen.

## Next
- Step (e) chunks via `run_battery.py` (driver now accepts seeds in S; check `frozen()` is True first). First chunk
  suggestion (STATE c2): seed 2021, H=192: tfps_zero + tfps_gauss_unit (~13 min) + baseline lr grid (8 runs), within ~45 min.

## Patches in force
- Unchanged from step (c) part 2 (P1-P8, B1-B2; num_workers 0 via argv).

## 2026-11-28 ~06:12-06:58 UTC — step (e) chunk 1: seed 2021, H = 192
- `frozen()` True before any scored call. Driver/wrapper files unchanged since FREEZE (provenance `rc_commit` 93b0103,
  `rc_dirty` true only because `runs/` was untracked during the runs).
- Finished (9 of 110 planned runs at n = 5): tfps_zero 192/2021 (stop 23, one segment, ~6.0 min), tfps_gauss_unit
  192/2021 (stop 23, ~5.8 min); dlinear 192/2021 at lr 1e-4, 5e-4, 1e-3, 5e-3 (all 4); patchtst 192/2021 at lr 5e-3,
  1e-3, 5e-4 (~3.1 / 2.7 / 6.1 min). 0 failed runs; no .pth in run dirs; runs/ = 1.1 MB.
- Not run this chunk: patchtst 192/2021 lr 1e-4 (could exceed the 10-min tool cap; baselines don't resume) and
  tfps_gauss 192/2021 (~27-30 min, needs a session of its own, resumable by `--budget`).
- No aggregation, no comparison between arms, no verdict computed (step f/g come later).

## Open issue found this chunk (engineering, metadata only; no code changed after FREEZE)
- **Baseline `es.json` / metrics.json truncate the val-loss list at the best epoch.** `rc_patchtst.py:48-55`
  (RCEarlyStopping) dumps es.json only when the checkpoint improves, so metrics.json `val_losses`, `stop_epoch` and
  `val_finite` cover epochs 1..best, not 1..stop. Evidence: patchtst lr1e-3 es_len 7 vs 27 epochs in train.log;
  lr5e-3 10 vs 30; lr5e-4 39 vs 59; dlinear lr1e-4 87 vs 100, 5e-4 40 vs 60, 1e-3 24 vs 44, 5e-3 10 vs 30.
  Selection inputs are unaffected (`val_loss_min_repr` and `best_epoch` are written at the improving epoch, which is
  exactly what §5 selects on). The §5 rule "a run with any non-finite val loss cannot be selected" is NOT checked by
  `val_finite` past the best epoch; checked by hand from train.log this chunk: all 7 runs print finite Vali Loss at
  every epoch (0 non-numeric). The 2-epoch self-test could not catch it (best == stop).
  Proposed fix (needs Gustav's OK, wrapper change after FREEZE): write the per-epoch val list every epoch; for runs
  already finished, step (f) reads the full list from train.log (7-decimal print) as the second code path.

## Next
- Chunk 2 suggestion: patchtst 192/2021 lr 1e-4 (alone, timeout ≤ 590 s; if it times out, report and split
  differently), then tfps_gauss 192/2021 with `--budget 540` segments (resumable) across the rest of the session.

## Patches in force
- Unchanged from step (c) part 2 (P1-P8, B1-B2; num_workers 0 via argv).

## 2026-11-28 ~09:12-09:50 UTC — step (e) chunk 2: seed 2021, H = 192
- `frozen()` True before any scored call; driver/wrapper files unchanged since FREEZE (provenance `rc_commit` 2452a59,
  `rc_dirty` true only from untracked `runs/` and `aborted/`).
- **tfps_gauss 192/2021 finished** in 3 resumable segments (`--budget 540`: 09:23-09:32, 09:32-09:41, 09:41-09:47Z;
  epochs 1-38, 39-77, 78-100). stop 100, released-criterion best epoch 97, pure-val-MSE best epoch 3, aux share of val
  loss at saved epoch 0.675, all 100 val losses / val MSEs finite (val loss 5.86 → 3.30). ~24 min. As PREREG §5 predicted.
- **patchtst 192/2021 lr 1e-4 NOT finished: killed by the 10-min tool cap** (timeout 590 s at 09:22:55Z after 68 epochs,
  ~8.7 s/epoch; lr 1e-4 has not early-stopped by epoch 68, so it needs up to ~14.5 min). Baseline runs do not resume
  (`run_battery.py base` restarts from scratch), so this run cannot finish in a foreground call as the driver stands.
  Not a "failed run" under §5 (no non-finite value; infrastructure kill). Partial log moved out of `runs/` to
  `aborted/patchtst_192_seed2021_lr1e-4_toolcap_0913Z.log` (test lines not read); run dir removed so it can be re-run.
- Totals: **10 of 110 scored runs finished, 0 failed, 1 aborted by tool cap.** Seed 2021 H=192 complete except patchtst
  lr 1e-4. No aggregation, no comparison between arms, no verdict computed. No .pth in run dirs; runs/ = 1.5 MB.

## Open issues
- (new, blocking for PatchTST lr 1e-4 at every seed/H, likely 10 runs at n = 5) baseline runs longer than ~9.5 min can't
  run under the tool cap. Options for Gustav: (A) OK a post-FREEZE engineering change giving `rc_patchtst.py` /
  `run_battery.py base` per-epoch resume (as P7 for TFPS, with a bitwise self-test interrupted == uninterrupted);
  (B) allow one background `nohup` process per session for these runs, polled in the foreground (protocol says
  foreground only). Without a decision the agent keeps running everything else and leaves the lr 1e-4 runs for last.
- (from chunk 1, still open) baseline es.json truncates the val-loss list at the best epoch — DECISION NEEDED in DIGEST.

## Next
- Chunk 3: seed 2021 H=336: tfps_zero + tfps_gauss_unit (~15 min) + dlinear lr grid (4) + patchtst lr 5e-3/1e-3/5e-4
  (timeout ≤ 590 s each; any that hits the cap goes to `aborted/` and waits on the decision above).

## Patches in force
- Unchanged from step (c) part 2 (P1-P8, B1-B2; num_workers 0 via argv).

## 2026-11-28 ~12:12-12:50 UTC — step (e) chunk 3: seed 2021, H = 336
- `frozen()` True before any scored call; driver/wrapper files unchanged since FREEZE (provenance `rc_commit` c3666b1,
  `rc_dirty` true only from untracked `runs/`).
- Finished (10 runs this chunk): tfps_zero 336/2021 (stop 22, ~6.8 min), tfps_gauss_unit 336/2021 (stop 22, ~6.5 min),
  dlinear 336/2021 lr 1e-4 / 5e-4 / 1e-3 / 5e-3 (all 4, ~2 min total), patchtst 336/2021 lr 5e-3 / 1e-3 / 5e-4
  (~2.4 / 2.5 / 3.4 min), **tfps_gauss 336/2021** in 2 resumable segments (`--budget 540`: epochs 1-32, 33-42; ~13 min).
- tfps_gauss at H = 336 early-stopped at epoch 42 (released best epoch 22, pure-val-MSE best epoch 2), unlike H = 192
  (100 epochs). Aux share of val loss at the saved epoch 0.012 (H = 192: 0.675); D relative change ~0.93 (time/freq),
  max |D| at end ~1.55. Plausibly the H = 336 config's lr 5e-3 (vs 5e-4 at 192) decays the D constraint terms fast;
  descriptive only, not checked further. Budget consequence: H = 336 gauss runs cost ~13 min, not ~30.
- Checks on all 10: per-window file N = 2545, every Vali Loss in train.log finite (count per log = full epoch list,
  baseline es.json still truncated at best epoch — known issue), no .pth in run dirs; runs/ = 2.8 MB.
- Not run: patchtst 336/2021 lr 1e-4 (waits on the tool-cap decision; same issue as 192).
- Totals: **20 of 110 scored runs finished, 0 failed, 1 aborted (patchtst 192/2021 lr 1e-4).** Seed 2021 complete
  except the two patchtst lr 1e-4 runs. No aggregation, no comparison between arms, no test number read, no verdict.

## Open issues
- Unchanged: (1) PatchTST lr 1e-4 vs the 10-min tool cap (decision A/B in DIGEST); (2) baseline es.json val-list
  truncation (metadata fix decision).

## Next
- Chunk 4: seed 42 (next in S = {2021, 42, 123, 456, 7}), H = 192: tfps_zero + tfps_gauss_unit + dlinear grid + patchtst lr
  5e-3/1e-3/5e-4, then tfps_gauss 192 segments as time allows (~24 min; resumes across sessions).

## Patches in force
- Unchanged from step (c) part 2 (P1-P8, B1-B2; num_workers 0 via argv).

## 2026-11-28 ~15:12-16:10 UTC — step (e) chunk 4: seed 42, H = 192
- `frozen()` True before any scored call; driver/wrapper/PREREG unchanged since the FREEZE commit (git diff empty);
  provenance `rc_commit` ae78f08, `rc_dirty` true only from untracked `runs/`.
- Finished (10 runs): tfps_zero 192/42 (stop 24, ~6.2 min), tfps_gauss_unit 192/42 (stop 24, ~6.5 min), dlinear 192/42
  lr 1e-4 / 5e-4 / 1e-3 / 5e-3 (all 4, ~2.7 min total), patchtst 192/42 lr 5e-3 / 1e-3 / 5e-4 (~3.8 / 3.2 / 5.9 min),
  **tfps_gauss 192/42** in 3 resumable segments (`--budget 540`; epochs 1-37, 38-74, 75-100; ~25 min): stop 100,
  released best epoch 81, pure-val-MSE best epoch 4, aux share of val loss at saved epoch 0.674, D rel change ~0.31-0.32,
  max |D| end ~3.3-3.7. Same pattern as seed 2021 at H = 192.
- Checks on all 10: per-window file N = 2689 float32, every Vali Loss in train.log finite (count = full epoch list:
  dlinear 100/92/41/53, patchtst 52/29/32 — es.json still truncated at best epoch, known issue), no .pth; runs/ = 4.2 MB.
- Not run: patchtst 192/42 lr 1e-4 (tool-cap decision pending).
- Note: TFPS resume state lives in `work/ckpt/` (gitignored, lost with the container), so a tfps_gauss 192 run (~25 min)
  must finish within one session; H = 336 gauss (~13 min) is easier.
- Totals: **30 of 110 scored runs finished, 0 failed, 1 aborted (patchtst 192/2021 lr 1e-4).** No aggregation, no
  comparison between arms, no test number read, no verdict.

## Open issues
- Unchanged: (1) PatchTST lr 1e-4 vs the 10-min tool cap (decision A/B in DIGEST) — now 3 runs waiting; (2) baseline
  es.json val-list truncation (metadata fix decision).

## Next
- Chunk 5: seed 42, H = 336: tfps_zero + tfps_gauss_unit + dlinear grid + patchtst lr 5e-3/1e-3/5e-4 + tfps_gauss 336
  (~13 min), all within one session (~45 min).

## Patches in force
- Unchanged from step (c) part 2 (P1-P8, B1-B2; num_workers 0 via argv).

## 2026-11-28 ~18:12-19:00 UTC — step (e) chunk 5: seed 42, H = 336
- `frozen()` True before any scored call; driver/wrapper/PREREG unchanged since the FREEZE commit (git diff empty);
  provenance `rc_commit` b56ec31, `rc_dirty` true only from untracked `runs/`.
- Finished (10 runs): tfps_zero 336/42 (stop 24, ~7.6 min), tfps_gauss_unit 336/42 (stop 22, ~7.0 min),
  **tfps_gauss 336/42** in 2 resumable segments (`--budget 540`; epochs 1-31, 32-47; ~14 min): stop 47, released best
  epoch 27, pure-val-MSE best epoch 3, aux share of val loss at saved epoch 0.005, D rel change ~0.94, max |D| end
  ~1.35 / 0.96 — same pattern as seed 2021 at H = 336 (early stop, small aux share). dlinear 336/42 lr 1e-4 / 5e-4 /
  1e-3 / 5e-3 (all 4, ~2 min total), patchtst 336/42 lr 5e-3 / 1e-3 / 5e-4 (~3.5 / 2.5 / 3.0 min).
- Checks on all 10: per-window file N = 2545 float32, every Vali Loss in train.log finite (counts dlinear 62/62/48/35,
  patchtst 34/25/30, tfps 47/22/24; baseline es.json still truncated at best epoch — known issue), no .pth; runs/ = 5.5 MB.
- Not run: patchtst 336/42 lr 1e-4 (tool-cap decision pending).
- Totals: **40 of 110 scored runs finished, 0 failed, 1 aborted (patchtst 192/2021 lr 1e-4).** Seeds 2021 and 42
  complete except patchtst lr 1e-4 (4 cells waiting). No aggregation, no comparison between arms, no test number read,
  no verdict.

## Open issues
- Unchanged: (1) PatchTST lr 1e-4 vs the 10-min tool cap (decision A/B in DIGEST) — now 4 runs waiting; (2) baseline
  es.json val-list truncation (metadata fix decision).

## Next
- Chunk 6: seed 123, H = 192: tfps_gauss 192 first (~25 min, must finish in one session), then tfps_zero +
  tfps_gauss_unit, then dlinear grid + patchtst lr 5e-3/1e-3/5e-4 as time allows (leftovers roll to the next chunk).

## Patches in force
- Unchanged from step (c) part 2 (P1-P8, B1-B2; num_workers 0 via argv).
