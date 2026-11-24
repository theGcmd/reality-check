# TIMEDISTILL-1 — state

## Done
- 2026-11-24 step (a) design review r1: **NOT_LICENSED** (not FOREGONE). 0 runs scored. Details: `_design_review.json`.

## Blocking issues found (all must be fixed in the r2 prereg)
1. Seeds cannot vary: `run.py:11-14` hardcodes `fix_seed=2021`; `--seed` (run.py:179) is never read. Needs a wrapper patch + self-test.
2. LR schedule/patience unpinned: harness default `lradj type1` halves lr every epoch (`utils/tools.py:13-14`), patience 5. Official PatchTST = its `type3` (const <3 epochs, then x0.9) + patience 100 (`patchtst_official/.../run_longExp.py:80,84`). TimeDistill's equivalent is `type4` (`utils/tools.py:22-23`); TimeDistill's own `type3` is a different rule.
3. P-official runs Time-Series-Library's PatchTST: attention and head dropout tied to `--dropout` (`models/PatchTST.py:56,71-72`), fixed sinusoidal PE, no residual attention, single final BatchNorm, `--fc_dropout/--head_dropout` unused.
4. The thresholds' basis "0.429 ± 0.004 (5 seeds, Table 8)" is not in arXiv v3 (Table 8 is a teacher ablation; no ± anywhere).
5. Verdict rules asymmetric; conflate baseline gap (P-released vs P-official) with headline (P-official vs 0.429). Proposed split F1/F2 with symmetric δ (see statistics lens).
6. No resume in the harness (`exp/exp_long_term_forecasting.py:153`; only best state_dict saved); every single-horizon run (~2.1-2.6 h) exceeds a session chunk. Needs a resume wrapper + equivalence self-test.
7. PREREG.md:8 misattributes drop_last=False and the "default configurations" quote to the paper (drop_last is code; quote is about teachers, App. C).

## Verified facts (no leakage)
- Selection on validation only (exp:354, reload :364-365); test evaluated each epoch but not used. Scaler train-only; drop_last False; test windows 2785/2689/2545/2161 at L=720; MSE on normalised scale.
- Table 12: PatchTST 0.428/0.476/0.543/0.546 (avg 0.498); TimeDistill 0.373/0.411/0.439/0.495 (avg 0.429).

## Cost (measured)
P-official ~95 s/epoch, 100 epochs/horizon ~2.6 h, 3 seeds x 4 H ~32 h; P-released ~8.4 h for 1 seed. Total ~40 CPU h.

## Next
Status BLOCKED pending owner decisions (see `_design_review.json` decisions_needed). Then step (b): rewrite prereg + review r2.

## Patches in force
None (no wrapper yet).
