# GTR-SOLAR-1 — state

## Done
- 2026-11-25: PREREG.md draft r1 written from the paper (arXiv 2602.10847 HTML) and code (gtr @ d941619); commit 811b933.
- 2026-11-25: step (a) design review r1 -> **NOT_LICENSED**. 0 runs, nothing scored. 5 lenses (Fable on "answer built in"):
  foregone (session) not foregone; Fable FOREGONE on the verdict bands; statistics NOT_LICENSED; fairness NOT_LICENSED;
  leakage LICENSED. Details: `_design_review.json`.

## Key review findings
- Verdict bands overlap (SHRINKS vs VANISHES) and SHRINKS is unreachable with 5 seeds at ~2% seed CV (CI ~5-7 pp wide).
- ANCHOR censors the gain verdict exactly when GTR fails to reproduce; band too loose for a 5-seed mean; G at 3e-3 may
  get only 3 seeds.
- CycleNet's published Solar lr is 0.01 (per fairness lens reading ACAT-SCUT/CycleNet; NOT re-checked, repo not pinned),
  outside the paper's grid; C's scored number undefined. CycleNet result.txt 5-seed avg 0.2104 matches GTR Table 1 0.210.
- Paper (App. A.3) says baselines used their original settings, so the claim-as-stated test is G-released vs C-published.
- Harness: setting string lacks lr (run.py:128-136) -> wrapper must isolate checkpoints/results per run.
- No leakage found: val-only early stopping, scaler on train, cycle_index = position mod 144, all test windows kept.

## Next
- BLOCKED on owner decisions (see DIGEST 2026-11-25 / `_design_review.json` decisions_needed). Then step (b): rewrite
  prereg r2 (disjoint ordered bands, Welch t on log ratio, float64 two-path check at 1e-5, per-run isolation, line refs,
  GTR fixed Dropout 0.1 note, no CPU patch needed) and re-review.

## Patches in force
- None (wrapper not built).
