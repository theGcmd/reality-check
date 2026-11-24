# Reality Check queue

**Gustav controls this file.** Agents work the FIRST row that is `approved: yes` and `READY` or `IN_PROGRESS`, one
step per session, strictly in order. Agents may change only the `status` column (to `IN_PROGRESS`, `BLOCKED`, or
`DONE-CANDIDATE`).

Approval basis: Gustav approved the plan on 2026-11-24 (step 2 = run pre-registered checks on the shortlist leads,
blind panel, then he decides about contacting authors). Each item still goes through design review and a FREEZE
before anything is scored.

| # | item | prereg | approved | status |
|---|------|--------|----------|--------|
| 1 | TFPS-1 — does TFPS's ETTh1 result depend on its uninitialised routing matrix, and does it survive same-harness PatchTST / DLinear with every test window kept? | `audits/tfps/PREREG.md` | yes | BLOCKED (design review r1 NOT_LICENSED (FOREGONE); awaiting Gustav's arm decisions, see DIGEST) |
| 2 | TIMEDISTILL-1 — does TimeDistill's ETTh1 average (0.429) beat PatchTST run with PatchTST's own published config in TimeDistill's harness? | `audits/timedistill/PREREG.md` | yes | READY (draft prereg; start with design review) |
| 3 | GTR-SOLAR-1 — does GTR's Solar gain survive baselines given the same lr search GTR got? | to be written: `audits/gtr/PREREG.md` | yes | READY (first step: draft the prereg from the paper and code, then design review) |

Interleaving: while an item is waiting on a long battery, the agent keeps to the first runnable row; it does not
jump ahead to save time.

## CPU budget notes (measured 2026-11-24, this container: 2 cores)
- TFPS, ETTh1, input 96: ~14 s per training epoch, ~25-30 s per epoch with its per-epoch val+test passes.
- PatchTST at TimeDistill's released config (d_model 512, input 720): 752 s per epoch. Run that anchor arm with ONE
  seed here unless Gustav provides a GPU.
- GTR: ~3 s per epoch on ETTh1.
