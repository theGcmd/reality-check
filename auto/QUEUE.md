# Reality Check queue

**Gustav controls this file.** Agents work the FIRST row that is `approved: yes` and `READY` or `IN_PROGRESS`, one
step per session, strictly in order. Agents may change only the `status` column (to `IN_PROGRESS`, `BLOCKED`, or
`DONE-CANDIDATE`).

Approval basis: Gustav approved the plan on 2026-11-24 (step 2 = run pre-registered checks on the shortlist leads,
blind panel, then he decides about contacting authors). Each item still goes through design review and a FREEZE
before anything is scored.

| # | item | prereg | approved | status |
|---|------|--------|----------|--------|
| 1 | TFPS-1 — does TFPS's ETTh1 result depend on its uninitialised routing matrix, and does it survive same-harness PatchTST / DLinear with every test window kept? | `audits/tfps/PREREG.md` | yes | READY (owner, 2026-11-25 03:00 London): step (b) rewrite PREREG.md per OWNER DECISIONS 1-8, then design review r2 |
| 2 | TIMEDISTILL-1 — does TimeDistill's ETTh1 average (0.429) beat PatchTST run with PatchTST's own published config in TimeDistill's harness? | `audits/timedistill/PREREG.md` | yes | BLOCKED (design review r1 NOT_LICENSED, not foregone; owner decisions needed, see audits/timedistill/_design_review.json; then step (b) rewrite + review r2) |
| 3 | GTR-SOLAR-1 — does GTR's Solar gain survive baselines given the same lr search GTR got? | to be written: `audits/gtr/PREREG.md` | yes | BLOCKED (prereg r1 drafted; design review r1 NOT_LICENSED, Fable lens FOREGONE on verdict bands; owner decisions needed, see audits/gtr/_design_review.json; then step (b) rewrite + review r2) |

Interleaving: while an item is waiting on a long battery, the agent keeps to the first runnable row; it does not
jump ahead to save time.

## CPU budget notes (measured 2026-11-24, this container: 2 cores)
- TFPS, ETTh1, input 96: ~14 s per training epoch, ~25-30 s per epoch with its per-epoch val+test passes.
- PatchTST at TimeDistill's released config (d_model 512, input 720): 752 s per epoch. Run that anchor arm with ONE
  seed here unless Gustav provides a GPU.
- GTR: ~3 s per epoch on ETTh1.
