# Reality Check (private)

Does a published ML result survive a fair test? Each audit takes a paper's own released code, reproduces the headline,
then runs the checks the paper skipped (baseline tuned as hard as the method, more seeds, matched budgets, every test
window kept, no selection on test) under a verdict written down before anything is scored: SURVIVES / SHRINKS /
VANISHES / INCONCLUSIVE.

- `SHORTLIST.md` — step 1: 10 time-series forecasting papers (2025-26), CPU feasibility, leads.
- `NOTES.md` — working notes and ground-truth pointers.
- `audits/<item>/` — prereg, wrappers, per-run files, candidate findings.
- `harness/` — pinned environment (`setup.sh`, `requirements.lock`, `repos.lock`), CPU redirect, smoke harness.
- `auto/` — the unattended agent loop: `AGENT_PROTOCOL.md`, `QUEUE.md` (owner-controlled), `DIGEST.md`, `STATE.md`.

Everything here is a candidate until re-derived from raw per-run files and reviewed. Nothing leaves this repo
without Gustav.
