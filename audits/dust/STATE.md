# DUST-1 — state

## Done
- 2026-11-27 ~15:13-15:25 UTC, step (a): prereg r1 drafted (`PREREG.md`) from `SOURCE_NOTES.md` + pinned code
  b20f7c03; design review r1 → **NOT_LICENSED (not foregone)**. 5/5 lenses (Fable ran answer-built-in)
  LICENSED_WITH_FIXES, 18 blocking items, consolidated into 8 in `_design_review.json` (raw lenses in `_r1/`).
- CPU cost probe `timing_cpu.py` (random tokens, seed 999, no data, nothing scored) → `timing_cpu.log`: 37.7M params;
  fwd+bwd of one 2048-token sequence ≈ 2.9-3.2 s → 8-sequence step ≈ 23.5 s with accumulation; full val pass ≈ 9 min.
  A batched 8×2048 backward OOMs in 7 GB, so every arm needs micro-batch 1 + gradient accumulation (exact, per the
  leakage lens; clip once on the summed gradient).
- Code facts found: the released backprop baseline lacks Dust's value_embeds (0.3) / lambdas (0.03) lr groups
  (`baselines/backprop.py:22-28` vs `dust.py:453-467`); both scripts refuse non-CUDA (`backprop.py:59-60`).

## Next
- Blocked on: P1 (paper check → `audits/dust/PAPER_EXTRACT.md`: measured 1M Dust and backprop cells with spreads,
  the 1M claim sentence, backprop tuning grids, use of val), P2 (data files), and owner decisions in
  `_design_review.json`. Then step (b): prereg r2 applying the 8 consolidated blocking fixes, then review r2.
- Queue says drafting + r1 only, so no further step until Gustav updates row 2.

## Patches in force
- None (no wrapper yet). Planned: CPU path (remove CUDA refusal), gradient accumulation, val-only/test-refusal guard.
