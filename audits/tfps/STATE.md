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
- BLOCKED on Gustav's decisions (see DIGEST 2026-11-24 DECISION NEEDED). Then step (b): rewrite PREREG.md per review
  and decisions, re-review (round 2 of max 3).

## Patches in force (wrapper `run_tfps_dinit.py`, not yet used for scoring)
- `layers.Cluster.EDESC.__init__` (layers/Cluster.py:30 allocation) — D init asis/zero/orth.
- `data_provider.data_factory.DataLoader` (data_factory.py:17-20) — test drop_last off when TFPS_TEST_DROP_LAST=0
  (currently crashes downstream in test(); see above).

## 2026-11-24 20:00 London — owner decisions recorded
- PREREG.md now has OWNER DECISIONS 1-8. Row BLOCKED until 03:00 London 2026-11-25 (owner timing); then step (b).
