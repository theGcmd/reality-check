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
