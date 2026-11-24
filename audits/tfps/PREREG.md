# TFPS audit — pre-registration (DRAFT, not frozen; nothing below has been scored)

Paper: "Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift" (TFPS),
NeurIPS 2025, arXiv 2410.09836. Code: github.com/syrGitHub/TFPS (shallow clone, 2026-11-24).

## What the paper claims (ETTh1, input 96, Table 1, single runs)
TFPS MSE 0.398 / 0.423 / 0.484 / 0.488 at H = 96 / 192 / 336 / 720; best baseline per horizon 0.387 (iTransformer,
TSLANet) / 0.434 (DLinear) / 0.491 (TSLANet, iTransformer) / 0.492 (TFDNet-IK). So on ETTh1 the claim rests on
H = 192, 336, 720, with margins 0.011 / 0.007 / 0.004. The mechanism: subspace bases D give each patch an affinity s
that routes it to pattern-specific experts; ablation "PI → Linear" raises H=96 MSE 0.398 → 0.414.

## What the released code does (verified 2026-11-24)
- `layers/Cluster.py:30`: `self.D = Parameter(torch.Tensor(n*d, n*d))` allocates without initialising.
  `Initialization_D` (k-means + SVD, as in the EDESC method TFPS builds on) is imported at `exp/exp_main.py:24` and
  never called. D's starting values are whatever was in memory: 3 fresh processes, same seed: 4635 / 4586 / 4616
  NaN entries of 12544, finite absmax 6e-41 / 1.9e25 / 6e-41.
- With D = 0 exactly, every affinity is eta/(eta+1) before normalisation, so s is uniform, top-1 routing always picks
  the same expert, and the gradient with respect to D is zero, so D never leaves 0 (to be confirmed empirically).
- Test loader `drop_last=True`, batch 128: the last 97 of 2785 test windows are dropped at H=96, 113 of 2161 at H=720.

## Questions
Q1. Is the released code runnable as published? (Already answered on this machine: NaN from epoch 1. Report as-is.)
Q2. Does the ETTh1 result depend on D's initialisation? Arms: D_zero (routing provably degenerate), D_orth
    (orthogonal init: unit-norm, mutually orthogonal bases, matching the paper's constraints).
Q3. With every test window kept, does TFPS beat PatchTST and DLinear trained in the same code, same seeds, same
    lr search?

## Design (to be adversarially reviewed before freezing)
- ETTh1, input 96, H ∈ {192, 336} (the horizons that carry the ETTh1 claim), seeds {2021, 42, 123, 456, 7}.
- TFPS: released script config (16 experts, top-1, lr 5e-4, batch 128, 100 epochs, patience 20), arms D_zero / D_orth.
- Baselines in TFPS's own repo (`models/PatchTST.py`, `models/DLinear.py`), same data pipeline, batch 128, lr chosen
  per seed on validation from {1e-4, 5e-4, 1e-3, 5e-3}; TFPS keeps its tuned config (advantage TFPS).
- Every arm scored twice: with the released test loader (drop_last=True) and with every test window kept.
- Metric: test MSE at the checkpoint the repo's own early stopping selects on validation.

## Pre-registered verdicts (per horizon; SE from 5 seeds, Welch)
- MECHANISM-INERT if |mean(D_zero) − mean(D_orth)| < 2·SE and D_zero's routing is constant.
- SURVIVES if mean(TFPS best arm) < min(mean PatchTST, mean DLinear) − 2·SE.
- SHRINKS if TFPS is lower than both baselines but by less than 2·SE.
- VANISHES if TFPS is not lower than the best baseline.
- Anchor (not gating): paper's TFPS numbers; report our D_orth/drop_last=True distance from them.

## Not claimed
Nothing about other datasets, other horizons, or whether the authors' own runs had finite D. This tests the released
code on ETTh1 only.
