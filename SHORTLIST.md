# Reality Check: step 1 shortlist (2026-11-24)

**Field:** long-term time-series forecasting (LTSF). The standard ETT benchmark (ETTh1/h2/m1/m2) comes from its official
GitHub repo and Solar/Electricity/Traffic/Exchange from a GitHub mirror, so audits run here without Google Drive.

**Kill point (step 1): fewer than 3 of 10 run on CPU → switch field.** Result: **PASSED** — 8 of the 10 below run end
to end on this CPU. TFPS trains only after its uninitialised matrix is given defined values; ReNF trains but its final
test step crashes as released.

**After defining TFPS's D** (1 seed, 3 epochs, H=96, feasibility only, not a result): D = 0 → test MSE 0.421,
D orthogonal → 0.419. The code is auditable once D is defined.

Smoke test = the repo's own ETTh1 script, first config, 1 epoch, on this 2-core CPU box, in a pinned Python 3.11 /
torch 2.4.1 / numpy 1.26 environment. No repo code edited; a CPU redirect maps hard-coded `.cuda()` calls to CPU.

| # | paper (venue) | runs on CPU? | s / epoch | leads for step 2 (candidates, not findings) |
|---|---|---|---|---|
| 1 | **TFPS** — pattern-specific experts (NeurIPS 2025) | runs, **loss NaN from epoch 1** | 14 | Cluster bases `D` that route the experts are **uninitialised memory** (`torch.Tensor(112,112)`); the k-means init routine is imported but never called. 3 fresh processes, same seed: ~4,600 of 12,544 entries NaN, values up to 1.9e25. Also: test loader drops the last partial batch (3.5–5.2% of test windows); TFPS grid-searched, baselines at "unified parameters"; single runs. |
| 2 | **TimeDistill** — MLP distillation (KDD 2026) | yes (one GPU-memory logging call stubbed) | 9 (iTransformer teacher, input 720) | Reports PatchTST at ETTh1 avg MSE **0.498** (input 720). Its PatchTST script uses 1 layer, width 512, 10 epochs; PatchTST's own ETTh1 setting is 3 layers, width 16, up to 100 epochs, and PatchTST's paper reports **0.413** (input 512). TimeDistill's headline is 0.429, so a properly configured PatchTST may beat it. Only TimeDistill has seed variance. |
| 3 | **GTR** — global temporal retrieval (ICLR 2026) | yes | 3 | Headline gain is on Solar (+8.2% MSE). LR searched for GTR only; baselines at published settings; only GTR has variance. Cheapest to audit. |
| 4 | **SiGMA** — single multi-scale operator (ICML 2026) | yes | 35 | Own ETTh1 table: not best at 3 of 4 horizons; "55 of 80 settings" vs 64 implied; 3 runs, no std. |
| 5 | **Dynamic TMoE** — drift-aware MoE (ICML 2026) | yes | 37 | ETTh1 0.429 vs 0.432 best baseline, no seeds or variance; grid search for method only. Drift detector updates are gated on `self.training` (checked: no test leak there). |
| 6 | **TimePerceiver** (NeurIPS 2025) | yes | 69 | Some baseline numbers copied from another paper; "54 of 64 std ≤ 0.0005" conflicts with its own Table 6. |
| 7 | **TimeCapsule** (KDD 2025) | yes | 23 | Same authors as ReNF; to inspect. |
| 8 | **KUP-BI** — retrieval plug-in (ICML 2026) | yes | 7 (plus ~70 s library build) | Library is train-only (checked in code). Joint-tuning arm tunes lr/batch for KUP-BI, not for the bare backbone. |
| 9 | **SARAF** — retrieval-augmented (KDD 2026) | yes | 7 (plus retrieval build) | ETTh1 0.415 vs RAFT 0.418, 3 runs, no std. Database train-only per paper. |
| 10 | **ReNF** (arXiv 2025) | trains; **final test crashes** as released | 32 | Lookback searched for ReNF only; 5-seed std reported as 0.000 everywhere. |

Does not run as released: **CAST-Norm** (KDD 2026) — `exp/exp_long_term_forecasting.py` imports `exp.exp_basic`, which
is not in the repository (single commit, 34 files).

Dropped: Selective Learning (NeurIPS 2025; listed code link is a different paper's repo), CombinationTS (a re-evaluation
framework, not a SOTA claim), LLM/foundation-model papers (need Hugging Face weights, unreachable here), SAMBA
(needs CUDA-only `mamba-ssm`).

## Recommended step 2 order
1. **TFPS** — run with D zero-initialised vs orthogonally initialised vs as released, 5 seeds, every test window kept,
   against PatchTST and DLinear trained in the same code with the same lr search. Question: does the headline
   depend on uninitialised memory, and does it survive a fair baseline?
2. **TimeDistill** — PatchTST at input 720 with its official ETTh1 config, 3–5 seeds, in TimeDistill's own harness.
   Question: does 0.429 still beat a properly configured PatchTST?
3. **GTR on Solar** — baselines given the same lr search GTR got, 5 seeds.

Each gets a pre-registered verdict (survives / shrinks / vanishes) before anything is scored.
