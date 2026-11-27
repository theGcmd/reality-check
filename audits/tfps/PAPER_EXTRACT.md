# TFPS paper extract (reviewing chat, 2026-11-27, Gustav supplied the link)

Source: arXiv 2410.09836 v2 (submitted 2024-10-13; v2 2025-10-01), "Learning Pattern-Specific Experts for Time Series
Forecasting Under Patch-level Distribution Shift", Sun, Xie, Eldele, Chen, Hu, Wu. Fetched via WebFetch of
https://arxiv.org/pdf/2410.09836v2 in two parts (offset 0 and 100000 of 136235 characters).

**Reliability:** WebFetch returns a model's reading of the PDF text, not the PDF. Text in "quotes" below was returned
as exact quotes. Table 1 cells were "reconstructed from garbled layout" by the fetch and column names were inferred:
check every number you rely on against the PDF before a verdict depends on it (Gustav can supply the PDF).

## Initialisation of D (exact quotes)
- Algorithm 1 (Appendix L) input: "random Gaussian distribution-initialized subspace".
- "Technically, we set D as 512, n as 2." (Algorithm 1 caption; refers to a dimension, not initialisation.)
- No sentence about k-means or SVD initialisation was found in either part.

## Settings (main text / appendix, as returned)
- Optimizer Adam. lr grid {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.005} (0.005 listed twice as printed).
  Experts K_t, K_f ∈ {1, 2, 4, 8}. "The best parameters are selected based on the lowest validation loss."
- Input length: "we searched input lengths among 96, 192, 336, and 512, and report the best-performing configuration
  for each forecasting length" (returned in the appendix part, described for TFPS). Table 1 caption: "The input
  lengths are L = 104 for ILI and L = 96 for others." These two statements need reconciling against the PDF.
- Baselines: "We rerun all the experiments with codes provided by their official implementation." (Note: the
  2026-11-24 summarised fetch said baselines used "unified parameters"; that was a paraphrase and is superseded
  by this quote unless the PDF says both.)
- Seeds / runs / error bars: none reported; checklist marks error bars NA. All TFPS results are single values.
- α_t = α_f = 1e-3 (Appendix J); β ≈ 0.1 recommended. Figure 7 expert-number experiments use lr 0.0001.
- Batch size, epochs, patience, hardware: not found in the returned text.

## Table 1, ETTh1, MSE / MAE (columns as inferred by the fetch, printed order)
| H | TFPS | TSLANet | FITS | iTransformer | TFDNet-IK | PatchTST | TimesNet | DLinear | FEDformer |
|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.398/0.413 | 0.387/0.405 | 0.395/0.403 | 0.387/0.405 | 0.396/0.409 | 0.413/0.419 | 0.389/0.412 | 0.398/0.410 | 0.385/0.425 |
| 192 | 0.423/0.423 | 0.448/0.436 | 0.445/0.432 | 0.441/0.436 | 0.451/0.441 | 0.460/0.445 | 0.441/0.442 | 0.434/0.427 | 0.441/0.461 |
| 336 | 0.484/0.461 | 0.491/0.487 | 0.489/0.463 | 0.491/0.463 | 0.495/0.462 | 0.497/0.463 | 0.491/0.467 | 0.499/0.477 | 0.491/0.473 |
| 720 | 0.488/0.476 | 0.505/0.486 | 0.496/0.485 | 0.509/0.494 | 0.492/0.482 | 0.501/0.486 | 0.512/0.491 | 0.508/0.503 | 0.501/0.499 |
(No Avg row printed for ETTh1 in the returned text.) The round-2 lens's recollection "DLinear H=336 ≈ 0.481" does not
match this extract (0.499); treat both as unverified until checked against the PDF.

## Table 2 ablation, ETTh1 MSE (row-to-component mapping UNREADABLE in the returned text)
- Full model: 0.398 / 0.423 / 0.484 / 0.488 (H = 96 / 192 / 336 / 720).
- Other variants (printed order, mapping unknown): 96: 0.401, 0.401, 0.414, 0.455, 0.503, 0.552; 192: 0.459, 0.451,
  0.460, 0.507, 0.535, 0.583; 336: 0.486, 0.494, 0.501, 0.539, 0.558, 0.591; 720: 0.492, 0.509, 0.500, 0.576, 0.583,
  0.594.
