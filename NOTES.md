# Reality Check — step 1 working notes (ground truth = files under smoke/ and repos/)

## Environment
- 2 CPU cores, 7 GB RAM, no GPU. Reachable: GitHub (git clone + raw), PyPI, uv python builds.
  Not reachable: Google Drive, Baidu, Hugging Face, OpenML, Zenodo, arXiv API, download.pytorch.org.
- Data: ETTh1/ETTh2/ETTm1/ETTm2 from github.com/zhouhaoyi/ETDataset (official source).
  laiguokun/multivariate-time-series-data mirrors electricity/traffic/exchange/solar (raw .txt, no dates).
  Weather, ILI not reachable.
- Pinned env (now `work/venv` via harness/setup.sh; originally `tslenv`): Python 3.11.15, torch 2.4.1, numpy 1.26.4, pandas 2.2.3, sklearn 1.4.2.
  `sitecustomize.py` CPU redirect (CPU_REDIRECT=1) maps .cuda()/cuda devices to CPU without editing repo code.
- Smoke = first config of the repo's own ETTh1 script, 1 epoch, our ETT path.

## Static red flags found so far
- test loader drop_last=True: TFPS (NeurIPS 2025), SAMBA/SDE (KDD 2025), Time-Energy-Model repo.
  TFPS batch 128 -> 97 of 2785 test windows dropped at H=96 (3.5%), 113 of 2161 at H=720 (5.2%).
- TFPS layers/Cluster.py:30: `self.D = Parameter(torch.Tensor(112,112))` = UNINITIALISED memory.
  `Initialization_D` (k-means/SVD init) imported in exp/exp_main.py:24 but never called.
  D routes the sparse MoE (backbone_cluster_time.py:78-82). Checked 3 fresh processes, same seed:
  4635 / 4586 / 4616 NaN entries of 12544, finite absmax 6e-41 / 1.9e25 / 6e-41. Seed does not control it.
  CPU smoke: Train/Vali/Test loss NaN from epoch 1.
- Paper-side (from arXiv text, to be re-checked against PDFs):
  - TFPS: grid search over lr and expert counts for TFPS; baselines "unified parameters"; single runs, no error bars.
  - SiGMA: ETTh1 not best at 96/192/336 by its own table; "55 of 80 settings" vs 64 implied; 3 runs, no std.
  - TimePerceiver: some DeformableTST numbers copied; "54 of 64 std <= 0.0005" conflicts with its own Table 6.
  - GTR: lr search for GTR, baselines at published settings; only GTR has variance.
  - Dynamic TMoE: ETTh1 0.429 vs 0.432 best baseline, no seeds/variance; grid search for method only.
- Selective Learning (NeurIPS 2025): awesome-list code link points to a different paper's repo. Dropped.
- CombinationTS (ICML 2026): itself a modular re-evaluation framework, not a SOTA claim. Dropped as target.

## Competitors / adjacent
- Google Paper Assistant Tool (text/maths; ICML 2026 + STOC pilots, 4,700 papers). No code execution.
- RECLAIM, PaperBench, FactReview: reproduce numbers; FactReview partly checks fairness.
- SOTAVerified: community reproduction tracker with an agent API (reproductions, not fairness).
- "There are no Champions in Supervised LTSF" (Brigato et al., arXiv 2502.14045): field-level reality check
  of 8 models x 14 datasets. Our angle is per-paper, automated, with the paper's own code.

## TFPS feasibility after defining D (audits/tfps/feas_*.log; NOT a result: 1 seed, 3 epochs, H=96)
- D=zero: finite training, test MSE 0.4213. D=orth: finite, test MSE 0.4192. Code is auditable once D is defined.
- vali()'s "Test Loss" includes the auxiliary clustering losses, so it differs from the final pure MSE.

## TimeDistill PatchTST config (scripts/long_term_forecast/ETTh1/PatchTST.sh + run.py defaults)
- e_layers 1, n_heads 16, d_model 512, d_ff 2048, dropout 0.1, 10 epochs, patience 5, batch 32, lr 1e-4.
- PatchTST paper Table 3 ETTh1 (ar5iv 2211.14730): /64 (L=512) 0.370/0.413/0.422/0.447 (avg 0.413);
  /42 (L=336) 0.375/0.414/0.431/0.449 (avg 0.417).
- CAST-Norm: exp/exp_basic.py missing from the repo (34 files, 1 commit) -> does not run as released.
- ReNF: trains (32 s/epoch); test() crashes at models/ReNF_alpha.py:31 (unpack) as released.
- PatchTST at TimeDistill's released config (d_model 512, input 720): 752 s per epoch on this CPU (3.07 s/batch).
  A full anchor arm (10 epochs x 4 horizons x 3 seeds) would be ~25 CPU-hours: run 1 seed only, or on a rented GPU.
