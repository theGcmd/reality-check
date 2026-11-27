# DUST-1 — source notes (reviewing chat, 2026-11-27)

Paper: "Dust: Pretraining Transformers Without Backpropagation", Q Labs (Dahal, Mandal, Gülbahar, Vegesna),
October 2026, https://qlabs.sh/research/dust . Code: github.com/qlabs-eng/dust @ b20f7c03eac630b6441ba6f254128c1761dc45ad
(pinned in harness/repos.lock).

**Status of these notes:** the paper text below comes from WebFetch, which returns a paraphrased summary, not the
page itself. Treat every paper-side number here as UNVERIFIED until checked against the page; Table 1 and Appendix F
were not readable. Code-side facts were read directly from the pinned commit.

## Paper (paraphrased via WebFetch)
- Method: zeroth-order. Gaussian noise added to each linear layer's output, independently per token; each token's
  noise rewarded by its centered loss reduction; reward-weighted noise ⊗ layer input = weight-gradient estimate.
  Attention internals scored by effect on attention output; layer types jittered in separate passes.
- Setup: GPT-style; base 8 layers × width 512; size sweep at 10M tokens: ~2M / 7.3M / 38M / 243M params. FineWeb,
  BPE-4096. Budgets 100k / 1M / 10M / 20M tokens, **one epoch**. Seq 2048, batch 8 sequences (16k tokens).
  Held-out val and test: 544 sequences each. Test loss reported at best-validation checkpoint. 3 seeds per cell
  (Adam comparison: 5 seeds Dust/backprop, 3 EGGROLL).
- Backprop baseline (main): SGD with momentum, constant lr, same momentum/lr grid as Dust, tuned per budget and
  population. Adam variant at 1M tokens re-tunes backprop lr. Weight decay / dropout / schedule: not stated.
- Measured (Fig. 4 table, 10M tokens, test loss; columns 2M / 7.3M / 38M / 243M params):
  backprop 5.180 / 5.066 / 5.015 / 5.048; 16k draws 5.171 / 5.065 / 5.036 / 5.086.
- 20M tokens: backprop test 4.633; Dust power-law limit 4.431 (95% bootstrap 3.89-4.58), fit ≈ 1.53·P^-0.145 + 4.43
  over 5 population cells; authors call it loosely constrained, an extrapolation, not a measured value.
- Adam at 1M tokens: Dust fitted limit 5.248 (95% 5.17-5.30) vs backprop 5.361.
- Authors' own limits: compute efficiency not the focus; orders of magnitude more efficiency needed; gains beyond 1M
  tokens rely on extrapolation; gap to backprop grows slightly with model size.

## Code facts (read directly)
- README: needs Linux + NVIDIA GPU with bf16; torch 2.10.0 cu128; default population 16,384 on 8 GPUs; supported
  populations 256 / 1024 / 4096 / 16384 on 1 / 1 / 4 / 8 GPUs. `prepare_data.py` streams FineWeb from Hugging Face
  (NOT reachable from this workspace) into data/train_1M.pt, train_10M.pt, val.pt, test.pt.
- `baselines/backprop.py:16-19`: 1M → SGD lr 0.175, momentum 0.98, **61 steps**; 10M → lr 0.2, momentum 0.95,
  610 steps. `:22-28`: embedding (`wte`) lr fixed at 1000.0; weight_decay 0.0; constant lr, no schedule, no Adam.
- `dust.py:48-65`: per-population (lr, momentum) for Dust at 1M and 10M.
- 61 steps × 16,384 tokens ≈ 1M tokens: backprop gets 61 optimizer steps at a batch size chosen for Dust (Dust's
  population lives in the tokens of the batch). Whether backprop at a smaller batch (more steps over the same
  tokens), or AdamW with a schedule, does better on the same data is not tested in the released code.

## Candidate audit question (for the prereg; owner approved adding DUST-1 on 2026-11-27)
At a budget the paper measures (1M tokens, one epoch, same model/data/val/test), does Dust's measured test loss
still beat backprop when backprop is configured the way transformers are normally trained on that data (AdamW,
warmup + decay, batch size and lr tuned on validation), at matched tokens? Report also the batch-matched framing
the paper uses. Compare against the paper's measured Dust numbers first; re-running Dust needs GPUs.

## Practical blockers
- Data: FineWeb via Hugging Face is unreachable here. Gustav can run `prepare_data.py --tokens 1M` on his laptop
  and commit the four .pt files (small) under a data path the harness reads, or a GPU box can fetch them.
- Compute: backprop arms on the 8×512 model at 1M tokens are plausibly CPU-feasible here (to be measured);
  Dust arms need a GPU.
