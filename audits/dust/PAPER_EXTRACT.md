# DUST-1 — paper extract, check P1 (reviewing chat, 2026-11-27 19:30 London)

Source: https://qlabs.sh/research/dust (Gustav supplied the link on 2026-11-26), fetched via WebFetch, which returns a
model's reading of the page. Text in "quotes" was returned as verbatim.

## What the page does and does not contain
- **Table 1 (the per-cell values) is NOT on the page** as fetched; it is referenced but not included. Figures 1 and 2
  show curves/points without numeric labels. So no measured per-cell 1M or 100k test loss (Dust or backprop, mean or
  spread) is available from the page. The pinned repo (b20f7c0) contains no results files either.
- Figure 3 (Adam, 1M tokens) has the only printed numbers: Dust fitted power-law limit 5.248 (95% bootstrap 5.17-5.30;
  a limit, not a measured cell) and a backprop label 5.361 (caption does not say whether measured mean or fit).
- Seeds: Section 3.1 "three seeds per cell"; Figure 3 caption: five for Dust and backprop, three for EGGROLL.

## The claims, verbatim
- SGD, Section 3.2: "At 100k and 1M tokens Dust ends below backprop, from a few hundred draws at 100k and from a
  thousand at 1M."
- Adam, Section 3.3: "Dust closes on backprop with a large population and the interval on its limit sits below
  backprop"

So the paper DOES claim a measured advantage at 1M tokens with SGD (from ~1,000 draws), stated qualitatively; the
numbers behind it (Table 1) are not available to us. This bears on the review's Q0-MOOT estimate (~65-75%), which
assumed no measured 1M Dust advantage is claimed.

## Grids and selection
- Section 3.1: backprop and Dust "share one momentum and learning-rate grid"; the grid values are NOT on the page.
  The released code's chosen values are in `dust.py:48-65` and `baselines/backprop.py:16-19`.
- Section 2.3: the grid search "keeps the setting that lowers the loss most" on a small token budget; the split used
  is not named. Test loss is reported "at best validation checkpoint" (checkpoint selection, not grid selection).
- Adam: backprop learning rates were re-tuned; grid not given.

## Still needed for P1
Table 1's measured values (1M and 100k, every population, SGD and Adam, mean ± spread). Options: ask the authors (Gustav's
call; nothing leaves the repo without him), or find a PDF/arXiv version of the paper that includes the table.
