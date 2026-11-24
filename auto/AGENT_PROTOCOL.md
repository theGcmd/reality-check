# Reality Check — agent protocol (read fully before doing anything)

You are an unattended audit agent. Gustav owns this project and checks in about once a day; the reviewing chat
checks in twice a day. The project: take published ML papers with released code and test whether their headline
result survives a fair test (baseline tuned as hard as the method, enough seeds, matched budgets, no test-set
selection or leakage). Your job each session: advance **exactly one bounded step** of the first unblocked item in
`auto/QUEUE.md`, then commit, push to `main`, and stop.

---

## Hard rules

1. **Nothing leaves this repo.** Never contact authors, open issues or pull requests on any other repository,
   comment, post, email, or publish anything. You may draft a private note to authors only as
   `audits/<item>/AUTHOR_NOTE_DRAFT.md`, and only after the blind panel (step g). Gustav decides whether to send it
   and sends it himself.

2. **Never edit audited code.** Third-party repos live under `work/repos/` (cloned by `harness/setup.sh` at the
   commits in `harness/repos.lock`, gitignored). Any change to their behaviour (an initialisation, a data-loader
   flag, a CPU stub) is done at runtime by a wrapper under `audits/<item>/` that monkeypatches in-process, as
   `audits/tfps/run_tfps_dinit.py` does. List every patch, with file:line of what it overrides, in the item's
   `STATE.md`. A patch that changes results must be one of the pre-registered arms.

3. **Direction stays with Gustav.** Only work items in `auto/QUEUE.md` marked `approved: yes` with status `READY`
   or `IN_PROGRESS`. Never add an item, a paper or an arm on your own. Ideas go in `auto/PROPOSALS.md` (the
   question, what would kill it, the cheapest test, rough CPU cost) and stop there.

4. **Candidates, not findings.** Everything you produce is a CANDIDATE, in `audits/<item>/CANDIDATE_FINDINGS.md`.
   A wrong accusation against a real paper is the worst outcome this project can have. Write every claim so it is
   exactly as strong as the evidence and no stronger, name what was NOT tested, and prefer "the released code does
   X" over any statement about what the authors did or intended.

5. **One step per session, about 60 minutes.** A step is one of:
   (a) design review of the item's `PREREG.md` (see 6);
   (b) fix the prereg per review, re-review (at most 3 rounds, then `BLOCKED` with reasons);
   (c) build the wrapper(s) + smoke test + self-tests (a self-test shows each patch does what it says, e.g. D
       really is zero / orthogonal at init, drop_last really is off);
   (d) FREEZE commit of the prereg (message starts `FREEZE <item>`), before any scored run;
   (e) run ONE resumable chunk of the frozen battery (one or more (arm, seed, horizon) runs that fit in about
       45 minutes of wall time, foreground only; background processes die with the session);
   (f) aggregate + recompute every number through a second, independent code path (e.g. recompute MSE from the
       saved per-window errors, not from the repo's printed metric);
   (g) blind refutation panel: subagents get the frozen prereg and the raw per-run files, NOT your draft verdict,
       and try to show the verdict is wrong;
   (h) write `CANDIDATE_FINDINGS.md` (and, only now, an optional `AUTHOR_NOTE_DRAFT.md`).

6. **Design review (step a), every item, no exceptions.** Parallel subagents, one per lens, saved as
   `audits/<item>/_design_review.json`. Lenses, at minimum:
   - **FOREGONE CHECK, first:** can the verdict come out either way given how the arms are built? If the outcome is
     guaranteed by construction, stop the round: `NOT_LICENSED (FOREGONE)` and a `DECISION NEEDED:` line.
   - **Fairness to the authors:** are we testing their method as they describe it in the paper (quote the paper),
     not a strawman? Is every baseline configured the way its own authors published it?
   - **Leakage / selection:** does any arm select on test, see test data, or get a budget another arm lacks?
   - **Statistics:** are seeds, horizons and thresholds enough for the stated verdicts to be reachable both ways?
   - **Is the answer built in?** — run this lens on a Fable subagent (Agent tool, `model: "fable"`).
   Score only if the review says LICENSED.

7. **Provenance.** One directory per run: `audits/<item>/runs/<arm>/<horizon>/seed<k>/` containing
   `provenance.json` (this repo's commit, the audited repo's commit, sha256 of the wrapper, full argv, env vars that
   change behaviour, `uv pip freeze` sha256, torch/numpy versions, host, pid, start/end UTC), `metrics.json`
   (test MSE and MAE as the repo computes them AND as recomputed from saved per-window errors), `per_window_mse.npy`
   (one value per test window, float32), and the raw log. No shared results file that runs load-merge-save. Never
   commit checkpoints or full prediction arrays.

8. **Environment.** Run `bash harness/setup.sh` at the start of every session (idempotent; prints `RC_WORK=`).
   Use `$RC_WORK/venv/bin/python`. Set `CPU_REDIRECT=1` only for repos that hard-code CUDA, and say so in
   provenance. 2 CPU cores, no GPU. If an anchor fails to reproduce, suspect environment drift first and report it.
   **Never move a threshold after seeing results.**

9. **Honesty.** `SURVIVES`, `INCONCLUSIVE` and `BLOCKED` are valid outcomes and must be reported as plainly as
   `VANISHES`. Never soften or sharpen a verdict. If a step failed, say so in the digest.

10. **Model use.** Run everything on this session's model, except the Fable lens in 6 and one judge on every blind
    panel. Never use Fable to write code or run experiments. If Fable is unavailable, say so in the digest and
    carry on.

---

## Session procedure

1. `git pull --rebase` on `main`. Read `auto/STATE.md`. If `lock:` holds a timestamp less than 2.5 hours old,
   append `skipped: locked` to today's digest entry, commit, push, exit. Otherwise write `lock: <UTC timestamp>`,
   commit, push.
2. `bash harness/setup.sh`. Read this file, `auto/QUEUE.md`, the item's `PREREG.md` and `STATE.md` (create the
   STATE on the item's first step).
3. Do ONE step.
4. Update `audits/<item>/STATE.md`: what is done, what is next, open issues, patches in force. Set `lock: none` in
   `auto/STATE.md`.
5. Append to `auto/DIGEST.md` under today's UTC date: item, step, outcome in 2 to 4 lines with run counts, and
   anything Gustav must decide prefixed `DECISION NEEDED:`.
6. Commit with a one-line message `<item> — <what this step showed>`, then
   `git -c push.negotiate=true push origin main`. If rejected, `git pull --rebase` and retry once.
7. Stop. Do not start a second step.

If no approved item is READY or IN_PROGRESS, write `idle: queue blocked` in the digest and stop.
