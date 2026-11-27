# Design review round 3 (last allowed) — common brief for every lens

You are one lens of a blind design review for the Reality Check project (audits whether ML papers' headline
results survive fair tests). Repo: /home/claude/reality-check (do NOT git commit or push; write only your own output file).

Read:
- audits/tfps/PREREG.md (r3 draft, commit bc7055d) — the design under review. Includes OWNER DECISIONS 1-11 at the end:
  those are fixed by the project owner; you may say a decision causes a problem, but do not ask to undo it silently.
- audits/tfps/PAPER_EXTRACT.md — the paper text available (WebFetch reading of arXiv 2410.09836v2; Table 1 cells
  were reconstructed by the fetch and must be checked against the PDF before FREEZE — the prereg makes that a precondition).
- audits/tfps/_design_review_r2.json — round-2 findings, to check whether each was fixed.
- Released code: work/repos/tfps (TFPS, commit 83a11827) and work/repos/patchtst_official/PatchTST_supervised.
  Data: work/data or as harness/setup.sh placed it (ETTh1.csv). Python: work/venv/bin/python (torch 2.4.1 CPU, 2 cores).

Rules: never edit anything under work/repos. Never fetch the web. Nothing leaves the repo. No scored training runs
(a few-second forward probe or arithmetic is fine if your lens allows running code; keep any probe under ~10 minutes,
foreground only). Cite file:line for every code claim. Be exactly as strong as the evidence.

Verdict scale: LICENSED (score as written), LICENSED_WITH_FIXES (only non-blocking fixes, each concrete enough to
apply mechanically before FREEZE without a new review), NOT_LICENSED (at least one blocking issue). Use "blocking"
only for something that would make a verdict wrong, unreachable by construction, or unfair to the authors.

Output: write ONE JSON file at audits/tfps/_r3/<lens>.json with keys:
  lens, model, round (3), prereg_reviewed ("audits/tfps/PREREG.md r3 @ bc7055d"), verdict,
  r2_issues_status (list of {issue, fixed: yes/partly/no, note}) — for issues in your lens's area,
  issues (list of {severity: blocking|major|minor, issue, evidence, fix}), summary (<= 120 words).
Then reply with just the verdict and a 3-line summary.
