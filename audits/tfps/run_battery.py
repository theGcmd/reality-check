"""TFPS-1 battery driver (PREREG r3.1 §5-§8). One (arm, H, seed[, lr]) run per call; segment-resumable.

  python run_battery.py tfps <arm> <H> <seed> [--budget S]          TFPS arm (tfps_gauss | tfps_gauss_unit | tfps_zero)
  python run_battery.py base <arm> <H> <seed> <lr>                   baseline (patchtst | dlinear), separate process
  python run_battery.py selftest                                     pre-FREEZE driver self-test (seed 999, 2 epochs, VAL split)

Scored runs (seed != 999) are refused unless a commit whose message starts 'FREEZE tfps' exists in this repo's history
and PREREG.md is unchanged since it. Run dir: audits/tfps/runs/<arm>/<H>/seed<k>[/lr<x>]/ with provenance.json,
metrics.json, per_window_mse.npy (float32; primary = mean of K=5 draws per window for TFPS), per-draw / noise-off /
pure-val-MSE arrays, routing.json, epochs.json, train.log. Checkpoints stay in $RC_WORK/ckpt (never committed).
Re-running a finished run is refused (no overwrite); re-running an unfinished TFPS run resumes it.
"""
import os, sys, json, time, shutil, subprocess, datetime, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

RC_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RC_WORK = os.path.abspath(os.environ.get("RC_WORK", os.path.join(RC_ROOT, "work")))
RUNS = os.path.join(HERE, "runs")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def frozen():
    """True iff a 'FREEZE tfps' commit exists and PREREG.md has not changed since it."""
    log = subprocess.run(["git", "log", "--format=%H %s"], cwd=RC_ROOT, capture_output=True, text=True).stdout.splitlines()
    fr = [l.split()[0] for l in log if l.split(" ", 1)[1].startswith("FREEZE tfps")] if log else []
    if not fr:
        return False
    d = subprocess.run(["git", "diff", "--quiet", fr[0], "--", "audits/tfps/PREREG.md"], cwd=RC_ROOT).returncode
    return d == 0


def routing_summary(rec):
    """A (noisy top-1 == argmax s), H_r (entropy of expert usage, nats), mean row max / range of s; per router."""
    out = {}
    for w, r in rec.items():
        idx = np.concatenate(r["idx"]).astype(int); am = np.concatenate(r["am"]).astype(int)
        p = np.bincount(idx, minlength=4) / len(idx); p = p[p > 0]
        out[w] = {"A": float((idx == am).mean()), "H_r": float(-(p * np.log(p)).sum()), "rows": int(len(idx)),
                  "s_rowmax": float(np.concatenate(r["rmax"]).mean()), "s_rowrange": float(np.concatenate(r["rrange"]).mean())}
    return out


def mean_routing(list_of_summaries):
    keys = ("A", "H_r", "s_rowmax", "s_rowrange")
    return {w: {k: float(np.mean([s[w][k] for s in list_of_summaries])) for k in keys} for w in list_of_summaries[0]}


def save_tfps_scores(run_dir, scores, released_printed, N_rel_batch=128):
    """Writes the per-window files + metrics.json. 'repo' numbers = utils.metrics on the concatenated passes is NOT
    used here; the second code path (step f) recomputes from the saved arrays."""
    m = {"released_test_printed": released_printed}  # repo's own test(): own noise draw, drop_last on
    for sel, sc in scores.items():
        D = np.stack([d["mse"] for d in sc["draws"]]).astype(np.float64)          # K x N
        Dm = np.stack([d["mae"] for d in sc["draws"]]).astype(np.float64)
        np.save(os.path.join(run_dir, f"{sel}_draws_mse.npy"), D.astype(np.float32))
        np.save(os.path.join(run_dir, f"{sel}_draws_mae.npy"), Dm.astype(np.float32))
        np.save(os.path.join(run_dir, f"{sel}_nonoise_mse.npy"), sc["nonoise"]["mse"].astype(np.float32))
        np.save(os.path.join(run_dir, f"{sel}_nonoise_mae.npy"), sc["nonoise"]["mae"].astype(np.float32))
        per_draw = D.mean(axis=1)                                               # mean over all N windows, per draw
        N = D.shape[1]; P = (N // N_rel_batch) * N_rel_batch
        m[sel] = {"checkpoint_sha256": sc["checkpoint_sha256"], "n_windows": int(N), "split_set_type": sc["set_type"],
                  "per_draw_mse": per_draw.tolist(), "per_draw_mae": Dm.mean(axis=1).tolist(),
                  "score_mse": float(per_draw.mean()), "score_mae": float(Dm.mean(axis=1).mean()),
                  "mean_abs_draw_dev": float(np.abs(per_draw - per_draw.mean()).mean()),
                  "draw_var": float(per_draw.var(ddof=1)),
                  "released_loader_score_mse": float(D[:, :P].mean(axis=1).mean()), "released_loader_rows": int(P),
                  "nonoise_mse": float(sc["nonoise"]["mse"].astype(np.float64).mean()),
                  "nonoise_mae": float(sc["nonoise"]["mae"].astype(np.float64).mean()),
                  "noise_seeds": [d["noise_seed"] for d in sc["draws"]]}
        m[sel]["routing_draws_mean"] = mean_routing([routing_summary(d["rec"]) for d in sc["draws"]])
        m[sel]["routing_nonoise"] = routing_summary(sc["nonoise"]["rec"])
    # primary per-window file: released selection, mean over the K draws per window (score = its mean)
    np.save(os.path.join(run_dir, "per_window_mse.npy"), np.load(os.path.join(run_dir, "released_draws_mse.npy")).astype(np.float64).mean(0).astype(np.float32))
    json.dump(m, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)
    return m


def run_tfps_cli(arm, H, seed, budget, selftest=False, train_epochs=100, runs_root=RUNS):
    import rc_tfps as R
    prefreeze = (seed == R.TIMING_SEED)
    if not prefreeze and not frozen():
        sys.exit("refused: scored seed before 'FREEZE tfps' commit (or PREREG.md changed since FREEZE)")
    if selftest:
        assert prefreeze
    run_dir = os.path.join(runs_root, arm, str(H), f"seed{seed}")
    if os.path.exists(os.path.join(run_dir, "metrics.json")) or os.path.exists(os.path.join(run_dir, "done.json")):
        sys.exit(f"refused: {run_dir} already finished (no overwrite)")
    ckpt_root = os.path.join(RC_WORK, "ckpt" + ("_selftest" if selftest else ""), "tfps", arm, str(H), f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    pv_path = os.path.join(run_dir, "provenance.json")
    pv = json.load(open(pv_path)) if os.path.exists(pv_path) else {"segments": []}
    seg = {"start_utc": utc()}
    if selftest:
        R.RC.score_flag = "val"  # set again after reset inside run_tfps (below)
    orig_reset = R.RC.reset

    def _reset():
        orig_reset()
        if selftest: R.RC.score_flag = "val"
    R.RC.reset = _reset
    info = R.run_tfps(arm, H, seed, run_dir, ckpt_root, train_epochs=train_epochs, prefreeze=prefreeze, instrument=True,
                      log_path=os.path.join(run_dir, "train.log"), train_impl="resumable", seg_budget_s=budget)
    seg.update({"end_utc": utc(), "wall_s": info["wall_s"], "ended_after_epoch": info["segment_ended_after_epoch"],
                "finished": info["finished"], "filtered_lines": info["filtered_lines"]})
    wrappers = [os.path.join(HERE, f) for f in ("rc_tfps.py", "run_battery.py")]
    base = R.provenance(info["argv"], wrappers, {"arm": arm, "H": H, "seed": seed, "prefreeze": prefreeze,
                                                "score_split": R.RC.score_flag, "train_impl": "resumable (P7)",
                                                "d_generator_seeds": list(R.d_seeds(seed)),
                                                "ckpt_dir": info["ckpt_dir"], "CPU_REDIRECT": os.environ.get("CPU_REDIRECT")})
    base["segments"] = pv["segments"] + [dict(seg, rc_commit=base["rc_commit"], pid=base["pid"])]
    json.dump(base, open(pv_path, "w"), indent=1)
    if not info["finished"]:
        print(json.dumps({"run": run_dir, "finished": False, "after_epoch": info["segment_ended_after_epoch"]}))
        return None
    ck = info["ckpt_dir"]
    shutil.copy(os.path.join(ck, "epochs.json"), os.path.join(run_dir, "epochs.json"))
    shutil.copy(os.path.join(ck, "es.json"), os.path.join(run_dir, "es.json"))
    shutil.copy(os.path.join(ck, "done.json"), os.path.join(run_dir, "done.json"))
    ep = json.load(open(os.path.join(ck, "epochs.json"))); es = json.load(open(os.path.join(ck, "es.json")))
    d_final = {w: R.torch.load(os.path.join(ck, "checkpoint.pth"))[f"model_{n}.cluster.D"] for w, n in (("time", "time"), ("freq", "frequency"))}
    summ = {"stop_epoch": len(ep), "best_epoch_released": es["best_epoch"], "best_epoch_valmse": R.RC.best_valmse_epoch,
            "n_candidate_epochs_valmse": len(ep),
            "aux_share_at_saved_epoch": (float(ep[es["best_epoch"] - 1]["val_loss"]) - float(ep[es["best_epoch"] - 1]["val_mse"])) / float(ep[es["best_epoch"] - 1]["val_loss"]),
            "D_rel_change": {w: (float((d_final[w] - R.RC.d_init[w]).norm() / R.RC.d_init[w].norm()) if float(R.RC.d_init[w].norm()) > 0 else None) for w in d_final}
                            if R.RC.d_init else None,
            "D_maxabs_final": {w: float(d_final[w].abs().max()) for w in d_final}}
    if R.RC.scores is not None:
        m = save_tfps_scores(run_dir, R.RC.scores, R.RC.released_printed)
        m["training"] = summ
        json.dump(m, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)
    else:
        json.dump({"training": summ, "note": "pre-FREEZE: no test pass"}, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)
    for junk in ("results", "test_results"):
        shutil.rmtree(os.path.join(run_dir, junk), ignore_errors=True)  # repo's pred.npy etc. (full predictions: never committed)
    pv = json.load(open(pv_path)); pv["metrics_written_utc"] = utc()
    pv["file_sha256"] = {f: R.sha256(os.path.join(run_dir, f)) for f in sorted(os.listdir(run_dir)) if f.endswith((".npy", ".json")) and f != "provenance.json"}
    json.dump(pv, open(pv_path, "w"), indent=1)
    print(json.dumps({"run": run_dir, "finished": True, "stop_epoch": summ["stop_epoch"]}))
    return run_dir


def run_base_cli(arm, H, seed, lr, selftest=False, train_epochs=100, runs_root=RUNS):
    import rc_patchtst as P
    prefreeze = (seed == P.TIMING_SEED)
    if not prefreeze and not frozen():
        sys.exit("refused: scored seed before 'FREEZE tfps' commit (or PREREG.md changed since FREEZE)")
    run_dir = os.path.join(runs_root, arm, str(H), f"seed{seed}", f"lr{lr:g}")
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        sys.exit(f"refused: {run_dir} already finished (no overwrite)")
    shutil.rmtree(run_dir, ignore_errors=True); os.makedirs(run_dir)  # baselines never resume: restart from scratch
    ckpt_root = os.path.join(RC_WORK, "ckpt" + ("_selftest" if selftest else ""), "tfps", arm, str(H), f"seed{seed}", f"lr{lr:g}")
    shutil.rmtree(ckpt_root, ignore_errors=True)
    t0 = utc()
    mode = "score" if not prefreeze else ("valscore" if selftest else "prefreeze")
    if mode == "valscore":
        orig = P.full_test_loader
        P.full_test_loader = lambda args: (lambda ds: (ds, P.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)))(P._data_provider(args, "val")[0])
    info = P.run(arm, H, seed, lr, run_dir, ckpt_root, train_epochs=train_epochs, mode=mode, instrument=True,
                 log_path=os.path.join(run_dir, "train.log"))
    es = json.load(open(os.path.join(info["ckpt_dir"], "es.json")))
    m = {"lr": lr, "val_loss_min_repr": es["val_loss_min_repr"], "best_epoch": es["best_epoch"], "stop_epoch": len(es["val_losses"]),
         "val_losses": es["val_losses"], "val_finite": bool(np.all(np.isfinite([float(v) for v in es["val_losses"]])))}
    if P.ST.scores is not None:
        s = P.ST.scores; N = len(s["mse"]); Pn = (N // 128) * 128
        np.save(os.path.join(run_dir, "per_window_mse.npy"), s["mse"].astype(np.float32))
        np.save(os.path.join(run_dir, "per_window_mae.npy"), s["mae"].astype(np.float32))
        m.update({"n_windows": N, "score_mse": float(s["mse"].mean()), "score_mae": float(s["mae"].mean()),
                  "released_loader_score_mse": float(s["mse"][:Pn].mean()), "released_loader_rows": Pn,
                  "released_test_printed": P.ST.printed, "checkpoint_sha256": hashlib.sha256(open(s["checkpoint"], "rb").read()).hexdigest()})
    if mode == "valscore":
        P.full_test_loader = orig
    json.dump(m, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)
    for junk in ("results", "test_results"):
        shutil.rmtree(os.path.join(run_dir, junk), ignore_errors=True)
    sys.path.insert(0, HERE)
    import importlib.util
    pv = {"rc_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=RC_ROOT, capture_output=True, text=True).stdout.strip(),
          "rc_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=RC_ROOT, capture_output=True, text=True).stdout.strip()),
          "audited_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(P.REPO), capture_output=True, text=True).stdout.strip(),
          "wrapper_sha256": {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in ("rc_patchtst.py", "run_battery.py")},
          "argv": info["argv"], "env": {k: os.environ.get(k) for k in ("CPU_REDIRECT", "OMP_NUM_THREADS", "PYTHONHASHSEED")},
          "uv_pip_freeze_sha256": hashlib.sha256(subprocess.run(["uv", "pip", "freeze", "--python", os.path.join(RC_WORK, "venv", "bin", "python")], capture_output=True, text=True).stdout.encode()).hexdigest(),
          "torch": P.torch.__version__, "numpy": np.__version__, "host": os.uname().nodename, "pid": os.getpid(),
          "torch_threads": P.torch.get_num_threads(), "start_utc": t0, "end_utc": utc(), "wall_s": info["wall_s"],
          "arm": arm, "H": H, "seed": seed, "lr": lr, "mode": mode, "ckpt_dir": info["ckpt_dir"]}
    pv["file_sha256"] = {f: hashlib.sha256(open(os.path.join(run_dir, f), "rb").read()).hexdigest() for f in sorted(os.listdir(run_dir)) if f.endswith((".npy", ".json"))}
    json.dump(pv, open(os.path.join(run_dir, "provenance.json"), "w"), indent=1)
    print(json.dumps({"run": run_dir, "finished": True, "stop_epoch": m["stop_epoch"]}))
    return run_dir


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "tfps":
        b = float(a[a.index("--budget") + 1]) if "--budget" in a else None
        run_tfps_cli(a[1], int(a[2]), int(a[3]), b)
    elif a[0] == "base":
        run_base_cli(a[1], int(a[2]), int(a[3]), float(a[4]))
    elif a[0] in ("_st_tfps", "_st_base"):  # internal: self-test children (seed 999, 2 epochs, VAL split, scratch root)
        root = a[-1]
        if a[0] == "_st_tfps":
            run_tfps_cli(a[1], 192, 999, float(a[2]) if a[2] != "none" else None, selftest=True, train_epochs=2, runs_root=root)
        else:
            run_base_cli(a[1], 192, 999, float(a[2]), selftest=True, train_epochs=2, runs_root=root)
    else:
        raise SystemExit(__doc__)
