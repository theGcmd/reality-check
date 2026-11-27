"""Pre-FREEZE timing (PREREG §8): seed 999 only, stdout line-filtered, no test pass, result files deleted.
Usage: python timing_tfps.py <arm> <H> <train_epochs> <seg_budget_s>  (re-run to resume; prints epoch count + s/epoch only)"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_tfps as R
arm, H, E, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
base = os.path.join(R.RC_WORK, "timing_tfps", f"{arm}_{H}_e{E}")
info = R.run_tfps(arm, H, 999, os.path.join(base, "run"), os.path.join(base, "ck"), train_epochs=E, prefreeze=True,
                  instrument=True, log_path=os.path.join(R.HERE, "selftest", f"timing_{arm}_{H}_e{E}.log"),
                  train_impl="resumable", seg_budget_s=B)
ep = json.load(open(os.path.join(info["ckpt_dir"], "epochs.json")))
es = json.load(open(os.path.join(info["ckpt_dir"], "es.json")))
out = {"arm": arm, "H": H, "seed": 999, "epochs_done": len(ep), "finished": info["finished"], "segment_wall_s": info["wall_s"],
       "best_epoch_released_criterion": es["best_epoch"],
       "aux_share_last": (float(ep[-1]["val_loss"]) - float(ep[-1]["val_mse"])) / float(ep[-1]["val_loss"])}
if info["finished"]:
    out["stop_epoch"] = json.load(open(os.path.join(info["ckpt_dir"], "done.json")))["stop_epoch"]
log = os.path.join(R.HERE, "selftest", f"timing_{arm}_{H}_e{E}.json")
prev = json.load(open(log)) if os.path.exists(log) else {"segments": []}
prev["segments"].append(out); json.dump(prev, open(log, "w"), indent=1)
print(json.dumps(out))
