"""Pre-FREEZE self-test of run_battery.py (PREREG §7-§8). Seed 999, H=192, 2 epochs, scores the VAL split only (no test
number is computed). Children run in fresh processes into a scratch runs root under $RC_WORK. Writes selftest/battery.json."""
import os, sys, json, subprocess, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RC_WORK = os.path.abspath(os.environ.get("RC_WORK", os.path.join(HERE, "..", "..", "work")))
ROOT = os.path.join(RC_WORK, "selftest_battery", "runs")
PY = sys.executable; DRV = os.path.join(HERE, "run_battery.py")
RES = []


def check(name, ok, **info):
    RES.append({"test": name, "pass": bool(ok), **info})
    print(("PASS " if ok else "FAIL ") + name + (" " + json.dumps(info, default=float) if info else ""), flush=True)


def child(*a):
    r = subprocess.run([PY, DRV, *a], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


shutil.rmtree(os.path.join(RC_WORK, "selftest_battery"), ignore_errors=True)
shutil.rmtree(os.path.join(RC_WORK, "ckpt_selftest"), ignore_errors=True)
os.makedirs(ROOT)

# refusals
rc, out, err = child("tfps", "tfps_gauss", "192", "2021")
check("driver refuses a scored TFPS seed before FREEZE", rc != 0 and "refused" in (out + err))
rc, out, err = child("base", "patchtst", "192", "42", "0.0001")
check("driver refuses a scored baseline seed before FREEZE", rc != 0 and "refused" in (out + err))

# TFPS gauss: budget 0 -> segment after epoch 1, then resume to finish (2 epochs)
segs = []
for _ in range(3):
    rc, out, err = child("_st_tfps", "tfps_gauss", "0", ROOT)
    assert rc == 0, err[-3000:]
    segs.append(json.loads([l for l in out.splitlines() if l.startswith("{")][-1]))
    if segs[-1]["finished"]: break
check("TFPS run segments and resumes through the driver (2 epochs -> 2 segments)", len(segs) == 2 and segs[-1]["finished"],
      segments=[s.get("after_epoch") for s in segs])
rc, out, err = child("_st_tfps", "tfps_zero", "none", ROOT)
assert rc == 0, err[-3000:]
rc2, out2, err2 = child("_st_tfps", "tfps_zero", "none", ROOT)
check("driver refuses to overwrite a finished run", rc2 != 0 and "refused" in (out2 + err2))

N_VAL = None
for arm in ("tfps_gauss", "tfps_zero"):
    d = os.path.join(ROOT, arm, "192", "seed999")
    pv = json.load(open(os.path.join(d, "provenance.json"))); m = json.load(open(os.path.join(d, "metrics.json")))
    need = ("rc_commit", "audited_commit", "wrapper_sha256", "argv", "env", "uv_pip_freeze_sha256", "torch", "numpy", "host", "pid", "segments", "file_sha256")
    seg_ok = all(k in s for s in pv["segments"] for k in ("start_utc", "end_utc", "pid"))
    check(f"{arm}: provenance.json has every §7 field incl. per-segment start/end UTC and pid", all(k in pv for k in need) and seg_ok,
          n_segments=len(pv["segments"]), score_split=pv.get("score_split"))
    pw = np.load(os.path.join(d, "per_window_mse.npy")); D = np.load(os.path.join(d, "released_draws_mse.npy"))
    N_VAL = len(pw)
    check(f"{arm}: per_window_mse.npy float32, one value per window, draws K=5 x N", pw.dtype == np.float32 and D.shape == (5, len(pw))
          and m["released"]["n_windows"] == len(pw), N=len(pw))
    check(f"{arm}: self-test scored the VAL split (set_type 1), never test", m["released"]["split_set_type"] == 1 and m["valmse"]["split_set_type"] == 1
          and pv.get("score_split") == "val")
    check(f"{arm}: routing recorded per named router (time, freq)", set(m["released"]["routing_draws_mean"]) == {"time", "freq"})
    r1 = abs(float(pw.astype(np.float64).mean()) - m["released"]["score_mse"])
    r2 = max(abs(float(D[k].astype(np.float64).mean()) - m["released"]["per_draw_mse"][k]) for k in range(5))
    check(f"{arm}: score recomputed from saved float32 arrays agrees with metrics.json (<1e-6)", r1 < 1e-6 and r2 < 1e-6, d_score=r1, d_draw=r2)
    for sel in ("valmse",):
        check(f"{arm}: pure-val-MSE checkpoint scored + files written", os.path.exists(os.path.join(d, f"{sel}_draws_mse.npy")) and sel in m)
    ep = json.load(open(os.path.join(d, "epochs.json")))
    check(f"{arm}: per-epoch log has s row range per router (val pass)", all("s_rowrange_time" in e and "s_rowrange_freq" in e and np.isfinite(e["s_rowrange_time"]) for e in ep),
          rowrange_time=[e["s_rowrange_time"] for e in ep])
    no_big = not any(os.path.isdir(os.path.join(d, j)) for j in ("results", "test_results")) and not any(f.endswith(".pth") for f in os.listdir(d))
    check(f"{arm}: no checkpoints / prediction arrays in the run dir", no_big)
    if arm == "tfps_zero":
        rz = m["released"]["routing_draws_mean"]
        check("tfps_zero: D stays 0, s row range == 0 (val log + scoring pass), A ~ 0.25 (premise checks feed Q2 rule 0)",
              all(v == 0 for v in m["training"]["D_maxabs_final"].values()) and all(e["s_rowrange_time"] == 0 and e["s_rowrange_freq"] == 0 for e in ep)
              and all(rz[w]["s_rowrange"] == 0 for w in rz) and all(abs(rz[w]["A"] - 0.25) < 0.02 for w in rz),
              A={w: rz[w]["A"] for w in rz}, H_r={w: rz[w]["H_r"] for w in rz})
    else:
        rn = m["released"]["routing_nonoise"]
        check("tfps_gauss: noise-off routing has A == 1 exactly (indices == argmax s)", all(rn[w]["A"] == 1.0 for w in rn))

for arm, lr in (("patchtst", "0.0001"), ("dlinear", "0.005")):
    rc, out, err = child("_st_base", arm, lr, ROOT)
    assert rc == 0, err[-3000:]
    d = os.path.join(ROOT, arm, "192", "seed999", f"lr{float(lr):g}")
    m = json.load(open(os.path.join(d, "metrics.json"))); pv = json.load(open(os.path.join(d, "provenance.json")))
    pw = np.load(os.path.join(d, "per_window_mse.npy"))
    check(f"{arm}: run dir per lr, provenance + metrics + per-window file; recompute agrees (<1e-6)",
          pw.dtype == np.float32 and len(pw) == m["n_windows"] == N_VAL and abs(float(pw.astype(np.float64).mean()) - m["score_mse"]) < 1e-6
          and all(k in pv for k in ("rc_commit", "audited_commit", "wrapper_sha256", "argv", "uv_pip_freeze_sha256", "start_utc", "end_utc", "pid"))
          and m["val_finite"] and float(m["val_loss_min_repr"]) == min(float(v) for v in m["val_losses"]), N=len(pw))

json.dump({"tests": RES, "note": "VAL split scored; no test number computed"}, open(os.path.join(HERE, "selftest", "battery.json"), "w"), indent=1, default=float)
print(f"{sum(r['pass'] for r in RES)}/{len(RES)} PASS")
