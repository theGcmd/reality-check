"""TFPS-1 step (c) self-tests (PREREG r3.1 §8). Seed 999 only; prints PASS/FAIL and diffs, never a test MSE/MAE.
Usage: python selftest_tfps.py fast|train   (writes audits/tfps/selftest/<name>.json)"""
import os, sys, json, math, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_tfps as R
from rc_tfps import RC, torch, np
import exp.exp_main as E
import utils.metrics as UM

OUT = os.path.join(R.HERE, "selftest"); os.makedirs(OUT, exist_ok=True)
RES = []


def check(name, ok, **info):
    RES.append({"test": name, "pass": bool(ok), **info})
    print(("PASS " if ok else "FAIL ") + name + (" " + json.dumps(info, default=float) if info else ""), flush=True)


def build(arm, H=192, seed=999):
    RC.reset(); RC.arm, RC.seed = arm, seed
    R.install_hooks(True)
    args = R.build_args(R.tfps_argv(arm, H, seed, "/nonexistent"))
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)  # as run_longExp.py:112-115
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        exp = E.Exp_Main(args)
    R.name_routers(exp.model)
    return exp, args, torch.get_rng_state().clone()


def fast():
    # seed-set disjointness
    train = set(R.S) | set(R.EXT) | {R.TIMING_SEED}
    allsd = sorted(train)
    dset = {x for s in allsd for x in R.d_seeds(s)}
    nset = {R.noise_seed(k, s) for s in allsd for k in range(1, R.K_DRAWS + 1)}
    check("seed sets pairwise disjoint", not (train & dset) and not (train & nset) and not (dset & nset)
          and len(dset) == 2 * len(allsd) and len(nset) == R.K_DRAWS * len(allsd))

    exps = {a: build(a) for a in R.ARMS}
    Dg = {w: RC_d for w, RC_d in []}
    g, gu, z = (exps[a][0].model for a in R.ARMS)
    Gt, Gf = g.model_time.cluster.D.detach(), g.model_frequency.cluster.D.detach()
    ref_t = R.make_D(tuple(Gt.shape), R.d_seeds(999)[0], "tfps_gauss")
    check("gauss: time-D is the 900000+seed draw, freq-D the 910000+seed draw",
          torch.equal(Gt, ref_t) and torch.equal(Gf, R.make_D(tuple(Gf.shape), R.d_seeds(999)[1], "tfps_gauss")))
    check("gauss: finite, mean~0, sd~1", bool(torch.isfinite(Gt).all() and torch.isfinite(Gf).all())
          and abs(float(Gt.mean())) < 0.02 and abs(float(Gt.std()) - 1) < 0.02,
          shape=list(Gt.shape), mean=float(Gt.mean()), sd=float(Gt.std()))
    Ut = gu.model_time.cluster.D.detach()
    check("gauss_unit == gauss/sqrt(112) exactly; mean column norm ~1",
          torch.equal(Ut, Gt / math.sqrt(112)) and torch.equal(gu.model_frequency.cluster.D.detach(), Gf / math.sqrt(112))
          and abs(float(Ut.norm(dim=0).mean()) - 1) < 0.05, colnorm=float(Ut.norm(dim=0).mean()))
    check("time-D != freq-D", not torch.equal(Gt, Gf))
    check("zero: D == 0 at init", float(z.model_time.cluster.D.abs().max()) == 0 and float(z.model_frequency.cluster.D.abs().max()) == 0)
    sds = {a: exps[a][0].model.state_dict() for a in R.ARMS}
    nonD = [k for k in sds["tfps_gauss"] if not k.endswith("cluster.D")]
    eq = all(torch.equal(sds["tfps_gauss"][k], sds[a][k]) for a in R.ARMS[1:] for k in nonD)
    check("all non-D parameters bitwise equal across arms at construction", eq, n_tensors=len(nonD))
    check("global RNG state after construction equal across arms",
          all(torch.equal(exps["tfps_gauss"][2], exps[a][2]) for a in R.ARMS[1:]))

    # zero stays zero after 3 optimiser steps (released loss: forecast + aux, Adam)
    exp, args, _ = exps["tfps_zero"]
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        tr_ds, tr_dl = exp._get_data("train")
    opt = exp._select_optimizer(); crit = exp._select_criterion(); exp.model.train()
    it = iter(tr_dl); grads = []
    for _ in range(3):
        bx, by, _, _ = next(it); bx = bx.float(); by = by.float()
        opt.zero_grad()
        st, sf, out = exp.model(bx)
        lt = exp.model.model_time.cluster.total_loss(pred=st, target=exp._refined_subspace_affinity(st.data), dim=28, n_clusters=4, beta=0.1)
        lf = exp.model.model_frequency.cluster.total_loss(pred=sf, target=exp._refined_subspace_affinity(sf.data), dim=28, n_clusters=4, beta=0.1)
        loss = crit(out[:, -args.pred_len:, :], by[:, -args.pred_len:, :]) + lt + lf
        loss.backward(); grads.append(float(exp.model.model_time.cluster.D.grad.abs().max())); opt.step()
    zt = float(exp.model.model_time.cluster.D.abs().max()); zf = float(exp.model.model_frequency.cluster.D.abs().max())
    check("zero: D still == 0 after 3 Adam steps", zt == 0 and zf == 0, max_grad_time=max(grads))

    # router: noise-off -> indices == argmax; noise-on -> bit-identical to released forward under same RNG
    router = R._M.NoisyTopkRouter_Cluster(1); router.eval(); router._rc_name = "t"
    logits = torch.softmax(torch.randn(5000, 4), -1)
    RC.noise_off = True; o, idx = router(logits); RC.noise_off = False
    check("router noise-off: indices == argmax s on every row", torch.equal(idx[:, 0], logits.argmax(-1)))
    st = torch.get_rng_state(); o1, i1 = router(logits)
    torch.set_rng_state(st); o2, i2 = R._released_router_forward(router, logits)
    check("router noise-on (eval): bit-identical to released forward", torch.equal(o1, o2) and torch.equal(i1, i2))
    router.train(); RC.noise_off = True
    st = torch.get_rng_state(); o1, i1 = router(logits); torch.set_rng_state(st); o2, i2 = R._released_router_forward(router, logits)
    RC.noise_off = False
    check("router train mode ignores the eval-only switch", torch.equal(o1, o2) and torch.equal(i1, i2))
    RC.record = True; RC.rec = {}; router.train(); router(logits); rec_train = dict(RC.rec)
    router.eval(); router(logits); RC.record = False
    check("router recording: eval only", "t" not in rec_train and len(RC.rec["t"]["idx"]) == 1)

    # A ceiling: one-hot s, n = 4, 100k rows, released router
    torch.manual_seed(12345)
    n = 100000; hot = torch.randint(0, 4, (n,)); s1 = torch.nn.functional.one_hot(hot, 4).float()
    _, ii = R._released_router_forward(router, s1)
    A1 = float((ii[:, 0] == hot).float().mean())
    s_u = torch.full((n, 4), 0.25); _, iu = R._released_router_forward(router, s_u)
    Au = float((iu[:, 0] == 0).float().mean())
    check("A ceiling simulated (one-hot s, n=4, 100k rows)", 0.25 < A1 < 1, A_ceiling_onehot=A1, A_uniform=Au, rows=n)
    json.dump({"A_ceiling_onehot_n4": A1, "A_uniform_n4": Au, "rows": n, "seed": 12345},
              open(os.path.join(OUT, "A_ceiling.json"), "w"), indent=1)

    # eval pass: concatenation, keep-all, released-loader prefix, metrics.MSE agreement (values NOT printed)
    exp, args, _ = exps["tfps_gauss"]
    with contextlib.redirect_stdout(io.StringIO()):
        ds, dl_full = R.full_test_loader(args)
        _, dl_rel = exp._get_data("test")
    torch.manual_seed(R.noise_seed(1, 999)); m_full, a_full = R.eval_pass(exp.model, dl_full, args)
    torch.manual_seed(R.noise_seed(1, 999)); m_rel, _ = R.eval_pass(exp.model, dl_rel, args)
    N = len(ds); p = (N // 128) * 128
    check("eval pass: rows == dataset length (drop_last off)", len(m_full) == N, N=N, released_rows=len(m_rel))
    check("eval pass: first floor(N/128)*128 rows reproduce drop_last pass (same RNG) to 1e-6",
          len(m_rel) == p and abs(float(m_full[:p].mean()) - float(m_rel.mean())) < 1e-6
          and float(np.abs(m_full[:p] - m_rel).max()) < 1e-6, abs_diff=float(abs(m_full[:p].mean() - m_rel.mean())))
    # independent: collect preds/trues with the repo's metric
    torch.manual_seed(R.noise_seed(1, 999)); P, T = [], []
    exp.model.eval()
    with torch.no_grad():
        for bx, by, _, _ in dl_full:
            _, _, out = exp.model(bx.float()); P.append(out[:, -args.pred_len:, :].numpy()); T.append(by.float()[:, -args.pred_len:, :].numpy())
    P = np.concatenate(P); T = np.concatenate(T)
    d = abs(float(m_full.mean()) - float(UM.MSE(P, T))); d2 = abs(float(a_full.mean()) - float(UM.MAE(P, T)))
    check("eval pass: mean per-window MSE/MAE == utils.metrics MSE/MAE on concatenated arrays to 1e-6", d < 1e-6 and d2 < 1e-6,
          abs_diff_mse=d, abs_diff_mae=d2)
    torch.manual_seed(R.noise_seed(1, 999)); m_b, _ = R.eval_pass(exp.model, dl_full, args)
    check("eval pass: same noise seed -> identical per-window MSE", np.array_equal(m_full, m_b))
    torch.manual_seed(R.noise_seed(2, 999)); m_c, _ = R.eval_pass(exp.model, dl_full, args)
    check("eval pass: different noise seed -> different routing (draws differ)", not np.array_equal(m_full, m_c))

    import extension_trigger
    msg = extension_trigger._selftest(); check("extension_trigger " + msg, "PASS" in msg)
    json.dump(RES, open(os.path.join(OUT, "fast.json"), "w"), indent=1, default=float)


def train_tests():
    """2-epoch runs, seed 999, H=192: instrumentation on vs off; EarlyStopping and val-hook tests; timing."""
    import shutil
    base = os.path.join(R.RC_WORK, "selftest_tfps"); shutil.rmtree(base, ignore_errors=True)
    out = {}
    for inst in (True, False):
        tag = "on" if inst else "off"
        rd, ck = os.path.join(base, "run_" + tag), os.path.join(base, "ckpt_" + tag)
        info = R.run_tfps("tfps_gauss", 192, 999, rd, ck, train_epochs=2, prefreeze=True, instrument=inst,
                          log_path=os.path.join(OUT, f"train2_gauss192_{tag}.log"))
        info["rng_end"] = torch.get_rng_state().clone(); info["np_end"] = np.random.get_state()[1].copy()
        info["sd"] = torch.load(os.path.join(info["ckpt_dir"], "checkpoint.pth"))
        if inst:
            info["epochs"] = list(RC.epochs); info["es"] = json.load(open(os.path.join(info["ckpt_dir"], "es.json")))
            info["best_valmse_epoch"] = RC.best_valmse_epoch
        out[tag] = info
    on, off = out["on"], out["off"]
    check("instrumentation leaves training unchanged: checkpoint state_dict bitwise equal",
          all(torch.equal(on["sd"][k], off["sd"][k]) for k in on["sd"]))
    check("instrumentation leaves training unchanged: global torch + numpy RNG equal at end",
          torch.equal(on["rng_end"], off["rng_end"]) and np.array_equal(on["np_end"], off["np_end"]))
    vl = [float(e["val_loss"]) for e in on["epochs"]]
    check("val hook fired exactly once per epoch (val loader only)", len(on["epochs"]) == 2)
    check("EarlyStopping: val_loss_min == min per-epoch val loss; saved epoch matches",
          float(on["es"]["val_loss_min_repr"]) == min(vl) and on["es"]["best_epoch"] == 1 + int(np.argmin(vl)))
    # restore the saved RNG state on the saved checkpoint -> reproduces recorded val loss / val MSE
    RC.reset(); RC.arm, RC.seed = "tfps_gauss", 999; R.install_hooks(True)
    args = R.build_args(R.tfps_argv("tfps_gauss", 192, 999, "/nonexistent"))
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        exp = E.Exp_Main(args); vds, vdl = exp._get_data("val")
        exp._get_profile(exp.model)  # thop adds total_ops/total_params buffers, as in train() (exp_main.py:142)
    be = on["es"]["best_epoch"]; exp.model.load_state_dict(on["sd"])
    RC.ckpt_dir = os.path.join(base, "replay"); os.makedirs(RC.ckpt_dir, exist_ok=True)
    rng = torch.load(os.path.join(on["ckpt_dir"], f"valrng_e{be}.pt"))
    torch.set_rng_state(rng["torch"]); np.random.set_state(rng["numpy"]); random.setstate(rng["python"])
    RC.epoch = be - 1
    vloss = exp.vali(vds, vdl, exp._select_criterion())
    rec = RC.epochs[-1]
    check("val hook: restoring saved RNG on saved checkpoint reproduces val loss and val MSE to 1e-6",
          abs(float(vloss) - float(on["epochs"][be - 1]["val_loss"])) < 1e-6
          and abs(float(rec["val_mse"]) - float(on["epochs"][be - 1]["val_mse"])) < 1e-6)
    sd_v = torch.load(os.path.join(on["ckpt_dir"], "checkpoint_valmse.pth"))
    vm = [float(e["val_mse"]) for e in on["epochs"]]
    check("pure-val-MSE checkpoint saved at argmin val MSE (ties earliest); checkpoint.pth untouched",
          on["best_valmse_epoch"] == 1 + int(np.argmin(vm)) and os.path.exists(os.path.join(on["ckpt_dir"], "checkpoint.pth")))
    check("pre-FREEZE: no result files, no test_results dir in run dirs",
          not any(f.startswith("result") or f in ("test_results", "results") for t in ("on", "off") for f in os.listdir(os.path.join(base, "run_" + t))))
    timing = {t: {"wall_s_2_epochs": out[t]["wall_s"], "filtered_lines": out[t]["filtered_lines"]} for t in out}
    json.dump({"tests": RES, "timing": timing, "per_epoch_on": on["epochs"]}, open(os.path.join(OUT, "train.json"), "w"), indent=1, default=float)
    print(json.dumps(timing))


def seg(tag, impl, budget):
    """One process: a 4-epoch run, seed 999, H=192, arm gauss; writes final RNG + records to <base>/<tag>.final.pt."""
    base = os.path.join(R.RC_WORK, "selftest_tfps_resume")
    b = None if budget == "none" else float(budget)
    info = R.run_tfps("tfps_gauss", 192, 999, os.path.join(base, "run_" + tag), os.path.join(base, "ck_" + tag),
                      train_epochs=4, prefreeze=True, instrument=True, log_path=os.path.join(OUT, f"resume_{tag}.log"),
                      train_impl=impl, seg_budget_s=b)
    if info["finished"]:
        torch.save({"rng": torch.get_rng_state(), "np": np.random.get_state()[1], "epochs": RC.epochs,
                    "ckpt_dir": info["ckpt_dir"]}, os.path.join(base, tag + ".final.pt"))
    print(json.dumps({"tag": tag, "finished": info["finished"], "seg_after": info["segment_ended_after_epoch"], "wall_s": info["wall_s"]}))


def resume_tests():
    import subprocess, shutil
    base = os.path.join(R.RC_WORK, "selftest_tfps_resume"); shutil.rmtree(base, ignore_errors=True); os.makedirs(base)
    py = sys.executable; me = os.path.abspath(__file__); walls = {}
    def go(tag, impl, budget):
        r = subprocess.run([py, me, "seg", tag, impl, budget], capture_output=True, text=True)
        last = [l for l in r.stdout.splitlines() if l.startswith("{")]
        assert r.returncode == 0, r.stderr[-2000:]
        return json.loads(last[-1])
    walls["released"] = go("released", "released", "none")["wall_s"]
    walls["resumable_uninterrupted"] = go("resum", "resumable", "none")["wall_s"]
    segs = []
    while True:
        o = go("seg", "resumable", "0"); segs.append(o)
        if o["finished"] or len(segs) > 6: break
    def load(tag):
        f = torch.load(os.path.join(base, tag + ".final.pt"))
        f["sd"] = torch.load(os.path.join(f["ckpt_dir"], "checkpoint.pth")); f["sdv"] = torch.load(os.path.join(f["ckpt_dir"], "checkpoint_valmse.pth"))
        return f
    A, B, C = load("released"), load("resum"), load("seg")
    def same(x, y):
        return (all(torch.equal(x["sd"][k], y["sd"][k]) for k in x["sd"]) and all(torch.equal(x["sdv"][k], y["sdv"][k]) for k in x["sdv"])
                and torch.equal(x["rng"], y["rng"]) and np.array_equal(x["np"], y["np"])
                and [e["val_loss"] for e in x["epochs"]] == [e["val_loss"] for e in y["epochs"]]
                and [e["val_mse"] for e in x["epochs"]] == [e["val_mse"] for e in y["epochs"]])
    check("resumable train() (uninterrupted, 4 epochs) bitwise equal to released train(): checkpoints, val losses, RNG", same(A, B))
    check("run interrupted after EVERY epoch (4 processes) and resumed == uninterrupted at epoch 4, bitwise", same(B, C) and len(segs) == 4,
          segments=[s["seg_after"] for s in segs])
    json.dump({"tests": RES, "wall_s": walls, "segments": segs}, open(os.path.join(OUT, "resume.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "seg":
        seg(*sys.argv[2:5])
    else:
        {"fast": fast, "train": train_tests, "resume": resume_tests}[sys.argv[1]]()
