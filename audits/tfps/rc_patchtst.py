"""TFPS-1 baseline wrapper: PatchTST / DLinear in the official PatchTST repo (patchtst_official @ 204c21ef,
PatchTST_supervised/). No model patches. Separate process from rc_tfps (module names collide).

Hooks (file:line overridden in PatchTST_supervised/):
  B1 utils/tools.py:40-69 via exp/exp_main.py:4,112  EarlyStopping subclass: repr(val_loss_min) + best epoch, per-epoch
                                     val losses (instrument only).
  B2 exp/exp_main.py:219-310         Exp_Main.test replaced by the audit scoring pass (full test set, drop_last off,
                                     concatenation; deterministic, one pass). Pre-FREEZE: no test pass, except the
                                     self-test mode 'compare', which records only |released-loader score - printed MSE|.
Arms: patchtst (official ETTh1 arch, scripts/PatchTST/etth1.sh:30-42), dlinear; individual 0; seq_len 96; batch 128;
train_epochs 100; patience 20 (official default 100, :80); lradj type3; lr in {1e-4, 5e-4, 1e-3, 5e-3}; num_workers 0.
"""
import os, sys, io, re, json, time, random, hashlib, runpy, subprocess, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RC_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RC_WORK = os.path.abspath(os.environ.get("RC_WORK", os.path.join(RC_ROOT, "work")))
REPO = os.path.join(RC_WORK, "repos", "patchtst_official", "PatchTST_supervised")
DATA_ROOT = os.path.join(RC_WORK, "repos", "ETDataset", "ETT-small") + "/"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import torch
import exp.exp_main as _E
import utils.tools as _T
import utils.metrics as _UM
from data_provider.data_factory import data_provider as _data_provider
from torch.utils.data import DataLoader

LR_GRID = (1e-4, 5e-4, 1e-3, 5e-3)
PUBLISHED_LR = {"patchtst": 1e-4, "dlinear": 5e-3}
TIMING_SEED = 999


class _St:
    def reset(self):
        self.ckpt_dir = None; self.instrument = True; self.mode = "prefreeze"; self.epoch = 0
        self.val = []; self.es = None; self.printed = None; self.scores = None


ST = _St(); ST.reset()


class RCEarlyStopping(_T.EarlyStopping):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.best_epoch = None; ST.es = self

    def __call__(self, val_loss, model, path):
        ST.epoch += 1; ST.val.append(repr(float(val_loss)))
        return super().__call__(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        super().save_checkpoint(val_loss, model, path)
        self.best_epoch = ST.epoch
        json.dump({"val_loss_min_repr": repr(float(val_loss)), "best_epoch": ST.epoch, "val_losses": ST.val},
                  open(os.path.join(ST.ckpt_dir, "es.json"), "w"), indent=1)


def full_test_loader(args):
    ds, _ = _data_provider(args, "test")
    return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)


@torch.no_grad()
def eval_pass(model, loader, args):
    model.eval(); mse, mae = [], []
    for bx, by, _, _ in loader:
        out = model(bx.float())[:, -args.pred_len:, 0:]; by = by.float()[:, -args.pred_len:, 0:]
        e = out.double() - by.double()
        mse.append((e ** 2).mean(dim=(1, 2)).numpy()); mae.append(e.abs().mean(dim=(1, 2)).numpy())
    return np.concatenate(mse), np.concatenate(mae)


_orig_test = _E.Exp_Main.test
_orig_metric = _E.metric


def _test(self, setting, test=0):
    if ST.mode == "prefreeze":
        return
    ck = os.path.join(ST.ckpt_dir, "checkpoint.pth")
    self.model.load_state_dict(torch.load(ck))
    if ST.mode in ("compare", "score"):
        cap = {}

        def _m(p, t):
            r = _orig_metric(p, t); cap["mae"], cap["mse"] = float(r[0]), float(r[1]); return r
        _E.metric = _m
        try:
            _orig_test(self, setting)  # repo's own released-loader test (prints; filtered in compare mode)
        finally:
            _E.metric = _orig_metric
        ST.printed = cap
    ds, dl = full_test_loader(self.args)
    m, a = eval_pass(self.model, dl, self.args)
    ST.scores = {"checkpoint": ck, "n_windows": len(ds), "mse": m, "mae": a}


def install(instrument=True):
    ST.instrument = instrument
    _E.EarlyStopping = RCEarlyStopping if instrument else _T.EarlyStopping
    _E.Exp_Main.test = _test


_FILTER = re.compile(r"test|mse:|mae:", re.I)


class LineFilter(io.TextIOBase):
    def __init__(self, sink): self.sink = sink; self.buf = ""; self.dropped = 0

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if _FILTER.search(line): self.dropped += 1
            else: self.sink.write(line + "\n")
        return len(s)

    def flush(self): self.sink.flush()


def argv(arm, H, seed, lr, ckpt_root, train_epochs=100):
    a = ["run_longExp.py", "--random_seed", str(seed), "--is_training", "1", "--root_path", DATA_ROOT,
         "--data_path", "ETTh1.csv", "--model_id", f"{arm}_{H}_s{seed}_lr{lr:g}",
         "--model", {"patchtst": "PatchTST", "dlinear": "DLinear"}[arm], "--data", "ETTh1", "--features", "M",
         "--seq_len", "96", "--pred_len", str(H), "--enc_in", "7", "--des", "Exp", "--train_epochs", str(train_epochs),
         "--patience", "20", "--lradj", "type3", "--itr", "1", "--batch_size", "128", "--learning_rate", f"{lr:g}",
         "--individual", "0", "--num_workers", "0", "--checkpoints", ckpt_root]
    if arm == "patchtst":
        a += ["--e_layers", "3", "--n_heads", "4", "--d_model", "16", "--d_ff", "128", "--dropout", "0.3",
              "--fc_dropout", "0.3", "--head_dropout", "0", "--patch_len", "16", "--stride", "8"]
    return a


def run(arm, H, seed, lr, run_dir, ckpt_root, train_epochs=100, mode="prefreeze", instrument=True, log_path=None):
    if mode != "score":
        assert seed == TIMING_SEED, "pre-FREEZE runs must use seed 999"
    os.makedirs(run_dir, exist_ok=True); os.makedirs(ckpt_root, exist_ok=True)
    ST.reset(); ST.mode = mode; install(instrument)
    av = argv(arm, H, seed, lr, ckpt_root, train_epochs)
    orig_train = _E.Exp_Main.train

    def _train(self, setting):
        ST.ckpt_dir = os.path.join(ckpt_root, setting); os.makedirs(ST.ckpt_dir, exist_ok=True)
        return orig_train(self, setting)
    _E.Exp_Main.train = _train
    cwd = os.getcwd(); os.chdir(run_dir); old = sys.argv; sys.argv = av
    logf = open(log_path or os.path.join(run_dir, "train.log"), "w")
    sink = logf if mode == "score" else LineFilter(logf)
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(sink):
            runpy.run_path(os.path.join(REPO, "run_longExp.py"), run_name="__main__")
    finally:
        _E.Exp_Main.train = orig_train; sys.argv = old; os.chdir(cwd); sink.flush(); logf.close()
    subs = [d for d in os.listdir(ckpt_root) if os.path.isdir(os.path.join(ckpt_root, d))]
    assert len(subs) == 1 and os.path.exists(os.path.join(ckpt_root, subs[0], "checkpoint.pth")), subs
    if mode != "score":
        for junk in ("test_results", "results"):
            subprocess.run(["rm", "-rf", os.path.join(run_dir, junk)])
        for f in os.listdir(run_dir):
            if f.startswith("result") and f.endswith(".txt"): os.remove(os.path.join(run_dir, f))
    return {"wall_s": time.time() - t0, "ckpt_dir": os.path.join(ckpt_root, subs[0]), "argv": av,
            "filtered_lines": getattr(sink, "dropped", 0)}


def selftest():
    """seed 999, H=192, 2 epochs: (1) instrumentation on/off bitwise; (2) ES record; (3) released-loader score ==
    repo printed test MSE to 1e-5 (only the difference is recorded); per arm. Prints no MSE/MAE."""
    import shutil
    out_dir = os.path.join(HERE, "selftest"); res = []; timing = {}
    base = os.path.join(RC_WORK, "selftest_patchtst"); shutil.rmtree(base, ignore_errors=True)

    def check(n, ok, **i):
        res.append({"test": n, "pass": bool(ok), **i}); print(("PASS " if ok else "FAIL ") + n + (" " + json.dumps(i) if i else ""), flush=True)
    for arm in ("patchtst", "dlinear"):
        r = {}
        for inst in (True, False):
            t = f"{arm}_{'on' if inst else 'off'}"
            info = run(arm, 192, 999, PUBLISHED_LR[arm], os.path.join(base, "run_" + t), os.path.join(base, "ck_" + t),
                       train_epochs=2, mode="prefreeze", instrument=inst, log_path=os.path.join(out_dir, f"train2_{t}.log"))
            info["rng"] = torch.get_rng_state().clone(); info["np"] = np.random.get_state()[1].copy()
            info["sd"] = torch.load(os.path.join(info["ckpt_dir"], "checkpoint.pth"))
            if inst: info["es"] = json.load(open(os.path.join(info["ckpt_dir"], "es.json")))
            r[t] = info; timing[t] = info["wall_s"]
        on, off = r[f"{arm}_on"], r[f"{arm}_off"]
        check(f"{arm}: instrumentation leaves training unchanged (state_dict, torch+numpy RNG bitwise)",
              all(torch.equal(on["sd"][k], off["sd"][k]) for k in on["sd"]) and torch.equal(on["rng"], off["rng"])
              and np.array_equal(on["np"], off["np"]))
        vl = [float(v) for v in on["es"]["val_losses"]]
        check(f"{arm}: EarlyStopping val_loss_min == min per-epoch val loss; saved epoch matches",
              len(vl) == 2 and float(on["es"]["val_loss_min_repr"]) == min(vl) and on["es"]["best_epoch"] == 1 + int(np.argmin(vl)))
        t = f"{arm}_cmp"
        info = run(arm, 192, 999, PUBLISHED_LR[arm], os.path.join(base, "run_" + t), os.path.join(base, "ck_" + t),
                   train_epochs=2, mode="compare", instrument=True, log_path=os.path.join(out_dir, f"train2_{t}.log"))
        m = ST.scores["mse"]; p = (len(m) // 128) * 128
        d = abs(float(m[:p].mean()) - ST.printed["mse"])
        check(f"{arm}: released-loader score (first floor(N/128)*128 windows) == repo printed test MSE to 1e-5",
              d < 1e-5 and len(m) == ST.scores["n_windows"], abs_diff=d, N=len(m))
        ST.scores = None; ST.printed = None
    json.dump({"tests": res, "timing_wall_s_2_epochs": timing}, open(os.path.join(out_dir, "baselines.json"), "w"), indent=1)
    print(json.dumps(timing))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
