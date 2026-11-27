"""TFPS-1 audit wrapper (PREREG r3.1 §5, §6, §8). Never edits work/repos/tfps; patches in-process only.

Patches (file:line overridden, all at 83a11827):
  P1 layers/Cluster.py:30            EDESC.__init__: D := gauss | gauss_unit | zero from a dedicated torch.Generator
                                     (900000+seed for the 1st EDESC built = time, 910000+seed for the 2nd = freq).
  P2 layers/SparseMoE.py:57-66       NoisyTopkRouter_Cluster.forward: eval-only noise switch (RC.noise_off) and
                                     eval-only recording of (indices, argmax s, row max, row range). Train path and
                                     switch-on path are the released ops in the released order.
  P3 utils/tools.py:40-69 via exp/exp_main.py:4,152  EarlyStopping subclass: records repr(val_loss_min) + best epoch.
  P4 exp/exp_main.py:73-135          Exp_Main.vali hook, keyed on the val-loader object (built at :139): RNG state
                                     saved before each val pass, val MSE (criterion part) + aux parts recorded,
                                     checkpoint_valmse.pth saved at the lowest val MSE (ties -> earliest).
  P5 exp/exp_main.py:50-52           Exp_Main._get_data: remembers the val loader object; num_workers forced 0 by argv.
  P6 exp/exp_main.py:277-397         Exp_Main.test replaced by the audit scoring pass (full test set, drop_last off,
                                     concatenation, K=5 eval-noise draws + noise-off), unless RC.prefreeze, in which
                                     case NO test pass of any kind is run.
"""
import os, sys, io, re, json, math, time, random, hashlib, runpy, socket, subprocess, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RC_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RC_WORK = os.path.abspath(os.environ.get("RC_WORK", os.path.join(RC_ROOT, "work")))
REPO = os.path.join(RC_WORK, "repos", "tfps")
DATA_ROOT = os.path.join(RC_WORK, "repos", "ETDataset", "ETT-small") + "/"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

S = [2021, 42, 123, 456, 7]
EXT = [1, 2, 3, 4, 5]
TIMING_SEED = 999
K_DRAWS = 5
ARMS = ("tfps_gauss", "tfps_gauss_unit", "tfps_zero")
HCFG = {192: dict(lr=5e-4, T=4, F=4), 336: dict(lr=5e-3, T=4, F=4)}  # scripts/etth1.sh:77-89, :133-145


def d_seeds(seed):
    return 900000 + seed, 910000 + seed


def noise_seed(k, seed):
    return 10000 * k + seed


class _RC:
    """Mutable audit state shared by the patches."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.arm = None; self.seed = None
        self.edesc_count = 0; self.d_init = {}
        self.noise_off = False; self.record = False; self.rec = {}
        self.val_loader = None; self.epoch = 0; self.epochs = []
        self.best_valmse = math.inf; self.best_valmse_epoch = None
        self.ckpt_dir = None; self.prefreeze = True; self.instrument = True
        self.aux_rec = None; self.val_rng = {}; self.es = None; self.scores = None; self.seg_budget_s = None


RC = _RC()

# ---------------------------------------------------------------- P1: D init
import layers.Cluster as _C
_orig_edesc_init = _C.EDESC.__init__


def make_D(shape, gen_seed, mode):
    if mode == "zero":
        return torch.zeros(shape)
    g = torch.Generator().manual_seed(gen_seed)
    G = torch.randn(shape, generator=g)
    if mode == "tfps_gauss":
        return G
    if mode == "tfps_gauss_unit":
        return G / math.sqrt(shape[0])
    raise ValueError(mode)


def _edesc_init(self, *a, **k):
    _orig_edesc_init(self, *a, **k)
    if RC.arm is None:  # every audit arm defines D; refusing avoids silently using uninitialised memory
        raise RuntimeError("RC.arm not set: D would be uninitialised")
    which = "time" if RC.edesc_count == 0 else "freq"
    gseed = d_seeds(RC.seed)[0 if which == "time" else 1]
    mode = "zero" if RC.arm == "tfps_zero" else RC.arm
    with torch.no_grad():
        self.D.copy_(make_D(tuple(self.D.shape), gseed, mode))
    self._rc_name = which
    RC.d_init[which] = self.D.detach().clone()
    RC.edesc_count += 1


_C.EDESC.__init__ = _edesc_init

# ---------------------------------------------------------------- P2: router
import layers.SparseMoE as _M


def _router_forward(self, logits):
    if (not self.training) and RC.noise_off:
        noisy_logits = logits
    else:
        noise = torch.randn_like(logits) * F.softplus(logits)
        noisy_logits = logits + noise
    top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
    zeros = torch.full_like(noisy_logits, float('-inf'))
    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
    router_output = F.softmax(sparse_logits, dim=-1)
    if (not self.training) and RC.record:
        r = RC.rec.setdefault(getattr(self, "_rc_name", "?"), {"idx": [], "am": [], "rmax": [], "rrange": []})
        r["idx"].append(indices[:, 0].detach().cpu().numpy().astype(np.int8))
        r["am"].append(logits.argmax(-1).detach().cpu().numpy().astype(np.int8))
        r["rmax"].append(logits.max(-1).values.detach().cpu().numpy())
        r["rrange"].append((logits.max(-1).values - logits.min(-1).values).detach().cpu().numpy())
    return router_output, indices


_released_router_forward = _M.NoisyTopkRouter_Cluster.forward
_M.NoisyTopkRouter_Cluster.forward = _router_forward


def name_routers(model):
    model.model_time.SaprseMoE.router._rc_name = "time"
    model.model_frequency.SaprseMoE.router._rc_name = "freq"

# ---------------------------------------------------------------- P3-P6: experiment hooks
import exp.exp_main as _E
import utils.tools as _T
from data_provider.data_factory import data_provider as _data_provider
from torch.utils.data import DataLoader


class RCEarlyStopping(_T.EarlyStopping):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.best_epoch = None
        RC.es = self

    def save_checkpoint(self, val_loss, model, path):
        super().save_checkpoint(val_loss, model, path)
        self.best_epoch = RC.epoch
        if RC.instrument:
            _write_json(os.path.join(RC.ckpt_dir, "es.json"),
                        {"val_loss_min_repr": repr(float(val_loss)), "best_epoch": RC.epoch})


_orig_get_data = _E.Exp_Main._get_data


def _get_data(self, flag):
    ds, dl = _orig_get_data(self, flag)
    if flag == "val":
        RC.val_loader = dl
    return ds, dl


class _RecCriterion:
    def __init__(self, crit):
        self.crit = crit; self.vals = []

    def __call__(self, a, b):
        l = self.crit(a, b); self.vals.append(l.item()); return l


_orig_total_loss = _C.EDESC.total_loss


def _total_loss(self, *a, **k):
    out = _orig_total_loss(self, *a, **k)
    if RC.aux_rec is not None:
        RC.aux_rec.setdefault(getattr(self, "_rc_name", "?"), []).append(out.item())
    return out


_orig_vali = _E.Exp_Main.vali


def _vali(self, vali_data, vali_loader, criterion):
    if not RC.instrument or vali_loader is not RC.val_loader:
        return _orig_vali(self, vali_data, vali_loader, criterion)
    assert getattr(vali_data, "set_type", None) == 1
    RC.epoch += 1
    rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    rc = _RecCriterion(criterion)
    RC.aux_rec = {}
    loss = _orig_vali(self, vali_data, vali_loader, rc)
    aux = RC.aux_rec; RC.aux_rec = None
    valmse = float(np.average(rc.vals))
    m = self.model
    Dt, Df = m.model_time.cluster.D.detach(), m.model_frequency.cluster.D.detach()
    rec = {"epoch": RC.epoch, "val_loss": repr(float(loss)), "val_mse": repr(valmse),
           "aux_time": float(np.average(aux.get("time", [np.nan]))), "aux_freq": float(np.average(aux.get("freq", [np.nan]))),
           "D_fro_time": float(Dt.norm()), "D_fro_freq": float(Df.norm()),
           "D_colnorm_time": float(Dt.norm(dim=0).mean()), "D_colnorm_freq": float(Df.norm(dim=0).mean())}
    RC.epochs.append(rec)
    RC.val_rng[RC.epoch] = rng
    if np.isfinite(valmse) and valmse < RC.best_valmse:
        RC.best_valmse, RC.best_valmse_epoch = valmse, RC.epoch
        torch.save(m.state_dict(), os.path.join(RC.ckpt_dir, "checkpoint_valmse.pth"))
        torch.save(rng, os.path.join(RC.ckpt_dir, "valrng_valmse.pt"))
    torch.save(rng, os.path.join(RC.ckpt_dir, f"valrng_e{RC.epoch}.pt"))
    _write_json(os.path.join(RC.ckpt_dir, "epochs.json"), RC.epochs)
    return loss


def full_test_loader(args):
    ds, _ = _data_provider(args, "test")
    return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)


@torch.no_grad()
def eval_pass(model, loader, args):
    """Per-window MSE/MAE in normalised space, mean over pred_len x 7 (as utils/metrics.py). Concatenates batches."""
    model.eval()
    mse, mae = [], []
    for bx, by, _, _ in loader:
        bx = bx.float(); by = by.float()
        _, _, out = model(bx)
        out = out[:, -args.pred_len:, 0:]; by = by[:, -args.pred_len:, 0:]
        e = (out.double() - by.double())
        mse.append((e ** 2).mean(dim=(1, 2)).numpy()); mae.append(e.abs().mean(dim=(1, 2)).numpy())
    return np.concatenate(mse), np.concatenate(mae)


def score_checkpoint(exp, ckpt, seed, tag):
    sd = torch.load(ckpt); exp.model.load_state_dict(sd)
    ds, dl = full_test_loader(exp.args)
    out = {"checkpoint": ckpt, "checkpoint_sha256": sha256(ckpt), "n_windows": len(ds), "draws": []}
    for k in range(1, K_DRAWS + 1):
        torch.manual_seed(noise_seed(k, seed)); RC.rec = {}; RC.record = True
        m, a = eval_pass(exp.model, dl, exp.args)
        RC.record = False
        out["draws"].append({"k": k, "noise_seed": noise_seed(k, seed), "mse": m, "mae": a, "rec": RC.rec})
    RC.noise_off = True; RC.rec = {}; RC.record = True
    m, a = eval_pass(exp.model, dl, exp.args)
    RC.noise_off = False; RC.record = False
    out["nonoise"] = {"mse": m, "mae": a, "rec": RC.rec}
    return out


def _test(self, setting, test=0):
    if RC.prefreeze:
        return  # leakage guard: no test pass at all before FREEZE
    RC.scores = {"released": score_checkpoint(self, os.path.join(RC.ckpt_dir, "checkpoint.pth"), RC.seed, "released"),
                 "valmse": score_checkpoint(self, os.path.join(RC.ckpt_dir, "checkpoint_valmse.pth"), RC.seed, "valmse")}


class SegmentDone(Exception):
    """Raised after a completed epoch when the segment's wall budget is spent; state is in resume.pt."""


def resumable_train(self, setting):
    """P7: replaces Exp_Main.train (exp_main.py:137-275) with the same operations in the same order, plus
    per-epoch resumable state (model, Adam, OneCycleLR, EarlyStopping, RC records, torch/numpy/python RNG).
    Bitwise equivalence to the released train() and resume == uninterrupted are self-tested (selftest_tfps resume)."""
    args = self.args
    train_data, train_loader = self._get_data(flag='train')
    vali_data, vali_loader = self._get_data(flag='val')
    test_data, test_loader = self._get_data(flag='test')
    print(self.model)
    self._get_profile(self.model)
    path = os.path.join(args.checkpoints, setting); os.makedirs(path, exist_ok=True)
    RC.ckpt_dir = path
    train_steps = len(train_loader)
    early_stopping = _E.EarlyStopping(patience=args.patience, verbose=True)
    model_optim = self._select_optimizer(); criterion = self._select_criterion()
    scheduler = _E.lr_scheduler.OneCycleLR(optimizer=model_optim, steps_per_epoch=train_steps, pct_start=args.pct_start,
                                           epochs=args.train_epochs, max_lr=args.learning_rate)
    start = 0; rp = os.path.join(path, "resume.pt")
    if os.path.exists(rp):
        st = torch.load(rp)
        self.model.load_state_dict(st["model"]); model_optim.load_state_dict(st["optim"])
        scheduler.load_state_dict(st["sched"])
        for k, v in st["es"].items(): setattr(early_stopping, k, v)
        for k, v in st["rc"].items(): setattr(RC, k, v)
        torch.set_rng_state(st["rng_torch"]); np.random.set_state(st["rng_numpy"]); random.setstate(st["rng_python"])
        start = st["epoch"]
        print(f"[audit] resumed after epoch {start}")
    t_seg = time.time()
    for epoch in range(start, args.train_epochs):
        train_loss = []
        self.model.train(); epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device); batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device); batch_y_mark = batch_y_mark.float().to(self.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(self.device)
            s_time, s_frequency, outputs = self.model(batch_x)
            s_tilde_time = self._refined_subspace_affinity(s=s_time.data)
            s_tilde_frequency = self._refined_subspace_affinity(s=s_frequency.data)
            n_z = args.c_out * args.d_model
            T_dim = int(n_z / args.T_num_expert); F_dim = int(n_z / args.F_num_expert)
            loss_cluster_time = self.model.model_time.cluster.total_loss(pred=s_time, target=s_tilde_time, dim=T_dim,
                                                                         n_clusters=args.T_num_expert, beta=args.beta)
            loss_cluster_frequency = self.model.model_frequency.cluster.total_loss(pred=s_frequency, target=s_tilde_frequency,
                                                                                   dim=F_dim, n_clusters=args.F_num_expert, beta=args.beta)
            outputs = outputs[:, -args.pred_len:, 0:]
            batch_y = batch_y[:, -args.pred_len:, 0:].to(self.device)
            loss_fore = criterion(outputs, batch_y)
            loss = loss_fore + args.alpha * loss_cluster_time + args.gama * loss_cluster_frequency
            train_loss.append(loss.item())
            loss.backward(); model_optim.step()
            if args.lradj == 'TST':
                raise RuntimeError("lradj TST not used in this audit")
        train_loss = np.average(train_loss)
        if not np.isfinite(train_loss):
            print("[audit] non-finite train loss")
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        vali_loss = self.vali(vali_data, vali_loader, criterion)
        test_loss = self.vali(test_data, test_loader, criterion)  # released per-epoch test pass (RNG stream); never logged
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))
        early_stopping(vali_loss, self.model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        _E.adjust_learning_rate(model_optim, scheduler, epoch + 1, args)
        if RC.seg_budget_s is not None and (time.time() - t_seg) > RC.seg_budget_s and epoch + 1 < args.train_epochs:
            st = {"model": self.model.state_dict(), "optim": model_optim.state_dict(), "sched": scheduler.state_dict(),
                  "es": {k: getattr(early_stopping, k, None) for k in ("counter", "best_score", "early_stop", "val_loss_min", "best_epoch")},
                  "rc": {k: getattr(RC, k) for k in ("epoch", "epochs", "best_valmse", "best_valmse_epoch")},
                  "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
                  "epoch": epoch + 1}
            torch.save(st, rp + ".tmp"); os.replace(rp + ".tmp", rp)
            raise SegmentDone(epoch + 1)
    _write_json(os.path.join(path, "done.json"), {"stop_epoch": RC.epoch, "early_stop": bool(early_stopping.early_stop)})
    self.model.load_state_dict(torch.load(path + '/' + 'checkpoint.pth'))
    return self.model


def install_hooks(instrument=True):
    RC.instrument = instrument
    _E.EarlyStopping = RCEarlyStopping if instrument else _T.EarlyStopping
    _E.Exp_Main._get_data = _get_data
    _E.Exp_Main.vali = _vali
    _E.Exp_Main.test = _test
    _C.EDESC.total_loss = _total_loss if instrument else _orig_total_loss


# ---------------------------------------------------------------- helpers
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _write_json(p, obj):
    with open(p + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(p + ".tmp", p)


_FILTER = re.compile(r"test|mse:|mae:", re.I)


class LineFilter(io.TextIOBase):
    """Drops any line containing 'Test Loss', 'test', 'mse:' or 'mae:' (case-insensitive) — PREREG §8 leakage guard."""
    def __init__(self, sink):
        self.sink = sink; self.buf = ""; self.dropped = 0

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if _FILTER.search(line): self.dropped += 1
            else: self.sink.write(line + "\n")
        return len(s)

    def flush(self):
        self.sink.flush()


def tfps_argv(arm, H, seed, ckpt_root, train_epochs=100):
    c = HCFG[H]
    return ["run_longExp.py", "--random_seed", str(seed), "--is_training", "1", "--root_path", DATA_ROOT,
            "--data_path", "ETTh1.csv", "--model_id", f"{arm}_{H}_s{seed}", "--model", "PatchTST_MoE_cluster",
            "--data", "ETTh1", "--features", "M", "--seq_len", "96", "--pred_len", str(H), "--enc_in", "7",
            "--e_layers", "3", "--n_heads", "4", "--d_model", "16", "--d_ff", "128", "--dropout", "0.3",
            "--fc_dropout", "0.3", "--head_dropout", "0", "--patch_len", "16", "--stride", "8",
            "--T_num_expert", str(c["T"]), "--T_top_k", "1", "--F_num_expert", str(c["F"]), "--F_top_k", "1",
            "--beta", "0.1", "--des", "Exp", "--train_epochs", str(train_epochs), "--itr", "1",
            "--batch_size", "128", "--learning_rate", str(c["lr"]), "--num_workers", "0",
            "--checkpoints", ckpt_root]


def build_args(argv):
    """Parse argv with run_longExp.py's own parser (run up to parse_args, nothing else)."""
    src = open(os.path.join(REPO, "run_longExp.py")).read()
    head = src.split("    args = parser.parse_args()")[0].replace("if __name__ == '__main__':", "if True:")
    g = {}
    old = sys.argv; sys.argv = argv
    try:
        exec(compile(head + "\n    args = parser.parse_args()\n", "run_longExp_parser", "exec"), g)
    finally:
        sys.argv = old
    a = g["args"]; a.use_gpu = False
    return a


def run_tfps(arm, H, seed, run_dir, ckpt_root, train_epochs=100, prefreeze=True, instrument=True, log_path=None,
             train_impl="released", seg_budget_s=None):
    """One training run through the released run_longExp.py (runpy), cwd = run_dir. Returns the Exp-trained model's
    state via the checkpoint dir. Pre-FREEZE: stdout filtered, no test pass, result files deleted."""
    assert arm in ARMS and H in HCFG
    if prefreeze:
        assert seed == TIMING_SEED, "pre-FREEZE runs must use seed 999"
    os.makedirs(run_dir, exist_ok=True); os.makedirs(ckpt_root, exist_ok=True)
    RC.reset(); RC.arm, RC.seed, RC.prefreeze, RC.seg_budget_s = arm, seed, prefreeze, seg_budget_s
    assert train_impl in ("released", "resumable") and (seg_budget_s is None or train_impl == "resumable")
    install_hooks(instrument)
    argv = tfps_argv(arm, H, seed, ckpt_root, train_epochs)
    cwd = os.getcwd(); os.chdir(run_dir)
    logf = open(log_path or os.path.join(run_dir, "train.log"), "a")
    sink = LineFilter(logf) if prefreeze else logf
    # setting dir is created by train(); RC.ckpt_dir is resolved lazily on first save via the setting prefix
    old_argv = sys.argv; sys.argv = argv
    orig_train = _E.Exp_Main.train

    def _train(self, setting):
        RC.ckpt_dir = os.path.join(ckpt_root, setting); os.makedirs(RC.ckpt_dir, exist_ok=True)
        return orig_train(self, setting)
    _E.Exp_Main.train = _train if train_impl == "released" else resumable_train
    t0 = time.time(); seg_done = None
    try:
        with contextlib.redirect_stdout(sink):
            runpy.run_path(os.path.join(REPO, "run_longExp.py"), run_name="__main__")
    except SegmentDone as e:
        seg_done = int(e.args[0])
    finally:
        _E.Exp_Main.train = orig_train; sys.argv = old_argv; os.chdir(cwd); sink.flush(); logf.close()
    wall = time.time() - t0
    subs = [d for d in os.listdir(ckpt_root) if os.path.isdir(os.path.join(ckpt_root, d))]
    assert len(subs) == 1, subs
    assert os.path.exists(os.path.join(ckpt_root, subs[0], "checkpoint.pth"))
    for junk in ("test_results", "results"):
        p = os.path.join(run_dir, junk)
        if os.path.isdir(p): subprocess.run(["rm", "-rf", p])
    for f in os.listdir(run_dir):
        if f.startswith("result") and f.endswith(".txt"): os.remove(os.path.join(run_dir, f))
    return {"wall_s": wall, "ckpt_dir": os.path.join(ckpt_root, subs[0]), "argv": argv, "segment_ended_after_epoch": seg_done,
            "finished": seg_done is None,
            "filtered_lines": getattr(sink, "dropped", 0)}


def provenance(argv, wrapper_files, extra=None):
    def git(cmd, cwd):
        return subprocess.run(["git"] + cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()
    py = os.path.join(RC_WORK, "venv", "bin", "python")
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", py], capture_output=True, text=True).stdout
    p = {"rc_commit": git(["rev-parse", "HEAD"], RC_ROOT), "rc_dirty": bool(git(["status", "--porcelain"], RC_ROOT)),
         "audited_commit": git(["rev-parse", "HEAD"], REPO),
         "wrapper_sha256": {os.path.basename(f): sha256(f) for f in wrapper_files},
         "argv": argv, "env": {k: os.environ.get(k) for k in ("CPU_REDIRECT", "OMP_NUM_THREADS", "PYTHONHASHSEED")},
         "uv_pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
         "torch": torch.__version__, "numpy": np.__version__, "host": socket.gethostname(), "pid": os.getpid(),
         "torch_threads": torch.get_num_threads()}
    if extra: p.update(extra)
    return p
