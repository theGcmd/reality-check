"""Run TFPS's own run_longExp.py with a defined initialisation of the cluster bases D.
Does not edit the repo: patches layers.Cluster.EDESC.__init__ in-process.
TFPS_D_INIT = asis | zero | orth.  Optional TFPS_TEST_DROP_LAST = 0 to keep every test window."""
import os, sys, runpy, torch
REPO = os.environ.get("TFPS_REPO") or os.path.join(os.environ.get("RC_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "work")), "repos", "tfps")
REPO = os.path.abspath(REPO)
os.chdir(REPO); sys.path.insert(0, REPO)
import layers.Cluster as C
_orig = C.EDESC.__init__
def _init(self, *a, **k):
    _orig(self, *a, **k)
    mode = os.environ.get("TFPS_D_INIT", "asis")
    with torch.no_grad():
        if mode == "zero": self.D.zero_()
        elif mode == "orth": torch.nn.init.orthogonal_(self.D)
        elif mode != "asis": raise ValueError(mode)
    print(f"[audit] D init={mode} nonfinite={int((~torch.isfinite(self.D)).sum())} absmax={float(self.D.abs().nan_to_num(0).max()):.3g}", flush=True)
C.EDESC.__init__ = _init
if os.environ.get("TFPS_TEST_DROP_LAST") == "0":
    import data_provider.data_factory as F
    _dl = F.DataLoader
    def _DataLoader(ds, *a, **k):
        if k.get("shuffle") is False and k.get("drop_last") is True and getattr(ds, "set_type", None) == 2:
            k["drop_last"] = False
        return _dl(ds, *a, **k)
    F.DataLoader = _DataLoader
sys.argv = ["run_longExp.py"] + sys.argv[1:]
runpy.run_path("run_longExp.py", run_name="__main__")
