# CPU redirect for repos that hard-code CUDA. Active only when CPU_REDIRECT=1.
# Smoke/feasibility harness only: leaves repo code untouched.
import os
if os.environ.get("CPU_REDIRECT") == "1":
    import torch, functools
    def _fix(d):
        if isinstance(d, str) and d.startswith("cuda"): return "cpu"
        if isinstance(d, torch.device) and d.type == "cuda": return torch.device("cpu")
        if isinstance(d, int): return "cpu"
        return d
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    _tto = torch.Tensor.to
    def _tensor_to(self, *a, **k):
        a = tuple(_fix(x) if (isinstance(x, (str, torch.device))) else x for x in a)
        if "device" in k: k["device"] = _fix(k["device"])
        return _tto(self, *a, **k)
    torch.Tensor.to = _tensor_to
    _mto = torch.nn.Module.to
    def _module_to(self, *a, **k):
        a = tuple(_fix(x) if (isinstance(x, (str, torch.device))) else x for x in a)
        if "device" in k: k["device"] = _fix(k["device"])
        return _mto(self, *a, **k)
    torch.nn.Module.to = _module_to
    for fn in ["set_device", "synchronize", "empty_cache", "manual_seed", "manual_seed_all"]:
        setattr(torch.cuda, fn, lambda *a, **k: None)
    for name in ["zeros","ones","randn","rand","randint","arange","tensor","empty","full","eye","linspace","zeros_like","ones_like","randn_like","as_tensor","from_numpy_","FloatTensor_"]:
        f = getattr(torch, name, None)
        if f is None: continue
        def _w(f):
            @functools.wraps(f)
            def g(*a, **k):
                if "device" in k: k["device"] = _fix(k["device"])
                return f(*a, **k)
            return g
        setattr(torch, name, _w(f))
    _load = torch.load
    def _l(*a, **k):
        k["map_location"] = "cpu"
        return _load(*a, **k)
    torch.load = _l
    torch.cuda.mem_get_info = lambda *a, **k: (0, 0)
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.memory_allocated = lambda *a, **k: 0
