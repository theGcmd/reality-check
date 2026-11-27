"""Q1 (PREREG §4, descriptive): does the RELEASED code give a finite D on this machine? Each call is one fresh process
running the released run_longExp.py (runpy, released H=192 argv from scripts/etth1.sh:77-89, seed 2021) up to the
point where Exp_Main.train would start; no audit patch is installed (D is left as the released code allocates it,
layers/Cluster.py:30). Exp_Main.train is replaced by a counter that reports non-finite entries of both D and exits,
so no data is loaded and nothing is trained or tested.
Usage: python q1_released.py run   (one process)   |   python q1_released.py 10   (spawns 10 fresh processes)"""
import os, sys, json, subprocess, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
RC_WORK = os.path.abspath(os.environ.get("RC_WORK", os.path.join(HERE, "..", "..", "work")))
REPO = os.path.join(RC_WORK, "repos", "tfps")


def one():
    sys.path.insert(0, REPO); os.chdir(os.path.join(RC_WORK)); os.makedirs("q1_scratch", exist_ok=True); os.chdir("q1_scratch")
    import contextlib, io
    import torch
    import exp.exp_main as E

    def _count(self, setting):
        out = {}
        for w, m in (("time", self.model.model_time), ("freq", self.model.model_frequency)):
            D = m.cluster.D.detach()
            out[w] = {"nonfinite": int((~torch.isfinite(D)).sum()), "numel": D.numel(),
                      "nan": int(torch.isnan(D).sum()), "inf": int(torch.isinf(D).sum())}
        print("Q1JSON " + json.dumps(out), file=sys.__stdout__, flush=True)
        raise SystemExit(0)
    E.Exp_Main.train = _count
    sys.argv = ["run_longExp.py", "--random_seed", "2021", "--is_training", "1", "--root_path", "/nonexistent/",
                "--data_path", "ETTh1.csv", "--model_id", "q1", "--model", "PatchTST_MoE_cluster", "--data", "ETTh1",
                "--features", "M", "--seq_len", "96", "--pred_len", "192", "--enc_in", "7", "--e_layers", "3",
                "--n_heads", "4", "--d_model", "16", "--d_ff", "128", "--dropout", "0.3", "--fc_dropout", "0.3",
                "--head_dropout", "0", "--patch_len", "16", "--stride", "8", "--T_num_expert", "4", "--T_top_k", "1",
                "--F_num_expert", "4", "--F_top_k", "1", "--beta", "0.1", "--des", "Exp", "--train_epochs", "100",
                "--itr", "1", "--batch_size", "128", "--learning_rate", "0.0005", "--num_workers", "0"]
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(os.path.join(REPO, "run_longExp.py"), run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        one()
    else:
        n = int(sys.argv[1]); rows = []
        for i in range(n):
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "run"], capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith("Q1JSON ")]
            rows.append(json.loads(line[0][7:]) if line else {"error": r.stderr[-500:]})
        any_nf = [("error" in x) or x["time"]["nonfinite"] > 0 or x["freq"]["nonfinite"] > 0 for x in rows]
        res = {"processes": n, "processes_with_nonfinite_D": int(sum(any_nf)), "fraction_finite": 1 - sum(any_nf) / n,
               "per_process": rows, "context": "released run_longExp.py via runpy, H=192 argv, seed 2021, CPU, torch 2.4.1; "
               "stopped at Exp_Main.train entry (no data, no training)"}
        json.dump(res, open(os.path.join(HERE, "selftest", "Q1_proper.json"), "w"), indent=1)
        print(json.dumps({k: res[k] for k in ("processes", "processes_with_nonfinite_D", "fraction_finite")}))
        print([(x.get("time", {}).get("nonfinite"), x.get("freq", {}).get("nonfinite")) for x in rows])
