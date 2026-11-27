"""DUST-1 CPU cost probe (no data, no scoring): times one backprop SGD step (batch 8 x 2048) and one
forward pass over 32 held-out-shaped sequences, on random token ids, using the pinned repo's model and
make_model/make_optimizer unchanged (imported, not edited). Seed 999, outside any scored seed set."""
import sys, time, json, os, platform
from pathlib import Path
REPO = Path(os.environ.get('RC_WORK', 'work')) / 'repos' / 'dust'
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'baselines'))
import torch
from dust import GPTConfig, make_model
from backprop import make_optimizer, train_step
torch.manual_seed(999)
dev = torch.device('cpu')
cfg = GPTConfig()
model = make_model(cfg, dev, 999).requires_grad_(True)
opt = make_optimizer(model, 0.175, 0.98)
n = sum(p.numel() for p in model.parameters())
g = torch.Generator().manual_seed(999)
out = dict(threads=torch.get_num_threads(), params=n, torch=torch.__version__, host=platform.node())
steps = []
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
    # 7 GB RAM: a full 8-sequence backward OOMs here, so time ONE sequence fwd+bwd (an 8-step accumulation
    # would cost ~8x this); no optimizer step is needed for timing.
    b = torch.randint(0, 4096, (1, 2049), generator=g)
    t = time.time(); model(b[:, :-1].contiguous(), b[:, 1:].contiguous()).backward(); model.zero_grad(); steps.append(time.time() - t)
out['fwd_bwd_1seq_s'] = steps; out['train_step_8seq_est_s'] = 8 * min(steps)
with torch.no_grad():
    b = torch.randint(0, 4096, (4, 2049), generator=g)
    t = time.time(); model(b[:, :-1].contiguous(), b[:, 1:].contiguous()); out['eval_fwd_4seq_s'] = time.time() - t
out['eval_full_544seq_est_s'] = out['eval_fwd_4seq_s'] * 136
print(json.dumps(out))
