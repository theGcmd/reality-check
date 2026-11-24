#!/bin/bash
# Reality Check — idempotent environment + code + data setup. Run at the start of every session.
# Fresh container: ~5-10 min (torch + CUDA wheels are ~2.5 GB from PyPI; download.pytorch.org is blocked here).
# Everything it creates lives under $RC_WORK (default <repo>/work, gitignored). Third-party code is never edited.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${RC_WORK:-$ROOT/work}
mkdir -p "$WORK/repos" "$WORK/data"

# 1. Pinned Python 3.11 env (many audited repos predate numpy 2 / pandas 3)
command -v uv >/dev/null || pip install --break-system-packages -q uv
uv python install 3.11 >/dev/null 2>&1
[ -x "$WORK/venv/bin/python" ] || uv venv -q -p 3.11 "$WORK/venv"
VIRTUAL_ENV="$WORK/venv" uv pip install -q -r "$ROOT/harness/requirements.lock"
SP=$("$WORK/venv/bin/python" -c "import site;print(site.getsitepackages()[0])")
cp "$ROOT/harness/sitecustomize.py" "$SP/sitecustomize.py"   # CPU redirect, active only with CPU_REDIRECT=1

# 2. Third-party code and data at pinned commits (fetch by SHA, shallow)
grep -vE '^\s*(#|$)' "$ROOT/harness/repos.lock" | while read -r name url sha; do
  d="$WORK/repos/$name"
  if [ "$(git -C "$d" rev-parse HEAD 2>/dev/null || true)" != "$sha" ]; then
    rm -rf "$d"; git init -q "$d"; git -C "$d" remote add origin "$url"
    git -C "$d" fetch -q --depth 1 origin "$sha"
    git -C "$d" checkout -q FETCH_HEAD
  fi
done

# 3. Data layout
ln -sfn "$WORK/repos/ETDataset/ETT-small" "$WORK/data/ETT-small"
mkdir -p "$WORK/data/solar"
[ -s "$WORK/data/solar/solar_AL.txt" ] || gunzip -c "$WORK/repos/lai/solar-energy/solar_AL.txt.gz" > "$WORK/data/solar/solar_AL.txt"

# 4. Verify data checksums (recorded 2026-11-24)
cd "$WORK/data"
md5sum -c --quiet - <<'EOF'
8381763947c85f4be6ac456c508460d6  ETT-small/ETTh1.csv
51a229a3fc13579dd939364fefe9c7ab  ETT-small/ETTh2.csv
82d6bd89109c63d075d99c1077b33f38  ETT-small/ETTm1.csv
7687e47825335860bf58bccb31be0c56  ETT-small/ETTm2.csv
de1d4810540a960549c2ce0f05c42974  solar/solar_AL.txt
EOF
"$WORK/venv/bin/python" -c "import torch, numpy, pandas; print('setup ok: torch', torch.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__)"
echo "RC_WORK=$WORK"
