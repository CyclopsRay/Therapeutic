#!/usr/bin/env bash
#SBATCH --job-name=txgnn-embed
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/txgnn-embed-%j.out
set -euo pipefail
# TxGNN drug embeddings (record 2026-10-07 13:40). MODE=both (default: released-checkpoint re-draw
# check, then seeded full_graph retrain), train, or ckpt. EXTRA= further args for the train step, e.g.
# "--finetune-epochs 20 --model-dir .../smoke --out .../smoke.npz" for a smoke run (not to be quoted).
MODE=${MODE:-both}
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/txgnn-gpu
SP="$ENV/lib/python3.8/site-packages/nvidia"
export LD_LIBRARY_PATH="$SP/cuda_runtime/lib:$SP/cublas/lib:$SP/cusparse/lib:${LD_LIBRARY_PATH:-}"
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$ENV/bin/python" -c "import torch, sys; ok = torch.cuda.is_available(); print('cuda available', ok, torch.__version__, torch.version.cuda); sys.exit(0 if ok else 3)"
"$ENV/bin/python" -c "import dgl, numpy, pandas; print('dgl', dgl.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__)"
S=/home/yuqi.lei/Therapeutic/scripts/txgnn_drug_embeddings.py
if [ "$MODE" = ckpt ] || [ "$MODE" = both ]; then
  # the check never writes embeddings; a missing checkpoint or a failure must not block the retrain
  CK=$(find /nfs-stor/yuqi.lei/Therapeutic/raw/txgnn/ckpt -name config.pkl -printf '%h\n' 2>/dev/null | head -1)
  if [ -n "$CK" ]; then
    "$ENV/bin/python" "$S" --mode ckpt --ckpt "$CK" || echo "ckpt check failed (exit $?); continuing"
  else
    echo "no released checkpoint under raw/txgnn/ckpt; skipping the re-draw check"
  fi
fi
if [ "$MODE" = train ] || [ "$MODE" = both ]; then
  "$ENV/bin/python" "$S" --mode train --require-gpu ${EXTRA:-}
fi
date --iso-8601=seconds
