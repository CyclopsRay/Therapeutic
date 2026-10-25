#!/usr/bin/env bash
#SBATCH --job-name=l1000-e2e
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-e2e-%j.out
set -euo pipefail
# Integrated end-to-end model, phase 1 (unseen drug), arms skel / kg_a / kg_b (record 2026-10-09).
# SLICE=P1-24|P2-24, SPLIT=ik14|platemap, EXTRA= further args (e.g. --folds 0 --max-epochs 2 --tag _smoke).
# Round 2 (record 2026-10-11): EXTRA="--anchor krr [--rewire 0] --arms ... --tag _krr[_rw0]".
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/gnn-gpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${ENV_ROOT}/bin/python" -c 'import torch, torch_geometric, numpy, scipy, pandas, sys; ok = torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, ok, "pyg", torch_geometric.__version__, "numpy", numpy.__version__, "pandas", pandas.__version__); sys.exit(0 if ok else 3)'
cd /home/yuqi.lei/Therapeutic/scripts
sha256sum run_l1000_e2e.py run_l1000_unseen_drug.py
"${ENV_ROOT}/bin/python" run_l1000_e2e.py --device cuda --slice "${SLICE:-P1-24}" --split "${SPLIT:-ik14}" ${EXTRA:-}
# optional second run in the same allocation (GPU QOS: 4 submitted / 2 running jobs per user)
if [ -n "${EXTRA2:-}" ]; then
  "${ENV_ROOT}/bin/python" run_l1000_e2e.py --device cuda --slice "${SLICE:-P1-24}" --split "${SPLIT:-ik14}" ${EXTRA2}
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
date --iso-8601=seconds
