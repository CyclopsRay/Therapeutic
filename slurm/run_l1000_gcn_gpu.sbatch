#!/usr/bin/env bash
#SBATCH --job-name=l1000-gcn
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-gcn-%j.out
set -euo pipefail
# Pre-registered G3b GCN on GPU (record 2026-10-07): arms gcn_ppi / gcn_rewired / gcn_nograph.
# SLICE=P2-24|P1-24, SPLIT=ik14|platemap, EXTRA= further args (e.g. --folds 0 --max-epochs 3 --tag _smoke).
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/gnn-gpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"${ENV_ROOT}/bin/python" -c 'import torch, numpy, scipy, pandas, pyarrow, sys; ok = torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, ok, "numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__); sys.exit(0 if ok else 3)'
cd /home/yuqi.lei/Therapeutic/scripts
sha256sum run_l1000_gnn.py run_l1000_unseen_drug.py
"${ENV_ROOT}/bin/python" run_l1000_gnn.py --device cuda --slice "${SLICE:-P2-24}" --split "${SPLIT:-ik14}" \
  --arms gcn_ppi gcn_rewired gcn_nograph mlp_target ${EXTRA:-}
date --iso-8601=seconds
