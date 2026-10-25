#!/usr/bin/env bash
#SBATCH --job-name=l1000-gnn
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-gnn-%j.out
set -euo pipefail
# Approach-2 GNN and MLP controls for the L1000 unseen-drug branch (G3b, report/10_09_record.md).
# SLICE=P2-24|P1-24|P1-6, SPLIT=ik14|platemap, EXTRA= further args (e.g. --arms, --folds, --max-epochs).
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c 'import torch, numpy, scipy, pandas; print("torch", torch.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__)'
cd /home/yuqi.lei/Therapeutic/scripts
"${ENV_ROOT}/bin/python" run_l1000_gnn.py --slice "${SLICE:-P2-24}" --split "${SPLIT:-ik14}" ${EXTRA:-}
date --iso-8601=seconds
