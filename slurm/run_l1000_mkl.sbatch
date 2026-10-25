#!/usr/bin/env bash
#SBATCH --job-name=l1000-mkl
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-mkl-%j.out
set -euo pipefail
# Integrated-model step 1 (PLAN.md Phase 7): multiple-kernel KRR with tiers and view nulls.
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
cd /home/yuqi.lei/Therapeutic/scripts
"${ENV_ROOT}/bin/python" run_l1000_mkl.py --slice "${SLICE:-P2-24}" --split "${SPLIT:-platemap}" ${EXTRA:-}
date --iso-8601=seconds
