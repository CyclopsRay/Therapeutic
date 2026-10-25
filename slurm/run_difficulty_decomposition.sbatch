#!/usr/bin/env bash
#SBATCH --job-name=difficulty-decomp
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/difficulty-decomp-%j.out

set -euo pipefail

# Descriptive decomposition of already-frozen per-condition results.
# No refitting, no model selection, no gate, no leakage surface.
# PLAN.md direction C (identifiability and decomposition).

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
git -C "${PROJECT_ROOT}/external/Rhaister" rev-parse HEAD
"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_difficulty_decomposition.py"
date --iso-8601=seconds
