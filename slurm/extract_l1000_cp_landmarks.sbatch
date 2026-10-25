#!/usr/bin/env bash
#SBATCH --job-name=l1000-cp-extract
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-cp-extract-%j.out

set -euo pipefail

# Every trt_cp Level-5 signature x 978 landmarks, for the unseen-drug branch.
# ACCESSION=GSE92742 (Phase I, default) or GSE70138 (Phase II); PERT_TYPES="trt_sh.cgs trt_oe" for genetic perturbations.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
df -h /nfs-stor/yuqi.lei | tail -1

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/extract_l1000_cp_landmarks.py" \
  --accession "${ACCESSION:-GSE92742}" --threads "${SLURM_CPUS_PER_TASK:-8}" ${PERT_TYPES:+--pert-types ${PERT_TYPES}}

date --iso-8601=seconds
