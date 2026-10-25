#!/usr/bin/env bash
#SBATCH --job-name=kd-concord
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/kd-concord-%j.out
set -euo pipefail
# Model-free gate: drug response vs its targets' L1000 knockdown signatures (record 10_16, 2026-10-14 06:10).
# Needs GSE92742_trt_sh.cgs_landmarks.npz from extract_l1000_cp_landmarks.sbatch (PERT_TYPES="trt_sh.cgs trt_oe").
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
cd /home/yuqi.lei/Therapeutic/scripts
sha256sum analyze_kd_target_concordance.py
for s in ${SLICES:-P1-24 P2-24}; do
  "${ENV_ROOT}/bin/python" analyze_kd_target_concordance.py --slice "$s" --seed 0 --draws 200
done
date --iso-8601=seconds
