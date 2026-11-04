#!/usr/bin/env bash
#SBATCH --job-name=strata-all-models
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/strata-all-models-%j.out
# strata_all_models.py (no training; record 2026-11-04 22:00).
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
date --iso-8601=seconds; hostname
sha256sum /home/yuqi.lei/Therapeutic/scripts/strata_all_models.py
"$ENV/bin/python" -I -u /home/yuqi.lei/Therapeutic/scripts/strata_all_models.py
date --iso-8601=seconds
