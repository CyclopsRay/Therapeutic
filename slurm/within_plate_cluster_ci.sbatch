#!/usr/bin/env bash
#SBATCH --job-name=within-cluster
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/within-cluster-%j.out
# Clustered bootstrap intervals for PCC_within (scripts/within_plate_cluster_ci.py; reviewer T3; no training).
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
date --iso-8601=seconds; hostname
sha256sum /home/yuqi.lei/Therapeutic/scripts/within_plate_cluster_ci.py
"$ENV/bin/python" -I -u /home/yuqi.lei/Therapeutic/scripts/within_plate_cluster_ci.py
date --iso-8601=seconds
