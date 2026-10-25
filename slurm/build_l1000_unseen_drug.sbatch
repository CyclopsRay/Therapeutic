#!/usr/bin/env bash
#SBATCH --job-name=l1000-slices
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-slices-%j.out
set -euo pipefail
# Pre-registered L1000 unseen-drug slices P1-24, P1-6, P2-24 (report/10_09_record.md).
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" /home/yuqi.lei/Therapeutic/scripts/build_l1000_unseen_drug.py
date --iso-8601=seconds
