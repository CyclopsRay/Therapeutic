#!/usr/bin/env bash
#SBATCH --job-name=case-scan
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/case-scan-%j.out
# Case-study scan: per-compound PCC_plate of every held-out compound (scripts/case_study_scan.py; no training; record 2026-11-04).
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas; print("numpy", numpy.__version__, "pandas", pandas.__version__)"
sha256sum /home/yuqi.lei/Therapeutic/scripts/case_study_scan.py /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/sdst.npz
"$ENV/bin/python" -I -u /home/yuqi.lei/Therapeutic/scripts/case_study_scan.py
date --iso-8601=seconds
