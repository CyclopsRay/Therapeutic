#!/usr/bin/env bash
#SBATCH --job-name=oracle-final
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/oracle-final-%j.out
# Uniform oracle on the final drug-blind run, same rows as UnDetect / XPert / ensemble (scripts/oracle_final.py; no
# training; record 2026-11-03). Deterministic: no random numbers are drawn.
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/manifest.json
sha256sum /home/yuqi.lei/Therapeutic/scripts/oracle_final.py /nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/final_cpu/final.json
"$ENV/bin/python" -I -u /home/yuqi.lei/Therapeutic/scripts/oracle_final.py
date --iso-8601=seconds
