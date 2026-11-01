#!/usr/bin/env bash
#SBATCH --job-name=finalize-sdst
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/finalize-sdst-%j.out
# Final drug-blind numbers on the CPU backbone run (scripts/finalize_sdst.py; no training).
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"
sha256sum /home/yuqi.lei/Therapeutic/scripts/finalize_sdst.py
"$ENV/bin/python" -I -u /home/yuqi.lei/Therapeutic/scripts/finalize_sdst.py
date --iso-8601=seconds
