#!/usr/bin/env bash
#SBATCH --job-name=view-ci
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/view-ci-%j.out
set -euo pipefail
# Fig. 3b intervals: compound-bootstrap 95% CIs of response similarity per view bin (scripts/view_bins_ci.py;
# no training; record 2026-11-04). Deterministic (seed 0 in the script).
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__, 'torch', torch.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/manifest.json
sha256sum /home/yuqi.lei/Therapeutic/scripts/view_bins_ci.py /home/yuqi.lei/Therapeutic/scripts/analyze_sdst_compounds.py /home/yuqi.lei/Therapeutic/scripts/run_sdst_hybrid.py
"$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/view_bins_ci.py
date --iso-8601=seconds
