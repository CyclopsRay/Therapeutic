#!/usr/bin/env bash
#SBATCH --job-name=ctx-sanity
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/ctx-sanity-%j.out
set -euo pipefail
# Context x compound sanity checks on the drug-blind benchmark (no network training): line-similarity structure of
# compound responses, oracle headroom from v3 saved predictions, linear drug-free backbones. Deterministic (seed 0).
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname
"$ENV/bin/python" -c "import numpy, pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/manifest.json
SCRIPT=${SCRIPT:-sanity_context_interaction.py}  # or sanity_response_line_kernel.py
sha256sum /home/yuqi.lei/Therapeutic/scripts/sanity_context_interaction.py /home/yuqi.lei/Therapeutic/scripts/$SCRIPT
"$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/$SCRIPT
date --iso-8601=seconds
