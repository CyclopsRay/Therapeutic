#!/usr/bin/env bash
#SBATCH --job-name=sdst-hybrid
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/sdst-hybrid-%j.out
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/manifest.json
sha256sum /home/yuqi.lei/Therapeutic/scripts/run_sdst_hybrid.py
"$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/run_sdst_hybrid.py ${EXTRA:-}
date --iso-8601=seconds
