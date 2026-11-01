#!/usr/bin/env bash
#SBATCH --job-name=sdst-hybrid-cpu
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/sdst-hybrid-cpu-%j.out
# CPU copy of run_sdst_hybrid.sbatch (e.g. the ablation arms on the CPU backbone cache: --backbone-cache cache_v1); arguments via EXTRA.
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/manifest.json
sha256sum /home/yuqi.lei/Therapeutic/scripts/run_sdst_hybrid.py /nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/cache_v1/fold_*/cache.npz
"$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/run_sdst_hybrid.py ${EXTRA:-}
date --iso-8601=seconds
