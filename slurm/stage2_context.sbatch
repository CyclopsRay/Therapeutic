#!/usr/bin/env bash
#SBATCH --job-name=stage2
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/stage2-%j.out
# Second stage (scripts/stage2_context.py --v2) on cache_gpu_s<SEED>, tag s2gpu_s<SEED>; then PCC_plate contrasts vs pool_kta.
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
date --iso-8601=seconds; hostname; nproc; echo "seed ${SEED:?}"
"$ENV/bin/python" -c "import numpy, pandas, torch; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'torch', torch.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid/cache_gpu_s$SEED/fold_*/summary.json | grep -h '"seconds"\|"mlp_epoch"'
sha256sum /home/yuqi.lei/Therapeutic/scripts/stage2_context.py /home/yuqi.lei/Therapeutic/scripts/analyze_sdst_centred.py
"$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/stage2_context.py --v2 --cache "cache_gpu_s$SEED" --tag "s2gpu_s$SEED"
"$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/analyze_sdst_centred.py --tag "s2gpu_s$SEED" --ref pool_kta
date --iso-8601=seconds
