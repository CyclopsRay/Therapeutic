#!/usr/bin/env bash
#SBATCH --job-name=undetect-cache-gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/undetect-cache-gpu-%j.out
# GPU copy of cache_undetect_cpu.sbatch (same backbone hardware as v3): one cache per seed in SEEDS, tag cache_gpu_s<seed>.
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/manifest.json
sha256sum /home/yuqi.lei/Therapeutic/scripts/run_sdst_hybrid.py /home/yuqi.lei/Therapeutic/scripts/run_undetect_v2.py /home/yuqi.lei/Therapeutic/scripts/cache_undetect.py
for S in ${SEEDS:-1729}; do
  echo "== seed $S"; "$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/cache_undetect.py --seed "$S" --tag "cache_gpu_s$S" ${EXTRA:-}
done
date --iso-8601=seconds
