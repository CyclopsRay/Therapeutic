#!/usr/bin/env bash
#SBATCH --job-name=undetect-cache
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/undetect-cache-%j.out
# UnDetect part cache for second-stage experiments (scripts/cache_undetect.py); arguments via EXTRA.
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/manifest.json
sha256sum /home/yuqi.lei/Therapeutic/scripts/run_sdst_hybrid.py /home/yuqi.lei/Therapeutic/scripts/run_undetect_v2.py /home/yuqi.lei/Therapeutic/scripts/cache_undetect.py
"$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/cache_undetect.py ${EXTRA:-}
date --iso-8601=seconds
