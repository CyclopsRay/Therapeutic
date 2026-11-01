#!/usr/bin/env bash
#SBATCH --job-name=undetect-v2c
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/undetect-v2c-%j.out
# CPU copy of run_undetect_v2.sbatch (GPU queue ~1.5 days on 2026-10-31); pass a separate --tag via EXTRA so GPU and CPU
# outputs never share a directory. The backbone MLP is not bit-identical to the GPU runs: compare arms within a run.
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/manifest.json
sha256sum /home/yuqi.lei/Therapeutic/scripts/run_sdst_hybrid.py /home/yuqi.lei/Therapeutic/scripts/run_undetect_v2.py
"$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/run_undetect_v2.py ${EXTRA:-}
date --iso-8601=seconds
