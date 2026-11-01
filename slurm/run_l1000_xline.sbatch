#!/usr/bin/env bash
#SBATCH --job-name=l1000-xline
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-xline-%j.out
set -euo pipefail
# Cross-line multimodal kernel model for unseen drugs (exploratory sanity check, 2026-10-28).
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__)"
cd /home/yuqi.lei/Therapeutic/scripts
"${ENV_ROOT}/bin/python" run_l1000_xline.py --slice "${SLICE:-P2-24}" --split "${SPLIT:-ik14}" ${EXTRA:-}
date --iso-8601=seconds
