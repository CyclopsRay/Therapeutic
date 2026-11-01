#!/usr/bin/env bash
#SBATCH --job-name=prep-sdst
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/prep-sdst-%j.out
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
date --iso-8601=seconds; hostname
"$ENV/bin/python" -c "import numpy, pandas, h5py; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'h5py', h5py.__version__)"
"$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/prep_xpert_sdst.py ${EXTRA:-}
date --iso-8601=seconds
