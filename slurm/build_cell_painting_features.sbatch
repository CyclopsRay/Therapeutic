#!/usr/bin/env bash
#SBATCH --job-name=cp-features
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/cp-features-%j.out
set -euo pipefail
# LINCS Cell Painting (A549 48 h) per-compound morphology features, joined to L1000 InChIKeys.
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" /home/yuqi.lei/Therapeutic/scripts/build_cell_painting_features.py
date --iso-8601=seconds
