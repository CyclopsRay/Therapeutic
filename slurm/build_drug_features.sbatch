#!/usr/bin/env bash
#SBATCH --job-name=drug-features
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/drug-features-%j.out
set -euo pipefail
# RDKit standardization + Morgan/MACCS/RDKit-2D/scaffolds + target/MOA sets (unseen-drug branch).
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" /home/yuqi.lei/Therapeutic/scripts/build_drug_features.py
date --iso-8601=seconds
