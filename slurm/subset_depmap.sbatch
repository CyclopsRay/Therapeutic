#!/bin/bash
#SBATCH --job-name=ther-depmap
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/subset-depmap-%j.out
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep

hostname
date --iso-8601=seconds
"${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -m pip freeze
"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/subset_depmap.py"
date --iso-8601=seconds

