#!/bin/bash
#SBATCH --job-name=ther-lincs-l5
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/fetch-lincs-level5-%j.out
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT=/home/yuqi.lei/Therapeutic

hostname
date --iso-8601=seconds
"${PROJECT_ROOT}/scripts/fetch_data.sh" lincs-level5
date --iso-8601=seconds
