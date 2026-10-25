#!/usr/bin/env bash
#SBATCH --job-name=e2e-kg
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/e2e-kg-%j.out
set -euo pipefail
# PrimeKG subgraph for the integrated end-to-end model (record 2026-10-09). Output derived/e2e_kg/v1.
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
date --iso-8601=seconds; hostname
cd /home/yuqi.lei/Therapeutic/scripts
sha256sum build_e2e_kg.py
"${ENV_ROOT}/bin/python" build_e2e_kg.py ${ARGS:-}
date --iso-8601=seconds
