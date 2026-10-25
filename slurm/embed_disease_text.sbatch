#!/usr/bin/env bash
#SBATCH --job-name=e2e-distext
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/e2e-distext-%j.out
set -euo pipefail
# Frozen SapBERT (names) / PubMedBERT (definitions) features of PrimeKG disease nodes (record 2026-10-09).
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/text-embed
export HF_HOME=/nfs-stor/yuqi.lei/Therapeutic/caches/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
date --iso-8601=seconds; hostname
cd /home/yuqi.lei/Therapeutic/scripts
sha256sum embed_disease_text.py
"${ENV_ROOT}/bin/python" embed_disease_text.py ${ARGS:-}
date --iso-8601=seconds
