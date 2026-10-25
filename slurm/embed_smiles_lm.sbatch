#!/usr/bin/env bash
#SBATCH --job-name=smiles-lm
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/smiles-lm-%j.out
set -euo pipefail
# Frozen ChemBERTa-77M-MLM / MoLFormer-XL embeddings of standardized SMILES (unseen-drug branch).
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/text-embed
export HF_HOME=/nfs-stor/yuqi.lei/Therapeutic/caches/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" /home/yuqi.lei/Therapeutic/scripts/embed_smiles_lm.py ${ARGS:-}
date --iso-8601=seconds
