#!/usr/bin/env bash
#SBATCH --job-name=ppi-info
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/ppi-info-%j.out
set -euo pipefail
# Exploratory, model-free: does STRING wiring carry response information beyond degree? (report/10_09_record.md)
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
cd /home/yuqi.lei/Therapeutic/scripts
for s in ${SLICES:-P2-24 P1-24}; do
  "${ENV_ROOT}/bin/python" analyze_ppi_information.py --slice "$s"
done
date --iso-8601=seconds
