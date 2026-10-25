#!/usr/bin/env bash
#SBATCH --job-name=l1000-struct
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-struct-%j.out
set -euo pipefail
# Q1 descriptive: chemical similarity vs transcriptional similarity on L1000 (report/10_09_record.md).
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
cd /home/yuqi.lei/Therapeutic/scripts
for s in ${SLICES:-P1-24 P1-6 P2-24}; do
  "${ENV_ROOT}/bin/python" analyze_l1000_structure_response.py --slice "$s"
done
date --iso-8601=seconds
