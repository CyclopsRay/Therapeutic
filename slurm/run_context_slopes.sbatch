#!/usr/bin/env bash
#SBATCH --job-name=context-slopes
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/%x-%j.out
set -euo pipefail
# Pre-registered E1 (report/10_02_record.md) on Tahoe: cell-context slopes.
# ARMS selects a subset so arms can run as parallel jobs; OUTPUT_ROOT per subset.
PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
date --iso-8601=seconds; hostname; echo "ARMS=$ARMS OUTPUT_ROOT=$OUTPUT_ROOT"
"$ENV_ROOT/bin/python" -c 'import numpy, pandas; print("numpy", numpy.__version__, "pandas", pandas.__version__)'
cd "$PROJECT_ROOT/scripts"
"$ENV_ROOT/bin/python" run_context_slopes.py --arms $ARMS --output-root "$OUTPUT_ROOT" ${EXTRA_ARGS:-}
date --iso-8601=seconds
