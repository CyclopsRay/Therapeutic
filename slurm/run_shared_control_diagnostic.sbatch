#!/usr/bin/env bash
#SBATCH --job-name=shared-control-diag
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/shared-control-diag-%j.out

set -euo pipefail

# How much of Rhaister-O's diagonal advantage is the DMSO control appearing on
# both sides of the equation?
#
# Target is Delta = treated minus plate-matched DMSO; the predictor feature is
# that same line's DMSO.  Shared estimation noise induces a mechanical
# association that transfers to held-out lines and that neither drug-mean nor
# reference-fitted PC1 removal can strip.
#
# Plates are halved at random.  Conditions are restricted to those measured only
# on half A, so every retained target's control lies in half A.  The 'same' arm
# builds the feature from half A, the 'cross' arm from half B; both average 7
# plates so feature noise is matched.  Targets are the frozen upstream deltas --
# nothing is recomputed.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
OUTPUT_ROOT=/nfs-stor/yuqi.lei/Therapeutic/results/shared_control_diagnostic
RHAISTER_DATASET_REVISION=c7963cf334bec0683225d41c9586d900ca6303a2
RANDOM_SEED=0
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c 'import numpy, pandas, pyarrow; print("numpy", numpy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__)'
git -C "${PROJECT_ROOT}/external/Rhaister" rev-parse HEAD
echo "tahoe-de-rhaister revision ${RHAISTER_DATASET_REVISION}"
echo "random seed ${RANDOM_SEED}"
echo "output root ${OUTPUT_ROOT}"

cd "${PROJECT_ROOT}/scripts"
"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_shared_control_diagnostic.py" \
  --n-plate-splits 5 \
  --min-treatments 200 \
  --rhaister-lambda 1.0 \
  --rhaister-blend 0.95 \
  --seed "${RANDOM_SEED}" \
  --data-revision "${RHAISTER_DATASET_REVISION}" \
  --output-root "${OUTPUT_ROOT}"

date --iso-8601=seconds
