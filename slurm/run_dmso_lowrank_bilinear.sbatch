#!/usr/bin/env bash
#SBATCH --job-name=dmso-lowrank-bilinear
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/dmso-lowrank-bilinear-%j.out

set -euo pipefail

# Track A: does a low-rank cross-gene bilinear model recover structure that
# Rhaister-O's per-gene diagonal map leaves behind?
#
# Both arms consume only the plate-matched Tahoe DMSO profile -- the same input
# the frozen Rhaister-O baseline uses.  No DepMap, no L1000, no target-line
# treated response.  This is therefore a modelling question and is not governed
# by the failed DepMap modality-expansion gate.
#
#   dmso_lowrank_direct      anchor = drug-dose mean, target = drug-mean residual
#   dmso_lowrank_correction  anchor = frozen Rhaister-O, target = truth minus
#                            leave-one-reference-line-out Rhaister-O
#
# All 25 held-out lines are evaluated, since every line has a DMSO profile.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
OUTPUT_ROOT=/nfs-stor/yuqi.lei/Therapeutic/results/dmso_lowrank_bilinear
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
"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_dmso_lowrank_bilinear.py" \
  --component-grid 1 2 3 5 8 10 15 20 25 30 35 40 \
  --lambda-grid 0.1 1 10 100 1000 10000 100000 \
  --rhaister-lambda 1.0 \
  --rhaister-blend 0.95 \
  --min-improvement 0.01 \
  --n-permutations 100 \
  --n-bootstrap 2000 \
  --seed "${RANDOM_SEED}" \
  --data-revision "${RHAISTER_DATASET_REVISION}" \
  --output-root "${OUTPUT_ROOT}"

date --iso-8601=seconds
