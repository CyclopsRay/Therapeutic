#!/usr/bin/env bash
#SBATCH --job-name=depmap-expr-capacity
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/depmap-expr-capacity-%j.out

set -euo pipefail

# Capacity sweep for the DepMap-expression residual ridge.
#
# Motivation: the original run searched component grid {0, 2, 5, 10} and all
# five outer splits selected 10 -- the top of the grid.  The capacity cap was
# binding, so the reported 0.2087 drug-mean-residual line-macro Pearson is a
# capacity-limited estimate rather than the ceiling of DepMap expression.
# This run widens both the component and the ridge-penalty grids and writes to
# a separate output root; the frozen original results are not touched.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
OUTPUT_ROOT=/nfs-stor/yuqi.lei/Therapeutic/results/depmap_expression_ridge_capacity_sweep
RHAISTER_DATASET_REVISION=c7963cf334bec0683225d41c9586d900ca6303a2
DEPMAP_RELEASE='Public 24Q4'
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
echo "DepMap release ${DEPMAP_RELEASE}"
echo "random seed ${RANDOM_SEED}"
echo "output root ${OUTPUT_ROOT}"
echo "NOTE: PCA on ~45 reference lines yields at most ~44 components;"
echo "      component values above that are skipped by the runner."

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_depmap_expression_ridge.py" \
  --component-grid 2 5 10 15 20 25 30 35 40 \
  --lambda-grid 0.1 1 10 100 1000 10000 100000 \
  --n-permutations 20 \
  --n-bootstrap 2000 \
  --seed "${RANDOM_SEED}" \
  --data-revision "${RHAISTER_DATASET_REVISION}" \
  --output-root "${OUTPUT_ROOT}"

date --iso-8601=seconds
