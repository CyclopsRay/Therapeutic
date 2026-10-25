#!/usr/bin/env bash
#SBATCH --job-name=coef-sharing
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/coef-sharing-%j.out

set -euo pipefail

# PLAN.md Phase 4 Track A: shrink Rhaister-O's own per-(treatment, gene)
# diagonal slopes toward the pooled per-gene slope.
#
# Motivation. Rhaister-O estimates roughly 1.5M scalar slopes from 45 reference
# lines apiece, and the frozen baseline already computes both endpoints of the
# shrinkage continuum: fully independent slopes score 0.631456 full-delta
# line-macro Pearson, fully pooled slopes score 0.489322. Everything between is
# untested. Job 249519 failed by ADDING degrees of freedom and concluded the
# diagonal restriction acts as beneficial regularization; this moves in the
# opposite direction.
#
# Scope. Input is the plate-matched Tahoe DMSO profile only, exactly what the
# frozen Rhaister-O consumes. No new modality, so the DepMap expansion gate in
# CLAUDE.md does not apply -- same reasoning that governed job 249519.
#
# Inner selection is leave-one-reference-line-out drug-residual Pearson, NOT
# LOO RMSE, because job 249519 showed RMSE is a poor proxy for the outer
# endpoint. What RMSE would have chosen is recorded per split for comparison.
#
# Pass --splits 5 for a cheap single-split smoke run; the predeclared gate is
# only meaningful over all five.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
# OUTPUT_ROOT and SPLITS may be overridden with sbatch --export so a cheap
# single-split smoke run writes somewhere separate and never pollutes the real
# result directory.
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs-stor/yuqi.lei/Therapeutic/results/coefficient_sharing}
N_BOOTSTRAP=${N_BOOTSTRAP:-20000}
RHAISTER_DATASET_REVISION=c7963cf334bec0683225d41c9586d900ca6303a2
RANDOM_SEED=0
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c 'import numpy, pandas, pyarrow; print("numpy", numpy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__)'
git -C "${PROJECT_ROOT}/external/Rhaister" rev-parse HEAD
echo "tahoe-de-rhaister revision ${RHAISTER_DATASET_REVISION}"
echo "random seed ${RANDOM_SEED}"
echo "output root ${OUTPUT_ROOT}"
echo "splits: ${SPLITS:-5 6 7 8 9}"
echo "targets are the frozen upstream Cell-Eval deltas; nothing is recomputed"

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_coefficient_sharing.py" \
  --splits ${SPLITS:-5 6 7 8 9} \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --seed "${RANDOM_SEED}" \
  --data-revision "${RHAISTER_DATASET_REVISION}" \
  --output-root "${OUTPUT_ROOT}"

date --iso-8601=seconds
