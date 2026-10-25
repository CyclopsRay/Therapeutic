#!/usr/bin/env bash
#SBATCH --job-name=plate-count-sweep
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/plate-count-sweep-%j.out

set -euo pipefail

# Absolute sensitivity of the pinned diagonal path to DMSO feature noise.
#
# Job 258115 matched both arms at seven averaged plates so the noise magnitude
# would cancel in the subtraction; it is therefore blind to the absolute noise
# level.  This sweep varies how many plates the feature averages (1, 2, 3, 4, 7,
# 10, 14) with five independent draws per count, holding the frozen targets, the
# full condition set, the pinned lambda/blend, the five outer splits and all 25
# held-out lines fixed.  Output is a curve of drug-residual Pearson against m.
#
# The m=14 point is this sweep's own internal reference, not the headline
# 0.5380: every point uses a plain mean over a plate subset, whereas the pinned
# Rhaister-O feature averages plate-matched centroids per observed condition.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
OUTPUT_ROOT=/nfs-stor/yuqi.lei/Therapeutic/results/plate_count_sweep
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
echo "targets are the frozen upstream Cell-Eval deltas; nothing is recomputed"

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_plate_count_sweep.py" \
  --plate-counts 1 2 3 4 7 10 14 \
  --n-draws 5 \
  --n-bootstrap 20000 \
  --seed "${RANDOM_SEED}" \
  --data-revision "${RHAISTER_DATASET_REVISION}" \
  --output-root "${OUTPUT_ROOT}"

date --iso-8601=seconds
