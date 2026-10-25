#!/usr/bin/env bash
#SBATCH --job-name=ref-count-sweep
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/ref-count-sweep-%j.out

set -euo pipefail

# Is the strict zero-shot task method-limited or data-limited?
#
# Subsample the 45 reference cell lines to m of them, refit the pinned diagonal
# path, evaluate on the five held-out lines. If the curve is still climbing at
# m=45 the bottleneck is the screen and more cell lines would buy accuracy; if
# it has plateaued, the task is saturated at this scale and the string of
# negative modelling results is a statement about the problem rather than about
# our attempts.
#
# Independently checks the Rhaister paper's claim that zero-shot metrics "do not
# appear to have plateaued" in the number of reference contexts, at the
# drug-residual level the paper does not report.
#
# Targets are the frozen upstream Cell-Eval deltas; nothing is recomputed. The
# treatment set is fixed once from all 45 lines so the target does not move
# with m.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs-stor/yuqi.lei/Therapeutic/results/reference_count_sweep}
RHAISTER_DATASET_REVISION=c7963cf334bec0683225d41c9586d900ca6303a2
RANDOM_SEED=0
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
git -C "${PROJECT_ROOT}/external/Rhaister" rev-parse HEAD
echo "tahoe-de-rhaister revision ${RHAISTER_DATASET_REVISION}"
echo "random seed ${RANDOM_SEED}"
echo "output root ${OUTPUT_ROOT}"

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_reference_count_sweep.py" \
  --splits ${SPLITS:-5 6 7 8 9} \
  --n-draws 5 \
  --n-bootstrap 20000 \
  --seed "${RANDOM_SEED}" \
  --data-revision "${RHAISTER_DATASET_REVISION}" \
  --output-root "${OUTPUT_ROOT}"

date --iso-8601=seconds
