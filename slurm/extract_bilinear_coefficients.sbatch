#!/usr/bin/env bash
#SBATCH --job-name=bilinear-coef
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/bilinear-coef-%j.out

set -euo pipefail

# Materialize B_hat for the frozen low-rank bilinear fit.
#
# run_dmso_lowrank_bilinear.py predicts through W = D_q (D'D + lam P)^-1 D' and
# never forms B_hat = (D'D + lam P)^-1 D' Y, so the cross-gene response programs
# were never inspectable.  This job recovers them at the *already selected*
# (K, lambda) of each split and arm, read out of the frozen ridge_selection.csv.
# No new model selection, no new fitting decision, no gate change.  The frozen
# bilinear output directory is not written to.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
OUTPUT_ROOT=/nfs-stor/yuqi.lei/Therapeutic/results/dmso_lowrank_coefficients
BILINEAR_ROOT=/nfs-stor/yuqi.lei/Therapeutic/results/dmso_lowrank_bilinear
RHAISTER_DATASET_REVISION=c7963cf334bec0683225d41c9586d900ca6303a2
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
echo "selection source ${BILINEAR_ROOT}"
echo "output root ${OUTPUT_ROOT}"
echo "pinned rhaister lambda 1.0 blend 0.95 (same as the frozen bilinear run)"

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/extract_bilinear_coefficients.py" \
  --bilinear-root "${BILINEAR_ROOT}" \
  --data-revision "${RHAISTER_DATASET_REVISION}" \
  --output-root "${OUTPUT_ROOT}"

date --iso-8601=seconds
