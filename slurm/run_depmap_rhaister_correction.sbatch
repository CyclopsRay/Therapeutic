#!/usr/bin/env bash
#SBATCH --job-name=depmap-rhaister-corr
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/depmap-rhaister-corr-%j.out

set -euo pipefail

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
RHAISTER_DATASET_REVISION=c7963cf334bec0683225d41c9586d900ca6303a2
DEPMAP_RELEASE='Public 24Q4'
RANDOM_SEED=0
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c 'import numpy, pandas, pyarrow; print("numpy", numpy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__)'
git -C "${PROJECT_ROOT}/external/Rhaister" rev-parse HEAD
echo "tahoe-de-rhaister revision ${RHAISTER_DATASET_REVISION}"
echo "DepMap release ${DEPMAP_RELEASE}"
echo "random seed ${RANDOM_SEED}"

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_depmap_rhaister_correction.py" \
  --component-grid 2 5 10 \
  --lambda-grid 1 10 100 \
  --n-permutations 20 \
  --n-bootstrap 2000 \
  --seed "${RANDOM_SEED}" \
  --data-revision "${RHAISTER_DATASET_REVISION}"

date --iso-8601=seconds
