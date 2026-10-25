#!/usr/bin/env bash
#SBATCH --job-name=rhaister-o-delta
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/rhaister-o-delta-%j.out

set -euo pipefail

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
RHAISTER_DATASET_REVISION=c7963cf334bec0683225d41c9586d900ca6303a2
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
echo "random seed ${RANDOM_SEED} (pipeline is deterministic; no stochastic estimator)"

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_rhaister_o_compact.py" \
  --split all \
  --save-predictions

date --iso-8601=seconds
