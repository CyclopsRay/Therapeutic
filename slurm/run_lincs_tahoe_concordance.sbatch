#!/usr/bin/env bash
#SBATCH --job-name=lincs-tahoe-sync
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/lincs-tahoe-sync-%j.out

set -euo pipefail

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
RANDOM_SEED=0
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c 'import h5py, numpy, pandas, pyarrow; print("h5py", h5py.__version__, "numpy", numpy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__)'
echo "LINCS accessions GSE92742 GSE70138"
echo "random seed ${RANDOM_SEED}"

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_lincs_tahoe_concordance.py" \
  --max-dose-ratio 2 \
  --bootstrap 2000 \
  --permutations 500 \
  --threads "${SLURM_CPUS_PER_TASK:-4}" \
  --seed "${RANDOM_SEED}"

date --iso-8601=seconds
