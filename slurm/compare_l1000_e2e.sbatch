#!/usr/bin/env bash
#SBATCH --job-name=l1000-e2e-cmp
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --exclude=cn-02,cn-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-e2e-cmp-%j.out
set -euo pipefail
# Paired contrasts for e2e runs (compare_l1000_e2e.py). RUNS="P1-24_ik14_seed0:_krr:_krr_rw0 P2-24_ik14_seed0::"
# = space-separated run:tag:also entries (also may be empty; several also-tags separated by '+').
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/gnn-gpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
date --iso-8601=seconds; hostname
"${ENV_ROOT}/bin/python" -c 'import numpy, pandas; print("numpy", numpy.__version__, "pandas", pandas.__version__)'
cd /home/yuqi.lei/Therapeutic/scripts
sha256sum compare_l1000_e2e.py
for entry in ${RUNS:?set RUNS}; do
  IFS=: read -r run tag also <<< "${entry}"
  echo "== ${run}${tag} (also: ${also:-none})"
  "${ENV_ROOT}/bin/python" compare_l1000_e2e.py --run "${run}" --tag "${tag}" ${also:+--also ${also//+/ }}
done
date --iso-8601=seconds
