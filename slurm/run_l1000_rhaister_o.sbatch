#!/usr/bin/env bash
#SBATCH --job-name=l1000-rhaister
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/l1000-rhaister-%j.out

set -euo pipefail

# Rhaister-O's estimator on LINCS L1000, leave-one-cell-line-out.
#
# Same estimator, same strict whole-line holdout, same metric as the Tahoe
# benchmark. Different response (Level-5 z-scores), different genes (978
# landmarks, only 79 shared with the Tahoe panel), different baseline (DepMap
# log2 TPM+1, because L1000 Level 5 carries no untreated profile and we hold no
# Level 3), and 8-9 contexts against Tahoe's 45.
#
# The absolute number is therefore NOT comparable to Tahoe's 0.5380. The run
# also emits the drug-mean and global-mean floors on this very slice, so the
# framing of the result can be settled afterwards without refitting.
#
# Decompresses the 5.4 GB Level-5 GCTX to scratch, reads once, deletes it, and
# caches the landmark submatrix so refits are cheap.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c 'import numpy, pandas, h5py; print("numpy", numpy.__version__, "h5py", h5py.__version__)'
df -h /nfs-stor/yuqi.lei | tail -1

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_l1000_rhaister_o.py" \
  --accession "${ACCESSION:-GSE70138}" \
  --hours "${HOURS:-24 h}" \
  --threads "${SLURM_CPUS_PER_TASK:-16}" \
  ${EXTRA_ARGS:-}

date --iso-8601=seconds
