#!/usr/bin/env bash
#SBATCH --job-name=eb-method
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/eb-method-%x-%j.out

set -euo pipefail
# Generic launcher: run one existing Tahoe method script unchanged on EmeraldBay
# through scripts/run_on_emerald_bay.py (patches the three data loaders).
# Required: SCRIPT (basename in scripts/), ARGS (script arguments). EB_DIR optional.
PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
export EB_DIR=${EB_DIR:-/nfs-stor/yuqi.lei/Therapeutic/derived/emerald_bay/sanity_v1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
date --iso-8601=seconds; hostname; echo "SCRIPT=$SCRIPT"; echo "ARGS=$ARGS"; echo "EB_DIR=$EB_DIR"
"$ENV_ROOT/bin/python" -c 'import numpy, pandas, pyarrow; print("numpy", numpy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__)'
cat "$EB_DIR/../pseudobulk_v1/provenance.json" 2>/dev/null || true
cd "$PROJECT_ROOT"
# shellcheck disable=SC2086
"$ENV_ROOT/bin/python" scripts/run_on_emerald_bay.py "scripts/$SCRIPT" $ARGS
date --iso-8601=seconds
