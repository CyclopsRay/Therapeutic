#!/usr/bin/env bash
#SBATCH --job-name=chem-shrink
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/chem-shrink-%j.out

set -euo pipefail

# Shrink Rhaister-O's per-(treatment, gene) slopes toward a CHEMISTRY-predicted
# target instead of a pooled per-gene mean.
#
# Job 272027 shrank uniformly and gained +0.00187 with an optimal blend of 0.9,
# i.e. almost no uniform shrinkage helps. Hypothesis: uniform pooling destroys
# signal by averaging dissimilar drugs, and chemically similar drugs should
# share slope structure.
#
# The diagonal structure is preserved and only the shrinkage TARGET changes, so
# a failure is attributable to chemistry carrying no transferable slope
# structure rather than to breaking the diagonal -- which job 249519 already
# showed fails at 45 reference lines.
#
# Arms: uniform_shrink (reproduces 272027 as an internal control), moa_only,
# morgan_only, chem_full (headline). The anchor is asserted per arm: forcing
# w=1 must reproduce the frozen array bit for bit.
#
# No new modality: the cell-side input is the same plate-matched Tahoe DMSO
# profile Rhaister-O consumes, so the DepMap expansion gate does not apply.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs-stor/yuqi.lei/Therapeutic/results/chemistry_shrinkage}
RHAISTER_DATASET_REVISION=c7963cf334bec0683225d41c9586d900ca6303a2
RANDOM_SEED=0
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c 'import numpy, pandas, pyarrow, rdkit; print("numpy", numpy.__version__, "pandas", pandas.__version__, "rdkit", rdkit.__version__)'
git -C "${PROJECT_ROOT}/external/Rhaister" rev-parse HEAD
echo "tahoe-de-rhaister revision ${RHAISTER_DATASET_REVISION}"
echo "seed ${RANDOM_SEED}   output ${OUTPUT_ROOT}"
echo "splits: ${SPLITS:-5 6 7 8 9}   arms: ${ARMS:-uniform_shrink moa_only morgan_only chem_full}"

"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_chemistry_shrinkage.py" \
  --splits ${SPLITS:-5 6 7 8 9} \
  --arms ${ARMS:-uniform_shrink moa_only morgan_only chem_full} \
  ${EXTRA_ARGS:-} \
  --n-permutations "${N_PERMUTATIONS:-50}" \
  --n-bootstrap "${N_BOOTSTRAP:-20000}" \
  --seed "${RANDOM_SEED}" \
  --data-revision "${RHAISTER_DATASET_REVISION}" \
  --output-root "${OUTPUT_ROOT}"

date --iso-8601=seconds
