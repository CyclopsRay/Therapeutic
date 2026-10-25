#!/usr/bin/env bash
#SBATCH --job-name=cell-feature-ablation
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/cell-feature-ablation-%j.out

set -euo pipefail

# Does a natural-language description of a cell line carry drug-response
# information beyond the structured metadata fields it is written from?
#
# Four arms share one response head, the five outer splits, the component and
# penalty grids, and reference-only fitting: DepMap expression, structured
# Model.csv fields, a frozen LM embedding of template text generated from
# exactly those fields, and a frozen LM embedding of redacted Cellosaurus
# comment text.  Each arm carries its own line-label permutation null.
#
# Text encoder weights and Cellosaurus text are pre-staged on the login node;
# this job runs offline.

PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
TEXT_ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/text-embed
TEXT_ROOT=/nfs-stor/yuqi.lei/Therapeutic/derived/cellosaurus
OUTPUT_ROOT=/nfs-stor/yuqi.lei/Therapeutic/results/cell_feature_ablation
RHAISTER_DATASET_REVISION=c7963cf334bec0683225d41c9586d900ca6303a2
TEXT_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
RANDOM_SEED=0
mkdir -p /nfs-stor/yuqi.lei/Therapeutic/logs

export HF_HOME=/nfs-stor/yuqi.lei/Therapeutic/caches/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

date --iso-8601=seconds
hostname
"${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c 'import numpy, pandas, pyarrow; print("numpy", numpy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__)'
"${TEXT_ENV_ROOT}/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
git -C "${PROJECT_ROOT}/external/Rhaister" rev-parse HEAD
echo "tahoe-de-rhaister revision ${RHAISTER_DATASET_REVISION}"
echo "text encoder ${TEXT_MODEL} (frozen, attention-masked mean pooling)"
echo "random seed ${RANDOM_SEED}"
echo "output root ${OUTPUT_ROOT}"

MODEL_PATH=$(find "${HF_HOME}/hub" -maxdepth 4 -type d -path '*SapBERT*/snapshots/*' | head -1)
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: no pre-staged SapBERT snapshot under ${HF_HOME}/hub" >&2
  exit 1
fi
echo "text encoder snapshot ${MODEL_PATH}"

echo
echo "=== stage 1: freeze text embeddings ==="
"${TEXT_ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/embed_cell_text.py" \
  --model-name "${TEXT_MODEL}" \
  --model-path "${MODEL_PATH}" \
  --output-root "${TEXT_ROOT}"

echo
echo "=== stage 2: cell-representation ablation ==="
cd "${PROJECT_ROOT}/scripts"
"${ENV_ROOT}/bin/python" "${PROJECT_ROOT}/scripts/run_cell_feature_ablation.py" \
  --component-grid 2 5 10 15 20 25 30 35 40 \
  --lambda-grid 0.1 1 10 100 1000 10000 100000 \
  --n-permutations 20 \
  --n-bootstrap 2000 \
  --seed "${RANDOM_SEED}" \
  --data-revision "${RHAISTER_DATASET_REVISION}" \
  --template-embeddings "${TEXT_ROOT}/embeddings_template.npz" \
  --cellosaurus-embeddings "${TEXT_ROOT}/embeddings_cellosaurus.npz" \
  --output-root "${OUTPUT_ROOT}"

date --iso-8601=seconds
