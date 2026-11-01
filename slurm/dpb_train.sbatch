#!/usr/bin/env bash
#SBATCH --job-name=dpb-train
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/dpb-train-%j.out
# Bai et al. 2026 main_benchmark reproduction: MODELS / FOLDS / EPOCHS (optional) / RESULTS via --export.
set -euo pipefail
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
W=/nfs-stor/yuqi.lei/Therapeutic/repro/dpb_bench; ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/dpb-benchmark
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8 HF_HOME=/nfs-stor/yuqi.lei/hf_cache PATH="$ENV/bin:$PATH"
cd "$W"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
sha256sum data/XPert/processed_data/l1000_sdst_78453.h5ad | cut -c1-16
python experiments/main_benchmark/train_all.py --models ${MODELS} --folds ${FOLDS:-0 1 2 3 4} \
  --results_dir "${RESULTS:-$W/results/main_benchmark}" ${EPOCHS:+--epochs $EPOCHS} --continue_on_fail
date --iso-8601=seconds
