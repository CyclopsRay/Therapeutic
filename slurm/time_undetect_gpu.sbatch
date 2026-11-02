#!/usr/bin/env bash
#SBATCH --job-name=undetect-time
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/undetect-time-%j.out
# Wall-time and memory of UnDetect on one A100-SXM4-40GB, the GPU of XPert job 357100 (paper: Method, Computational
# cost; record 2026-11-03). Stage 1 = cache_undetect.py (backbone, then kernel transfer and leave-one-out terms; the
# log line "mlp valid ... Ns" is the backbone time, summary.json "seconds" the fold total). Stage 2 =
# stage2_context.py --v2 (all interaction arms, an upper bound for ctx_pc alone; numpy on the node's CPUs).
# Peak GPU memory from torch.cuda.max_memory_allocated (PyTorch tensors only); host peak RSS from getrusage(RUSAGE_CHILDREN). Seed 1729 as cache_v1.
set -euo pipefail
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu
OUT=/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid
TAG=timing_gpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat /nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1/manifest.json
sha256sum /home/yuqi.lei/Therapeutic/scripts/run_sdst_hybrid.py /home/yuqi.lei/Therapeutic/scripts/cache_undetect.py /home/yuqi.lei/Therapeutic/scripts/stage2_context.py
echo "seed 1729"
mkdir -p "$OUT/$TAG"
timed() { "$ENV/bin/python" -I -c 'import resource, subprocess, sys, time
t = time.time(); rc = subprocess.run(sys.argv[1:]).returncode
print("wall_s", round(time.time() - t, 1), "peak_rss_MB", round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024), flush=True)
sys.exit(rc)' "$@"; }
echo "== stage 1 start $(date +%s)"
timed "$ENV/bin/python" -I -c 'import runpy, sys, torch
sys.argv = ["cache_undetect.py", "--seed", "1729", "--tag", sys.argv[1]]
runpy.run_path("/home/yuqi.lei/Therapeutic/scripts/cache_undetect.py", run_name="__main__")
print("peak_gpu_alloc_MB", round(torch.cuda.max_memory_allocated() / 2**20), flush=True)' "$TAG"
echo "== stage 1 end $(date +%s)"
echo "== stage 2 start $(date +%s)"
timed "$ENV/bin/python" -I /home/yuqi.lei/Therapeutic/scripts/stage2_context.py --v2 --cache "$TAG" --tag "${TAG}_s2"
echo "== stage 2 end $(date +%s)"
grep -h '"seconds"\|"mlp_epoch"' "$OUT/$TAG"/fold_*/summary.json
date --iso-8601=seconds
