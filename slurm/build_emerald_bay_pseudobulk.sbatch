#!/usr/bin/env bash
#SBATCH --job-name=eb-pseudobulk
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/eb-pseudobulk-%j.out

set -euo pipefail
# EmeraldBay single cells -> (cell_line, sample, half) sums on the Tahoe 2K panel.
PROJECT_ROOT=/home/yuqi.lei/Therapeutic
ENV_ROOT=/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep
REV=${REV:-f2a0be6b02f731553657f0115c345b20bb020ede}
RAW=/nfs-stor/yuqi.lei/Therapeutic/raw/emerald-bay/$REV
OUT=${OUT:-/nfs-stor/yuqi.lei/Therapeutic/derived/emerald_bay/pseudobulk_v1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
date --iso-8601=seconds; hostname; echo "rev $REV"
"$ENV_ROOT/bin/python" -c 'import numpy, pandas, pyarrow; print("numpy", numpy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__)'
"$ENV_ROOT/bin/python" "$PROJECT_ROOT/scripts/build_emerald_bay_pseudobulk.py" \
  --raw "$RAW" --out "$OUT" \
  --panel /nfs-stor/yuqi.lei/Therapeutic/raw/tahoe-de-rhaister/definition/static_2k_genes.json \
  --tahoe-gene-meta /nfs-stor/yuqi.lei/Therapeutic/raw/tahoe-100m/metadata/gene_metadata.parquet \
  --workers 8 --seed 0
date --iso-8601=seconds
