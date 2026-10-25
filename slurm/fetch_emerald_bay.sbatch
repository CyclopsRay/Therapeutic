#!/usr/bin/env bash
#SBATCH --job-name=fetch-emerald
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/fetch-emerald-%j.out

set -euo pipefail
# Download the 116 EmeraldBay expression shards (57.7 GB), pinned to one HF revision,
# then write a size + sha256 manifest. Raw data are immutable: existing complete
# files are skipped (curl -C - resumes partial ones).
REV=${REV:-f2a0be6b02f731553657f0115c345b20bb020ede}
D=/nfs-stor/yuqi.lei/Therapeutic/raw/emerald-bay/$REV
BASE=https://huggingface.co/datasets/tahoebio/EmeraldBay/resolve/$REV
mkdir -p "$D/expression_data"
date --iso-8601=seconds; hostname; curl --version | head -1; echo "rev $REV"; df -h /nfs-stor/yuqi.lei | tail -1

seq -f '%05g' 0 115 | xargs -P 8 -I{} bash -c '
  f="expression_data/train-{}-of-00116.parquet"
  curl --fail --location --silent --show-error --retry 8 --retry-delay 10 -C - -o "'"$D"'/$f" "'"$BASE"'/$f" && echo "ok $f"'

cd "$D"
n=$(ls expression_data/*.parquet | wc -l); echo "shards present: $n"
test "$n" -eq 116
find . -type f ! -name SHA256SUMS ! -name MANIFEST.tsv -printf '%P\t%s\n' | sort > MANIFEST.tsv
find . -type f ! -name SHA256SUMS ! -name MANIFEST.tsv -print0 | sort -z | xargs -0 -P 8 -n 8 sha256sum > SHA256SUMS
echo "total bytes: $(awk -F'\t' '{s+=$2} END {print s}' MANIFEST.tsv)"
date --iso-8601=seconds
