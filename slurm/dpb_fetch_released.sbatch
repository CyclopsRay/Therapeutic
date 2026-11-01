#!/usr/bin/env bash
#SBATCH --job-name=dpb-released
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --exclude=cn-02,cn-20
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/dpb-released-%j.out
# Bai et al. 2026 companion archive v2.0.0 (Zenodo 22095049, MIT): released per-run predictions of the seven models and
# the MLP (benchmark_predictions.zip, 9.35 GB) plus manifest and docs. Raw, immutable; used to check our reproduction
# and to compute PCC_plate on their own predictions.
set -euo pipefail
date --iso-8601=seconds; hostname; aria2c --version | head -1
D=/nfs-stor/yuqi.lei/Therapeutic/raw/bai2026_archive; mkdir -p "$D"; cd "$D"
curl -sSLf https://zenodo.org/api/records/22095049 -o record.json
python3 - <<'EOF' > files.tsv
import json
for f in json.load(open("record.json"))["files"]:
    if f["key"] in ("benchmark_predictions.zip", "MANIFEST_SHA256.csv", "README.md", "DATA_DICTIONARY.md", "benchmark_metrics.zip"):
        print(f["key"], f["links"]["self"], f["checksum"].split(":")[1], sep="\t")
EOF
while IFS=$'\t' read -r key url md5; do
  aria2c -c -x 8 -s 8 -k 64M --file-allocation=none --max-tries=0 --retry-wait=20 --timeout=60 --lowest-speed-limit=100K \
    --summary-interval=600 --console-log-level=warn -o "$key" "$url"
  echo "$md5  $key" | md5sum -c
  printf '%s\t%s\t%s\n' "$key" "$url" "$(date -u --iso-8601=seconds)" >> SOURCE
done < files.tsv
sha256sum benchmark_predictions.zip >> SHA256SUMS
unzip -l benchmark_predictions.zip | tail -3
date --iso-8601=seconds
