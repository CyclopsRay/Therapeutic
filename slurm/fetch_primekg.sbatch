#!/usr/bin/env bash
#SBATCH --job-name=fetch-primekg
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/fetch-primekg-%j.out
set -euo pipefail
# PrimeKG v2.1 (Harvard Dataverse doi:10.7910/DVN/IXA7BM, CC0 1.0 on the Dataverse record) for the
# drug-target / KG overlap audit of the unseen-drug branch. Raw, immutable; checksums recorded.
OUT=/nfs-stor/yuqi.lei/Therapeutic/raw/primekg_v2.1
mkdir -p "$OUT"; cd "$OUT"
date --iso-8601=seconds; hostname
API="https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/IXA7BM"
curl -s -f "$API" > dataset.json
python3 - <<'PY' > files.tsv
import json
d=json.load(open('dataset.json'))['data']['latestVersion']
for f in d['files']:
    df=f['dataFile']; print(df['id'], df['filename'], df.get('filesize'), df.get('md5',''), sep='\t')
PY
cat files.tsv
for name in nodes.tab drug_features.tab kg.csv README.txt; do
  id=$(awk -F'\t' -v n="$name" '$2==n{print $1}' files.tsv)
  case "$name" in *.tab) q="?format=original";; *) q="";; esac
  [ -s "$name" ] || curl -s -f -L -o "$name" "https://dataverse.harvard.edu/api/access/datafile/${id}${q}"
  ls -la "$name"
done
md5sum nodes.tab drug_features.tab kg.csv README.txt > MD5SUMS_local.txt; sha256sum nodes.tab drug_features.tab kg.csv README.txt > SHA256SUMS
cat MD5SUMS_local.txt
date --iso-8601=seconds
