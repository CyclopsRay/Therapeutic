#!/usr/bin/env bash
#SBATCH --job-name=txgnn-build
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=cn-02,cn-20
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/txgnn-build-%j.out
set -euo pipefail
# Build both TxGNN envs, fetch TxGNN's own PrimeKG (Dataverse doi:10.7910/DVN/CNQV69; NOT the
# raw/primekg_v2.1 release) and the README checkpoint into raw/txgnn with sizes and sha256, and
# link them into the TxData working folder derived/txgnn/data (TxGNN writes splits there).
date --iso-8601=seconds; hostname
VARIANT=cpu bash /home/yuqi.lei/Therapeutic/repro/envs/build_txgnn.sh
VARIANT=gpu bash /home/yuqi.lei/Therapeutic/repro/envs/build_txgnn.sh
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/txgnn-cpu
RAW=/nfs-stor/yuqi.lei/Therapeutic/raw/txgnn
WORK=/nfs-stor/yuqi.lei/Therapeutic/derived/txgnn/data
mkdir -p "$RAW/primekg" "$RAW/ckpt" "$WORK"
cd "$RAW/primekg"
for pair in 7144484:kg.csv 7144482:node.csv 7144483:edges.csv; do
  id=${pair%%:*}; f=${pair#*:}
  [ -s "$f" ] || "$ENV/bin/python" -c "
import requests
r = requests.get('https://dataverse.harvard.edu/api/access/datafile/$id', stream=True, timeout=120); r.raise_for_status()
with open('$f.part', 'wb') as fh:
    for c in r.iter_content(1 << 20): fh.write(c)
" && mv "$f.part" "$f"
  ln -sfn "$RAW/primekg/$f" "$WORK/$f"
done
cd "$RAW/ckpt"
[ -s txgnn_ckpt.download ] || "$ENV/bin/gdown" --id 1fxTFkjo2jvmz9k6vesDbCeucQjGRojLj -O txgnn_ckpt.download
file txgnn_ckpt.download
case "$(file -b txgnn_ckpt.download)" in
  Zip*) "$ENV/bin/python" -m zipfile -e txgnn_ckpt.download . ;;
  gzip*|*tar*) tar -xf txgnn_ckpt.download ;;
esac
find "$RAW/ckpt" -name config.pkl -o -name model.pt
find "$RAW" -type f -printf '%s\t%p\n' | sort -k2 > "$RAW/files.tsv"
find "$RAW" -type f ! -name files.tsv ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > "$RAW/SHA256SUMS"
cat "$RAW/files.tsv"
date --iso-8601=seconds
