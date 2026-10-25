#!/usr/bin/env bash
#SBATCH --job-name=txgnn-fetch
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --exclude=cn-02,cn-14
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/txgnn-fetch-%j.out
set -euo pipefail
# Fetch TxGNN's own PrimeKG (Dataverse doi:10.7910/DVN/CNQV69; NOT raw/primekg_v2.1) into raw/txgnn
# and link it into derived/txgnn/data. TLS to Dataverse's S3 resets intermittently on some nodes
# (cn-02, cn-14 on 2026-10-07; cn-20 earlier, though it worked on 2026-10-03): retry with backoff.
# The released checkpoint (Google Drive) is best-effort: only the re-draw side check needs it.
date --iso-8601=seconds; hostname
PY=/nfs-stor/yuqi.lei/Therapeutic/envs/txgnn-cpu/bin/python
RAW=/nfs-stor/yuqi.lei/Therapeutic/raw/txgnn
WORK=/nfs-stor/yuqi.lei/Therapeutic/derived/txgnn/data
mkdir -p "$RAW/primekg" "$RAW/ckpt" "$WORK"
# Stage downloads on node-local disk, then move: on 2026-10-08 writes from ciai-login-2 to /l failed
# with OSError EFAULT ("Bad address") after ~500 MB of kg.csv, while the network transfer itself was fine.
STAGE=${FETCH_STAGE:-${TMPDIR:-/tmp}/txgnn-fetch-$USER}
mkdir -p "$STAGE"
cd "$RAW/primekg"
for pair in 7144484:kg.csv 7144482:node.csv 7144483:edges.csv; do
  id=${pair%%:*}; f=${pair#*:}
  for attempt in 1 2 3 4 5 6; do
    [ -s "$f" ] && break
    echo "$(date -Is) $f attempt $attempt"
    "$PY" -c "
import requests
r = requests.get('https://dataverse.harvard.edu/api/access/datafile/$id', stream=True, timeout=120); r.raise_for_status()
n = 0
with open('$STAGE/$f.part', 'wb') as fh:
    for c in r.iter_content(1 << 20): fh.write(c); n += len(c)
want = r.headers.get('Content-Length')
assert want is None or int(want) == n, f'short read {n} of {want}'
print('$f', n, 'bytes')
" && mv "$STAGE/$f.part" "$RAW/primekg/$f" || { rm -f "$STAGE/$f.part"; sleep $((attempt * 60)); }
  done
  [ -s "$f" ] || { echo "FAILED to fetch $f"; exit 5; }
  ln -sfn "$RAW/primekg/$f" "$WORK/$f"
done
cd "$RAW/ckpt"
if [ ! -s txgnn_ckpt.download ]; then
  "$(dirname "$PY")/gdown" --fuzzy "https://drive.google.com/file/d/1fxTFkjo2jvmz9k6vesDbCeucQjGRojLj/view" \
    -O txgnn_ckpt.download || echo "checkpoint download failed (optional; retrain does not need it)"
fi
if [ -s txgnn_ckpt.download ]; then
  file txgnn_ckpt.download
  case "$(file -b txgnn_ckpt.download)" in
    Zip*) "$PY" -m zipfile -e txgnn_ckpt.download . ;;
    gzip*|*tar*) tar -xf txgnn_ckpt.download ;;
  esac
  find "$RAW/ckpt" -name config.pkl -o -name model.pt
fi
find "$RAW" -type f -printf '%s\t%p\n' | sort -k2 > "$RAW/files.tsv"
find "$RAW" -type f ! -name files.tsv ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > "$RAW/SHA256SUMS"
cat "$RAW/files.tsv"
date --iso-8601=seconds
