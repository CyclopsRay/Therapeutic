#!/usr/bin/env bash
#SBATCH --job-name=dpb-prepare
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=cn-02,cn-20
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/dpb-prepare-%j.out
# main_benchmark prepare steps A, A2, B, C, D, F (experiments/_shared/prepare_steps.py), run once before any training
# job (step D edits the h5ad in place). PertDiT steps D2/E are left out until its data is fetched.
set -euo pipefail
date --iso-8601=seconds; hostname
W=/nfs-stor/yuqi.lei/Therapeutic/repro/dpb_bench; ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/dpb-benchmark
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
cd "$W"
# XPert configs read reference/XPert/HG_data/; fill it from Figshare HG_data.zip if the clone does not carry it
if [ ! -s reference/XPert/HG_data/saved_embedding/HG_drug_embeddings.npy ]; then
  mkdir -p reference/XPert/tmp_hg && unzip -o -q data/XPert/figshare_28955141/HG_data.zip -d reference/XPert/tmp_hg
  src=$(dirname "$(dirname "$(find reference/XPert/tmp_hg -name HG_drug_embeddings.npy | head -1)")")
  mkdir -p reference/XPert/HG_data && cp -r "$src"/. reference/XPert/HG_data/ && rm -rf reference/XPert/tmp_hg
fi
"$ENV/bin/python" - <<'PY'
import sys; sys.path.insert(0, "experiments/_shared")
import prepare_steps as s
ok = s.run_steps([("A extract processed_data", s.step_extract_processed_data),
                  ("A2 extract unimol arr", s.step_extract_xpert_unimol_arr),
                  ("B CIGER gene_feature_nm", s.step_generate_gene_feature_nm),
                  ("C TranSiGen VAE", s.step_verify_transigen_vae),
                  ("D drug-blind splits", s.step_generate_drug_blind_splits),
                  ("F XPert HG_data", s.step_verify_xpert_hg_data)])
sys.exit(0 if ok else 1)
PY
ls -la data/XPert/processed_data/
date --iso-8601=seconds
