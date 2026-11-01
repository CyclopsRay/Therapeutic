#!/usr/bin/env bash
#SBATCH --job-name=dpb-setup
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --exclude=cn-02,cn-20
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/dpb-setup-%j.out
# Reproduction of Bai et al. 2026 (drug-perturbation-benchmark @ 55edbfe) main_benchmark, 2026-10-31.
# 1) conda env `dpb-benchmark` with the pins of environment/install_pip.sh (prebuilt wheels only);
# 2) the seven upstream repos at the pinned commits (reference/fetch_upstream.sh clone_pinned, code only);
# 3) only the data main_benchmark needs: XPert processed_data.zip (already in raw/, same md5), Figshare 28955141
#    all_drugs_unimol_arr.zip + HG_data.zip, CIGER/DeepCE repo bundles, TranSiGen release v1.0.
# MultiDCP / PRnet Zenodo data are needed only by original_* tasks and are not fetched. PertDiT: dpb_fetch_pertdit.
set -euo pipefail
date --iso-8601=seconds; hostname
W=/nfs-stor/yuqi.lei/Therapeutic/repro/dpb_bench
ENV=/nfs-stor/yuqi.lei/Therapeutic/envs/dpb-benchmark
export MAMBA_ROOT_PREFIX=/nfs-stor/yuqi.lei/micromamba PIP_CACHE_DIR=/nfs-stor/yuqi.lei/pip-cache
MM=/nfs-stor/yuqi.lei/bin/micromamba
cd "$W"

# ---------- 1. environment
if [ ! -x "$ENV/bin/python" ]; then
  "$MM" create -y -p "$ENV" -c conda-forge python=3.9 pip git unrar
fi
PIP="$ENV/bin/python -m pip"
$PIP install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
$PIP install torch-scatter==2.1.2 torch-sparse==0.6.18 pyg-lib==0.4.0 --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
$PIP install torch-geometric==2.6.1
$PIP install numpy==1.26.4 pandas==2.3.0 scipy==1.13.1 scikit-learn==1.4.0 matplotlib==3.8.2 seaborn==0.13.2 h5py==3.10.0 \
  pyyaml==6.0.1 tqdm==4.66.1 requests==2.28.1
$PIP install scanpy==1.9.8 anndata==0.10.9 rdkit==2024.3.2
$PIP install einops==0.8.2 torchmetrics==1.6.0 pytorch-lightning==1.9.5 diffusers==0.30.2
$PIP install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0.post1/flash_attn-2.6.0.post1+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
$PIP install unimol-tools==0.1.4.post1
$PIP install pandas==2.3.0
$PIP install transformers==4.36.2 sentencepiece==0.2.0 "zenodo_get>=1.5"
$PIP freeze > "$W/environment/FREEZE_dpb-benchmark.txt"
"$ENV/bin/python" -c "import torch, torch_geometric, rdkit, scanpy, anndata; print('torch', torch.__version__, 'pyg', torch_geometric.__version__, 'rdkit', rdkit.__version__, 'scanpy', scanpy.__version__)"

# ---------- 2. upstream code at pinned commits (as fetch_upstream.sh clone_pinned)
declare -A URL=([CIGER]=https://github.com/pth1993/CIGER.git [DeepCE]=https://github.com/pth1993/DeepCE.git
  [MultiDCP]=https://github.com/XieResearchGroup/MultiDCP.git [PertDiT]=https://github.com/wangkekekeke/PertDiT.git
  [PRnet]=https://github.com/Perturbation-Response-Prediction/PRnet.git [TranSiGen]=https://github.com/myzhengSIMM/TranSiGen.git
  [XPert]=https://github.com/GSanShui/XPert.git)
declare -A COMMIT=([CIGER]=81c16f107cf957ca73840e6fab41de174c85ee8f [DeepCE]=9b0d04fa920ef73df578b070cfcc3982effdde42
  [MultiDCP]=36ecdef9598de703b61fe0d1b91bcfeeba844274 [PertDiT]=596d681f816d3184d7a0a63a133ce68d5838fae0
  [PRnet]=f19174bde3ed2633f54c7831799cc38c4ffc7a0d [TranSiGen]=8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a
  [XPert]=d53ff497e465692c80fac7b899e3cc5cc8e7bfed)
GIT="$ENV/bin/git"
for M in CIGER DeepCE MultiDCP PertDiT PRnet TranSiGen XPert; do
  d="reference/$M"
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then echo "[skip clone] $M"; continue; fi
  rm -rf "$d"; "$GIT" clone --quiet "${URL[$M]}" "$d"; "$GIT" -C "$d" checkout --quiet --detach "${COMMIT[$M]}"
  echo "[cloned] $M $("$GIT" -C "$d" rev-parse HEAD)"; rm -rf "$d/.git"
done

# ---------- 3. data needed by main_benchmark
mkdir -p data/CIGER data/DeepCE data/TranSiGen/LINCS2020 data/XPert/figshare_28955141
[ -s data/CIGER/drug_smiles.csv ] || cp -r reference/CIGER/CIGER/data/. data/CIGER/
[ -s data/DeepCE/drugs_smiles.csv ] || cp -r reference/DeepCE/DeepCE/data/. data/DeepCE/
[ -s data/TranSiGen/LINCS2020/idx2smi.pickle ] || cp -r reference/TranSiGen/data/. data/TranSiGen/
REL=https://github.com/myzhengSIMM/TranSiGen/releases/download/v1.0
for f in processed_data.h5 KPGT_emb2304.pickle; do
  [ -s "data/TranSiGen/LINCS2020/$f" ] || curl -sSLf --retry 5 -o "data/TranSiGen/LINCS2020/$f" "$REL/$f"
done
# XPert: processed_data.zip = raw/xpert_sdst copy (figshare 58181296, md5 4a58bc01...), linked, not re-downloaded
ln -sfn /nfs-stor/yuqi.lei/Therapeutic/raw/xpert_sdst/processed_data.zip data/XPert/processed_data.zip
for spec in "54296132 all_drugs_unimol_arr.zip 4cea4b57b6143fbe9ee5f3f8d860607b" "54292856 HG_data.zip 17677573799fc37577f001c33cc2921d"; do
  set -- $spec; out="data/XPert/figshare_28955141/$2"
  [ -s "$out" ] || curl -sSLf --retry 5 -o "$out" "https://ndownloader.figshare.com/files/$1"
  echo "$3  $out" | md5sum -c
done
find data -maxdepth 3 -type f -newer "$W/README.md" -exec ls -la {} \; | head -50
du -sh data reference
date --iso-8601=seconds
