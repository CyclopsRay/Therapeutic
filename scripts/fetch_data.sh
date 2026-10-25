#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${THERAPEUTIC_DATA_ROOT:-/nfs-stor/yuqi.lei/Therapeutic}"
TAHOE_REV="2dc57900b7981cfcf5e211527169a0b006546a95"
RHAISTER_DATA_REV="c7963cf334bec0683225d41c9586d900ca6303a2"

usage() {
  echo "Usage: $0 {tahoe-smoke|rhaister-benchmark|depmap-24q4|lincs-metadata|lincs-level5}"
}

download() {
  local url="$1"
  local destination="$2"
  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]]; then
    echo "exists: $destination"
    return
  fi
  curl --fail --location --silent --show-error --retry 5 \
    --continue-at - --output "${destination}.part" "$url"
  mv -- "${destination}.part" "$destination"
  echo "downloaded: $destination"
}

fetch_tahoe_smoke() {
  local base="https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/${TAHOE_REV}"
  local root="${DATA_ROOT}/raw/tahoe-100m"
  local name
  for name in cell_line_metadata drug_metadata gene_metadata sample_metadata; do
    download "${base}/metadata/${name}.parquet" \
      "${root}/metadata/${name}.parquet"
  done
  download "${base}/data/train-00000-of-03388.parquet" \
    "${root}/sample/train-00000-of-03388.parquet"
}

fetch_rhaister_benchmark() {
  local base="https://huggingface.co/datasets/tahoebio/tahoe-de-rhaister/resolve/${RHAISTER_DATA_REV}"
  local root="${DATA_ROOT}/raw/tahoe-de-rhaister"
  local plate file split
  for plate in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    download "${base}/cell_eval/plate_plate${plate}.parquet" \
      "${root}/cell_eval/plate_plate${plate}.parquet"
  done
  for file in cell_line_centroids_dmso.parquet \
              cell_line_centroids_dmso_hvg.parquet \
              control_expression.parquet; do
    download "${base}/zeroshot/${file}" "${root}/zeroshot/${file}"
  done
  for file in dataset.toml static_2k_genes.json; do
    download "${base}/definition/${file}" "${root}/definition/${file}"
  done
  for split in 5 6 7 8 9; do
    file="zeroshot_generalization_converted_cell_lines_3b.toml"
    download "${base}/definition/${split}_holdout/${file}" \
      "${root}/definition/${split}_holdout/${file}"
  done
}

fetch_depmap_24q4() {
  # 24Q4 is used because its immutable Figshare links remain scriptable. The
  # current 26Q1 catalogue is recorded in data/README.md but requires the
  # DepMap portal's human-verification flow as of 2026-09-07.
  local root="${DATA_ROOT}/raw/depmap/DepMap_Public_24Q4"
  download "https://ndownloader.figshare.com/files/51065795" "${root}/README.txt"
  download "https://ndownloader.figshare.com/files/51065297" "${root}/Model.csv"
  download "https://ndownloader.figshare.com/files/51065489" \
    "${root}/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
  download "https://ndownloader.figshare.com/files/51064667" \
    "${root}/CRISPRGeneEffect.csv"
  download "https://ndownloader.figshare.com/files/51065324" \
    "${root}/OmicsCNGene.csv"
  download "https://ndownloader.figshare.com/files/51065750" \
    "${root}/OmicsSomaticMutationsMatrixHotspot.csv"
  download "https://ndownloader.figshare.com/files/51065747" \
    "${root}/OmicsSomaticMutationsMatrixDamaging.csv"
}

fetch_lincs_metadata() {
  local root="${DATA_ROOT}/raw/lincs-l1000"
  local p1="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl"
  local p2="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl"
  download "${p1}/GSE92742_Broad_LINCS_cell_info.txt.gz" "${root}/GSE92742/cell_info.txt.gz"
  download "${p1}/GSE92742_Broad_LINCS_gene_info.txt.gz" "${root}/GSE92742/gene_info.txt.gz"
  download "${p1}/GSE92742_Broad_LINCS_pert_info.txt.gz" "${root}/GSE92742/pert_info.txt.gz"
  download "${p1}/GSE92742_Broad_LINCS_sig_info.txt.gz" "${root}/GSE92742/sig_info.txt.gz"
  download "${p1}/GSE92742_SHA512SUMS.txt.gz" "${root}/GSE92742/SHA512SUMS.txt.gz"
  download "${p1}/GSE92742_Broad_LINCS_README.pdf" "${root}/GSE92742/README.pdf"
  download "${p2}/GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz" "${root}/GSE70138/cell_info.txt.gz"
  download "${p2}/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz" "${root}/GSE70138/gene_info.txt.gz"
  download "${p2}/GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz" "${root}/GSE70138/pert_info.txt.gz"
  download "${p2}/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz" "${root}/GSE70138/sig_info.txt.gz"
  download "${p2}/GSE70138_SHA512SUMS.txt.gz" "${root}/GSE70138/SHA512SUMS.txt.gz"
  download "${p2}/GSE70138_Broad_LINCS_README.pdf" "${root}/GSE70138/README.pdf"
}

fetch_lincs_level5() {
  local root="${DATA_ROOT}/raw/lincs-l1000"
  local p1="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl"
  local p2="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl"
  local f1="GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
  local f2="GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz"
  download "${p1}/${f1}" "${root}/GSE92742/${f1}"
  download "${p2}/${f2}" "${root}/GSE70138/${f2}"
  printf '%s  %s\n' \
    '6a3115cf3aaa402bb1bc098678b52b9c03632fc21087050e3e1a56c24a60196c4809bbac3797978dab63a47d955ff7a2c6110025a6e6d3cef4c5f6e010208a2a' \
    "${root}/GSE92742/${f1}" | sha512sum --check -
  printf '%s  %s\n' \
    '9d078903d3d028f37b0bae584c1afe48f5bab0082e61f6e542de471674c78946cff710132b8e95beb997a39dc2dd6442fdda0fbb1cc9b2399ef21cf636975c9f' \
    "${root}/GSE70138/${f2}" | sha512sum --check -
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  tahoe-smoke) fetch_tahoe_smoke ;;
  rhaister-benchmark) fetch_rhaister_benchmark ;;
  depmap-24q4) fetch_depmap_24q4 ;;
  lincs-metadata) fetch_lincs_metadata ;;
  lincs-level5) fetch_lincs_level5 ;;
  *) usage; exit 2 ;;
esac
