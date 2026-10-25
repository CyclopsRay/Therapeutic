#!/usr/bin/env bash
#SBATCH --job-name=fetch-net
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs-stor/yuqi.lei/Therapeutic/logs/fetch-net-%j.out
set -euo pipefail
# Network sources for the RECOMB control-ladder study (PLAN.md Phase 6, E1/E7). Raw files are
# immutable; SOURCE records URL and UTC time, SHA256SUMS the checksums. HuRI is taken from BioGRID
# (Luck et al. 2020, PMID 32296183) because interactome-atlas.org serves a mismatched TLS certificate.
R=/nfs-stor/yuqi.lei/Therapeutic/raw/networks
date --iso-8601=seconds; hostname
get() {  # dir url file
  mkdir -p "$R/$1"; curl -sSfL --retry 3 --max-time 3000 "$2" -o "$R/$1/$3"
  printf '%s\t%s\t%s\n' "$3" "$2" "$(date -u --iso-8601=seconds)" >> "$R/$1/SOURCE"
  (cd "$R/$1" && sha256sum "$3" >> SHA256SUMS); ls -la "$R/$1/$3"
}
get biogrid "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-LATEST.tab3.zip" BIOGRID-ORGANISM-LATEST.tab3.zip
get omnipath "https://omnipathdb.org/interactions?datasets=omnipath&genesymbols=yes&fields=sources,references,curation_effort&format=tsv&license=academic" omnipath_interactions.tsv
get collectri "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&fields=sources,references&format=tsv&license=academic" collectri_interactions.tsv
get hgnc "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt" hgnc_complete_set.txt
get gene2pubmed "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz" gene2pubmed.gz
cd "$R/biogrid" && unzip -o -q BIOGRID-ORGANISM-LATEST.tab3.zip 'BIOGRID-ORGANISM-Homo_sapiens*' && ls -la
date --iso-8601=seconds
