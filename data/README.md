# Data Inventory and Acquisition

Bulk data are stored at `/nfs-stor/yuqi.lei/Therapeutic`; `bulk-data` at the
project root is a convenience symlink. Run `scripts/fetch_data.sh` with one
named target to reproduce an acquisition. Do not download into `$HOME`.

## Tahoe-100M

Source: `tahoebio/Tahoe-100M`, pinned Hugging Face revision
`2dc57900b7981cfcf5e211527169a0b006546a95` (repository last modified
2025-07-23). License: CC0-1.0.

Present locally:

- four core metadata tables (`cell_line`, `drug`, `gene`, `sample`);
- one 28,225-cell raw-count expression shard for format/QC smoke tests.

The complete Hugging Face dataset is about 429 GB. It includes 3,388
expression Parquet shards plus a 2.29 GB observation table and a very large
long-form differential-expression configuration. Full acquisition is deferred
until we choose between streaming aggregation, Arc h5ad shards, and the compact
Rhaister derivative.

## Tahoe Rhaister benchmark

Source: `tahoebio/tahoe-de-rhaister`, pinned revision
`c7963cf334bec0683225d41c9586d900ca6303a2`. License: CC0-1.0.

Present locally (~1.2 GB):

- all 14 `cell_eval` plate files: one row per cell-line/treatment condition and
  2,000 pseudobulk delta-expression genes;
- DMSO cell centroids, 2,000-gene plate centroids, and the long-form control
  reference table;
- the dataset definition, fixed gene panel, and five complete-cell-line
  zero-shot split files.

This is the recommended first benchmark because it matches the prior Rhaister
results and uses plate-matched controls without reprocessing 95.6 million cells.
Its raw-to-derived implementation is mirrored at `external/Rhaister`.

## DepMap

Current public release found on 2026-09-07: DepMap Public 26Q1 (2026-04-01).
The desired primary file is
`OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` (305 MB), with `Model.csv`
for identifiers; dependency, CNV, and damaging/hotspot mutation matrices are
secondary modalities. The portal's no-captcha catalogue exposes 26Q1 filenames
and checksums but not download URLs, while the normal endpoint triggers human
verification.

For an immediately reproducible baseline, `depmap-24q4` downloads the immutable
Figshare-backed DepMap Public 24Q4 files: model metadata, unbatch-corrected
protein-coding log2(TPM+1) expression, Chronos gene effect, gene-level copy
number, and hotspot/damaging mutation matrices. We will use expression first
and compare/upgrade to 26Q1 only if its pipeline changes matter.

Prepared with Slurm job `212289` (completed `0:0` in 00:03:46):

- `derived/depmap_24q4/tahoe_observed_Model.csv`: 49 mapped Tahoe lines;
- baseline expression Parquet: 47 lines × 19,193 features;
- Chronos gene-effect Parquet: 38 lines × 17,916 features;
- gene-level CNV Parquet: 49 lines × 38,590 features;
- hotspot mutation Parquet: 49 lines × 542 features;
- damaging mutation Parquet: 49 lines × 19,097 features;
- `gene_columns.csv` and `summary.json` record feature names and missing IDs.

The missing coverage is real and must be handled explicitly. Do not silently
filter all experiments to the 38-line intersection.

## LINCS L1000

Sources: NCBI GEO GSE92742 (Phase I) and GSE70138 (Phase II). Metadata,
official checksum lists, README files, and the two Level 5 expression matrices
are present locally:

- Phase I Level 5: 473,647 signatures × 12,328 genes, 21,328,033,748 bytes;
- Phase II Level 5: 118,050 signatures × 12,328 genes, 5,365,179,698 bytes.

Slurm job `212309` downloaded these files on `cn-06` and completed with exit
`0:0` in 00:31:21. Both match the SHA-512 values in the official GEO checksum
lists. Do not decompress or transpose the complete matrices merely for
convenience; use the frozen signature/gene eligibility lists to extract the
smallest necessary subset.

Level 5 is the appropriate first auxiliary matrix (replicate-aggregated
signatures). Level 3 is preferable only if we later need replicate-aware
reprocessing. The assay directly measures 978 landmark genes and infers the
remainder, so direct and inferred genes must not be treated as equally observed.

The metadata overlap audit finds 7 Tahoe cell-line name matches in Phase I and
3 in Phase II, and maps 946 of 978 L1000 landmark symbols to Tahoe. Drug
crosswalks use normalized names, PubChem CIDs where available, and RDKit-derived
exact InChIKeys. Phase I has 157 Tahoe drugs matched by any method, 131 with
structure evidence; Phase II has 147 and 79. The outputs retain match methods
and expose one-to-many perturbagen IDs, which still require salt/form review
before modeling.

The first expression-level external comparison was completed by final Slurm
job `218460`. It filters to structure-validated drugs, exact shared lines,
24-hour signatures, and the nearest dose within two-fold, then temporarily
decompresses each GCTX only long enough to extract selected columns. Compact
arrays are under `/nfs-stor/yuqi.lei/Therapeutic/derived/lincs_tahoe_concordance`
(8.7 MB); result tables and the manifest are under
`/nfs-stor/yuqi.lei/Therapeutic/results/lincs_tahoe_concordance` (1.5 MB). The
primary comparison has 79 directly measured landmarks shared with the
Rhaister 2K panel. All four NPZ archives load with `allow_pickle=False`; see
`reports/TAHOE_DEPMAP_L1000_BILINGUAL_SYNTHESIS.md` for results and caveats.

## Derived metadata

Run:

```bash
/path/to/therapeutic/bin/python scripts/prepare_metadata.py
```

Outputs under `/nfs-stor/yuqi.lei/Therapeutic/derived/metadata` include the 50
observed Tahoe cell lines and DepMap IDs, parsed Tahoe dose/sample metadata,
LINCS overlap candidates, the L1000-landmark/Tahoe gene intersection, and a
machine-readable summary.

## Preparation environment and verification

The minimal environment is
`/nfs-stor/yuqi.lei/Therapeutic/envs/therapeutic-prep`; recreate it from
`requirements-prep.txt`. Raw acquisition checksums are stored beside their
downloads where supplied by the source. Paper checksums are in
`papers/SHA256SUMS`.
