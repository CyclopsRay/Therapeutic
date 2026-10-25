# Weekly Experiment Record / 每周实验记录 — 2026-09-11

**Coverage / 覆盖时间：** project inception through 2026-09-11  
**Status / 状态：** first weekly record; it consolidates the discussion and
work performed before this reporting convention was introduced  
**Companion results report / 配套结果报告：** `09_11_report.md`

This file preserves questions, reasoning, protocols, decisions, corrections,
and future branches. Final tables belong in the companion report, but important
numbers are repeated here when needed to explain a decision.

本文件忠实保存问题、推理、实验方案、决策、纠正和后续分支。正式结果表放在
配套 report 中；如果某个数字是理解决策所必需的，会在这里少量重复。

## 1. Weekly-file convention / 每周文件约定

- A new `MM_DD_report.md` and `MM_DD_record.md` pair is created only after the
  user explicitly says that a new week has started following the Friday group
  meeting. / 只有用户明确告知“周五组会后进入新的一周”才建立下一对文件。
- `report` is English and result-only: tables, figures, provenance, raw links,
  and positive or negative findings. / `report` 用英文，只记录结果；正负结果都保留。
- `record` may be bilingual and preserves the entire experiment discussion,
  including abandoned and gated-off ideas. / `record` 可中英混用，保留完整讨论，
  包括放弃或未过 gate 的方案。
- `ONGOING.md` now contains only running Slurm jobs and newly completed jobs
  whose outputs remain under analysis. Analyzed history is removed promptly. /
  `ONGOING.md` 只保留仍在集群运行、或刚完成但尚未分析完的任务。
- `report/` is tracked in git as durable research memory. `reports/` (plural)
  remains the durable topic-report directory. / 单数 `report/` 是每周实验记录并纳入
  git；复数 `reports/` 是长期主题报告，两者不能混淆。2026-09-11 用户明确纠正了
  先前“加入 gitignore”的要求，确认采用 host-level 标准范式；临时的项目例外已撤销。

## 2. Scientific question / 科学问题

### Primary question

Can a model predict the 24-hour gene-expression change caused by a known drug
and dose in a completely unseen cancer cell line, when that line contributes
no treated response to training?

主要问题：如果一个癌细胞系的任何 treatment response 都没有参加训练，模型能否
仅借助该细胞系的 untreated/DMSO state、药物信息以及其他细胞系上的 response，
预测已知 drug-dose 在该新细胞系中 24 小时后的表达变化？

### The claim boundary we fixed

| Term | Meaning here | Not equivalent to |
|---|---|---|
| Unseen cell line | Every treated Tahoe condition from the target line is excluded from all fitting | Random held-out cells or conditions |
| Unseen line–drug pair | The pair is absent but both entities can appear elsewhere | A completely unseen line |
| Unseen drug | The compound is excluded from response training | A new line receiving a known drug |
| Few-shot line | Some treatment responses from the target line are provided | Strict zero-shot line |

All preprocessing learned from response outcomes—gene selection, scaling,
PCA/encoders, hyperparameters, and early stopping—must also exclude the outer
test line.

## 3. Starting from the data / 从数据单元理解任务

### 3.1 What is a cell line? / cell line 到底是什么？

A cell line is a named, maintained population of cells derived from one tumor
or tissue and propagated in culture. A549, for example, is a human lung-cancer
cell line. It is neither one cell nor a normal developmental lineage. Cells in
one well are biological sisters from the same catalogue line and culture
batch, but individual cells differ in cell cycle, stochastic state, and
technical capture.

Cell line 是从某个肿瘤/组织样本建立并长期体外培养的一群细胞，例如 A549 是肺癌
细胞系。它不是一个细胞，也不是发育轨迹意义上的 cell type。一次实验中的一个
well 放入同一命名细胞系的许多细胞；单细胞之间仍会因 cell cycle、状态和测序噪声
而不同。

Tahoe is a parallel-endpoint screen. The sequenced cells before and after
treatment are not the same physical cells followed through time. Replicate
wells are allocated to DMSO or a drug-dose, cultured for 24 hours, and then
destroyed during scRNA-seq. Therefore “before/after” is a comparison between
matched populations, not a cellular trajectory or pseudotime.

Tahoe 不是跟踪同一个细胞的发展过程。DMSO well 和 treatment well 是平行终点；
24 小时后测序会破坏细胞。所以这里的 before/after 是匹配群体之间的对照，不是
pseudotime。

### 3.2 What do DMSO and treatment look like? / DMSO 和 treatment 长什么样？

- DMSO is the vehicle control: cells receive the solvent and otherwise the same
  plate/time handling. In a matrix it is simply many cells—or their average
  gene vector—labelled `DMSO_TF`.
- Treatment is a separate well with the same line and plate receiving one drug
  at one dose. Tahoe uses 0.05, 0.5, and 5 µM and reads the cells at 24 hours.
- The meaningful comparison is plate-matched. A global DMSO average would mix
  plate/batch differences into the drug effect.

Concrete pseudobulk example:

```text
A549, plate 6, DMSO:      [GENE1=1.2, GENE2=3.1, GENE3=0.4, ...]
A549, plate 6, drug X 5µM:[GENE1=2.0, GENE2=2.6, GENE3=0.9, ...]
delta = treated - DMSO:   [       +0.8,        -0.5,        +0.5, ...]
```

“Pseudobulk” means averaging normalized expression across cells belonging to
one biological condition. It reduces single-cell sampling noise and makes one
`cell line × drug × dose` vector the evaluation unit. It does not turn the
experiment into bulk RNA-seq, and the number of cells in a well is not the
number of independent treatment replicates.

### 3.3 Do we consider cell type? / 是否考虑不同 cell type？

Within Tahoe, the major context variable is the cancer cell-line identity and
its lineage. These cultures are far less compositionally diverse than a patient
tumor, but they can still contain cell-state subpopulations. Our current
Rhaister-O evaluation predicts condition-level pseudobulk deltas; it does not
model single-cell state transitions or annotate subclusters.

HTAN changes the problem: patient tumors contain malignant cells, immune cells,
fibroblasts, endothelium, and spatial niches. Any HTAN analysis must stratify
malignant versus microenvironmental compartments rather than average the whole
tumor and call that a cell-line response.

## 4. Dataset roles / 三个主数据集各自做什么

| Dataset | A useful “row” | Inputs/readout | Role in this project |
|---|---|---|---|
| Tahoe-100M | One cell, or after aggregation one `line × drug × dose` vector | scRNA counts; plate, line, compound, dose, DMSO labels | Supervised perturbation target |
| DepMap | One catalogue cell line | Baseline expression, CNV, mutation, dependency | Independent cell-context features |
| L1000 | One perturbational signature (`sig_id`) | 12,328-row Level-5 vector: 978 measured landmark genes plus inferred genes, with line/drug/dose/time/replicate metadata | External cross-platform comparison or carefully excluded prior |

The “same cell line” between Tahoe and L1000 means the same named catalogue
model, checked by a versioned identity registry and stable IDs where possible.
It never means the same physical cells or batch. Normalized-name matching was
sufficient for the first audit, not for a future training pipeline.

L1000 gene overlap also has levels: 946/978 landmarks occur somewhere in the
full Tahoe gene universe, but only 79 directly measured landmarks overlap the
released Rhaister 2,000-gene panel. The larger 1,365-row matrix overlap mostly
contains L1000-inferred values; it cannot be treated as 1,365 equally direct
measurements.

## 5. Why Rhaister-O is the baseline / Rhaister-O 如何实现

Rhaister-O is the closest published method to the strict question. Ordinary
Rhaister sees a perturbation panel in the query line and is few-shot; the `-O`
variant does not.

For treatment `t`, gene `g`, and unseen/query line `q`, the compact logic is:

1. Build a plate-matched DMSO centroid for each line.
2. For every reference line, calculate the pseudobulk drug delta.
3. Estimate a treatment- and gene-specific diagonal slope across the 45
   reference lines. This learns how baseline expression relates to response.
4. Apply that slope to the query line's DMSO vector.
5. Blend the diagonal prediction 95/5 with an additive ALS prediction; use the
   additive fallback when reference coverage is incomplete.

Conceptually:

```text
query prediction(t,g) ≈ treatment mean(t,g)
                      + gamma(t,g) × query baseline context(g)
```

The implementation is `scripts/run_rhaister_o_compact.py`, with five released
outer splits `5_holdout` through `9_holdout`, 45 reference and five test lines
per split, and a frozen 2,000-gene order. This is a pseudobulk delta predictor,
not a model that generates individual post-treatment cells.

## 6. Ordered experiment sequence and decisions / 已执行的实验顺序

The user fixed this sequence, and it was completed in order:

1. **All 25 held-out lines + independent bootstrap.** We summarized each line
   and used 2,000 drug-cluster bootstrap replicates per line, avoiding the false
   precision obtained by bootstrapping individual cells.
2. **Drug-mean residual metric.** We subtracted the reference-only drug-dose
   mean to ask how much of `~0.63` is more than “this drug usually moves these
   genes in this direction.” Rhaister-O remained at `0.537951`.
3. **Generic-template/PCA removal.** We fit PC1 only on reference-line residual
   responses and projected it out at test time. Rhaister-O remained at
   `0.525927`; reference PC1 explained only 4.90–5.49%.
4. **Nearest-DMSO and kNN controls.** Fixed 1NN/3NN based on untreated profiles
   scored `0.298532/0.340531` on full delta and much lower after component
   removal. This rejected the idea that baseline-neighbor copying alone is a
   competitive solution.
5. **DepMap-expression residual ridge.** A standalone ridge found weak real
   signal (`0.208719` residual Pearson) but was far below Rhaister-O
   (`0.539430`).
6. **Cross-fitted DepMap correction and gate.** Adding the learned correction
   changed residual Pearson by only `+0.000357`, with 95% line-bootstrap CI
   `[-0.002679, 0.003610]`. The gate failed.

Decision: under the current preregistered ladder, do not proceed to DepMap CNV,
mutation, dependency, or an MLP. The negative gate result is scientifically
meaningful and must remain visible.

## 7. L1000 synchronized comparison / L1000 这条线

No L1000 expression experiment existed at the start; only data and preliminary
crosswalks had been prepared. The user authorized a small parallel comparison.
The frozen protocol required:

- exact shared catalogue lines;
- structure-validated drugs where possible;
- 24-hour signatures;
- nearest doses within two-fold;
- the 79 directly measured common landmark genes as the primary panel;
- no target-line L1000 treated response used as an input to Tahoe training.

The final comparison used A549, HT29, and HS578T. Phase I produced weak but
bootstrap/permutation-supported concordance (`r=0.060522`); Phase II did not
replicate it (`r=0.015311`, `p=0.107784`). Phase I/II reproducibility was only
`0.2025` on 31 common conditions. On aligned HS578T conditions, direct L1000
transfer (`0.0577/0.0408`) was far below the Tahoe drug mean
(`0.6814/0.6822`) and Rhaister-O (`0.8437/0.8597`).

Interpretation / 解释：

- This is not “no biological signal”; it is evidence that direct gene-wise
  cross-platform transfer is weak and unstable after strict matching.
- L1000 Level-5 values and Tahoe pseudobulk deltas have different measurement
  scales, normalization, cell batches, and gene inference. Raw concatenation
  under a single loss is not justified.
- A future L1000 prior must exclude all treated signatures of the target Tahoe
  line and use nested line-level validation.

## 8. Deep learning discussion / 能不能上 deep learning？

Deep learning is technically possible, but architecture is not the current
bottleneck. Rhaister-O already captures a strong in-platform treatment-transfer
signal, whereas the tested extra context and cross-platform signal are weak.
A nonlinear model can easily learn drug means, batch/platform identity, or
target-line leakage and appear better on an easy split.

Two previously discussed candidate tracks remain unexecuted:

### Track A — Tahoe full-2K low-rank residual model

- Target: residual around frozen Rhaister-O or drug mean across the full 2,000
  genes.
- Inputs: target-line DMSO context and treatment identity/dose, with outer-line
  exclusion.
- First controls: linear and low-rank/bilinear models.
- A neural model would be considered only after the linear residual signal is
  stable over lines and beats permutation/bootstrap uncertainty.

### Track B — 79-gene L1000 late fusion

- Build a target-line-excluded L1000 drug prior on the 79 directly measured
  genes.
- Calibrate it through a Tahoe-specific linear/low-rank head; do not merge raw
  matrices.
- Compare against frozen Rhaister-O and drug mean on exactly the same held-out
  lines/conditions.
- A nonlinear adapter is a later-stage option, not the first experiment.

An illustrative stronger improvement threshold of approximately `+0.01`
line-macro Pearson was discussed for future nonlinear work, but it was not
formally authorized as a new gate. The current status remains: no deep model
training has been approved.

## 9. HTAN question and portal audit / HTA5 与 HTA9

### 9.1 Why HTAN is a different task

HTAN atlases measure patient tumors across clinical timepoints and spatial or
multi-omic assays. They do not supply Tahoe's factorial
`cell line × controlled compound × dose × plate-matched DMSO` matrix. A patient
pre/post contrast includes therapy combinations, time, disease evolution,
sampling site, tumor purity, and cell-composition changes.

Therefore:

- We cannot run Rhaister-O directly on HTAN.
- We cannot use HTAN as a clean gene-level benchmark for a drug-dose prediction.
- We can test retrospective translational consistency: do pathways predicted
  from Tahoe move in the same direction in relevant patient tumors, and in
  which malignant or microenvironmental compartments?

The language must be “retrospective consistency,” “case study,” or
“translational bridge,” not “clinical validation.”

### 9.2 HTA5

HTA5 (DFCI) studies the cellular geography of therapeutic resistance across
breast cancer, melanoma, and colon cancer. Current public metadata contain 156
participants and 896 linked biospecimens from 133 participants, with extensive
metastatic tissue and scRNA/MERFISH/MxIF/CODEX/imaging data.

The portal shows eight participants with both pre- and post-treatment
biospecimens atlas-wide, but zero currently linked to both timepoints within
the scRNA or MERFISH data-file joins. This must not be reported as “no paired
data”: several processed H5AD records have empty or pooled biospecimen fields.
The next scientific audit would have to inspect file manifests and `.obs`
metadata.

Role if used: spatial and cell-compartment analysis of resistant versus
sensitive states; likely cross-sectional or carefully recovered pre/post case
studies.

### 9.3 HTA9

HTA9 (OHSU) is a metastatic-breast-cancer atlas with serial biopsies and
multi-omic/spatial modalities. Current metadata contain 33 participants and
1,866 linked biospecimens; 32/33 participants have multiple nonempty timepoint
labels. The strongest portal-visible paired coverage is:

- bulk RNA: 27/32 linked participants with biopsy 1 and 2;
- GeoMx: 23/31;
- scATAC: 8/17;
- scRNA: 5/8;
- scDNA: 3/4.

Role if used: a small longitudinal breast-cancer pilot linking Tahoe-predicted
CDK4/6, endocrine, PARP, or immunomodulatory pathway responses to within-patient
serial biopsy changes, stratified by malignant/immune/stromal compartments and
therapy metadata.

### 9.4 HTAN decision

HTA9 is the preferred first HTAN candidate because paired within-patient
structure is already visible in several modalities. HTA5 is the secondary
spatial/resistance candidate. No HTAN matrices were downloaded this week; this
was a metadata-first, read-only feasibility audit. Before any compute job, we
need a metadata-only manifest extraction that freezes release, access status,
participant, biopsy/timepoint, therapy exposure, assay, and file IDs.

## 10. Other usable datasets / 还有哪些数据可以用？

### Tier 1: transcriptomic perturbation bridge

1. **sci-Plex.** Approximately 650,000 original single-cell profiles from
   A549, K562, and MCF7 treated with 188 compounds across four doses. A549 and
   the 24-hour multi-dose structure make it the closest extra stress test.
2. **scPerturb.** A harmonized collection of 44 perturbation datasets. It is
   useful for portability and evaluation tooling, but heterogeneity across
   source experiments prevents treating it as one clean training matrix.

### Tier 2: orthogonal functional endpoint

3. **PRISM Repurposing.** Large-scale cell-line viability with DepMap IDs.
   This is a strong test of whether a predicted transcriptional program is
   associated with actual killing/growth inhibition, but it does not validate
   individual gene predictions.
4. **CTRP v2** and optionally **GDSC.** Similar viability/sensitivity roles.
   Use them only where compound/line/time exposure can be made comparable.

### Tier 3: patient relevance

5. **BeatAML.** Patient ex-vivo drug response plus omics. It is valuable for
   patient-level generalization but changes both disease context (AML) and
   endpoint (viability), so it should be a separate project branch.
6. **HTAN HTA9/HTA5.** Patient pathway and spatial consistency, as described
   above; no direct factorial drug-response labels.

Possible order if a new branch is authorized:

```text
sci-Plex portability pilot
    → PRISM/CTRP viability association
    → HTA9 longitudinal pathway case study
    → HTA5 spatial resistance case study
```

This order is a proposal, not an execution instruction. Dataset downloads or
large preprocessing require explicit selection and Slurm submission.

## 11. Complete run ledger / 已运行任务

| Date | Slurm job | Purpose | Outcome |
|---|---:|---|---|
| 2026-09-07 | 212289 | Subset DepMap Public 24Q4 modalities | Completed; Tahoe-matched Parquet outputs |
| 2026-09-07 | 212309 | Download official L1000 Phase I/II Level-5 matrices | Completed; both GEO SHA-512 checks passed |
| 2026-09-07 | 214284, 214286 | Build compact Rhaister cache and reproduce splits | Completed |
| 2026-09-07 | 214291 | Verify prediction archive path/provenance | Completed |
| 2026-09-07 | 214296 | First provenance-complete Rhaister archives | Completed but superseded: object-dtype labels required pickle |
| 2026-09-07 | 214307 | Final safe Rhaister-O delta archives | Completed; fixed-width Unicode labels, `allow_pickle=False` |
| 2026-09-07 | 214816 | Small split-5 component/bootstrap pilot | Completed |
| 2026-09-08 | 216184 | All-25-line bootstrap, residual, PC1, 1NN/3NN evaluation | Completed |
| 2026-09-08 | 216201 | Standalone DepMap-expression residual ridge | Completed; preliminary gate false |
| 2026-09-08 | 216239 | Cross-fitted DepMap correction to Rhaister-O | Completed; final gate false |
| 2026-09-09 | 218448 | Initial L1000 extraction/alignment | Extraction succeeded; cancelled after inefficient repeated NPZ decompression was identified |
| 2026-09-09 | 218459 | Cached L1000 rerun | Completed but superseded: object-dtype labels |
| 2026-09-09 | 218460 | Final safe L1000 comparison | Completed; fixed-width Unicode labels, `allow_pickle=False` |

No Therapeutic Slurm job was active when this record was closed. Repeated
cluster-accounting database failures prevented `sacct` queries for several
jobs; completed job logs and `scontrol` were used to verify final state and
exit codes.

## 12. Corrections and durable cautions / 纠正与注意事项

- `0.6315` is a full pseudobulk-delta Pearson, not proof of perfect biological
  response recovery. Drug averages and generic templates must be reported
  beside it.
- Bootstrap uncertainty must respect the experiment unit. Cells within a well
  are not independent drug-response replicates.
- Target-line DMSO is permitted in the current zero-shot definition; any
  treated target-line signature is not.
- L1000 “same cell line” is catalogue identity only. Freeze a Cellosaurus/
  DepMap/provider identity registry before training across datasets.
- L1000's 12,328 matrix rows mix measured landmarks and inferred genes. The 79
  direct landmarks are the clean primary overlap with the Rhaister 2K panel.
- The compact Rhaister reproduction is the delta branch. Full FC/FDR heads and
  the six-metric upstream reproduction require the separate approximately
  40.6-GB pDEX parquet.
- HTAN file counts are release-specific data-file records, not unique samples.
  Portal linkage can be incomplete for pooled/processed objects.
- HTAN patient pre/post changes are confounded by cell composition and clinical
  treatment context. Analyze malignant and microenvironmental compartments
  separately.

## 13. Decisions required before the next experiment / 下一步需要决定

1. Choose one primary next branch: sci-Plex portability, HTA9 metadata/processed
   pilot, Track A Tahoe low-rank residual, or Track B 79-gene L1000 late fusion.
2. If choosing HTA9, decide whether the first endpoint is bulk-RNA pathway
   change (larger paired cohort) or scRNA compartment-specific change (smaller,
   richer cohort). A sensible sequence is bulk-RNA feasibility followed by
   scRNA/GeoMx triangulation.
3. If revising the deep-learning gate, define the minimum effect, bootstrap
   unit, permutation null, and outer-line protocol before fitting.
4. Do not add DepMap CNV/mutation/dependency merely because they are available;
   doing so requires an explicitly revised scientific rationale after the
   expression-only negative result.

## 14. Source pointers / 原始链接

- HTAN portal: <https://data.humantumoratlas.org/>
- HTA5: <https://data.humantumoratlas.org/center/hta5>
- HTA9: <https://data.humantumoratlas.org/center/hta9>
- HTAN access guide: <https://docs.humantumoratlas.org/data_access/portal/>
- sci-Plex: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7289078/>
- scPerturb: <https://www.sanderlab.org/scPerturb/>
- PRISM: <https://depmap.org/repurposing/>
- CTRP: <https://portals.broadinstitute.org/ctrp/>
- BeatAML: <https://www.cancer.gov/ccg/blog/2019/beataml>
- GDSC: <https://www.cancerrxgene.org/>

## 15. Clarification: DepMap representation and Tahoe response counts

### How DepMap expression represents one line

For each catalogue cell line, Public 24Q4 supplies one baseline bulk-expression
vector with 19,193 protein-coding genes in log-transformed TPM space:

```text
x(cell line) = [expression(gene 1), ..., expression(gene 19,193)]
```

The Tahoe line is joined to the DepMap model through the released stable
identifier mapping, not a fuzzy name. Within each outer split, every gene is
centered and scaled using only the 42–43 expression-covered reference lines;
missing values use the reference-gene mean. PCA is then fitted on those
reference lines only. Thus the 19,193-dimensional vector becomes a small set
of cell-context coordinates. The standalone model selected 10 PCs and ridge
lambda 100 in all five splits; the correction model selected 2 PCs and lambda
100.

### What the multi-output ridge predicts

For the standalone model, the response target for each reference line is a
flattened tensor:

```text
Y(line) = truth(line, treatment, 2,000 genes)
          - reference drug-dose mean(treatment, 2,000 genes)
```

There are 731–778 treatments with complete reference coverage per split. A
single multi-output ridge maps the DepMap PC vector to the complete
`treatment × gene` residual tensor. Equivalently, it uses the target line's
DepMap similarity to form a regularized weighted combination of reference-line
residual tensors. The final standalone prediction is drug mean plus this
predicted residual.

For the correction model, each reference line first receives a genuinely
out-of-sample, leave-one-line-out Rhaister-O prediction. The ridge target is:

```text
correction target = truth - leave-one-line-out Rhaister-O
outer prediction  = frozen Rhaister-O + predicted DepMap correction
```

Cross-fitting is essential: an in-sample Rhaister residual would leak the
reference line's own treated response and would not resemble the error made on
a completely unseen outer line.

### Tahoe counting conventions

- Unique compounds: 379.
- Nonzero doses: 0.05, 0.5, and 5 µM.
- The Tahoe paper reports 1,135 drug-dose combinations.
- The frozen compact derivative has 1,137 distinct treatment labels, exactly
  `379 × 3`.
- The nominal complete line-treatment grid is `50 × 1,137 = 56,850`; the
  compact derivative contains 56,827 observed pseudobulk delta rows, so 23 are
  absent.
- Before aggregation there are 65,218 plate rows because some conditions occur
  on multiple plates. DMSO centroids are stored separately as 700 line-plate
  rows (`50 × 14`).
- The released five-split evaluation uses 19,268 selected outer-test conditions
  across 25 held-out lines; it is an evaluation subset, not the whole Tahoe
  matrix.

The paper's 1,135 and the derivative's 1,137 belong to different released
counting contexts. Preserve both with provenance; do not silently replace one
with the other.

## 16. Clarification: gene overlap and reference-only PCA

The current DepMap residual models do not require DepMap and Tahoe to have the
same columns. They implement a supervised mapping between two spaces:

```text
X: aligned cell lines × 19,193 DepMap baseline-expression genes
Y: the same aligned cell lines × (treatments × 2,000 Tahoe response genes)
```

The required join is therefore the cell-line row join. The output coefficients
can learn how any DepMap input gene or global expression program predicts each
Tahoe treatment-gene response. A gene intersection would become mandatory for
a direct per-gene identity comparison, joint PCA, or shared encoder, none of
which is used here.

An exact-symbol audit nevertheless finds 1,987/2,000 (`99.35%`) of the frozen
Tahoe output genes in the 19,193 DepMap expression columns. The 13 absent
symbols are `ABTB3`, `CFAP107`, `CFAP95`, `DEFB113`, `DYNLT5`, `HYCC1`,
`MROH5`, `NALF1`, `OR1P1`, `OR8J2`, `PRSS51`, `PTCHD3`, and `SPDYE15`.
DepMap raw names contain symbol and Entrez ID; the prepared
`gene_columns.csv` stores parsed symbols and IDs.

For each outer split, PCA is produced as follows:

1. select only the 42–43 DepMap-covered reference lines;
2. for each of 19,193 genes, impute nonfinite values with the reference-line
   mean, remove zero/near-zero-variance genes, and z-score using reference
   mean and standard deviation;
3. compute `X_ref_scaled = U S V^T` by SVD;
4. define reference PC scores as `U S` and query scores as
   `X_query_scaled V^T`;
5. rescale PC scores by their reference-line standard deviations;
6. select 0, 2, 5, or 10 leading PCs and ridge lambda 1, 10, or 100 using
   analytic leave-one-reference-line-out RMSE.

The outer held-out lines are transformed with reference means, scales, and
loadings only. They do not affect gene filtering, PCA directions, or
hyperparameter selection. The standalone drug-mean-residual model selected 10
PCs in all splits; the Rhaister-error correction selected 2 PCs.

## 17. Clarification: Tahoe-to-DepMap cell-line identity coverage

The shared entity is a catalogue cell line, not a physical cell, well, passage,
or sequencing batch. Tahoe's released `Cell_ID_DepMap`/CVCL identifier is joined
to DepMap `RRID`, and the resulting DepMap `ModelID` indexes the molecular
matrices. Name normalization is descriptive only and is not the authoritative
join.

Coverage in the frozen Public 24Q4 preparation is:

```text
50 Tahoe observed lines
  └─ 49 have an exact DepMap catalogue match
       └─ 47 have a row in the 24Q4 baseline-expression matrix
```

`hTERT-HPNE` is the one Tahoe observed line without a released DepMap ID. `C3A`
(`CVCL_1098`, `ACH-001021`) and `COLO 205` (`CVCL_0218`, `ACH-001039`) have
exact catalogue matches but no baseline-expression row in this release. In the
25-line Rhaister outer-test cohort, these are the two unavailable expression
cases, leaving 23/25 evaluable lines.

Verified examples include A549 (`CVCL_0023` → `ACH-000681`), HT-29
(`CVCL_0320` → `ACH-000552`), Hs 578T (`CVCL_0332` → `ACH-000148`), and PANC-1
(`CVCL_0480` → `ACH-000164`). Matching means the same established cell-line
model; culture and platform differences remain and must not be described as
the same biological sample.

## 18. Proposed DepMap-to-pseudo-Tahoe-DMSO bridge

Question: can the shared lines be used to generate a pseudo Tahoe-space
Rhaister-O response from a DepMap-only baseline?

Answer: feasible as a new cross-platform experiment, but not by direct input
substitution. Rhaister-O consumes a Tahoe DMSO feature with the fixed 2,000
genes. The released centroid table has 700 line-plate rows (`50 × 14`) plus two
identifier columns. The executable path takes `log1p`, aggregates by cell line,
and centers using outer reference lines. DepMap is bulk `TPM log1p` with 19,193
genes and a different platform distribution.

The proposed data flow is:

```text
DepMap baseline x_q (19,193 genes)
        ↓ bridge fitted only on shared outer-reference lines
pseudo Tahoe-DMSO z_q (2,000-gene log1p feature)
        ↓ frozen reference-line Rhaister-O
pseudo Tahoe drug delta (treatment × 2,000 genes)
```

The smallest candidate bridges are: (A) shrinkage gene-wise calibration on
the 1,987 direct gene overlaps, with reference means for 13 absent genes; and
(B) reference-only DepMap PCA to reference-only Tahoe-DMSO PCA followed by a
ridge map and reconstruction. Direct `19,193 × 2,000` regression is not
credible with only 42–43 outer-reference anchors.

For each of the 23 expression-covered outer test lines, the bridge must exclude
that line's Tahoe DMSO and all treated responses. It predicts pseudo-DMSO from
DepMap, after which frozen Rhaister-O produces the response. Required controls
are drug mean, direct DepMap residual ridge, nearest-line pseudo-DMSO, and the
original real-DMSO Rhaister-O as an upper reference. Evaluate both pseudo-DMSO
reconstruction and downstream full/drug-residual/PC1-removed response metrics
with line-level and drug-cluster uncertainty.

If successful, the bridge could be applied to DepMap-only lines, but only for
the Tahoe-trained drugs/doses and only within supported cell-context regions.
Its final output is a predicted Tahoe-scale pseudobulk delta, not a generated
single-cell population. This experiment is recorded as proposed and has not
been run or authorized yet.

## 19. Clarification: what is an input record in Tahoe?

Tahoe must be described at three levels. A raw row is one cell with
`BARCODE_SUB_LIB_ID`, `cell_line_id`, drug/sample/plate labels, and sparse
gene/count arrays. Dose is resolved through the sample/treatment metadata. A
biological condition groups cells by `cell line × drug × dose × plate` at the
fixed 24-hour endpoint. The compact Rhaister record then aggregates cells into
one fixed-2,000-gene treated-minus-plate-matched-DMSO delta vector.

Tahoe has cell-line identity and tissue/lineage-of-origin metadata, but not a
patient-tumor-style immune/stromal/epithelial cell-type mixture annotation.
Cells within one line may occupy different cell-cycle or stress states; the
current pseudobulk benchmark does not use those as explicit cell-type inputs.

A verified example is A549 (`CVCL_0023`) on plate 4, sample `smp_1783`, treated
with 8-Hydroxyquinoline at 0.05 µM for 24 hours. The retained smoke shard has
424 cell rows from 47 pooled lines in that sample, including 15 A549 rows. One
A549 row has barcode `01_020_053-lib_1105` and sparse counts including
`NDUFAF7=1`, `CD38=1`, `UBE3C=2`, `HIVEP2=2`, and `VIM=1`. Plate-4 DMSO wells
are `smp_1877` and `smp_1878`. The authoritative compact A549 delta includes
`CFH=+0.002041`, `GCLC=+0.025761`, `LASP1=-0.043518`, `TFPI=+0.111277`,
`CD44=-0.007296`, and `PDK4=+0.014861` on the processed log-expression scale.

For Rhaister-O specifically, the query inputs are the target line's 2,000-gene
Tahoe DMSO centroid and a categorical drug-dose treatment label. Reference-line
DMSO and treated deltas are the training data. There is no explicit cell-type,
chemical-structure, MOA, or variable-time input; plate enters through matched
control construction, and the endpoint is fixed at 24 hours.

## 20. Clarification: cell line versus cell type

A cell line is a named experimental material: a population established from a
particular donor/tissue specimen and maintained in culture through passages. A
cell type is a biological class defined by shared identity, structure,
function, and/or developmental program, such as epithelial cell, fibroblast,
or T cell. A cell state is a reversible or contextual condition such as
cycling, stressed, interferon-responsive, or apoptotic.

Consequently, `A549` may be used as a categorical label in a plotting column,
but it is ontologically a cell-line identity rather than a conventional cell
type. `lung` is its tissue/lineage of origin, and individual A549 cells may
occupy multiple states. Tahoe labels each sequenced cell by its source line;
it does not provide an intact-tumor mixture of immune, stromal, and malignant
cell types.

Concrete catalogue examples are A549 (`CVCL_0023`, lung carcinoma,
`ACH-000681`), HT-29 (`CVCL_0320`, colorectal adenocarcinoma, `ACH-000552`),
PANC-1 (`CVCL_0480`, pancreatic adenocarcinoma, `ACH-000164`), and Hs 578T
(`CVCL_0332`, breast carcinoma, `ACH-000148`). These provenance descriptions
identify the specimens/diseases from which the laboratory lines were
established; they do not turn `lung`, `colon`, `pancreas`, or `breast` into a
single-cell type label.

Definition/provenance sources:

- NCI Dictionary, cell line: <https://www.cancer.gov/publications/dictionaries/cancer-terms/def/cell-line>
- Cell Ontology: <https://obophenotype.github.io/cell-ontology/>
- Cellosaurus resource: <https://www.cellosaurus.org/>
- A549: <https://www.cellosaurus.org/CVCL_0023> and <https://www.atcc.org/products/ccl-185>
- HT-29: <https://www.cellosaurus.org/CVCL_0320> and <https://www.atcc.org/products/htb-38>
- PANC-1: <https://www.cellosaurus.org/CVCL_0480> and <https://www.atcc.org/products/crl-1469>
- Hs 578T: <https://www.cellosaurus.org/CVCL_0332> and <https://www.atcc.org/products/htb-126>

## 21. 2026-09-14 architecture consultation: virtual-cell encoders

Status: literature review and scientific discussion only. The user requested a
critical assessment before choosing a pivot. No training, feature extraction,
new modality, or revised gate was authorized or executed. The current weekly
pair remains 09_11; no week rollover was announced.

### Interpretation and proposed direction

The virtual-cell framing is scientifically appropriate for the existing
baseline-conditioned, unseen-line response task, but does not change its
estimand. Residual correction is a parameterization and complementarity test,
not a competing scientific objective. For any proposed full predictor f and
fixed baseline b, the same prediction can be written b + (f - b). The negative
DepMap-expression correction therefore rejects the tested low-capacity
correction route, not arbitrary nonlinear conditional prediction. Correcting
a strong model is not necessarily statistically easier: remaining errors have
less signal and cross-fitted residual targets contain estimation noise.

Reasons a different model could improve include shared cross-gene response
structure, parameter sharing across related drugs, nonlinear DMSO-by-drug or
DMSO-by-DepMap interactions, and genuinely complementary molecular features.
The previous treatment-by-gene ridge already allows treatment-specific
coefficients; simply adding a drug encoder does not introduce a previously
absent class of interaction. Generic pretraining and additional parameters
alone are not supporting evidence.

The consultation recommends betting against a large model that substitutes
cell-line prose for measured DMSO. There are only 45 response-labelled reference
contexts per split, although the many treatments help estimate shared drug
and gene structure. The likely failure is learning transferable average drug
programs and reference-line identity while failing on the additional
drug-specific context component. A small model retaining DMSO and borrowing
strength across genes/drugs is more defensible.

Protein sequence or structure embeddings describe gene products; a response
decoder must learn how transcript abundance changes in context. These are
different objectives. Fixed output genes do not require an unseen-gene
encoder. A deterministic encoder/MLP/decoder is still a regression model;
latent reconstruction alone does not establish a useful generative endpoint.
Concatenation neither requires equal dimensions nor aligns independently
pretrained spaces. Bilinear conditioning is an interpretable low-capacity
control, not an established universal winner over concatenation or attention.

SapBERT text features remain an optional frozen ablation. Its synonym/entity
alignment objective is poorly matched to quantitative, drug-specific context.
Compare text against the exact same structured metadata, retain DMSO, omit
line names and treatment-derived annotations, and avoid fine-tuning a large
encoder on 45 lines. A fixed cell-node KG embedding is also weak unless it
contains useful molecular attributes and supports induction to new nodes.
Chronos is an external functional perturbation assay, not an untreated
expression measurement; it is allowed by the project's declared input boundary
but needs that explicit description.

### Literature audit (scope matters)

- **MultiDCP**, PLOS Computational Biology (2022), is the closest positive
  architectural precedent found. It concatenates drug, gene, cell and dose
  representations; uses baseline-expression encoding, STRING/node2vec gene
  features and a chemical GCN; and reports leave-new-cell-line-out L1000
  differential-expression prediction on 15 overlapping lines. This supports
  feasibility, but is not a Tahoe/Rhaister-O or template-removed comparison.
  Source: <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010367>.
- **TxPert**, Nature Biotechnology (2026), reports four leave-one-line-out
  genetic-perturbation tests with treated target-line data withheld and target
  controls allowed. It also tests graph rewiring. Caveats: proprietary PxMap
  and TxMap are not released; a raw-control additive bypass is reported to be
  useful in the limited-context setting; its nearest-line comparator chooses
  similarity using shared perturbation deltas, so that comparator does not
  satisfy our baseline-only nearest-line rule. No Rhaister-O/template-removed
  superiority is established. Source: <https://www.nature.com/articles/s41587-026-03113-4>.
- **State** separates the gene-level State Embedding model from State
  Transition's attention across sets of cells. Its 2025 v2 main Tahoe protocol
  and separate query-context protocol should retain their existing labels.
  A Cell corrected proof is now online (2026-08-31), DOI
  <https://doi.org/10.1016/j.cell.2026.07.052>; do not silently treat the local
  v2 PDF as that final version. Source for v2 architecture and tasks:
  <https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full>.
- **SapBERT** is biomedical entity/synonym alignment, not a validated
  cell-description response encoder:
  <https://aclanthology.org/2021.naacl-main.334/>.
- **Carrera et al. text-embedding chapter**: the publisher's accessible
  abstract describes PubMed-derived cell-line clustering and topic coherence,
  not a held-out transcriptional-response benchmark. Full chapter was not
  accessible. Source:
  <https://link.springer.com/chapter/10.1007/978-3-031-98287-3_31>.
- **Zinchenko et al. encoding benchmark**, DOI
  <https://doi.org/10.64898/2025.12.15.694331>: only the reproduced author
  abstract was accessible; full methods were not verified. It concerns
  viability and gene dependency and reports advantages for raw expression,
  STRING and bioassay-pretrained chemical features in those tasks. Do not cite
  it as a strict Tahoe expression-delta result or as evidence about SapBERT.
- **CLDR** encodes regression labels into natural-language supervision; it
  does not establish a baseline-only cell-description encoder:
  <https://arxiv.org/abs/2312.10707>.
- **CODE-AE** transfers cell-line sensitivity labels toward patient-response
  prediction, rather than predicting transcriptome deltas:
  <https://www.nature.com/articles/s42256-022-00541-0>.
- **PDGrapher** includes a forward response module but its principal endpoint
  and speed comparison concern inverse therapeutic-target prediction; the
  per-perturbagen scGen/CellOT discussion belongs to that benchmark setup:
  <https://www.nature.com/articles/s41551-025-01481-x>.
- **chemCPA** supports unseen-drug chemistry representation and **GEARS**
  supports graph priors for unseen genetic perturbations; neither original
  result is a direct strict Tahoe unseen-line comparison:
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract.html>,
  <https://www.nature.com/articles/s41587-023-01905-6>.

### Small pilot proposed for discussion, not approved

Test shared drug-conditioned cross-gene correction using compact existing
pseudobulks. Retain frozen Rhaister-O, fit a small response-error basis to
reference-line cross-fitted errors, and compare a bilinear head with a
one-hidden-layer concatenation MLP on the same low-dimensional features.
Cross these heads with DMSO-only versus DMSO-plus-DepMap-expression inputs.
Use known drug-dose features fitted on reference data, and include a
rank-matched linear control. This directly tests nonlinear conditioning and
sharing without requiring ESM, a KG build, a text encoder, or L1000 fusion.
If text is itself decision-critical, add frozen metadata-text and identical
structured-metadata ablations to the same head instead of expanding the model.

Every inner fold must refit response bases, PCs, mean templates and residual
construction without its held-out line. Select all capacity, shrinkage and
early stopping using inner line validation; freeze one candidate before outer
evaluation. Inspect error-basis reconstruction on inner validation as a
capacity diagnostic. Report every prescribed baseline, full and
template-removed delta scores, delta RMSE, and absolute-expression secondary
metrics on identical cohorts. The 23-line matched PC1-removed Rhaister score
is 0.5271, distinct from the 25-line 0.5259 and the drug-residual 0.539430.

A possible predeclared practical gate is >=0.01 absolute improvement in
template-removed line-macro Pearson plus a positive paired interval, no
material delta-RMSE degradation, and improvement across most outer splits.
This threshold is a proposed decision rule, not a literature-derived minimum
or an approved replacement gate. Bootstrap drugs with doses grouped and
shared drug resampling across lines; separately represent uncertainty across
lines. Include a line-block feature permutation and DMSO-feature ablation to
test whether apparent benefit actually uses transferable context. Account for
overlap of outer training sets and label the already inspected benchmark as
retrospective. Training belongs in monitored Slurm allocations; under-week
compute is plausible for compact heads but wall time is not measured.

If only DMSO-based heads improve, the evidence supports response modeling but
does not establish DepMap complementarity. A reliable added DepMap benefit
would reopen that hypothesis; failure of all compact probes argues against
scaling this particular route, without proving all nonlinear models
impossible. Other-line deployment with only DepMap and no Tahoe-like DMSO is
a separate, stricter portability task. Culture/protocol/platform differences
also remain environmental shifts even if environment generalization is not
the primary endpoint.

## 22. 2026-09-14 expanded review: DepMap-only virtual-cell generalization

用户明确提出 virtual cell 方向，要求召集子代理做 perturbation 文献综述和
cell context 编码评估。原始架构设想：drug / gene / cell line 三个 encoder，
drug 使用化学表征，gene 使用 ESM-2 或 AlphaFold 等先验，cell line 转成自然语言
后用 SapBERT；可增加 drug/gene KG；concat 后通过 MLP 与 decoder 预测表达变化。
讨论同时涉及 unseen perturbation、unseen cell context、unseen environment。

Executed work: primary literature review and architecture synthesis only.
Two user-requested agents ran in parallel: `virtual_cell_literature`
(gpt-5.6-terra, high) and `cell_context_design` (gpt-6-astra, xhigh).
The model availability check confirmed both configured tiers; no model
training, pretrained-feature extraction, matrix download or Slurm job ran.
The week has not rolled over; this remains the 09_11 record.

The reviewed artifact is
[`../reports/VIRTUAL_CELL_ENCODERS_AND_GENERALIZATION.md`](../reports/VIRTUAL_CELL_ENCODERS_AND_GENERALIZATION.md),
including primary links, model scope table, cell-encoder alternatives,
architecture diagram, OOD split matrix and a proposed bounded experiment.

### Interpretation and design decisions for this proposal

- 本轮把“DepMap 为 baseline”暂解释为**部署时只有外部基线，没有目标细胞系的
  Tahoe DMSO 输入**。这是待冻结的部署假设。此前的 DMSO+DepMap residual
  correction 仍保留为独立问题；它的负结果不能否定 DepMap-only 的实用价值。
  Rhaister-O 在新设定中是额外输入的参照，不是同输入竞争模型或数学上界。
- 核心 cell encoder 先保留定量 expression；文本作为额外先验，比较相同事实的
  structured metadata 与 frozen text。不能把药物敏感性、耐药结论或处理后表达
  写入目标细胞系的描述，再宣称 baseline-only。泛用预训练也要核对语料暴露。
- Drug 输入用 SMILES/分子图，不能仅用分子式；剂量保留单位。是否允许外部靶点
  和 MOA 需要在 cold-drug 测试中标注 chemistry-only 或 target-informed。
- Gene 是表达读出维度，不是此处的扰动靶点。ESM 表示蛋白序列，AlphaFold
  提供结构先验，都不自带 expression decoder。先比较 learned gene readout
  与功能/网络先验，再决定是否增加 protein features。
- 可行的简化形式是 `h = F(E_cell(DepMap), E_drug(structure), dose)`，
  `delta_g = D(h, E_gene(g))`。Concat 不要求输入维数相同；投影到相同维度也
  不代表语义已对齐。先比较小型 concat MLP 与 bilinear/gated head。
- 主监督放在解码后的 Tahoe expression delta；latent 重构不是最终生物学目标。
  不可直接将 Tahoe delta 加到 DepMap log-TPM 上作为绝对表达。
- 按 new-line、new-drug、joint new-line/new-drug 分别拆分。Environment 要定义
  可测量的培养或观测变量；单一协议不能支持任意未见环境的外推主张。

### Literature findings and one adjudicated correction

MultiDCP and XPert give direct architectural precedent, so three-encoder
concatenation alone is not a novelty claim. XPert reports L1000 cold-cell and
cold-drug delta evaluation. CellWhisperer/LangCell have more relevant
transcriptome–text objectives than SapBERT, but no response-prediction result
for our task was established. scFoundation has official bulk-use examples;
BulkFormer's author repository removed its L1000 perturbation benchmark due
to modality mismatch. TxPert/GEARS supply genetic, not chemical, analogues.

The agents differed on MultiDCP's baseline pretraining boundary. Root checked
the primary S10 caption, which explicitly describes distinct train/dev/test
cell lines for both response prediction and autoencoder tasks. The synthesis
therefore acknowledges this stated split instead of inferring that all target
baselines entered training. Cross-module fold alignment and preprocessing
remain unaudited. This is a literature interpretation correction, not a
change to our measured numbers.

### Proposed next test, not an executed experiment

Start with DepMap-only known-drug transfer on the existing 23
expression-covered held-out lines and fixed 2K target. Retain drug mean and
the existing DepMap ridge as equal-input baselines. Compare compact
quantitative context encoders and small conditional heads; then add one
text/gene/chemical-prior source at a time. Include drug-only and line-block
context permutations, decoded-delta RMSE, drug/template-removed metrics and
paired line uncertainty. All transforms and selection stay within nested
reference folds. This inspected benchmark is retrospective.

A prospective decision rule should test meaningful gain over the best
equal-input baseline rather than require beating a model supplied with target
Tahoe DMSO. No new numerical expansion gate was adopted during this review.
The earlier gate continues to describe its original complementarity test;
the new design and future execution are distinct from that completed result.
