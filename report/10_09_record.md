# Therapeutic — Weekly Record, week ending Friday 2026-10-09

Reporting interval: [2026-10-02T18:00:00+04:00, 2026-10-09T18:00:00+04:00) — Asia/Dubai.

Plans, discussion, decisions, reasoning, and corrections (Chinese or English).

## 2026-10-03 — Unseen-drug branch on L1000: question, literature plan, pre-registration

### Question (user, 2026-10-03)

用户提出新方向:**给定细胞系 baseline 与一个未见过的药物(有化学结构、部分靶点/PPI 背景知识),
预测其转录扰动响应。** 倾向用 L1000(药物远多于 Tahoe)。两种思路:

1. **Retrieval(approach 1)**:药物编码空间足够稠密时,对新药在编码空间找最近的已知药物,
   用加权近邻的响应作为预测。依赖 encoder 是否把"响应相似"的药物放在一起。
2. **Graph inference(approach 2)**:把药物 embedding、靶点、PPI、通路等放进一个图,训练
   GNN/GAT 直接推断;依赖 PPI/靶点知识是否有用且完整。

并问:是否已有研究、弱点何在、能否引入新模态改进,以及 Rhaister-O 的经验哪些可迁移。
用户允许做 sanity check 实验并在有前景时继续改进模型。

This is the **unseen-drug** task, which CLAUDE.md lists as a distinct secondary experiment. It
is not the primary unseen-cell-line benchmark, and nothing here changes the frozen Rhaister-O
results or the expansion gate on that benchmark. Prior local evidence: E2 (job 284894,
`report/10_02_report.md` §2.5): on Tahoe, Morgan kNN / kernel ridge **lost** to a MOA-class mean
(residual 0.096 / 0.093 / 0.117 vs 0.148).

### Data audit before design (measured 2026-10-03, login node metadata only)

- GSE92742 (Phase I) has 205,034 trt_cp Level-5 signatures; 24 h / 10 µM covers 8,715 pert_ids.
  Only **34-47% of signatures per core line have TAS ≥ 0.2** (median TAS 0.15-0.19): most
  compounds barely move the transcriptome. Any metric averaged over all compounds is mostly
  measuring noise prediction.
- **Identity trap:** 20,350 SMILES-bearing trt_cp pert_ids collapse to 9,174 InChIKey
  connectivity blocks (first 14 characters). The largest blocks are Broad DOS libraries with
  up to 32 stereoisomers of one scaffold (e.g. `XYLRKRISJMFHFX`). Default 2D Morgan
  fingerprints are identical for stereoisomers, so a split by pert_id would hand chemistry
  kNN near-duplicates. **Folds must be grouped by InChIKey connectivity (ik14); a scaffold
  split is the stricter secondary.**
- Target/MOA annotation (LINCS2020 `compoundinfo_beta.txt`): 2,302 of the 8,715 24 h/10 µM
  pert_ids (26%) have ≥1 target. Approach 2 can only be evaluated on this KNOWN subset; it
  must be compared with chemistry restricted to the same subset and the same neighbor pool.
- GSE70138 (Phase II) 24 h / 10 µM: 1,730 compounds, each in ~all of HT29, MCF7, PC3, A375,
  HA1E, YAPC, HELA. Denser cross-cell coverage, mostly annotated drugs.

### Literature plan

Four Haiku fetch agents (drug encoders in perturbation models; chemical vs transcriptional
similarity; KG/PPI resources and graph models; new drug-side modalities). Their reports are
leads only; numbers quoted in the synthesis are re-verified from source text by me.

### Pre-registration (written before any L1000 unseen-drug result exists)

**Slices.** P1-24 (primary): GSE92742, 24 h, 10 µM, lines PC3, MCF7, VCAP, A549, HA1E,
A375, HCC515. P1-6 (replication, time): GSE92742, 6 h, 10 µM, lines PC3, MCF7, VCAP, A549,
A375, HT29, HCC515, HA1E, HEPG2. P2-24 (replication, release): GSE70138, 24 h, 10 µM, lines
HT29, MCF7, PC3, A375, HA1E, YAPC, HELA. Non-cancer differentiated lines (NPC, NEU, ASC, SKB,
PHH) are excluded a priori.

**Unit.** A compound is the InChIKey of the RDKit-standardized parent (largest organic
fragment, uncharged; `scripts/build_drug_features.py`). Signatures of all pert_ids/batches
sharing that key in one line are averaged. Fold = hash(ik14, seed 0) mod 5, identical in every
line (a compound is held out in the same fold everywhere). Scaffold split: hash(generic
Bemis-Murcko scaffold) mod 5, acyclic compounds grouped by ik14.

**Cohorts.** ALL; **ACTIVE (primary)** = (compound, line) with mean TAS ≥ 0.2; KNOWN = ≥1
annotated target; XCELL = compound measured in ≥2 lines of the slice. Models always train on
all training compounds of the line; cohorts only select which test rows are scored.

**Arms** (per line, prediction of the 978-landmark signature):
- `mean`: training-compound mean in that line.
- `knn_<enc>` for enc ∈ {morgan2 (Tanimoto), maccs (Tanimoto), rdkit2d (cosine, z-scored),
  chemberta, molformer (cosine, centered)}: m + α Σ softmax_topk(s/τ)(y_j − m);
  k ∈ {1,3,10,30,100}, τ ∈ {0.01,0.03,0.1,∞}, α ∈ {0.25,0.5,1}, chosen by inner 4-fold
  group CV on training compounds, maximizing the primary metric on ACTIVE training rows.
- `krr_morgan`: Tanimoto kernel ridge on the residual from `mean`, λ by exact LOO.
- KNOWN-only arms, neighbor pool = KNOWN training compounds: `knn_chem_known` (morgan2),
  `knn_target` (Jaccard of target sets), `knn_target_perm` (target sets shuffled among KNOWN
  compounds — control), `moa_mean` (mean residual of training compounds sharing a MOA label),
  `knn_ppi` (cosine of random-walk-with-restart propagation of targets over STRING v12
  physical links, combined score ≥ 400, restart 0.5, per-gene z-scored across KNOWN
  compounds), `knn_ppi_rewired` (same on a degree-preserving rewired graph — control),
  `knn_chem_target` (mean of Tanimoto and Jaccard).
- Ceilings / new-modality probes, labelled as such and never as models: `oracle_knn`
  (neighbors chosen by the true response in the same line: the best any retrieval could do
  with this library), `xcell_knn` (neighbors chosen by response similarity of the new drug
  in the *other* lines of the slice — one cheap measured context), `xcell_copy` (the drug's
  mean residual in the other lines, scaled by α).

**Metrics.** Per (compound, line): Pearson of full signature (**primary**), Pearson of
residual from `mean`, top-100 |z| overlap. Per line across compounds: Spearman of predicted vs
true residual norm (magnitude; Pearson is scale-invariant, lesson from `10_02` §4) and AUROC of
predicted norm for ACTIVE. Summary = line-macro mean. Uncertainty = bootstrap over ik14 groups
(2,000), resampled jointly across lines.

**Gates** (minimum meaningful effect +0.01, project convention):
- G1 approach 1 viable: best chemistry kNN − `mean`, ACTIVE, primary metric: bootstrap
  low > 0, ≥ 5/7 lines positive, in P1-24 and at least one replication slice; also repeated
  on the scaffold split. Report the share of the `oracle_knn` headroom captured.
- G2 learned SMILES encoders beat fingerprints: `knn_chemberta` or `knn_molformer` −
  `knn_morgan2`, low > 0 in P1-24 and one replication.
- G3 knowledge beyond chemistry (approach-2 signal): on KNOWN∩ACTIVE, `knn_target` or
  `knn_ppi` − `knn_chem_known` low > 0; and `knn_ppi` − `knn_ppi_rewired` low > 0, and
  `knn_target` − `knn_target_perm` low > 0. Without the graph controls passing, a GNN on PPI
  is not justified (lesson from `kg_neighbor` vs `kg_random`, `10_02` §2.4).
- G4 new-modality value: `xcell_knn` − best chemistry kNN on XCELL∩ACTIVE, low > 0.

Expectations written before results: G1 passes weakly (structure carries some signal for
active compounds, far below the oracle); G2 fails or is marginal; G3 target passes, PPI vs
rewired is uncertain; G4 passes by a wide margin.

**Amendments before any result (2026-10-03 ~06:00):**
- `knn_ppi` similarity: propagated profiles are **mean-centered across KNOWN compounds, not
  per-gene z-scored**. Reason: with restart 0.5 almost all mass sits on seeds and first
  neighbours; per-gene z-scoring would give the ~19k far genes with ~1e-6 values the same
  weight as the target neighbourhood, so the cosine would be dominated by degree-structure
  noise. Centering removes the shared hub profile, which is what the z-score was meant to do.
- kNN on indicator similarities (`moa_knn`, `knn_target`) uses the same top-k/softmax
  machinery; with small τ only the matching compounds carry weight, with τ = ∞ the top-k is
  padded by ties at 0. Inner CV decides. `moa_mean` in the pre-registration is therefore
  implemented as `moa_knn`.
- Environment: added scipy 1.16.3, scikit-learn 1.7.2, networkx 3.5 to
  `envs/therapeutic-prep` (numpy 2.2.6 / pandas 2.3.3 pinned unchanged), needed for sparse
  propagation and AUROC. Existing scripts do not import them.
- `krr_ppi` (cosine kernel on the centered propagated profiles, KNOWN pool) added as the
  parametric counterpart of `knn_ppi`; the same inner group CV selects λ for every KRR arm
  (instead of hat-matrix LOO, which stereoisomer near-duplicates in a training set would bias).

**Smoke run and post-smoke additions (2026-10-03 ~06:15; disclosed, gates unchanged).**
Smoke job 292899 ran the runner end-to-end on P2-24 without the SMILES-LM arms and with 200
bootstraps (output `results/l1000_unseen_drug/_smoke/`, **not to be quoted**). After seeing it,
and before any official run, two arm families were added; they are labelled post-smoke in every
table and are not part of G1-G4:
- KRR on the KNOWN pool for every knowledge kernel (`krr_chem_known`, `krr_target`,
  `krr_ppi`, `krr_ppi_rewired`, `krr_chem_target`), because the smoke run showed `krr_ppi`
  ahead of `knn_chem_known` and that contrast mixed a model class with a feature. The fair
  comparison is KRR vs KRR on the same pool.
- `xcell1_knn` / `xcell1_copy`: the same cross-line probes, but the new compound is "measured"
  in **one** other line only (stable hash choice). The all-other-lines probe overstates what a
  single cheap measurement would give.
Two bugs found and fixed before the official runs: SMILES-LM `.npz` files stored `inchikey` as an
object array (unloadable with `allow_pickle=False`; rewritten as `<U27`, embeddings unchanged),
and the PrimeKG fetch script aborted on `kg.csv` (`?format=original` only applies to ingested
tabular files).

**Independent code review (Sonnet subagent, 2026-10-03 ~06:30) and resulting amendments — all
before any official run.** Review notes:
`scratchpad/review_unseen_drug.md` (session scratch, summarized here). No leakage, no critical
bug; kNN, KRR, rewiring (degrees preserved, 97% of edges moved) and RWR verified against
naive/closed-form implementations; 60 slice rows recomputed from raw signatures, 0 mismatches.
Changes made because of it:
1. **Grid truncation (major).** In the smoke run chemistry kNN chose k = 100 (grid max) in
   74-100% of folds, every knowledge arm chose α = 0.25 (grid min), oracle/xcell chose α = 1.0
   (grid max), τ = 0.1 (largest finite) almost always. Grids widened to k ≤ 1000,
   τ ∈ {0.01 … 1, ∞}, α ∈ {0.05 … 2}, copy-α ≤ 2. Multi-α Pearson is computed analytically
   (unit-tested equal to the direct computation to 1e-16).
2. **LM arms required** (`--allow-missing-lm` only for smoke tests); LM file hashes and the arm
   list are written to `manifest.json`.
3. `knn_chem_select`: per line-fold, the chemistry encoder with the best inner-CV score. This is
   the honest "best chemistry kNN" for G1 and the comparator for G4 (picking the best encoder on
   the outer test would be optimistic).
4. KRR selection now uses the full inner-training mean (as the prediction does); scaffold-split
   bootstrap resamples scaffold groups; per-replicate empty lines are dropped, not scored 0;
   magnitude and cohort summaries are restricted to each arm's valid pool; output directories
   carry the seed; non-finite predictions abort the run; oracle-headroom share is reported.
5. MOA labels are no longer split on commas (3 labels such as "Indoleamine 2,3-dioxygenase
   inhibitor" were broken); drug table and slices rebuilt (jobs 292925, 292926).
6. **New confound arm, motivated by the review's suspicion** that chemistry kNN might exploit
   plate artifacts (Broad DOS compounds were plated together): `knn_morgan2_xplate` and
   `oracle_xplate_knn` exclude neighbours that share any L1000 plate (sig_id prefix) with the
   query. This mirrors the project's shared-control coupling diagnostic (job 258115).
KNOWN in all tables means "≥1 annotated target that maps into STRING v12 physical (score ≥ 400)",
about 1% fewer compounds than "any target annotated".

Official runs queued 2026-10-03 06:35 behind the rebuilds: 292927 (P1-24, ik14), 292928 (P1-6),
292929 (P2-24), 292930 (P1-24 scaffold), 292931 (P2-24 scaffold); Q1 descriptive re-run 292932+.

**Addendum G5 — measured morphology as the "new modality" (added 2026-10-03 ~06:45, official
runs held with `scontrol hold` while it was added, so every official run contains it; no
official result existed).** Data: LINCS Cell Painting (Way et al., Cell Systems 2022;
`broadinstitute/lincs-cell-painting`, CC0), A549, 48 h, batch `2016_04_01_a549_48hr_batch1`,
spherized DMSO-normalized consensus MODZ profiles (10,752 consensus rows, 1,026 features,
1,571 Broad samples). Joined to L1000 through the 13-character Broad ID = L1000 pert_id: 1,570
match; **1,550 of the 1,725 P2-24 compounds** and ~756 P1 compounds have a profile.
Arms (pool = training compounds with a 10 µM profile, cohort MORPH): `knn_chem_morphpool`
(Morgan on the same pool — the fair comparator), `knn_morph` (cosine of z-scored 10 µM
profile), `knn_morph_doses` (6 doses concatenated), `knn_chem_morph` (mean of Tanimoto and
morphology cosine).
**G5 gate:** `knn_morph` − `knn_chem_morphpool` on MORPH∩ACTIVE, primary metric, bootstrap
low > 0 in P2-24 and in P1-24 (≥ +0.01 to count as meaningful). Also reported, not gated:
`xcell1_knn` − `knn_morph` (one L1000 profile in another line vs one Cell Painting profile).
Expectation before results: morphology beats chemistry, but by less than a cross-line L1000
measurement, since it is a different assay at a different time point (48 h vs 24 h).
The morphology is measured in A549; for the other lines it is a cross-cell, cross-assay
phenotype. Per-line results will separate A549 (P1-24 only) from the rest.

**Addendum G3b — a trained GNN for approach 2 (written 2026-10-03 ~06:55, before any GNN or
official result; implementation delegated to a Sonnet subagent with a fixed spec;
`scripts/run_l1000_gnn.py`).** Same folds, cohorts, metrics and row schema as the kNN runner, so
rows pair one-to-one. Drug enters a STRING-physical graph (score ≥ 400) at its target nodes
(target indicator + learned vector + MLP of Morgan bits added to a learnable gene embedding);
2 GCN layers; per-line readout over [landmark node state ; target-node pooled state + chemistry].
Hyperparameters fixed a priori (Adam 1e-3, wd 1e-5, batch 64, ≤150 epochs, early stopping
patience 15 on one inner ik14 fold, criterion = full Pearson on ACTIVE validation rows). Arms:
`gnn_ppi`, `gnn_rewired` (same seed-0 degree-preserving graph as the kNN control),
`gnn_nograph` (identity adjacency), `mlp_target` (targets + chemistry, no graph), `mlp_chem`
(Morgan only, KNOWN pool), `mlp_chem_all` (Morgan only, all compounds).
**G3b gate:** on KNOWN∩ACTIVE in P2-24 and P1-24, `gnn_ppi` − `gnn_rewired` low > 0 **and**
`gnn_ppi` − `gnn_nograph` low > 0 **and** `gnn_ppi` − max-comparator low > 0, where the
comparator is `krr_chem_known` (the strongest pre-declared non-graph arm on the same pool).
Expectation: fails — the smoke run's `knn_ppi` vs `knn_ppi_rewired` difference was +0.001 and
this project has seen every added-parameter model fail its random-graph control (`10_02` §2.4).
The bit-for-bit rule applies: the GNN script's `mean` arm must equal the runner's to < 1e-9.

## 2026-10-03 晚 — Official L1000 unseen-drug results: first reading, and a plate confound

Jobs 292927 (P1-24), 292928 (P1-6), 292929 (P2-24), 292930/292931 (scaffold) all finished
without error (logs checked; `sacct` still unavailable). Numbers are in `10_09_report.md` §1.
Gate reading against the pre-registration (primary metric, ACTIVE cohort):

| gate | verdict | note |
|---|---|---|
| G1 chemistry kNN > mean | **pass** in all slices and both splits (+0.028 … +0.054, all lines) | but see the plate confound below |
| G2 SMILES LMs > Morgan | **fail**; ChemBERTa significantly worse in all 5 runs, MoLFormer ties | learned SMILES embeddings do not help retrieval |
| G3 knowledge > chemistry | `knn_ppi` − `knn_chem_known` +0.013/+0.017/+0.013 (pass); `knn_ppi` − rewired +0.002/+0.003/+0.003 (low > 0 but 3-5× below the +0.01 threshold) | target knowledge helps; PPI *topology* barely matters |
| G4 one measured context | **pass**, `xcell1_knn` − `knn_chem_select` +0.105/+0.098/+0.149 | the largest effect in the study |
| G5 morphology > chemistry | **pass**, +0.028 (P1-24), +0.095 (P2-24) | P2 size needs the confound check below |

**Plate confound (found by the pre-declared `*_xplate` arm).** Excluding neighbours that share
an L1000 plate with the query costs Morgan kNN +0.0197 (P1-24), +0.0239 (P1-6), +0.0050 (P2-24):
**about 40% of chemistry retrieval's gain in Phase I comes from same-plate neighbours.** The
oracle also loses 0.023-0.036. Two readings: (a) a batch artifact shared by co-plated wells
survives Level-5 plate normalization; (b) co-plated compounds are genuinely related (DOS
libraries were plated by scaffold family). At deployment a new compound is measured on its own
plate, so (a) can never be exploited and the same-plate-excluded number is the faithful one.
The P2-24 morphology gain could share a version of (a): Phase-II L1000 (`REP.*` plates) and the
Cell Painting batch were both run from Drug Repurposing Hub plate maps.

**Post-hoc analyses designed after seeing these results (exploratory, labelled as such):**
- X1 matched-similarity test: response r of compound pairs by Morgan-Tanimoto bin × same/
  different L1000 plate. At equal chemistry, an excess for same-plate pairs is the batch
  artifact. Also checks whether same-L1000-plate and same-Cell-Painting-plate-map pairs
  coincide.
- X2 re-run all three slices with `--exclude-same-plate`: every retrieval arm (chemistry,
  knowledge, morphology, oracle, xcell) loses same-plate neighbours; KRR zeroes the test
  kernel entries of same-plate training compounds (approximation, labelled). This is the
  deployment-faithful table.

## 2026-10-03 21:10 — Plate-controlled protocol, Tahoe replication, PrimeKG audit (designed after §1.5)

**X3 plate-map grouped split (`--split platemap`, all models).** Neighbour exclusion (X2) only
protects retrieval arms; a parametric model can still absorb plate structure. Union-find shows
that compounds measured on several plate maps connect every map into one component, so a pure
plate split is impossible. Rule implemented in `run.make_folds`/`plate_map_folds`: a compound is
held out in fold f only if every plate map (sig_id prefix, e.g. `CPC006`, `REP.A001`) it was
measured on hashes to f; it trains in fold f only if none of its maps hashes to f and no member of
its ik14 family is held out. Inner folds use the same rule inside the training set; the bootstrap
resamples plate maps. Login-node check: 0 leakage violations; test-eligible 6,045/8,119 (P1-24),
1,635/1,725 (P2-24). The ik14 masks were asserted identical to the old folds, and job 294087
re-runs P2-24 ik14 after the refactor for a bit-for-bit comparison with 292929.

**Tahoe replication (pre-declared before running).** `build_tahoe_unseen_drug.py`: one slice per
dose (T-0.05, T-0.5, T-5), 350 compounds × 50 lines, rows = frozen compact Rhaister deltas (2K
genes). ACTIVE := delta norm ≥ the line's median at that dose (no TAS on Tahoe; half the rows by
construction). Same runner with `--source Tahoe` (Tahoe targets/MOA from `drug_metadata`), ik14
and platemap splits (plates = Tahoe plates; same plate ⇒ shared DMSO control). Gates replicated:
G1 (chemistry − mean), G2 (LM − Morgan), G3 (`knn_ppi`/`knn_target` − `knn_chem_known`; ppi −
rewired), G4 (`xcell1_knn` − `knn_chem_select`). On Tahoe "another line" is one of 49 lines in the
same screen, so G4 there is the drug-measured-elsewhere (pair-holdout) regime, not a new assay.

**PrimeKG audit (done, `scripts/audit_primekg_overlap.py`, results/primekg_overlap/).** DrugBank's
vocabulary download returns HTTP 403 without an account, so PrimeKG DrugBank IDs were mapped to
InChIKeys through PubChem PUG-REST (5,500 of 6,282 drugs with protein edges). Numbers in
`10_09_report.md` §2.

**X4 (exploratory, post hoc) applicability domain.** `scripts/analyze_l1000_applicability.py`:
per scored row of the official ik14 runs, max Morgan Tanimoto to that line's training compounds
in the same fold; chemistry-kNN gain over `mean` by bin. Result in `10_09_report.md` §1.6:
the gain is ~0 below Tanimoto 0.3 and only becomes material above 0.6, while the oracle headroom
is flat (0.33-0.43) across bins — the response twins exist whatever the chemical neighbourhood.

**X5 (exploratory) — do PrimeKG/DrugBank targets improve the knowledge arms?** `--targets union`
adds PrimeKG `target` edges (DrugBank, mapped via PubChem; `derived/.../primekg_targets.tsv`,
4,965 InChIKeys) to the LINCS2020 targets, matching by full InChIKey then connectivity block.
Comparison is on rows KNOWN under the original annotation (paired with the 292929 output), plus
the enlarged KNOWN cohort descriptively. Job 294120 (P2-24, ik14).
- 294087 refactor verification: P2-24 ik14 rerun identical to 292929 on all 361,956 rows (max abs diff 0.0 for every metric).

**G3b deviation (2026-10-03 21:45, before any GNN result beyond a 3-epoch smoke).** The GNN subagent
stopped on a rate limit before writing code; I implemented `scripts/run_l1000_gnn.py` myself.
Smoke job 294094 (P2-24, fold 0, 3 epochs) ran end-to-end, but the specified per-drug 2-layer GCN
on 15,870 nodes took ~55 s/epoch on CPU (≈ 15-36 h for the full design). Replaced by a
**decoupled-propagation GNN** (SGC/SIGN family): the drug's target signal propagated 0-3 hops over
the normalized adjacency, read at each landmark (log-scaled, precomputed per graph), plus learnable
gene embeddings propagated A·E and A²·E each step; drug readout = target-node pooling of the
propagated embeddings + chemistry MLP; per-line head over [gene state; propagated drug signal;
drug readout]. Rewired and identity adjacencies are the controls, unchanged. Gate G3b unchanged.
Also: KRR arms under X2 masking (294074) are **not interpretable** — α is fitted with plate-mates
present and the masked cross-kernel breaks it (`krr_morgan2` − `mean` falls from +0.033 to
+0.001); KRR is judged under the platemap split (X3) only.

## 2026-10-04 00:15 — GNN gate G3b inconclusive under the pre-specified training; exploratory fix

Jobs 294193 (P2-24 ik14) and 294194 (P2-24 platemap) finished without error; the GNN script's
`mean` arm reproduces the kNN runner's bit for bit (max |Δ| = 0.0 on 11,676 / 11,059 rows). But
**every neural arm (three GNN variants and three MLPs) reached its best inner-validation score after
the first epoch and early-stopped at epoch 16** in every fold, so the kept weights are effectively
initial: full Pearson on KNOWN∩ACTIVE equals the mean's (P2-24 ik14: all GNN arms 0.3317 = mean
0.3317; MLPs 0.331-0.334), against 0.36-0.38 for kNN/KRR on the same rows. G3b therefore cannot
discriminate graph from no graph: **inconclusive, not negative.**
Diagnosis: MSE training produces residual predictions whose scale hurts the full-signature metric;
every kNN/KRR arm has an inner-CV shrinkage (α or λ) and the neural arms had none.
Exploratory fix (post hoc, labelled): `--shrink` selects an output scale α ∈ {0, 0.05 … 2} on the
inner validation split at every epoch (early stopping on the shrunk score) and applies it at test.
Jobs 294352 (P2-24 ik14, lr 1e-3), 294353 (P2-24 ik14, lr 3e-4), 294354 (P2-24 platemap). Job 294195 (P1-24,
old protocol) continues as a replication of the failure.
- 294195 (GNN P1-24, no-shrink protocol) cancelled 2026-10-04 00:25 after 44 min: would only replicate the epoch-1 failure; P1-24 will be rerun with --shrink if P2-24 shows learning.

**2026-10-04 00:50 — shrinkage run 294352 (P2-24 ik14).** With inner-validation output shrinkage the
MLPs learn (KNOWN∩ACTIVE full Pearson: mlp_chem 0.356, mlp_target 0.361, vs mean 0.330), but the
decoupled GNN still does not (gnn_ppi / rewired / nograph 0.333; early stop ~25 epochs, validation
≈ mean + 0.0005). The failure is the GNN's head (gene-specific output only through a small shared
MLP over [gene state; propagated signal; drug vector]), not the graph. Exploratory arms added so
the graph is tested inside an architecture that does learn: `mlp_ppi` = `mlp_target` plus the
PPI-propagated target signal at all 978 landmarks (hops 0-3, 3,912 features); controls
`mlp_ppi_rewired` (degree-preserving rewired graph) and `mlp_ppi_nograph` (identity, hop-0 only).
Job 294396.

## 2026-10-04 01:30 — Verdicts after the plate-map split and the learned-model runs

| gate | final reading (plate-map split unless noted) |
|---|---|
| G1 chemistry retrieval > mean | passes, but small: +0.023 / +0.029 / +0.028 (P1-24 / P1-6 / P2-24), 6-9% of the oracle headroom; **0 on Tahoe** after same-plate exclusion |
| G2 SMILES LMs > Morgan | **fails**: ChemBERTa −0.006 in all three slices (significantly worse), MoLFormer −0.003 to −0.004 |
| G3 knowledge > chemistry | target knowledge passes: chemistry + targets − chemistry +0.015 to +0.018 (KRR). PPI topology fails on size: real − rewired +0.0015-0.0016 (kNN), +0.0003-0.0006 (KRR) |
| G3b trained GNN | pre-specified training **inconclusive** (no learning); the decoupled GNN never learned. Exploratory MLP + PPI features: graph features +0.012 over none, **real − rewired −0.001 (ns)** |
| G4 one measured context | passes: +0.076 / +0.076 / +0.130 over best chemistry; Tahoe +0.05 / +0.18 |
| G5 Cell Painting > chemistry | passes: +0.033 / +0.018 / +0.055; the ik14 P2-24 +0.095 was ~42% shared plate layout |

Interpretation recorded for the next discussion (not a new result):
- The PPI graph's measurable value is node degree; the wiring adds ≤ 0.003 in every model family
  tried (kNN, KRR, MLP). A GNN on STRING is not justified for this endpoint unless a future design
  beats its rewired control.
- Learned models trail kernels on identical inputs at this sample size; future approach-2 work
  should use kernels on graph-derived features, or a much larger training set (LINCS 2020: ~33k
  compounds) before revisiting deep models.
- The decisive lever is a measured phenotype of the new drug used as the retrieval key. The next
  experiment worth pre-registering is a reference-panel design: which 1-3 reference lines (or one
  Cell Painting run) maximise downstream retrieval, evaluated under the plate-map split.

## 2026-10-04 — 用户问题:PPI 只提供"度"吗?是模型没学到,还是本来就学不到?(exploratory, before any result)

User question (2026-10-04): if every model gains no more from STRING than from a degree-preserving
rewired graph, is that a modelling failure or is there no wiring information for this endpoint?
Two further questions: how other models use PPI (gap analysis), and whether a larger output head or
another fusion would let the GNN learn.

Code reading behind the second question: `DecoupledGNN` predicts gene g of drug d as a shared
2-layer MLP over the concatenation [gene embedding; propagated signal at g; drug vector]; drug-gene
interaction exists only through ReLU. `FlatMLP` (which learns) ends in `Linear(256, lines x genes)`,
i.e. a per-gene dot product with the drug embedding. Concatenation-MLPs are known to learn dot
products poorly (Rendle et al., RecSys 2020). The GNN also learns 15,870 x 64 free node embeddings
from ~700 training compounds.

**Model-free test, job 294863** (`scripts/analyze_ppi_information.py`, slices P2-24 and P1-24;
designed before results; descriptive, no train/test split). KNOWN compounds, per line, on STRING v12
physical (score >= 400, the runner's graph) and functional (score >= 700), each vs its seed-0
degree-preserving rewiring:
- A locality: within each landmark gene, standardized |residual| when the drug's nearest target is
  1 hop away minus when it is >= 3 hops away; `A_rel` after dividing each signature by its own mean
  |residual| (removes drug magnitude, which hub-targeting drugs could confound). Null: target sets
  permuted among compounds (20 permutations).
- B pairs: residual-signature Pearson of drug pairs by target-set distance (0 shared, 1, 2, >= 3),
  different-plate pairs only (plate confound, §1.5).
- C degree: Spearman of mean log(1 + target degree) with residual norm (all KNOWN rows).
Reading rule fixed now: wiring information exists if real − rewired is positive in most lines for
A_rel or for B at distance 1-2; "degree only" if C is clearly non-zero while real ≈ rewired.

**294863 result and reading (2026-10-04 06:35).** Numbers in `10_09_report.md` §4.6. Against the
reading rule: wiring information exists (B 1-hop real − rewired > 0 in 7/7 lines in three of four
slice-graph combinations; A_rel in 3 of 4), and degree is also clearly informative (C ρ ≈ 0.2,
28/28). So "PPI = degree only" is too strong. Better statement: the usable wiring signal is
one-hop, small (about a tenth of sharing a target), and touches ~2-3% of landmarks per drug.
Our feature constructions diluted it: RWR and 0-3-hop propagation spread mass to 2-3 hops, where
real ≈ rewired, and hub-dominated profiles. The functional network (≥ 700) carries more gene-level
locality than the physical one. Caveats: descriptive, one rewiring seed, unweighted hops, no
train/test separation; not a gate.

## 2026-10-04 — DeepCE 复现:它的 PPI 基因特征到底贡献多少?(user request)

用户决定(2026-10-04):(1) 复现 DeepCE(Pham et al., *Nat Mach Intell* 2021),量化其 STRING 基因特征的
贡献;代码放 `repro/deepce/`(已加入 `.gitignore`,上游无 license,只做本地研究复现)。(2) 我们 GNN 的
点积输出头"有必要",但等 DeepCE 完成后再做。(3) 现阶段可用性验证不必过严:例如"图的价值主要是度"
在别的模型未提及前先作为一个保留的可能性,不再追加验证。

Setup: upstream `pth1993/DeepCE` @ 9b0d04f, branch `repro` with one compatibility patch
(`collections.abc.Iterable`); data (74 MB) moved to `/nfs-stor/yuqi.lei/Therapeutic/repro/deepce/data_upstream`
with SHA-256 list beside it; env `/nfs-stor/yuqi.lei/Therapeutic/envs/deepce-gpu` (torch 2.4.1+cu118,
rdkit 2024.03.6, numpy 1.26.4; job 294903). GPU QOS was full (promoterVI jobs), so training runs on CPU,
four gene variants of one seed in parallel per job (`repro/deepce/slurm/train_cpu.sbatch`).

What the code shows: the 978 x 128 `gene_vector.csv` goes through a linear layer and becomes the
query of a 2-layer drug-gene attention over the drug's atom embeddings (neural fingerprint GCN);
per gene, [attended vector; summed drug embedding; 4-d cell; 4-d dose] -> ReLU -> 128 -> 1. The
drug-gene interaction is a dot product (attention), which our DecoupledGNN lacked. The paper
ablates the attention blocks (0.4907 -> 0.4418 without attention) but never the gene features;
they are described as STRING embeddings (~19k nodes, ~12M edges; method in the supplement).
Split: train/dev/test are pert_id-disjoint (343/92/92 drugs; after the upstream filter 1,895/544/496
(drug, line, dose) conditions, one replicate chosen per condition by median rank). sig_ids carry
plate maps (`REP.A002_...`), so the plate confound of `10_09_report.md` §1.5 may apply; not checked.

Plan (exploratory): `run_deepce.py` mirrors upstream training; gene variants `ppi` (upstream),
`shuffled` (same vectors permuted across genes), `random` (Gaussian, matched per-dimension
mean/sd), `onehot` (978-d identity); seeds 0-2; 100 epochs; quote test Pearson at the best-dev
epoch (upstream also prints the best test epoch, which is optimistic). Expectation: all four within
seed noise, i.e. the gene vectors work as gene identity codes; PPI content counts only if `ppi`
beats `shuffled` and `random` by more than the seed spread. Baselines DeepCE did not report
(`repro/deepce/baselines.py`, login node, seconds) are in `10_09_report.md` §5.

**DeepCE ablation final (2026-10-04 20:40; `10_09_report.md` §5.3) and correction of the interim
§5.2.** Seeds 0/1 ran their last 6-7 epochs after the interim; `ppi` seed 0 moved its best-dev epoch
from 40 to 58 and its test Pearson from 0.464 to 0.481. Corrected contrasts: ppi − onehot −0.011
(was −0.017; still significant), ppi − random −0.004 (was −0.010), ppi − shuffled +0.012 (was
+0.008; now interval > 0). Reading: DeepCE's PPI-derived gene features do not contribute beyond a
gene identity code; a free one-hot gene embedding is best. That fits the PPI information test (§4.6)
and our own G3 results: with a per-gene output, a gene-side prior is replaceable by learned
identity once ~2k training signatures exist. The paper's 0.4907 is not reproduced with honest
selection; unknown epoch count (supplement) and seed variance are possible reasons. Not pursued
further (exploratory rigor). Next per user ordering: the dot-product output head for our GNN.

**Dot-product head for the GNN (2026-10-04 20:50; user ordering step 2; exploratory, before results).**
`DotGNN` in `scripts/run_l1000_gnn.py` keeps the DecoupledGNN encoder unchanged (free node
embeddings, 0-2-hop propagation, target-node pooling + chemistry -> drug vector d; landmark gene
vector g) and replaces the concatenation-MLP head with out[l, g] = d' W_l g + sum_k beta[l, g, k]
prop[g, k] + bias[l, g] (per-line bilinear dot product plus a per-gene coefficient on the 0-3-hop
propagated target signal). Arms `dot_ppi` / `dot_rewired` / `dot_nograph`, trained with `--shrink`
(lr 1e-3) next to `mlp_target`, `mlp_ppi`, `mlp_ppi_rewired` in the same job so the contrasts in
`scripts/compare_l1000_gnn.py` pair within one file. P2-24, ik14 and plate-map splits.
Questions: does the head make the GNN learn (dot_ppi − mean, vs gnn_ppi ≈ +0.001)? does it reach
the MLP/KRR level? and, kept as an open possibility only, does real − rewired move from ≈ 0?

**Dot-head result (2026-10-04 21:15; `10_09_report.md` §4.7).** The concatenation head was a real
bottleneck: with a dot-product head the no-graph GNN learns (+0.014 to +0.022 over the mean). But
(i) optimisation is fragile: in most graph-arm folds the model never beat the mean on inner
validation within 16-19 epochs, so shrinkage chose α = 0; (ii) when it does learn, graph
propagation of the free node embeddings hurts (−0.012 to −0.015 vs no graph), plausibly
over-smoothing the landmark gene vectors (A·E, A²·E) that the bilinear head needs to keep
distinct; (iii) all versions stay below the flat MLP and KRR on identical inputs. Real − rewired
remains ~+0.001 (kept as the open "degree only" possibility, not pursued). Candidate next steps,
not started: optimisation fixes for the dot head (lower lr / warm-up / longer patience, larger W
init, or gene vectors without propagation); kernel fusion (MKL) as the stronger fusion route at
this sample size.

## 2026-10-05 — 决定:冲 RECOMB 2027(PPI 先验的基准与批判)

User decision (2026-10-05): target RECOMB 2027 (abstract 10-28, full paper 11-04 AoE). Working
thesis: the measurable benefit of PPI/KG priors in perturbation-response models is mostly node
degree. Experiment list E0-E9 and schedule in `PLAN.md` Phase 6. Novelty check (≈ 10 searches,
`papers/README.md`): precedents exist outside perturbation prediction (Gillis & Pavlidis: degree in
gene-function prediction; Bertin 2019: random gene graphs; Caranzano 2025: pathway-informed nets ≈
sparsity-matched random); none found that audits published perturbation-response models with
degree-preserving controls; GSNN's +24% used a non-degree-preserving random graph, so it is the
decisive audit. RECOMB acceptance was ~16% in 2024 and 2025 (search summary). Jun Wen still wants an
integrated L1000 model; design sketched in `PLAN.md` Phase 7, scheduled after the RECOMB push
unless re-prioritised.

**User decisions, 2026-10-05 (second round).** (1) Audit list: DeepCE, GSNN, GEARS, our four model
families, and E6 is required too. (2) Jun Wen's integrated model runs in parallel with the paper
work: the paper should critique PPI and also offer a new direction. (3) GPU-heavy work: prepare the
code first and submit when the GPU QOS allows. (4) Authors: Yuqi Lei (first author), Hanzhe Liang,
Jun Wen. Week-1 work started: E0 control-ladder toolkit, E1 network downloads, code checks, MKL
version of the integrated model.

**2026-10-05 prep notes (data and code for E4-E6; no results yet).**
- GSNN's published random-graph control (`gsnn.models.utils.randomize`, `--randomize graph`) redraws all
  three edge sets at once: drug targets (input -> function) become random proteins, the function ->
  function graph becomes uniform random (degrees not kept), and every output reads from a random
  function node. Its +24% therefore mixes target identity, output mapping and wiring.
  `repro/gsnn/ladder.py` perturbs one edge set at a time (ff_rewired / ff_labels / ff_er / ff_empty,
  in_shuffle, out_shuffle) and also reproduces their all-three control (gsnn_random).
- PDGrapher loads a per-line PPI `edge_index_<LINE>.pt`; `repro/pdgrapher/make_null_edges.py` writes
  ladder versions next to it. Their chemical runs used up to 90 GPU-hours per line, so the audit will
  use 1-2 lines and fewer folds/epochs (deviation to be stated).
- GEARS: `repro/gears/run_gears_ladder.py` replaces the GO and/or co-expression graph after GEARS builds
  it (directed degree-preserving swaps carry each edge weight with its source slot).
- Compute-node egress: cn-02 and cn-20 reset TLS connections to S3 / Harvard Dataverse; cn-03 works.
  Downloads run with `--exclude=cn-02,cn-20`; two small Dataverse files (gene2go_all.pkl,
  essential_all_data_pert_genes.pkl) were fetched from the login node.
- Env `gnn-gpu` (python 3.11, torch 2.4.1+cu118, PyG 2.5.3, lightning 2.4, cell-gears 0.1.2) built.

**E1 reading (2026-10-05 09:05; `10_09_report.md` §6).** Across P1-24, P2-24 and Tahoe T-5 the pattern of
§4.6 holds and sharpens: (1) undirected PPI wiring adds little gene-level locality beyond degree,
while directed signaling (OmniPath) and TF -> target (CollecTRI) networks add a consistent +0.03 to
+0.07 (6/7-7/7 lines); (2) target degree predicts response magnitude only in curated networks, not
in HuRI, and not because of publication counts alone; (3) adjacency of two drugs' targets adds at
most +0.014 pair similarity over rewired graphs. Candidate "new direction" for the paper and for Jun
Wen's integrated model: use directed signaling -> TF -> target-gene structure (with its own
in/out-degree-preserving null) instead of undirected PPI. Next: add a CollecTRI/OmniPath view to the
MKL model and test it against its null; keep the claim at the level of these descriptive tests until
then.
- (11:25) PDGrapher data extracted (51 GB; job 296946). All ten per-line PPI files are byte-identical
  (one PPI, 10,716 nodes, 303,678 directed entries incl. one self-loop per node). The first null files
  dropped the self-loops; regenerated with self-loops kept in every null (`make_null_edges.py`, nulls
  computed once and copied). `repro/pdgrapher/run_pdgrapher_ladder.py` mirrors upstream chemical.py
  (backward data, supervision + cycle losses, multiplier 0.01) on selected folds and one GNN depth.
- GSNN graph after exp1 processing: 8,665 inputs, 8,894 function nodes (5,428 protein, 3,466 RNA),
  77,005 function edges, 527 outputs; 51,567 observations stored as one file each (gsnn-lib design;
  ~51k small files on /l). GSNN's cell-drug split drops test-only drugs: GSNN cannot do unseen drugs.
- GPU: no audit job could be submitted between 08:15 and 11:22 because the account's GPU submit quota
  (4 jobs) was held by promoterVI; `repro/gpu_queue.sh` submits the queued smokes as slots free.
- (16:00) GPU smokes: GEARS 297549 (Norman, GO graph real, 1 epoch, gpu-54, 392 s) ran end to end
  (not to be quoted); `run_gears_ladder.py` now also stores GEARS' own subgroup analysis (seen/unseen
  combos and singles, pearson_delta and non-dropout metrics). GSNN 297698 failed on a missing output
  parent directory (`os.mkdir`); sbatch now creates it, resubmission queued. PDGrapher 297711 queued.
  Queued next: GEARS Norman GO ladder (real x 3 model seeds; rewired, labels x 3 null seeds; er; empty;
  20 epochs).
- (16:15) Smoke failures and fixes: (a) gpu-51 exposes GPUs to nvidia-smi but torch sees no CUDA (GSNN
  and PDGrapher smokes landed there); all GPU sbatch files now exclude gpu-05,gpu-51 and abort when
  `torch.cuda.is_available()` is false; pending job 297743 updated with ExcNodeList. (b) GSNN HEAD
  (2026-05 refactor) removed `utils.get_activation` that gsnn-lib (last push 2025-02-07) calls; GSNN is
  pinned to 3bedcd0 (2025-01-31) on branch `repro-compat`. Note: GSNN commit dfc33dd (2024-12-20)
  "fixed graph randomization bug" post-dates the preprint, so the preprint's +24% random-graph result
  may have used the earlier randomisation. (c) PDGrapher's vendored PyG internals do not import under
  PyG 2.5; a separate env `pdgrapher-gpu` (python 3.9, torch 1.10.1+cu111, PyG 2.0.4, lightning 1.9.5)
  is being built as its README specifies.
- (20:45) GSNN smoke 298920 (gpu-03, 2 epochs, real graph) ran end to end on CUDA: ~77 s/epoch, test
  r_flat 0.154 after 2 epochs (not to be quoted). Queued the GSNN ladder (fold 0, default
  hyper-parameters, 100 epochs with patience 10): none, ff_rewired0-2, ff_labels0-2, ff_er0, ff_empty,
  in_shuffle0, out_shuffle0, gsnn_random0 (their all-three control), in three GPU jobs.
  PDGrapher smoke 299174 running on gpu-54 in the new env.
- (21:45) PDGrapher smoke 299174 (gpu-54, A549 fold 1, real PPI, 2 epochs) ran end to end in the legacy
  env (~785 s/epoch, 415 s of it re-evaluating the training set each epoch; test backward_avg_topk 664
  after 2 epochs, not to be quoted). Per-epoch train/test logging switched off in the runner (final
  evaluation unchanged: best-validation models restored, train and test evaluated as upstream), ~4 min
  per epoch. Queued: A549 fold 1, 50 epochs, real / rewired0-2 / labels0 / er0 / empty0 in two GPU jobs,
  after the GSNN ladder. Primary PDGrapher metric for the ladder: test backward_avg_topk (mean rank of
  the true targets among 10,716 genes; lower is better), with forward/backward R2 and Spearman.
- DeepCE E3 CPU seeds at epoch 41-47 of 60 (~1,000 s/epoch: three jobs share cn-03); ~4 h left.
- (2026-10-06 02:30) E3 done (`10_09_report.md` §5.4). Reading: in DeepCE the STRING prior on the gene
  side is replaceable by gene identity; node2vec on the real graph is not distinguishable from the
  degree-preserving or label-permuted nulls; the one significant contrast (real > uniform random graph,
  +0.023) mirrors §5.3's "misassigned structure hurts" (shuffled < random), not a wiring benefit.
- (2026-10-06 03:00) MKL gets a `reg` view (OmniPath + CollecTRI downstream influence, 1-2 hops, cosine
  between drugs) and its in/out-degree-preserving null; tiers T2reg / T3reg (+ nulls). Smoke (MCF7, one
  plate-map fold; not to be quoted): krr_reg 0.3455 vs null 0.3452 on KNOWN rows; adding it to T2/T3
  changes nothing. Interpretation: E1's directed-network signal is gene-level (genes downstream of the
  targets move more), which a drug-drug similarity kernel cannot use. Proposed next test (not started):
  a gene-level term on top of the MKL prediction, Δ[d, l, g] += β_l · s_d · x_dg, where x_dg is the
  signed downstream influence of drug d's targets on gene g through CollecTRI/OmniPath signs and s_d = −1
  for inhibitors / +1 for activators (from MOA text); one coefficient per line (and per gene as a
  variant), fitted on training drugs, against the same term on the rewired network. Full MKL runs with
  the reg view: 302096 (platemap), 302097 (ik14).
- (2026-10-06 03:30) MKL step 1 written up (`10_09_report.md` §7). Reading for the integrated-model
  direction: information value is tiered (chemistry +0.03 < knowledge +0.013-0.017, of which target
  identity carries the signal and one-hop STRING equals its rewired null < Cell Painting +0.05-0.09 <
  one measured response in another line +0.13). For Jun Wen's model this argues for a tiered design
  whose cheapest strong input is a measured phenotype, with PPI only as a degree-level feature.

## 2026-10-07 13:40 — TxGNN 作为 drug-side graph baseline:计划、预注册 gate、当前阻塞 (user request)

**Question (user, 2026-10-07).** Benchmark TxGNN (Huang et al., *Nat Med* 2024; mims-harvard/TxGNN,
pinned commit `0e0203c3939f7d12da27aaf10e55daaff5424023`, 2024-05-06, tarball sha256
`aa55af8069ba5dfec8d06434d42683c702d54aa049725daa88d103b5a5b00fd8`, under gitignored `repro/txgnn_src/`).

**Fit to our tasks.** TxGNN predicts drug–disease indication/contraindication links on PrimeKG; it has
no expression head. The only fair use is as a **drug representation** in the L1000 unseen-drug
retrieval arms (secondary task), alongside chem kNN / ChemBERTa / MolFormer. It is not a candidate for
the primary unseen-cell-line benchmark.

**Data provenance.** TxGNN downloads its own processed PrimeKG from Dataverse `doi:10.7910/DVN/CNQV69`
(files 7144484 kg.csv, 7144482 nodes, 7144483 edges). This is **not** the PrimeKG Dataverse v2.1
release (IXA7BM, files 61806xx) already under `raw/primekg_v2.1`; do not substitute one for the other.

**Code finding that changes the plan.** `TxGNN.load_pretrained()` calls `model_initialize()`, which
calls `initialize_node_embedding()`: every node's input feature `inp` is a fresh Xavier draw with
`requires_grad=False`, stored on the DGL graph, not in `model.pt`, and no seed is set. The released
Google Drive checkpoint therefore restores the trained layer weights on top of *new* random inputs,
so node embeddings read from it are not those of the trained model. Unless the checkpoint folder
turns out to contain the inputs (check on download), embeddings require **retraining with a fixed
seed and saving `inp`** (pretrain all edge types + drug–disease finetune; GPU job). For drug
embeddings no disease masking is needed: use `split='full_graph'`.

**Pre-registered gate (written before any TxGNN result).**
- Cohort `TX_ACTIVE`: ACTIVE rows whose compound maps to a TxGNN drug node (DrugBank → InChIKey via
  `derived/l1000_unseen_drug/primekg_drugbank_pubchem_inchikey.tsv`, full key then ik14 block).
- Arms (same kNN implementation and inner CV as every other arm; neighbour pool = covered training
  compounds): `knn_txgnn` (cosine on centred embeddings), `knn_chem_txpool` (Morgan2 on the same
  pool), `knn_txgnn_perm` (embeddings permuted among covered compounds), `knn_chem_tx`
  (0.5 Morgan2 + 0.5 TxGNN).
- **Primary contrast:** `knn_txgnn − knn_chem_txpool` on `TX_ACTIVE`, pearson_full, line-macro,
  ik14-group bootstrap (2,000). Pass = lower 95% bound > 0 on **both** P1-24 and P2-24, and
  `knn_txgnn − knn_txgnn_perm` lower bound > 0. Secondary: `knn_chem_tx − knn_chem_txpool`; and
  vs `knn_target` on rows that are both KNOWN and covered. Report the coverage-diluted gain over all
  ACTIVE rows (expected ~20% coverage on P1-24, per §2 audit).
- Prior expectation: small. PrimeKG targets already gave no gain on KNOWN rows (§4.2, −0.0013), and
  TxGNN's objective is therapeutic use, not mechanism.

**Blockers at 13:40 (Dubai).** (1) `sbatch` fails on both ciai-login-1 and ciai-login-2:
`spank_pyxis.so: cannot open shared object file` (plugstack `pyxis.conf`) — cluster-side, report to
HPC. (2) ciai-login-1 lacks git/curl, its CA store rejects GitHub, and the system python3.12 has no
`zoneinfo`/`http.client`; ciai-login-2 networking works. (3) Env build and data/ckpt download are
staged as `slurm/build_txgnn.sbatch` (+ `repro/envs/build_txgnn.sh`, python 3.8, torch 1.7.1+cpu,
DGL 0.5.2) and wait for Slurm; not run on a login node.
- (2026-10-07 17:00) TxGNN pipeline status. Envs `txgnn-cpu` (torch 1.7.1+cpu, dgl 0.5.2) and
  `txgnn-gpu` (torch 1.7.1+cu110, dgl-cu110 0.5.3 + nvidia-*-cu11 11.8 runtime/cuBLAS/cuSPARSE via
  LD_LIBRARY_PATH) built (jobs 306604, 306647). Data fetch failed: Dataverse S3 TLS handshake reset on
  cn-14 and on cn-10 (6/6 attempts, job 306657); Google Drive checkpoint "cannot retrieve public link".
  ciai-login-2 reaches Dataverse. Retrain and benchmarks (306658-306660) cancelled pending data.
  Code smoke 306649 (P2-24, **random N(0,1) embeddings**, `_smoke/`, not a result): TX coverage 990 /
  1,725 compounds; `knn_txgnn` − mean −0.0002, − permuted +0.0001 (inner CV shrinks a useless
  similarity to ~0, as intended); `knn_chem_txpool` − mean +0.0288; all TX contrasts computed.

## 2026-10-07 — Deck correction: GNN (approach 2) design and run log added; conclusion narrowed (user request)

用户指出 meeting deck 没有完整呈现 graph-inference 实验的设计与执行过程。Deck
(https://claude.ai/artifact/2LoAhJt4SFmJakNgZ7QiDC, version 13) 新增 `gnn-design`(输入、两次用图的
encoder、concat/dot 两种 head、对照、训练设置)和 `gnn-log`(G3b 预注册 → 10/03 偏离 → 294193-294195
inconclusive → 294352-294354 / 294396 / 296290-296291 exploratory),放在 `ppi-way2` 之前。

**Correction of framing.** `ppi-way2` 原标题 "Four model families, one answer: real = rewired" 过强:预注册
的 GNN 从未学到东西(G3b inconclusive),DotGNN 是 exploratory、单 seed、只有 P2-24、多数 fold 不稳定。
现标题 "Real ≈ rewired wherever a model learns";real-vs-rewired 证据来自能学习的模型(MLP + 传播、
kNN、KRR、DeepCE)加上无模型 E1 ladder。Still untested: pre-registered GCN (on GPU), P1-24, >1 seed.

**Why not GPU.** Record 没有 GNN 专门的理由;当时 GPU QOS 被 promoterVI 占满(见 10/04 DeepCE 条目),
`slurm/run_l1000_gnn.sbatch` 写在 `cscc-cpu-p`。CPU 太慢时选择了改设计而不是上 GPU,事后看应先上 GPU
保持预注册检验干净。是否补跑(GCN on GPU / DotGNN P1-24 / 3 seeds)待用户决定。

## 2026-10-07 — G3b re-run: the pre-registered GCN on GPU (user request; written before any GCN result)

**Why.** The 10/03 deviation replaced the pre-registered per-drug 2-layer GCN with the decoupled GNN
because it was too slow on CPU; G3b was then inconclusive (no learning). GPU is free now (0 jobs on the
account, idle A100s), so the original design gets its clean test.

**Implementation.** `scripts/run_l1000_gnn.py`, new class `GCN`, arms `gcn_ppi` / `gcn_rewired` /
`gcn_nograph` (not in the default arm list; old arms unchanged, still CPU by default). Exactly the G3b
text: node state H0 = E + 1[target]·(v + chemMLP(Morgan)); 2 GCN layers over D^-½(A+I)D^-½ (ReLU
between); per-line readout MLP over [landmark node state ; target-pooled (1/n) state + chemistry] →
7 lines, + bias[line, gene]. STRING physical ≥ 400; rewired = the same seed-0 degree-preserving graph;
no graph = identity adjacency. Training as pre-specified: Adam 1e-3, wd 1e-5, batch 64, ≤ 150 epochs,
patience 15 on the first inner split, full Pearson on ACTIVE validation rows, **no output shrinkage**.
Same folds/cohorts as the kNN runner (seed 0). `mlp_target` is re-run in the same job as a sanity anchor.
New `--device cuda`, `--init-seed`; sbatch `slurm/run_l1000_gcn_gpu.sbatch`, env `gnn-gpu`
(torch 2.4.1+cu118; pyarrow 17.0.0 added 2026-10-07 for parquet I/O, nothing else changed).

**Gate (= G3b unchanged).** Primary: ik14 split, KNOWN∩ACTIVE, P2-24 **and** P1-24, init seed 0:
`gcn_ppi` − `gcn_rewired` low > 0 **and** `gcn_ppi` − `gcn_nograph` low > 0 **and**
`gcn_ppi` − `krr_chem_known` low > 0 (line-macro bootstrap via `compare_l1000_gnn.py`). Also reported
against the project's +0.01 meaningful-effect bar. Secondary: plate-map split, same contrasts.
**Learning check:** if the GCN arms reach their best validation score within the first 2 epochs in a
majority of folds, the result is "did not learn" (inconclusive again), not a graph verdict; an output-
shrinkage re-run would then be exploratory. If seed 0 learns, init seeds 1 and 2 on ik14 follow
(robustness, not part of the gate). Expectation: fails on real − rewired, as every graph control so far.

## 2026-10-08 03:10 — G3b re-run result: fails (jobs 307277-307280; report §8)

预注册的 per-drug GCN 在 GPU 上跑完。P2-24 两个 split 10/10 fold 在 epoch 1 最佳(= mean,又没学起来);
P1-24 学到一点(+0.008 over mean),但 no-graph 一样 +0.008,real − rewired +0.0003,比 KRR(chem) 低 0.045,
比同 job 的 flat MLP 低 0.018。三个 gate 条件在主 split 两个 slice 上都不过 → **G3b fails**。
这堵上了 deck `gnn-log` 页 "Not yet tested: the original GCN" 那一项;P2-24 不学习说明在 ~700 个训练药物下
这类 graph 模型本身难训练,和 kernel-beats-neural 的一贯结论一致。仍未做:多 init seed(预注册里规定只在
seed 0 学起来时才做;P1-24 学了但 gate 已明确失败,补 seed 不会改变结论,建议不做,待用户定)。
Deck 尚未更新(等用户同意)。

## 2026-10-08 — DepMap↔Tahoe CCA shared space: does the same line align across datasets? (user request; written before any result)

**问题 (user).** 用 Tahoe 与 DepMap 共有的细胞系做 CCA,映射到一个 joint space,看同一细胞系在两个数据集里
是否有共享表示;再用这个共享表示预测 Tahoe 的 response,和同一细胞系的 Tahoe 真值比较。
注意:所有 Tahoe 线都在 DepMap 里,所以 "shared lines" = Tahoe 的 ~49 条有 DepMap 表达的线;没有 response 真值
的 DepMap 线不参与评估。This is an exploratory side check (not a gate-bearing model).

**Why this is cheap and well-posed.** Rhaister-O is linear in the held-out line's centred DMSO vector:
`pred = (1-b)·additive + b·(mu_y + gamma ⊙ z_c)`. Everything else stays frozen (gamma, mu_y, additive from
reference lines). We only replace `z_c` (real Tahoe DMSO) by an estimate from DepMap.

**Protocol (released splits 5–9, 5 held-out lines each; all fitting on outer reference lines only).**
- X = DepMap Public 24Q4 log TPM (protein coding); Y = Tahoe per-line mean log1p DMSO on the fixed 2,000 genes.
  Each side: reference-only standardisation + PCA (whitened, k comps), then CCA = SVD of the cross-covariance.
- (k, d) chosen from k ∈ {5,10,20}, d ≤ k by inner leave-one-reference-line-out reconstruction error of Y
  from X. Label-free: no treated response enters alignment or selection.
- **A. Identity alignment.** For each held-out line: rank of its own Tahoe DMSO embedding among all lines'
  Tahoe embeddings, querying with its DepMap embedding (cosine in canonical space). Report top-1/top-5 and
  rank percentile. Controls: direct gene-matched correlation without CCA; null = refit with permuted
  reference pairing (50 permutations/split). Also out-of-sample canonical correlations vs in-sample.
- **B. Response prediction.** Replace z by: real DMSO (= frozen Rhaister-O, checked by exact rebuild);
  CCA-mapped DMSO; nearest reference line in DepMap space; mean DMSO (z_c = 0, no line info); shuffled
  CCA-mapped DMSO (another held-out line's). Metrics full / drug_residual / train-PC1-removed Pearson,
  line-macro with line bootstrap. Headline quantity: fraction of the real-DMSO gain over mean-DMSO recovered
  by the CCA map, on drug_residual.
- Expectation: identity retrieval well above chance (same catalogue line); response recovery partial,
  ≤ real DMSO by construction. Script `scripts/run_depmap_tahoe_cca.py`, sbatch `slurm/run_depmap_tahoe_cca.sbatch`.

## 2026-10-08 15:20 — CCA result (job 308584; report §9)

结果:(1) 同一细胞系在两数据集间**确实有共享信号**:CCA 检索平均 rank 6.7/47,permuted null 23.8。
但 **CCA 不如不做 CCA**:直接按共同基因做相关,top-1 96% vs CCA 26%(更正后,见下)。原因是 n≈43 条参考线上
k=10 的 CCA 过拟合(in-sample ρ≈0.98,out-of-sample 部分维度为负)。CCA 找的是"两数据集都变化最大的共同
方向",而不是"把每条线区分开的方向"。
(2) 用 DepMap 经 CCA 映射出的 DMSO 喂给冻结的 Rhaister-O:drug-residual 0.249,恢复真实 DMSO 相对 mean-DMSO
增益的 46%,显著高于 shuffled (+0.164) 和 nearest-line;但比真实 DMSO 低 0.29。full-delta 0.490 vs 0.634。
结论:DepMap-only 部署可行但代价大;"在 DepMap 上得到很高 Pearson"不成立(gene space,truth = Tahoe delta)。

Deviation note: after submission the user clarified a different design — fit Rhaister-O inside the shared
space and score against *projected* Tahoe perturbed profiles. Not run. Raised caveats: projection keeps
only identity-aligned directions, Pearson over d≈5–20 coordinates is not comparable to 0.526, and absolute
post-treatment profiles would be inflated by identity. Awaiting user answers (delta vs absolute; per-
canonical-dimension gamma vs projected gene-space prediction).
Open, not run: direct gene-matched substitution (DepMap genes standardised into Tahoe DMSO scale) as a
simpler map, suggested by its better identity retrieval.

## 2026-10-08 — CCA shared-space Rhaister-O, absolute post-treatment target (user design; written before any result)

用户澄清 (1) target = **absolute post-treatment** expression; (2) 先做 **per-canonical-dimension** Rhaister-O。
Absolute profile per condition = log1p(plate-matched DMSO centroid, averaged over the condition's plates) + upstream
delta (delta is defined upstream as log1p(treated) − log1p(DMSO), so this is exact up to plate averaging).
Delta 是否"在 shared space 里怪"?不怪:Tahoe 侧投影是仿射的,P(DMSO+Δ) = P(DMSO) + L(Δ),两者都良定义;
absolute 只是多了 identity 项 P(DMSO),会抬高 Pearson,所以同时报告 delta 并加 identity-only baseline。

Protocol: same CCA as job 308584 (reference lines only, same (k, d) selection). Shared-space model per treatment t,
canonical dim j: target_ctj = mu_tj + gamma_tj (x_cj − x̄_j), ridge λ = 1, trained with x = reference lines' Tahoe DMSO
canonical coordinates. Deviation from Rhaister-O: no 95/5 ALS blend (ALS has no canonical-space analogue).
Held-out inputs: DepMap coordinates as-is (user's idea); ρ-shrunk DepMap (E[v|u] = ρu); real Tahoe DMSO
coordinates (upper bound); zero (mean, no line info); shuffled DepMap (another held-out line).
Baselines in the same space: identity-only (predict target = the input coordinates, zero drug effect; absolute
only), drug mean over reference lines. Metric: per-condition Pearson over the d coordinates, also RMSE;
line macro with line bootstrap. Also report retained fraction of held-out true gene-level delta (standardised)
after projection. Pearson over d ≈ 5–10 numbers is not comparable to gene-space 0.526.
Script `scripts/run_depmap_tahoe_cca_space.py`, sbatch `slurm/run_depmap_tahoe_cca_space.sbatch`.

## 2026-10-08 15:30 — Shared-space result (job 308619; report §10)

结论:absolute target 下 DepMap 输入得到 0.58(ρ-shrunk 0.61),真实 Tahoe DMSO 输入 0.95——**但全部来自 identity**:
"只预测该线自己的坐标、药物效应为零" 的 baseline 分别是 0.587 和 0.953,Rhaister-O 加上药物项反而略低
(−0.006 / −0.005)。原因是 CCA 空间本身几乎不含药物响应:held-out 线的基因级 delta 投影后再映射回去,
与原 delta 的 Pearson 只有 0.05–0.07。Delta target 下 DepMap 0.217 vs drug mean 0.171(+0.046, CI [0.006, 0.089]),
真实 DMSO 0.280。
所以用户的直觉部分成立:同一细胞系在 DepMap 与 Tahoe 之间有共享表示,absolute 指标上可以拿到看似高的 Pearson;
但这个数衡量的是"认出是哪条线",不是"预测药物响应",不能作为响应预测的结果报告。
§9 的 gene-space 版本(drug-residual 0.249,恢复 46%)是对响应预测更有信息量的数。
Next options (not run, for the user): (a) direct gene-matched DepMap→Tahoe substitution (better identity retrieval
in §9); (b) a response-aware joint space (e.g. fit the reduction on reference responses, then align) — must keep
target-line responses out of the fit.

## 2026-10-08 15:40 — Correction: direct identity control (job 308632 replaces 308584's direct numbers)

画图时发现 direct gene-correlation control 有数值 bug:参考线中方差为 0 的基因(DepMap 11–14 个,Tahoe 1–4 个)
被除以 ~1e-8,query 的 |z| 高达 4e7,淹没了相关。去掉这些基因后 direct top-1 **95.7%**(22/23),top-5 100%,
mean rank 1.09(原报 60.9% / 65.2% / 5.22)。CCA 与所有 response 数字不受影响(308632 与 308584 完全一致)。
解读更强:同一细胞系跨数据集几乎可以被完美识别,不需要学习;CCA 的差表现来自 n≈43 下的过拟合,
而不是缺少共享信号。这让"直接 gene-matched 替代 DMSO"成为更值得试的下一步(仍未运行)。
Figures: `report/figures/10_09_cca/` (script `scripts/plot_depmap_tahoe_cca.py`); interactive explorer requested by user
(data export `scripts/export_cca_explorer_data.py`).
Interactive explorer: https://claude.ai/artifact/Ku9pALrbWARFiQFdKkjHyw (private); local copy `report/figures/10_09_cca/line_match_explorer.html`.

## 2026-10-08 16:00 — 用户问:之前跑过 "Rhaister-O on DepMap" 吗?目标是给 unseen DepMap 线预测药物响应

Lookup: Rhaister-O itself was never run on DepMap input (KNOWLEDGE: "Never feed DepMap TPM log1p directly into
Rhaister-O"; a DepMap→pseudo-DMSO bridge was listed as unexecuted until today's CCA run). The DepMap-only
predictors that do exist, all on the same 23 held-out Tahoe lines, drug-residual line-macro Pearson:
DepMap ridge, whitened PCA (ORDERED_BASELINE Step 5A) 0.209 [0.159, 0.260]; capacity sweep identical (09_18 §1);
unwhitened ridge (10_02 §2.7) 0.260 [0.207, 0.315]; today's CCA-mapped DMSO into frozen Rhaister-O 0.249.
Rhaister-O with real Tahoe DMSO: 0.539.

Login-node side check (exploratory, all 47 lines, no response data): per gene, cross-line Pearson between DepMap
log-TPM and Tahoe log1p DMSO: median 0.75 (IQR 0.61–0.83; 85% of 1,976 genes > 0.5). A leave-one-line-out
per-gene linear calibration DepMap→Tahoe reconstructs a line's centred DMSO at mean r = 0.69, versus 0.38 for the
CCA map. Since §9 showed CCA-mapped response accuracy tracks reconstruction quality, a direct per-gene bridge
fed into frozen Rhaister-O is the obvious next test (not run; awaiting user). Tahoe covers 13 solid-tumour
lineages only, no haematopoietic lines.

## 2026-10-08 — Direct per-gene DepMap→Tahoe DMSO bridge into frozen Rhaister-O (user request; written before any result)

Same harness, splits, lines and metrics as §9 (job 308632); new arms only, output to a new directory
`results/depmap_tahoe_direct_map/` so 308632 stays frozen. For every Tahoe panel gene with a DepMap match and
non-zero reference variance on both sides, fit OLS `tahoe_dmso_g = a_g + b_g · depmap_g` on the split's reference
lines only; predict the held-out line's DMSO vector; unmatched genes take the reference mean (centred value 0).
Arms added: `direct_mapped`, `shuffled_direct` (another held-out line's mapped vector). Reported against real DMSO,
mean DMSO, CCA-mapped and shuffled. Headline: drug-residual line-macro Pearson and the fraction of the real-DMSO
gain recovered. Expectation from the side check (LOO reconstruction r 0.69 vs CCA 0.38): clearly above CCA's 0.249,
below 0.539. No response data enter the bridge.

## 2026-10-08 16:10 — Direct bridge result (job 308706; report §11)

结果超出预期:直接 per-gene 映射 DepMap→Tahoe DMSO 后喂给冻结的 Rhaister-O,drug-residual **0.406**
(CI [0.361, 0.442]),恢复真实 DMSO 增益的 75%;比 CCA 高 0.157,23/23 条线都更好;比之前所有 DepMap-only
预测器(0.21–0.26)高约 0.15。仍比真实 DMSO 低 0.134,没有任何一条线达到其真实 DMSO 分数的 90%。
线间表现与 DMSO 重建质量强相关(r = 0.78)→ 下一步提升应来自更好的 bridge(例如 shrinkage 斜率、
多基因 ridge、用更多 DepMap 线做无监督校准),而不是改 Rhaister-O。
为什么比 DepMap ridge 好:ridge 要从 45 条线直接学 DepMap→(treatment × gene) 响应;bridge 只学 DepMap→DMSO
(每基因 2 个参数),然后复用 Rhaister-O 已有的 DMSO→响应结构。Bridge 用的是"同一基因"这一已知对应。
308671 在 cn-20 split 9 静默死亡(无 traceback,sacct 不可用),308706 `--exclude=cn-20` 重跑成功;前 4 个 split
结果相同。Open for user: (a) scope of "instruction" for unseen DepMap lines; (b) whether to try a better bridge;
(c) add the direct arm to the explorer artifact.

2026-10-08 — Meeting deck slide `depmap` rewritten as the §11 result (user: keep it simple, positive claims): DepMap-only Rhaister-O 0.406 vs real DMSO 0.539, previous ridge 0.209, drug mean 0.000; CCA and caveats moved to speaker notes. Deck version 1791477324-7f9c. GCN (G3b) result still not in the deck.

## 2026-10-08 — End-to-end integrated model (Jun Wen design): proposal and critique (user request; discussion only, nothing run)

**Proposal (user + Jun Wen).** Input (molecule, gene, cell line), output the up/down response of that gene in
[-1, 1]. Two modules: (1) a "prior" GCN/RGCN over PrimeKG that produces a molecule encoding F_m and a gene
encoding F_g (gene node features possibly from pretrained DNA encoders such as Evo, DNABERT or AlphaGenome;
molecule node features from Morgan fingerprints; disease nodes in the KG); (2) an MLP over [cell-line
baseline, F_m, F_g] that outputs the response.

**Evidence in hand that constrains the design (this week's report):**
- GCN/RGCN prior: every graph encoder tried ties its degree-preserving rewired null and sits below chemistry
  KRR (§4.5, §4.7, §8 G3b fails: real − rewired +0.0003, GCN − KRR −0.045 on P1-24). A KG encoder
  is admissible only with the same control ladder.
- Concatenation head: the decoupled GNN with a shared concat head did not learn (§4.5); swapping in a
  dot-product head made it learn (§4.7). Per-(drug, gene) MLP over concatenated vectors carries the same risk.
- Gene encoder: with a fixed 978-landmark panel, a free one-hot gene embedding ≥ STRING/node2vec vectors
  (§5.3, §5.4). Pretrained gene priors can only matter when genes are unseen at test or data is tiny.
- Drug-side information is tiered (§7): target identity +0.035-0.045 over permuted; Cell Painting +0.05-0.09;
  one measured response in another line +0.13; one-hop PPI = rewired.
- PrimeKG covers only 1,567 / 8,119 P1-24 compounds (§2); 69% of P1-24 has no target knowledge at all. A
  transductive KG drug node does not exist for most test compounds.
- Directed signed regulatory networks (OmniPath, CollecTRI) carry gene-level locality beyond degree
  (+0.03-0.07, §6): the one graph signal found so far, and it is gene-level, not drug-drug.
- Cell line: tissue lineage as similarity hurts on Tahoe (-0.034, 10_02 §4); untreated baseline is the best
  cell descriptor; on L1000 (7-9 lines) a cell encoder cannot generalize, only memorize.

**Critique / proposed changes (Claude, for discussion with the user):** fix the split axis and target first;
continuous delta target with derived up/down calls instead of a forced [-1, 1] sign; bilinear/dot-product
decoder with a built-in line×gene mean instead of a pure concat MLP; inductive drug encoder (Morgan → KG space)
so drugs absent from PrimeKG still work; per-(line, gene) baseline expression as an input, not only a cell
vector; KG encoder pretrained and probed frozen before end-to-end fine-tuning; every module with its null.
Full reasoning in the chat of this date; decisions pending the user.

## 2026-10-08 — Integrated model: user decisions, round 1 (discussion; nothing run)

User decisions: (1) unseen drug first (L1000), unseen cell line second (Tahoe); (2) output is a continuous
number; (3) proposal: per-gene NB normalization (learned mean and dispersion per gene) so the network outputs
one number per (drug, line, gene); (4) train end to end (PrimeKG alone judged insufficient) — needs a time
estimate; (5) initialize disease nodes from PubMedBERT.

Claude's notes for the record:
- (3) NB applies to Tahoe counts only; L1000 Level 5 is already a per-gene moderated z-score (Level 3 is log2
  fluorescence), so L1000 keeps a Gaussian/MSE head. For Tahoe the mean must be per (line, plate, gene) — taken
  from the plate-matched DMSO (or the DepMap bridge for unseen lines) — with dispersion θ_g per gene and the network
  output Δ = log fold change: counts ~ NB(s · μ_{l,p,g} · exp(Δ_{d,l,g}), θ_g), fitted on pseudobulk counts with
  library-size offsets. Δ is the shared "one number"; platform-specific heads, consistent with
  `reports/L1000_TAHOE_IDENTITY_FUSION_AND_DL.md`.
- (4) PrimeKG v2.1 measured on disk: 129,375 nodes, 8,100,498 directed edges, 30 relations; drug_drug 2,672,628
  and anatomy_protein_present 3,036,406 are 70% of edges. Back-of-envelope full-graph 2-layer RGCN on A100-40GB:
  ~0.3-1 s/step full graph, ~0.1-0.3 s/step without those two relations; ~1-3 GPU-h per L1000 P1-24 run; a
  5-fold × 3-seed ladder with 4 KG arms ≈ 60 runs ≈ 60-180 GPU-h. Estimate only; a timing smoke should replace it.
- (5) Text source: PrimeKG `disease_features.tab` (113,534,270 bytes, same Dataverse record, not yet downloaded).
  Suggest SapBERT (PubMedBERT fine-tuned for UMLS synonyms) for names and PubMedBERT for definitions; the same
  text initialization for every non-drug, non-gene node type; null = embeddings shuffled among disease nodes.
  On L1000 unseen-drug, disease enters through drug–indication/contraindication edges, not as a line feature.

## 2026-10-09 00:40 — Integrated model round 1: user decisions and implementation (nothing quoted yet)

**User decisions (round 2).** KG ablation is low priority; only two KG arms: (a) PPI + drug–disease +
disease–gene; (b) drug–disease + disease–gene. The no-KG ladder only if time remains — "we'd like to include KG to
some extent". One seed at the start; the network structure will change rapidly (agile). Everything else follows the
plan of 2026-10-08. Maintain an interactive tracker of interim results and pipeline design:
https://claude.ai/artifact/74ath3hh5LkPM2cqd32Rwu (results in its database; design section republished on change).

**Implementation (Claude).**
- `scripts/build_e2e_kg.py` (job 309951): PrimeKG v2.1 relations protein_protein, indication, contraindication,
  off-label use, disease_protein, plus drug_protein (stored, used by neither arm), split by direction → 11 relation
  types; 32,024 nodes (19,093 gene/protein, 6,642 drug, 6,289 disease), 939,170 directed edges. Compounds with a
  drug–disease edge: P1-24 1,117 / 8,119 (14%), P1-6 1,121 / 7,626, P2-24 708 / 1,725 (41%), T-5 198 / 350.
  `disease_features.tab` downloaded on the login node (md5 8b38e95c… matches Dataverse file 6180618).
- `scripts/embed_disease_text.py`: SapBERT-from-PubMedBERT [CLS] of the name; PubMedBERT
  (BiomedBERT-base-uncased-abstract-fulltext, downloaded to the project HF cache) masked mean of
  "name. definition" (5,678 / 6,289 have a Mondo/UMLS/Orphanet definition).
- `scripts/run_l1000_e2e.py`: arms skel / kg_a / kg_b (kg_t = kg_a + drug–target defined, not scheduled). Every
  slice compound is its own drug node with DrugEnc(Morgan2) input; a compound matched to a PrimeKG drug (full
  InChIKey, else ik14) takes over that drug's edges; other drugs and genes are free embeddings; diseases are a
  linear map of the frozen text features. 2-layer RGCN (mean aggregation, residual, LayerNorm) over the whole graph
  at every step + 0.1 × DistMult link loss. Decoder pred[d,l,g] = <W_l h_d, e_g>/√128, e_g = free landmark
  embedding + KG gene state. Residual-from-line-mean target, masked MSE, AdamW 1e-3 / wd 1e-4, batch 128
  compounds, ≤200 epochs, patience 20, output shrinkage on inner validation. Same folds/rows as the runner.
- Deviation to note: drug–target edges are in neither of the user's arms, so the strongest knowledge signal so far
  (§7: target identity +0.035–0.045) is absent from round 1 by design; kg_t is ready if wanted.
- Bug found: `run_l1000_gnn.py` re-reads `sl["Y"]` from the NpzFile on every row (slow, not wrong; frozen results
  unaffected). Fixed in the new script only.
- Skeleton is run on CPU as an early look (`_skelcpu`); the quoted skel number will come from the GPU job that runs
  all three arms with identical settings.
- (01:15) GPU smoke 309991 (A100, ~0.1 s/step, 22 GB) exposed a scale bug: residual RGCN states grow, so KG arms
  started at MSE 6.7 (skel 2.5) with an unscaled DistMult link loss of 47 and sat at the mean (α = 0). Fix: LayerNorm
  on h_d before the decoder (all arms) and DistMult on LayerNormed states divided by √D. Smoke 310014 (25 epochs,
  fold 0, not quoted): all arms learn (val skel 0.339, kg_a 0.335, kg_b 0.334, mean 0.328); KG arms learn slower.
  Full runs 310004 (P1-24) / 310005 (P2-24) were held during the fix and released after. The CPU skeleton runs
  309999/310000 predate the fix (no LayerNorm on h_d); treat them as v0.
