# Unseen-drug response prediction: drug representations, retrieval, graphs and new modalities

Date: 2026-10-03. Status: literature review plus a pre-registered L1000/Tahoe sanity battery.
Numbers in this report are copied from `report/10_09_report.md` (results) and
`report/10_09_record.md` (protocol, amendments, reasoning); those files are authoritative.
This is the **unseen-drug** task (CLAUDE.md: a distinct secondary experiment). It does not
change the frozen Rhaister-O unseen-cell-line benchmark.

## 0. 一句话结论 / Bottom line

**化学结构能找到"像"的药，但找不到"响应像"的药；图结构几乎不增加信息；最有效的新模态是对新药做一次
便宜的表型测量。**

1. Many models encode drugs from SMILES (fingerprints, MACCS, graph networks, ChemBERTa/MoLFormer,
   Uni-Mol). On L1000 the learned SMILES language models do **not** beat a 2048-bit Morgan
   fingerprint for retrieving response-similar drugs. ChemBERTa is significantly worse in every
   slice, and MoLFormer ties.
2. The library **is** dense in response space: 65-73% of active compounds have a response twin
   (r ≥ 0.5) among the other compounds of the same line. But the chemical nearest neighbour is that
   twin only 10-12% of the time, whatever the encoder. Approach 1's bottleneck is the similarity
   function, not the size of the drug pool.
3. Target knowledge helps a little beyond chemistry (+0.013 to +0.019 full-signature Pearson on
   annotated compounds). Propagating targets over a real PPI network adds only +0.002 to +0.003
   over a degree-preserving random network. A trained GNN on the PPI graph (approach 2) is §4.4.
4. One cheap measurement of the new drug changes the picture. Its L1000 profile in a single
   *other* cell line, used to choose neighbours, beats the best chemistry by +0.10 to +0.15.
   Cell Painting morphology also beats chemistry, but its larger effect in Phase II is partly a
   shared-plate-layout artifact (§4.3).
5. A plate confound affects L1000 unseen-drug benchmarks. Co-plated compounds share a residual
   batch signal, and chemically related compounds were often co-plated. Excluding same-plate
   neighbours removes about 40% of chemistry retrieval's gain in Phase I. Published L1000
   cold-drug results do not report a plate-aware split.

## 1. Question and deployment boundary

Input at deployment: the cell line (seen in training; baseline expression known), the new
compound's structure, optionally its known targets / mechanism, and optionally one cheap
measurement of the compound (a different assay or a different line). Output: the 978-landmark
(L1000) or 2,000-gene (Tahoe) expression change of that line. Nothing measured on the new
compound in the target line may be used.

The user's two approaches:
- **Approach 1, retrieval.** Embed the new drug, find the nearest library drugs, and average
  their responses with weights.
- **Approach 2, graph inference.** Place the drug (embedding + targets) in a PPI/pathway graph
  and let a GNN/GAT infer the response.

Mathematically both are smoothers over drug-drug similarity. Approach 1 uses an explicit kernel.
Approach 2 learns the kernel implicitly through shared parameters, and the graph only matters
through the targets that connect a new drug to the training drugs. The experiments below test
both, with controls designed to separate "the graph helps" from "the target annotation helps".

## 2. Literature: who encodes drugs how, and what they report for unseen drugs

Claims below were checked against the source text where a link is given. Subagent-retrieved claims
that failed verification are listed in §2.5.

### 2.1 Drug encoders in transcriptional-response models

| work | data | drug representation | unseen-drug test | what it reports vs baselines |
|---|---|---|---|---|
| DeepCE, Pham et al., *Nat Mach Intell* 2021 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8009091/)) | L1000 Level 5, 7 lines, high-quality signatures (APC ≥ 0.7) | GCN "neural fingerprint" on the molecular graph + gene-gene/drug-gene attention | split by compound ("de novo"); no stereoisomer/scaffold grouping described | Pearson 0.491 vs kNN (Tanimoto) 0.391, vanilla NN 0.426, ECFP-NN 0.41, PubChem-NN 0.39, linear 0.18. No training-mean baseline. |
| chemCPA, Hetzel et al., NeurIPS 2022 ([arXiv](https://arxiv.org/abs/2204.13545)) | sci-Plex3 + L1000 transfer | compared RDKit fingerprints/descriptors, GCN, MPNN, Weave, GROVER, JT-VAE | 9 held-out drugs × 3 lines | pretrained graph encoders did not consistently beat RDKit features (supplement, via search excerpt; exact numbers not re-read) |
| MultiDCP, Wu et al., *PLoS Comput Biol* 2022 | L1000 + external baseline RNA | neural fingerprint + gene/cell/dose encoders | leave-new-line-out (primary); see the project's earlier audit | no Rhaister-style drug-residual comparison |
| TranSiGen, Tong et al., *Nat Commun* 2024 ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11199551/)) | L1000 | pretrained molecular representation + basal expression, VAE | novel compounds | compared with deep baselines |
| PRnet, Qi et al., *Nat Commun* 2024 | L1000 + sci-Plex | perturbation-conditioned generative model | novel compounds, pathways, lines | compared with deep baselines |
| PertDiff, Ma et al., *Bioinformatics* 2026, 42(7) btag482 | LINCS 2020, 8,316 drugs, 164 lines | molecular GNN + LLM cell description + control expression, diffusion | "cross-drug" | +10.2% R² over TranSiGen; **no mean or kNN baseline** |
| XPert, *Nat Mach Intell* 2026 (project audit, `VIRTUAL_CELL_ENCODERS_AND_GENERALIZATION.md`) | L1000 | Uni-Mol + drug-target/PPI features | cold-drug and cold-cell splits | not independently re-verified here |
| Agarwal & Bisht, "The Metric Picks the Winner", arXiv 2606.12639 (2026) | DRUG-seq, THP-1, 14,026 training compounds | Morgan linear, Tanimoto retrieval, ChemBERTa/frozen-chemistry deep fusion | Bemis-Murcko scaffold split | under standard metrics the linear fingerprint baseline wins; under the contest's weighted metric the deep fusion model wins. **The ranking depends on the metric.** |
| OP3 / NeurIPS 2023 competition, Szałata et al., NeurIPS 2024 D&B | 146 compounds, PBMC cell types | varied | **held-out cell types for seen compounds**, not unseen drugs | (often cited as a drug benchmark; it is not an unseen-drug test) |

Benchmarks of the field point the same way: Ahlmann-Eltze, Huber & Anders,
*Nat Methods* 2025 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/)) find that
foundation and deep models do not beat deliberately simple linear/additive baselines for genetic
perturbations, and Systema (Viñas Torné et al., *Nat Biotechnol* 2025) shows that systematic
perturbed-vs-control variation inflates common metrics.

### 2.2 Does chemical similarity predict transcriptional similarity?

- Sirci et al., *npj Syst Biol Appl* 2017 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5572457/)),
  CMap build 02 (1,309 compounds, Affymetrix, **not L1000**), 3D pharmacophore similarity, top-5%
  thresholds. Of 856,806 pairs, 0.3% are similar in both, 5.1% in structure only, 4.0% in
  transcription only, and 91% in neither. So about 6% of structurally similar pairs are
  transcriptionally similar, against a 4.3% base rate. The authors conclude the two views are
  largely independent.
- Iorio et al., *PNAS* 2010: communities in a transcriptional drug network are enriched for shared
  mechanism of action.
- Our L1000 measurement (§4.1) is consistent and more specific. Similarity is predictive only in
  the far tail (Tanimoto ≥ 0.6), and such pairs are rare.

### 2.3 Knowledge graphs, PPI and graph models

| resource / model | relevant facts |
|---|---|
| PrimeKG v2.1 (Chandak, Huang, Zitnik, *Sci Data* 2023; Dataverse doi:10.7910/DVN/IXA7BM, CC0 record) | 16,380 drug-target, 5,317 enzyme, 3,092 transporter and 864 carrier edges from DrugBank; 321k PPI edges. Drug nodes are DrugBank IDs without structures. |
| DrugBank | academic licence; the open vocabulary download returned HTTP 403 without an account (2026-10-03) |
| STRING v12 | physical subnetwork, score ≥ 400: 15,809 genes, 210,667 edges; shares only 60,645 edges with PrimeKG's PPI |
| LINCS 2020 `compoundinfo_beta` (Drug Repurposing Hub annotations) | targets for 2,302/8,715 Phase-I 24 h/10 µM compounds (26%) |
| GEARS, Roohani et al., *Nat Biotechnol* 2023 | GO graph for unseen *genetic* perturbations. An independent ablation (AblateCell, arXiv 2604.19606) found its GO-similarity and co-expression graphs contribute little next to the message-passing encoder. |
| PDGrapher, Gonzalez et al., *Nat Biomed Eng* | PPI/GRN as a causal-graph approximation, but solves the **inverse** problem (which targets reverse a state) on L1000 chemical perturbations in 9 lines |
| TxPert (*Nat Biotechnol* 2026), MAP (*Nat Mach Intell* 2026) | graph/KG-informed perturbation representations; genetic (TxPert), or line-drug pair splits (MAP) |

**No published chemical-perturbation model that we found reports a degree-preserving
random-graph control or a permuted-target control.** This project has twice found such
controls decisive (Reactome prior vs random, `report/10_02_report.md` §2.4; PPI vs rewired, §4.2
below).

### 2.4 New drug-side modalities

| modality | evidence |
|---|---|
| Cell Painting morphology | Way et al., *Cell Systems* 2022 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10246468/)): 1,327 Drug Repurposing Hub compounds × 6 doses in A549 profiled by both L1000 and Cell Painting; "shared and complementary information"; Cell Painting more reproducible. Haghighi et al., *Nat Methods* 2022: four paired datasets, > 28,000 perturbations. JUMP-CP: ~116k compounds (U2OS). |
| molecule-phenotype contrastive encoders | CLOOME (Sanchez-Fernandez et al., *Nat Commun* 2023): retrieval of the matching image from ~2,000 candidates at > 70× random. MolPhenix (Fradkin et al., NeurIPS 2024): 8.1× better zero-shot retrieval of active molecules from phenomic images. InfoAlign (Liu et al., ICLR 2025): aligns molecules to morphology and expression neighbourhoods. None of these reports L1000 response prediction for unseen drugs. |
| the drug measured in another context | Hodos et al., PSB 2018: cross-cell-line imputation of L1000 profiles reaches r = 0.68 (kNN and tensor completion). OP3 is effectively this regime. |

### 2.5 Subagent claims that failed verification (do not cite)

- "~20% of structurally similar compounds share expression (Musa et al. 2017)". The numbers
  belong to Sirci et al. 2017, use CMap build 02 and 3D pharmacophores, and imply about 6%.
- "PertDiff, Xiong et al. 2024". It is Ma et al., *Bioinformatics* 2026.
- "PMC10246468 = Haghighi 2022". It is Way et al. 2022.
- "35-45% of landmark genes predictable from morphology (R² > 0.6)". This was not found in a
  source.
- Several entries were marked UNVERIFIED by the agents themselves (CIGER venue, DTIAM authorship,
  etc.); they are omitted.

### 2.6 Weaknesses common to the published unseen-drug work

1. **No hard baselines.** Training-mean and plain Tanimoto-kNN baselines are often absent
   (PertDiff), or the comparison is to other deep models only. A response oracle is never
   reported, so the reader cannot tell how much of the attainable signal is captured.
2. **Leaky splits.** Splits are by compound identifier, not grouped by structure. In L1000
   Phase I, 20,350 SMILES-bearing compound IDs collapse to 9,174 connectivity blocks; DOS
   stereoisomer families have up to 32 members with identical 2D fingerprints.
3. **Plate/batch co-membership** is never controlled (§4.3).
4. **Full-signature metrics are dominated by the shared component.** Full Pearson includes the
   generic response every compound shares; residual and magnitude metrics are rarely reported.
   Pearson is scale-invariant and cannot see whether the drug does anything.
5. **Inert compounds.** Only 34-47% of Phase-I signatures have TAS ≥ 0.2. Averaging over inert
   compounds rewards predicting noise.
6. **No graph controls**, and KG coverage limited to known drugs (≈ 26-31% of L1000 compounds).

## 3. Experiments run here (protocol summary)

Pre-registration, amendments and every deviation: `report/10_09_record.md` (2026-10-03).
Scripts: `scripts/extract_l1000_cp_landmarks.py`, `build_drug_features.py`,
`embed_smiles_lm.py`, `build_l1000_unseen_drug.py`, `build_cell_painting_features.py`,
`run_l1000_unseen_drug.py`, `run_l1000_gnn.py`, `analyze_l1000_structure_response.py`,
`analyze_l1000_plate_confound.py`, `analyze_l1000_applicability.py`,
`audit_primekg_overlap.py`, `build_tahoe_unseen_drug.py`.

- Slices: P1-24 (GSE92742, 24 h, 10 µM, 7 lines, 8,119 compounds), P1-6 (6 h, 9 lines),
  P2-24 (GSE70138, 24 h, 10 µM, 7 lines, 1,725 compounds); Tahoe T-0.05/T-0.5/T-5 (350
  compounds × 50 lines).
- Splits: ik14 (stereoisomers grouped), Bemis-Murcko scaffold, and a plate-map-grouped split
  (no training compound shares a plate map with a test compound).
- Every kNN arm shares one implementation; arms differ only in the similarity matrix.
  Hyperparameters are chosen by inner group CV on the primary metric, and the grids were widened
  after a smoke run showed edge hits.
- Ceilings and probes: `oracle_knn` (neighbours by true response — the best any retrieval could
  do with this library); `xcell1_knn` (neighbours by the new drug's response in ONE other line);
  `knn_morph` (neighbours by Cell Painting profile).
- Controls: permuted targets, degree-preserving rewired PPI, identity-adjacency GNN, same-plate
  exclusion.

## 4. Results

(see §1 of `report/10_09_report.md` for the full tables with intervals)

### 4.1 Structure vs response, and library density

ACTIVE compounds (TAS ≥ 0.2), pairs of different connectivity blocks, means over lines
(job 292932):

| quantity | P1-24 | P1-6 | P2-24 |
|---|---|---|---|
| base rate P(r ≥ 0.5) | 0.018 | 0.012 | 0.018 |
| P(r ≥ 0.5 \| Morgan Tanimoto ≥ 0.85) | 0.381 | 0.270 | 0.470 |
| P(r ≥ 0.5 \| shares a target) / no shared target | 0.190 / 0.020 | 0.146 / 0.014 | 0.145 / 0.017 |
| stereoisomer pairs P(r ≥ 0.5) | 0.389 | 0.273 | 0.409 |
| rank correlation of similarity with response r (best encoder) | 0.032 | 0.024 | 0.076 |
| **active compounds with a response twin in the library** | **0.728** | **0.720** | **0.646** |
| chemical nearest neighbour is such a twin (any encoder) | 0.10-0.11 | 0.09-0.11 | 0.10-0.12 |

Applicability domain (X4, exploratory). The chemistry-kNN gain over the mean is ~0 when the
nearest training analogue has Tanimoto < 0.3, and +0.06 to +0.18 above 0.6. The oracle headroom
is 0.33-0.43 in every bin. Only 18% of active P2-24 test rows (mostly annotated drugs) have an
analogue ≥ 0.6.

### 4.2 Retrieval arms and gates (ik14 split; full-signature Pearson, ACTIVE)

| contrast | P1-24 | P1-6 | P2-24 |
|---|---|---|---|
| best chemistry kNN (inner-CV encoder choice) − mean | +0.051 | +0.054 | +0.035 |
| oracle kNN − mean (ceiling) | +0.409 | +0.421 | +0.338 |
| ChemBERTa − Morgan | −0.003* | −0.004* | −0.004* |
| MoLFormer − Morgan | +0.001 | +0.001 | −0.001 |
| MACCS − Morgan | −0.007* | −0.005* | −0.004* |
| Morgan kNN − same-plate-excluded Morgan kNN | +0.020* | +0.024* | +0.005* |
| target-Jaccard kNN − chemistry kNN (KNOWN pool) | +0.004 | +0.007* | +0.008* |
| PPI-propagated target kNN − chemistry kNN | +0.013* | +0.017* | +0.013* |
| PPI − degree-preserving rewired PPI | +0.002* | +0.003* | +0.003* |
| targets − permuted targets | +0.042* | +0.048* | +0.039* |
| chemistry + targets (kernel mean) − chemistry | +0.014* | +0.019* | +0.017* |
| one other line's L1000 profile (xcell1) − best chemistry | +0.105* | +0.098* | +0.149* |
| Cell Painting kNN − chemistry kNN (same pool) | +0.028* | +0.027* | +0.095* |

`*` = 95% ik14-group bootstrap interval excludes 0. The scaffold split gives the same ordering
(best chemistry − mean +0.039 / +0.028; ChemBERTa − Morgan −0.006 / −0.006). Chemistry captures
10-12% of the oracle headroom. Magnitude ("does the drug do anything?") is barely predictable
from chemistry: AUROC for ACTIVE from the predicted residual norm is 0.53-0.60, against
0.79-0.88 for the oracle.

### 4.3 Plate confound and plate-controlled re-estimates

At equal (negligible) chemical similarity, compounds sharing an L1000 plate are 3-4× more often
response-similar (P1-24 4.6% vs 1.4%; P2-24 5.9% vs 1.5%). Phase-II L1000 and the LINCS Cell
Painting batch share plate layouts (92.5% of same-L1000-plate pairs share a Cell Painting plate
map). Plate-controlled P2-24 (plate-map split, no training compound shares a plate map with a test
compound), P1-24 / P1-6 / P2-24: best chemistry − mean +0.023 / +0.029 / +0.028 (6-9% of the
oracle headroom); ChemBERTa − Morgan −0.006 in all three; chemistry + targets − chemistry (KRR)
+0.017 / +0.018 / +0.015; PPI − rewired (kNN) +0.0016 / +0.0016 / +0.0015; one other line −
chemistry +0.077 / +0.076 / +0.130; Cell Painting − chemistry +0.033 / +0.018 / +0.055.
Same-plate exclusion alone removes 41-44% of chemistry's ik14 gain in Phase I.

### 4.4 Trained GNN (approach 2) vs controls

The pre-specified training produced no learning in any neural arm (early stop at the first epoch);
G3b is inconclusive. With inner-validation output shrinkage (exploratory) MLPs learn but trail the
kernel methods on identical inputs (−0.005 to −0.029); the decoupled GNN still does not learn.
The informative test puts the graph inside an MLP that learns: PPI-propagated target features add
+0.012 [0.008, 0.017] over targets alone, and a degree-preserving rewired graph does as well
(real − rewired −0.001 [−0.004, +0.001]). The graph's useful content is node degree, not wiring.

### 4.5 Tahoe replication
Chemistry kNN − mean +0.017 (5 µM) / +0.021 (0.5 µM), and **−0.001 / −0.001 once same-plate
neighbours are excluded**: on Tahoe the whole chemistry-retrieval gain is plate co-membership
(drugs on a plate share its DMSO control). ChemBERTa ≤ Morgan; PPI − rewired +0.004; one other
line − chemistry +0.34 / +0.15. The Tahoe plate-map split is degenerate (2 of 5 folds tested).

## 5. My take on approach 1 vs approach 2, and what to do next (interim)

- The two approaches are the same estimator family: both smooth over drug-drug similarity. Approach
  2 learns the kernel and lets the graph shape it. What decides performance is **which information
  defines "similar"**, not retrieval vs message passing. Chemistry finds chemically similar drugs;
  response-similar drugs are mostly not chemically similar (§4.1).
- Approach 1's premise ("the pool is dense") is true in response space (≈ 70% of active compounds
  have a twin) and false in chemical space for annotated drugs (82% of active P2-24 test rows have
  no analogue at Tanimoto ≥ 0.6). Its weakness is the encoder, as suspected, but better *SMILES*
  encoders do not fix it.
- Approach 2's weakness is not PPI incompleteness: a degree-preserving random graph does almost as
  well (+0.001 to +0.004). What helps is the target annotation itself, and only for the 26-31% of
  compounds that have one.
- **The new modality worth adding is a cheap phenotypic measurement of the new drug** (one L1000
  profile in a reference line, or Cell Painting), used as the retrieval key: +0.13 and +0.055 over
  chemistry under plate control. This is the drug-side analogue of what made Rhaister-O work: a
  task-matched measured baseline beats any amount of prior knowledge. Fewest-modality story:
  structure + (optional) targets + one reference profile.
- A trained approach-2 model confirms it: graph features help an MLP, but a random graph with the
  same degrees helps equally, and learned models trail kernels at this sample size.
- Next: a reference-panel design (which 1-3 lines or one Cell Painting run make the best
  retrieval key), plate-map-held-out evaluation as the standard, and morphology-to-L1000 retrieval
  at scale (JUMP-CP), before any larger graph model.

## 6. Lessons carried over from the Rhaister-O work

| Rhaister-O lesson | applied here |
|---|---|
| hard mean baselines and residual (drug-residual) metrics | `mean` arm + residual Pearson + an explicit oracle ceiling |
| Pearson is scale-invariant; score magnitude separately | magnitude Spearman and ACTIVE AUROC reported; chemistry predicts activity barely (AUROC 0.53-0.60) |
| reference data quality matters (NCI-H596, reliability weighting) | ACTIVE cohort (TAS ≥ 0.2); inert compounds dominate L1000 |
| identity registry before modelling | standardized InChIKeys; stereoisomer families grouped in folds |
| shared-control coupling (job 258115) | the same artifact family appears as plate co-membership and decides the Tahoe result |
| reweighting beats adding parameters at small n | kNN ≈ KRR; the GNN gate tests the rest |
| random-graph controls (Reactome vs random) | PPI vs degree-preserving rewired PPI; target vs permuted targets |
| inner-CV criterion must match the endpoint; check grid edges; bit-for-bit reproduction | grids widened after edge hits; refactor verified identical to 0.0 |
| the best cell similarity is the measured untreated transcriptome | the best drug similarity is a measured response, not structure |
