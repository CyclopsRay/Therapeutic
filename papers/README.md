# Literature Library

Last reviewed: 2026-09-14. The PDFs in this directory are primary papers, not
secondary summaries. Verify them with `sha256sum -c papers/SHA256SUMS` from the
project root. “Strict zero-shot cell line” below means that no treated response
from the target line is available to model fitting or inference; an untreated
baseline from that line is allowed.

## Read first

1. **Zhang et al., Tahoe-100M** — dataset paper, bioRxiv v3 (2025),
   [`tahoe-100m_v3.pdf`](tahoe-100m_v3.pdf),
   [DOI](https://doi.org/10.1101/2025.02.20.639398). Describes the 24-hour
   Mosaic screen, 95.6 million full-QC cells, 50 assayed lines (47 used for most
   downstream analysis), 379 compounds, 1,135 drug-dose combinations, three
   doses, 14 plates, DMSO controls, and the plate-6/plate-14 replicate.
2. **Svensson et al., Rhaister / Rhaister-O** — bioRxiv preprint (2026),
   [`rhaister_2026.pdf`](rhaister_2026.pdf),
   [DOI](https://doi.org/10.64898/2026.06.09.731197),
   [model/code](https://huggingface.co/tahoebio/Rhaister). This is the closest
   prior to our primary question. Rhaister-O predicts Tahoe responses in five
   complete-cell-line splits using only target-line DMSO expression. The
   ordinary Rhaister model is a different, few-shot setting because it observes
   a drug panel in each target line.
3. **Adduri et al., State** — bioRxiv v2 (2025),
   [`state_v2.pdf`](state_v2.pdf),
   [DOI](https://doi.org/10.1101/2025.06.26.661135),
   [code](https://github.com/ArcInstitute/State). Its main Tahoe comparison is
   underrepresented-context generalization: 30% of perturbations in each of
   five test lines enter training. Its separate strict zero-shot task holds out
   a context in a query dataset after Tahoe pretraining; that is valuable, but
   it is not the same as a strict Tahoe-line benchmark.
4. **Gandhi et al., Tahoe-x1** — bioRxiv v1 (2025),
   [`tahoe-x1_v1.pdf`](tahoe-x1_v1.pdf),
   [DOI](https://doi.org/10.1101/2025.10.23.683759),
   [code](https://github.com/tahoebio/tahoe-x1). Tx1 embeddings are combined
   with State Transition and evaluated with released few-shot and strict
   zero-shot Tahoe split configurations. The paper also explicitly compares
   plate-matched and plate-agnostic control construction.
5. **Chen et al., SCALE** — arXiv v3 (2026),
   [`scale_2603.17380v3.pdf`](scale_2603.17380v3.pdf),
   [arXiv](https://arxiv.org/abs/2603.17380). Its Tahoe benchmark excludes five
   biologically distinct cell lines from training and predicts from matched
   control populations. It is therefore another direct strict-zero-shot
   comparator, although it uses one high-compute population-transport model
   rather than Rhaister-O's five released outer splits.
6. **Molina and Zhang, perturbation-response decomposition** — bioRxiv v1
   (2026), [`response-decomposition_v1.pdf`](response-decomposition_v1.pdf),
   [DOI](https://doi.org/10.64898/2026.07.24.740459),
   [code](https://github.com/xinyizhanglab/perturbation-decomposition). This is
   the most important evaluation critique: full-response scores mix global,
   perturbation, cell-line, and interaction components. Its Tahoe analysis
   reports that responses to the same compound in source lines transfer better
   than static chemistry descriptors after removal of a generic response
   template.

## Additional Tahoe models and neighboring tasks

- **Jiang et al., OCOO-T** — arXiv v1 (2026),
  [`ocoo-t_2606.12838v1.pdf`](ocoo-t_2606.12838v1.pdf),
  [arXiv](https://arxiv.org/abs/2606.12838). It uses a conditional
  flow-matching Transformer with an implicit control-population input, but its
  Tahoe evaluation follows the State protocol: the training partition still
  spans all 50 lines, while validation/test use selected conditions in five
  lines. Treat its reported result as few-shot condition completion, not a
  strict complete-line holdout.
- **Feng et al., MAP**, *Nature Machine Intelligence* (2026),
  [`map_2026.pdf`](map_2026.pdf),
  [DOI](https://doi.org/10.1038/s42256-026-01286-w),
  [code](https://github.com/MAGIC-AI4Med/MAP). MAP has strong multimodal drug
  representations, but its Tahoe context task holds out 5% of drugs per line
  across six lines. It is unseen *pairs*, not an unseen line. Its other task is
  genuinely unseen drugs.
- **Dai et al., PerturbLDM** — bioRxiv v1 (2026),
  [`perturbldm_v1.pdf`](perturbldm_v1.pdf),
  [DOI](https://doi.org/10.64898/2026.08.07.743610). It randomly holds out about
  30% of Tahoe drug-dose-line conditions while all 379 drugs, 47 lines, and
  three doses occur in both partitions. It is useful for conditional
  generation, but not evidence of unseen-line performance.
- **Xiong et al., LLM-Guided Retrieval (LGR)** — arXiv v1 (2026),
  [`lgr_2608.01734.pdf`](lgr_2608.01734.pdf),
  [arXiv](https://arxiv.org/abs/2608.01734). The paper labels one task
  “unseen-cell-line,” but LGR retrieves and averages other drug responses that
  were measured in the target line. That makes it a target-line panel/few-shot
  method, not our strict baseline-only setting. Appendix B also says 100 Tahoe
  lines although Tahoe has 50, so reproduce the exact split before relying on
  the claim.

## Virtual-cell encoder review added 2026-09-14 (online sources)

The annotated comparison, exact OOD boundaries, cell representation
alternatives and proposed architecture are in
[`../reports/VIRTUAL_CELL_ENCODERS_AND_GENERALIZATION.md`](../reports/VIRTUAL_CELL_ENCODERS_AND_GENERALIZATION.md).
No additional PDFs or weights were downloaded in this review.

- [MultiDCP, PLOS Computational Biology 2022](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010367):
  drug/gene/baseline-cell/dose design; S10 describes line-disjoint response
  and autoencoder partitions. Pipeline not reproduced.
- [XPert, Nature Machine Intelligence 2026](https://www.nature.com/articles/s42256-025-01165-w):
  baseline gene tokens, chemical/graph priors, L1000 cold-cell/cold-drug tests
  and explicit delta prediction. Not a Tahoe benchmark result.
- [PrePR-CT, Nature Machine Intelligence 2026](https://www.nature.com/articles/s42256-026-01202-2):
  control-derived context graph; inspect target-control availability in its
  cross-dataset Tahoe comparison.
- [TxPert, Nature Biotechnology 2026](https://www.nature.com/articles/s41587-026-03113-4):
  graph-conditioned genetic perturbations and line transfer; genetic
  intervention evidence remains distinct from chemical transfer.
- [chemCPA, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract.html):
  chemistry-conditioned unseen-compound prediction.
- [SapBERT, NAACL 2021](https://aclanthology.org/2021.naacl-main.334/),
  [LangCell, ICML 2024](https://proceedings.mlr.press/v235/zhao24u.html),
  [CellWhisperer, Nature Biotechnology 2025](https://www.nature.com/articles/s41587-025-02857-9):
  entity alignment versus cell/transcriptome–language alignment; distinct
  endpoints from perturbation prediction.
- [scFoundation, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02305-7)
  and [BulkFormer, Cell Systems 2026](https://doi.org/10.1016/j.cels.2026.101657):
  bulk-input candidate encoders; see the synthesis for assay and endpoint
  limitations, including BulkFormer's repository correction.

## Dataset foundations (online full text)

- **Subramanian et al.**, “A Next Generation Connectivity Map: L1000 Platform
  and the First 1,000,000 Profiles,” *Cell* (2017),
  [DOI](https://doi.org/10.1016/j.cell.2017.10.049),
  [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5990023/). L1000
  directly measures 978 landmark transcripts and uses those to infer much of
  the remaining transcriptome.
- **Ghandi et al.**, “Next-generation characterization of the Cancer Cell Line
  Encyclopedia,” *Nature* (2019),
  [DOI](https://doi.org/10.1038/s41586-019-1186-3),
  [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6697103/). Describes
  the baseline molecular profiles that underpin current DepMap/CCLE use.
- **Boehm and Golub et al.**, “The present and future of the Cancer Dependency
  Map,” *Nature Reviews Cancer* (2025),
  [DOI](https://doi.org/10.1038/s41568-024-00763-x). Current conceptual review
  and the citation requested by the DepMap portal.

The two PMC papers are linked rather than stored because automated PMC PDF
delivery returned browser-challenge HTML on this host; no invalid placeholder
files are retained.

## Unseen-drug branch: drug representations, similarity, network priors (added 2026-10-04)

Literature consulted for the L1000/Tahoe unseen-drug branch (`report/10_09_*`,
`reports/UNSEEN_DRUG_REPRESENTATION_RETRIEVAL_AND_GRAPHS.md` §2). No PDFs stored. "Checked" says
how far the claim was verified: *full text* / *abstract* (read from the source) / *summary*
(search-engine or fetch summary only; re-read before quoting numbers) / *secondary* (known only
through another paper's citation). Claims that failed verification are listed in that report's
§2.5 and must not be cited.

### Drug encoders and molecular-representation benchmarks

- **Praski, Adamczyk & Czech**, "Benchmarking Pretrained Molecular Embedding Models For Molecular
  Representation Learning", [arXiv 2508.06199](https://arxiv.org/abs/2508.06199) (v1 2025-08, v4
  2026-02). 25 models × 25 datasets: nearly all neural embeddings give negligible or no gain over
  ECFP; only CLAMP (fingerprint-based) is significantly better. Property prediction, not
  transcriptional response. Checked: abstract (v1, v4).
- **Deng et al.**, "A systematic study of key elements underlying molecular property prediction",
  *Nat Commun* 14:6395 (2023), [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10575948/).
  62,820 models; representation learning shows limited gains over fixed representations on most
  datasets. Checked: summary.
- **Pham et al., DeepCE**, *Nat Mach Intell* (2021), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8009091/),
  [code](https://github.com/pth1993/DeepCE). Molecular-graph GCN + drug-gene attention; gene
  features are STRING embeddings (~19k nodes, ~12M edges); Pearson 0.4907 vs kNN 0.3903 on a
  pert_id-disjoint L1000 split; gene features never ablated. Our reproduction (`repro/deepce/`,
  `report/10_09_report.md` §5, 2026-10-04): 0.455 ± 0.026 over 3 seeds with best-dev selection
  (paper 0.4907 not reached); STRING vectors no better than random vectors and below a one-hot
  gene code (−0.011); all variants ≈ a tuned Morgan kNN (0.451); the paper's NN/kNN baselines
  (~0.39) equal a per-line mean. Checked: full text (main) and code.
- **Hetzel et al., chemCPA**, NeurIPS 2022 ([arXiv](https://arxiv.org/abs/2204.13545)): pretrained
  graph encoders not consistently better than RDKit features. Checked: summary.
- **Agarwal & Bisht**, "The Metric Picks the Winner", arXiv 2606.12639 (2026), DRUG-seq: a Morgan
  linear baseline wins under standard metrics; the deep ChemBERTa fusion wins only under the
  contest metric. Checked: earlier session (report §2.1).
- **Szałata et al., Chem-PerturBridge**, [arXiv 2605.31522](https://arxiv.org/html/2605.31522v1)
  (2026): perturbation-pretrained compound embeddings beat Morgan modestly on an OP3 compound
  re-split (Spearman 0.264 → 0.275); no mean/kNN baseline, split grouping not described.
  Checked: full text (fetch summary).
- Other perturbation models with drug encoders (MultiDCP, TranSiGen, PRnet, PertDiff, XPert, MAP):
  see the report's §2.1 table and the virtual-cell section above.

### Chemical similarity versus activity or response

- **Martin, Kofron & Traphagen**, "Do structurally similar molecules have similar biological
  activity?", *J Med Chem* 45:4350 (2002), PMID 12213076. Daylight fingerprints: only 30% of
  compounds ≥ 0.85 Tanimoto to an active are active. Checked: abstract.
- **Landrum**, "Thresholds for 'random' in fingerprints the RDKit supports", [RDKit blog](https://greglandrum.github.io/rdkit-blog/posts/2021-05-18-fingerprint-thresholds1.html)
  (2021, updated 2025). Morgan2 bit-vector noise level: 95% of random ChEMBL pairs < 0.185,
  99% < 0.226 (counts: 0.248 / 0.297). Checked: full text.
- **Sirci et al.**, *npj Syst Biol Appl* (2017), CMap build 02: structural and transcriptional
  similarity largely independent. **Iorio et al.**, *PNAS* (2010): transcriptional drug network
  communities enriched for shared MOA. Checked: earlier session (report §2.2).

### Networks, PPI and graph models

- **Szklarczyk et al.**, "The STRING database in 2025: protein networks with directionality of
  regulation", *NAR* 53:D730 (2025), [doi:10.1093/nar/gkae1113](https://doi.org/10.1093/nar/gkae1113).
  v12.5 offers functional, physical and a new directed regulatory network. Checked: summary.
- **Szklarczyk et al.**, "The STRING database in 2023: protein-protein association networks and
  functional enrichment analyses for any sequenced genome of interest", *NAR* 51:D638 (2023),
  [PMC9825434](https://pmc.ncbi.nlm.nih.gov/articles/PMC9825434) — the v12.0 release we use
  (`raw/string_v12`, downloaded 2026-10-03, SHA-256 in `raw/annotations/SHA256SUMS_2026-10-03.txt`).
  STRING 2021 (*NAR* 49:D605, [PMC7779004](https://pmc.ncbi.nlm.nih.gov/articles/PMC7779004)): v11.5
  introduced the physical subnetwork. Scoring ([FAQ](https://string-db.org/help/faq/)): seven
  evidence channels, combined after removing and re-adding a random-pair prior. Checked: summary.
  Local counts (human v12.0): 19,699 proteins; functional 6.86 M undirected pairs (≥ 700: 236,930);
  physical 738,805 (≥ 400: 210,667).
- **Gillis & Pavlidis**, "The impact of multifunctional genes on 'guilt by association' analysis",
  *PLoS ONE* (2011), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3041792): node degree alone
  reaches AUC 0.63 vs 0.70 for the full PPI method; Spearman 0.96 between the two across tasks.
  Checked: full text (fetch summary). Companion: "'Guilt by association' is the exception rather
  than the rule in gene networks", *PLoS Comput Biol* (2012),
  [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3315453/). Checked: summary.
- **Evans et al., GSNN**, "Graph Structured Neural Networks for Perturbation Biology", bioRxiv
  (2024), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10925270/). Drug → protein (OmniPath) →
  TF → RNA (DoRothEA) → L1000 gene; split = unseen (drug, line) pairs. GSNN 0.51-0.54 vs NN
  0.47-0.52; GCN/GAT/GIN 0.06-0.26; true graph +24% over a random graph that does not preserve
  degree. Checked: full text (fetch summary).
- PLOS ONE (2023), [doi:10.1371/journal.pone.0281286](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0281286):
  cites Bertin et al. (2019; random networks nearly as good as biological ones for expression
  prediction) and Crawford et al. (2020; biological better after removing low-degree genes).
  Checked: secondary for Bertin and Crawford.
- **Isik et al.**, "Drug target prioritization by perturbed gene expression and network
  information", *Sci Rep* (2015), [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4663505/):
  drug targets are usually not differentially expressed; network neighbourhoods carry the signal.
  Checked: summary.
- **Nilsson et al., LEMBAS**, *Nat Commun* (2022), [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9163072/):
  RNN constrained by a signaling network, ligands → TF activities, r ≈ 0.8 under CV on 59 ligands
  in macrophages. Checked: summary.
- **CARNIVAL** ([Bioconductor](https://code.bioconductor.org/browse/CARNIVAL/)) and **CollecTRI**
  ([GitHub](https://github.com/saezlab/CollecTRI)): signed directed signaling + TF regulons for
  causal reasoning from expression footprints. Checked: summary.
- **TxPert** ([arXiv 2505.14919](https://arxiv.org/abs/2505.14919v1)): fuses STRING, GO and
  Recursion's PxMap/TxMap with multi-graph GNNs, genetic perturbations. Checked: summary.
- GEARS, AblateCell, PDGrapher, PrimeKG, DrugBank: see report §2.3.

- **Wong, Hill & Moccia**, "Simple controls exceed best deep learning algorithms and reveal
  foundation model effectiveness for predicting genetic perturbations", *Bioinformatics* 41(6)
  btaf317 (2025), [link](https://academic.oup.com/bioinformatics/article/41/6/btaf317/8142305):
  a CRISPR-informed mean beats GEARS and scGPT on three Perturb-seq sets; no graph ablation.
  Checked: summary.
- **Bertin et al.**, "Is graph-based feature selection of genes better than random?" (MLCB 2019,
  [arXiv 1910.09600](https://arxiv.org/abs/1910.09600)) and "Analysis of gene interaction graphs as
  prior knowledge for machine learning models" (2019): gene-expression dependencies are captured
  almost as well by random graphs as by gene interaction graphs. Checked: summary.
- **Caranzano et al.**, "Sparsity is All You Need: Rethinking Biological Pathway-Informed
  Approaches in Deep Learning", [arXiv 2505.04300](https://arxiv.org/abs/2505.04300) (2025): 15
  pathway-informed models; randomized counterparts with the same sparsity perform equally well, 3/15
  better. Closest prior art for a "priors ≈ random" claim (pathway masks, not perturbation response).
  Checked: abstract.
- **Chandak, Huang & Zitnik, PrimeKG**, *Sci Data* 10:67 (2023), [PMC9893183](https://pmc.ncbi.nlm.nih.gov/articles/PMC9893183/):
  PPI edges are the unweighted, undirected union of the Menche et al. interactome (TRANSFAC, MINT,
  IntAct, CORUM), BioGRID, STRING and HuRI; 642,150 directed = 321,075 undirected edges. Checked:
  full text (fetch summary).
- Audit candidates found 2026-10-05 (code status to confirm): **GSNN** code at
  [github.com/nathanieljevans/GSNN](https://github.com/nathanieljevans/GSNN) (docs on readthedocs);
  **Passigan, Zhu & Ning**, "Interpretable Perturbation Modeling Through Biomedical Knowledge
  Graphs", [arXiv 2512.22251](https://arxiv.org/abs/2512.22251) (PrimeKG++ with L1000 drug and
  line nodes, GAT on 978-landmark deltas, scaffold/random splits; reports that edge shuffling hurts;
  code not stated); **PrePR-CT** ([bioRxiv](https://www.biorxiv.org/content/10.1101/2024.07.24.604816.full.pdf);
  cell-type co-expression graphs for chemical perturbations); **X-Pert** (docs at
  x-pert.readthedocs.io). The DOI mapping of XPert / X-Pert / PrePR-CT to *Nat Mach Intell* articles
  in this file is inconsistent across sources; re-check before citing. Checked: summary.
- Novelty check 2026-10-05 (about ten web searches): "priors ≈ random" has precedent for gene
  function prediction (Gillis & Pavlidis: degree), expression feature graphs (Bertin) and
  pathway-informed networks (Caranzano: sparsity). No paper found that audits published PPI/KG
  perturbation-response models (chemical or genetic) with degree-preserving controls; GSNN's random
  control does not preserve degree. Not a systematic review; redo systematically before submission.

### New drug-side modalities

- Way et al. *Cell Systems* 2022 (LINCS Cell Painting + L1000), Haghighi et al. *Nat Methods*
  2022, CLOOME (*Nat Commun* 2023), MolPhenix (NeurIPS 2024), InfoAlign (ICLR 2025), Hodos et al.
  PSB 2018: see report §2.4. Checked: earlier session.

### Modelling methodology

- **Rendle, Krichene, Zhang & Anderson**, "Neural Collaborative Filtering vs. Matrix Factorization
  Revisited", RecSys 2020, [arXiv 2005.09683](https://arxiv.org/pdf/2005.09683): an MLP over
  concatenated embeddings learns a dot product poorly; a plain dot product wins. Motivates the
  dot-product head. Checked: summary.
- Ahlmann-Eltze, Huber & Anders, *Nat Methods* 2025, and Systema (Viñas Torné et al., *Nat
  Biotechnol* 2025): see report §2.1.
