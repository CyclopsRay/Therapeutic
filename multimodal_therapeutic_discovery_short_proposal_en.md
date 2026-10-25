# Semi-Supervised Multimodal Perturbation Modeling for Therapeutic Discovery

## Short Project Proposal for Discussion

### Motivation

Predicting how a cancer cell responds transcriptionally to a drug is a useful step toward therapeutic discovery, but perturbation datasets cover only a limited number of cellular contexts. Tahoe-100M provides a large single-cell perturbation resource, yet its experiments span only about 50 cancer cell lines. This project asks whether baseline molecular information can help a perturbation model transfer to cell lines that were not used for training.

This is intended as an exploratory course project. A complete data pipeline, a working baseline, and interpretable preliminary results would already constitute a successful outcome, even if the proposed method does not outperform the baseline. Promising results could motivate a larger follow-up project.

### Research Question

Can a semi-supervised multimodal model use baseline cell-line information to predict drug-induced gene-expression responses in held-out cancer cell lines?

For each cell line–drug–dose condition, the model will use:

- baseline gene expression;
- drug identity embedding;
- dose information;
- (additional DepMap features such as mutation, copy-number variation, and gene dependency, if they are sufficiently complete and biologically aligned).

The model will jointly predict:

1. post-treatment expression; and
2. expression change relative to baseline.

The primary objective is generalization to held-out cell lines. Generalization to unseen molecules is not a core claim because the initial model uses drug identity rather than chemical structure.

### Data

**Tahoe-100M — primary supervised dataset.** Tahoe contains more than 100 million perturbed cells across approximately 50 cancer cell lines, 379 compounds, three dose levels, and a 24-hour treatment condition. It will provide the labeled baseline-to-treatment examples. We will work with a restricted, biologically meaningful gene space rather than the full transcriptome. (Aggregate single-cell profiles into one pseudobulk signature per cell line–drug–dose condition.)

**DepMap — baseline representation data.** DepMap will primarily provide untreated cell-line expression profiles for representation pretraining and cell-context features. (Add mutation, CNV, and dependency modalities after checking coverage and overlap with Tahoe cell lines.)

**Optional supporting datasets.** (Use L1000 as weakly labeled pretraining data or as a secondary robustness check if gene, drug, and cell-line mappings are manageable.) (Use HTAN9 as a small pathway-level clinical case study for breast cancer and palbociclib.) HTAN9 should not be treated as a strict external perturbation benchmark because its longitudinal biopsies reflect combination therapy, long-term resistance, changing tumor composition, and clinical progression. (Reconsider HTA5 only if downloadable metadata reveals well-matched treatment and expression samples.)

### Proposed Method

The main model will contain:

- a cell encoder for baseline expression and available cell features;
- a learned drug-identity embedding and dose encoding;
- a cell–drug interaction module;
- two linked prediction heads for post-treatment expression and expression change.

Training will be staged:

1. Pretrain the cell encoder on untreated DepMap profiles using masked-feature reconstruction.
2. Fine-tune the full model on labeled Tahoe perturbations.
3. Encourage consistency between the two outputs, so that predicted post-treatment expression is approximately equal to baseline expression plus predicted change.

(Compare masked reconstruction with multimodal alignment, consistency regularization, pseudo-labeling, or L1000 weak-label pretraining if time permits.)

### Baseline and Evaluation

The initial baseline will be a conditional multilayer perceptron that concatenates baseline expression, drug embedding, and dose, and directly predicts expression change.

The primary evaluation will hold out several complete cell lines during training and repeat this across a small number of cross-validation splits. A random condition-level split can be included as a sanity check. Evaluation will use expression correlation and error, together with differential-expression or pathway-level agreement. We will also examine whether pretraining helps more for held-out cell lines than for cell lines already represented during training.

(For the HTAN9 case study, compare tumor-intrinsic pathway directions—such as cell-cycle/E2F or related resistance programs—rather than raw gene-level prediction error.)

### Expected Contribution and Minimum Deliverable

The methodological contribution is an exploratory semi-supervised multimodal perturbation architecture designed to transfer across cellular contexts. The minimum course deliverable is:

- a reproducible Tahoe preprocessing and training pipeline;
- one conditional MLP baseline;
- a staged DepMap-to-Tahoe pretraining experiment;
- held-out cell-line results with error analysis;
- (one optional L1000 or HTAN9 analysis).

### Points to Decide Together

- (Use pseudobulk signatures or retain a single-cell training objective.)
- (Select highly variable genes, landmark genes, or a curated pathway-focused gene set.)
- (Which DepMap modalities are complete enough to include beyond baseline expression.)
- (Whether L1000 adds useful supervision relative to its platform mismatch.)
- (Whether the HTAN9 palbociclib case study is feasible within the course timeline.)

### Key Resources

- [Tahoe-100M dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M)
- [DepMap](https://depmap.org/portal/)
- [HTAN9 project](https://data.humantumoratlas.org/center/hta9)
- [HTAN9 longitudinal CDK4/6 inhibitor study](https://doi.org/10.1101/2023.09.27.557464)
