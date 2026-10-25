# Tahoe-100M, DepMap, and LINCS L1000: Bilingual Experimental Synthesis

Last updated: 2026-09-09

---

# 中文版

## 摘要

本项目要回答的核心问题是：对于一个完全没有 Tahoe treatment response
训练数据的新癌细胞系，能否根据已知 drug、dose 和该细胞系的 baseline
状态，预测 24 小时后的表达变化？

我们依次完成了三条相互补充、但信息边界不同的实验线：

1. **Tahoe / Rhaister-O**：严格 held-out cell line 的主 benchmark；
2. **DepMap expression**：测试独立 baseline molecular context 是否能提供
   Rhaister-O 之外的增量信息；
3. **LINCS L1000**：作为外部 perturbation atlas，测试相同 cell line、相同
   compound、近似 dose、相同 24h 时间点的响应方向能否跨平台复现。

总体结论是：

- Rhaister-O 在完整 cell-line holdout 下表现很强，而且信号在去掉 drug
  mean 和一个训练集拟合的 generic residual component 后仍然存在。
- DepMap expression 确实含有 cell-specific response 信息，但不能稳定修正
  Rhaister-O 的剩余误差，因此没有理由直接扩展 CNV、mutation、dependency
  或 MLP。
- L1000 与 Tahoe 存在**很弱的 cross-platform concordance**。Phase I 的结果
  在 bootstrap 和 drug-label permutation 下成立，Phase II 没有在 primary
  measured-landmark Pearson 上复现。L1000 目前更适合作为 pathway/direction
  层面的外部证据，而不是直接替代 Tahoe prediction。

## 三个数据集在项目中的角色

| 数据 | 本项目中的角色 | 主要信息 | 不能混淆的限制 |
|---|---|---|---|
| Tahoe-100M | 监督训练和严格 unseen-line 测试 | 50 条实际 assayed lines、379 compounds、三个非零剂量、24h、单细胞聚合 delta | 主任务是 unseen cell line，不是 random condition holdout |
| DepMap Public 24Q4 | 独立 baseline context | bulk expression，以及尚未启用的 CNV/mutation/dependency | 没有 perturbation response；target baseline 可用但 target treatment outcome 不可用 |
| LINCS L1000 Phase I/II | 外部 perturbation 验证 | Level-5 signatures；978 landmark transcripts 直接测量，其余为推断 | bulk/平台/dose/cell coverage 与 Tahoe 不同，不能当作完全等价标签 |

相关数据与模型论文见 [`papers/README.md`](../papers/README.md)，包括
Tahoe-100M、Rhaister/Rhaister-O、State、Tahoe-x1、SCALE、response
decomposition、DepMap/CCLE 和 Connectivity Map L1000。

## 信息边界与评价原则

主任务是 strict zero-shot cell-line generalization：

- outer-test line 的所有 Tahoe treated responses 都从训练、PCA、模型选择和
  prediction 中排除；
- 可以使用 target line 的 DMSO baseline 或外部 DepMap baseline；
- drug 和 dose 已在 reference lines 中观测；
- 所有 response-derived preprocessing 都只在 reference lines 上拟合；
- Pearson 在每个 `(line, drug, dose)` 条件的基因维度上计算，然后按 line
  macro-average；
- bootstrap 以 drug 或 held-out line 为单位，而不是把单细胞当作独立样本。

L1000 实验是 external validation，不是 strict-zero-shot training
augmentation。我们没有把 held-out target line 的 L1000 treatment signature
输入 Tahoe 模型。

## 实验 1：复现 Rhaister-O 严格 unseen-line baseline

使用固定代码 commit `75fed20a1d97b05a09bf25c5341762141882106a`、固定
Tahoe derivative revision `c7963cf334bec0683225d41c9586d900ca6303a2` 和五个
released outer splits，我们在 25 条 distinct held-out lines、19,268 个测试条件、
固定 2,000 genes 上复现了 Rhaister-O expression-delta 路径。

| 模型 | Full-delta Pearson |
|---|---:|
| Zero delta | 0.0000 |
| Global gene mean | 0.1880 |
| Drug-dose mean | 0.4434 |
| Shared gene slope | 0.4893 |
| **Rhaister-O treatment-specific slope** | **0.6315 ± 0.0680 across splits** |

这个结果匹配论文/代码发布的约 0.63。Rhaister-O 相对 drug-dose mean 提高约
0.188 Pearson，并使 RMSE/MAE 各下降约 15.4%。

## 实验 2：确认 0.63 中有多少是 cell-specific signal

我们对 25 条 held-out lines 分别加入 2,000 次 drug-cluster bootstrap，并在
三个层级评价 frozen predictions：

| Evaluation level | Drug mean | DMSO 1NN | DMSO 3NN | Rhaister-O |
|---|---:|---:|---:|---:|
| Full delta | 0.4434 | 0.2985 | 0.3405 | **0.6314** |
| Drug-mean residual | 0.0000 | 0.1472 | 0.1388 | **0.5380** |
| Reference-only PC1 removed | -0.0028 | 0.1282 | 0.1174 | **0.5259** |

主要发现：

- 去掉 reference drug mean 后，Rhaister-O 仍有 0.5380，因此 headline score
  并非主要由所有 cell lines 共享的 drug response 构成。
- generic residual PC1 只解释每个 split 约 4.90–5.49% 的 residual variance；
  去掉后仍有 0.5259。
- 所有 25 条线的 Rhaister-O drug-cluster bootstrap interval 在三个层级都高于
  0，但 line heterogeneity 很大：PC1-removed Pearson 为 0.0648–0.6850。
- fixed DMSO 1NN/3NN 偶尔有用，但不能稳定传递 treatment-specific response。

这说明 Rhaister-O 的主要增益来自 treatment-specific baseline-to-response
mapping，而不是简单地复制相似细胞系或 generic stress template。

## 实验 3：DepMap expression 是否能预测 residual

### 3A. 独立预测 drug-mean residual

每个 outer split 内仅用 reference-line DepMap expression 拟合 scaling、PCA 和
ridge，并通过 reference-line LOOCV 选择 0/2/5/10 PCs 和 lambda 1/10/100。
Public 24Q4 expression 覆盖 23/25 held-out lines；C3A 和 COLO 205 明确标记为
unavailable，没有静默插补。

| Metric level | DepMap ridge | Rhaister-O（同一23-line cohort） |
|---|---:|---:|
| Full delta | 0.4732 | 0.6344 |
| Drug-mean residual | **0.2087** | **0.5394** |
| PC1 removed | 0.1803 | 0.5271 |

DepMap drug-residual Pearson 的 23-line bootstrap interval 为
`[0.1591, 0.2601]`，高于 permutation 95th percentile `0.0304`。因此
DepMap expression 的确携带可预测的 cell context，但比 task-matched 的
Rhaister-O 弱很多。

### 3B. DepMap 能否修正 Rhaister-O

为了直接测试增量价值，我们在每个 outer reference set 内先构建
leave-one-line-out Rhaister-O prediction，然后用
`truth - cross-fitted Rhaister-O` 作为 DepMap correction target，最后把
correction 加到 frozen outer-test Rhaister-O。

| Evaluation level | Frozen Rhaister-O | + DepMap correction | Improvement | 95% line bootstrap |
|---|---:|---:|---:|---:|
| Full delta | 0.634447 | 0.636036 | +0.001589 | [-0.000286, 0.003819] |
| Drug residual | 0.539430 | 0.539787 | **+0.000357** | **[-0.002679, 0.003610]** |
| PC1 removed | 0.527100 | 0.527491 | +0.000392 | [-0.002850, 0.003983] |

23 条线中只有 10 条 point estimate 提高；primary bootstrap interval 跨 0。
虽然实际 alignment 高于小规模 permutation null，但预先规定的稳定性 gate
失败。因此我们停止了 DepMap CNV、mutation、dependency 和 MLP 扩展。

## 实验 4：LINCS L1000 与 Tahoe 的同步外部对照

此前只有 L1000 数据下载和 crosswalk，没有表达签名实验。本轮首次完成了
小型 cross-platform validation。

### 匹配规则

- exact curated shared cell line；
- 只保留 `structure_validated=True` 的 compound crosswalk；
- LINCS `trt_cp` Level-5 signatures；
- 24h exactly；
- 对每个 Tahoe dose 选择最近的 LINCS dose，且 ratio 不超过 2；
- 同一 line/drug/dose 的多个 Level-5 signatures 取均值；
- primary panel：同时位于 Tahoe/Rhaister 2K panel 的 **79 个直接测量
  landmarks**；
- secondary panels：1,076 BING genes 和全部 1,365 distributed overlaps，
  其中大量基因是 L1000 推断值。

24h 条件最终只覆盖 A549、HT-29 和 HS-578T。Phase I 有 171 个 Tahoe 对齐
条件、97 drugs、199 unique L1000 signatures；Phase II 有 304 个条件、78
drugs、332 signatures。很多 dose 是近似匹配而非完全一致，例如 Tahoe 5 µM
常对应 LINCS 10 µM（Phase I）或 3.33 µM（Phase II）。

### Primary 79-landmark result

| L1000 release | Conditions / drugs / cells | Drug-macro Pearson | 95% drug bootstrap | Drug-permutation 95th | p |
|---|---:|---:|---:|---:|---:|
| Phase I / GSE92742 | 171 / 97 / 3 | **0.0605** | **[0.0360, 0.0859]** | 0.0413 | 0.0020 |
| Phase II / GSE70138 | 304 / 78 / 3 | 0.0153 | [-0.0041, 0.0351] | 0.0172 | 0.1078 |

Phase I 的 Spearman 为 0.0524、sign agreement 为 0.5191；Phase II 的
Spearman 为 0.0285、sign agreement 为 0.5079。Phase I 的弱正相关在 drug
bootstrap 和 within-cell/dose drug-label permutation 下都成立，说明并非完全
由 generic stress response 造成。但 Phase II primary Pearson 没有复现，不能
声称稳定的跨平台 gene-level reproduction。

在两个 Phase 都有的 31 个 line/drug/dose 条件、13 个 drugs 上，Phase-I 对
Phase-II 的 79-landmark Pearson 平均为 0.2025。也就是说两个 L1000 releases
彼此有一定一致性，但它们与 Tahoe 的一致性仍然弱，提示平台、处理批次、
dose formulation 和 response construction 都在影响结果。

### 与 frozen Rhaister 的完全对齐子集

在 25 条 Rhaister held-out lines 中，24h L1000 exact-line cohort 只有 HS-578T
与 released split 7 重合。因此这个表只能作为单线 case study：

| Release | Conditions / drugs | L1000 | Drug mean | Rhaister-O |
|---|---:|---:|---:|---:|
| Phase I, 79 landmarks | 33 / 11 | 0.0577 | 0.6814 | **0.8437** |
| Phase II, 79 landmarks | 16 / 6 | 0.0408 | 0.6822 | **0.8597** |

这里三者都在相同 Tahoe truth conditions 和相同 79 genes 上评价。L1000 是
外部 bulk signature，不是为 Tahoe 拟合的 predictor，所以这个数量级差异不
应解释成公平的模型 leaderboard；它说明“直接复制跨平台 signature”不足以
预测 Tahoe response。

## 综合解释

三条实验线给出一个连贯的图景：

1. **Tahoe 内部、task-matched baseline 最强。** Rhaister-O 使用 target-line
   DMSO expression，并对每个 treatment/gene 学习 response-specific slope，
   因而能恢复大量 cell-specific structure。
2. **静态 external context 有信号但大多冗余。** DepMap expression 可预测
   drug-mean residual，却不能稳定预测 Rhaister-O error。
3. **外部 perturbation response 可以提供方向证据，但平台转移很弱。** L1000
   Phase I 显示小的 drug-specific concordance，Phase II primary test 不稳定；
   直接 signature transfer 远不足以取代 Tahoe-trained model。
4. **negative result 仍有科学价值。** 它缩小了真正需要解释的部分：不是
   “能否预测 unseen line”，而是哪些 line/drug interactions 在 task-matched
   DMSO model 后仍然不可预测，以及外部数据何时能补充这些误差。

## 接下来可以做什么

建议按成本和证据强度排序：

1. **先分析 frozen predictions，不训练新模型。** 比较 DepMap correction
   帮助的 8 条和伤害的 13 条 lines，按 lineage、DMSO similarity、dose、
   response magnitude 和 drug/MOA 分层。
2. **对现有 L1000 子集做 dose/pathway sensitivity。** 分开 exact/tight dose
   与 2-fold boundary；在 measured landmarks 和 BING panel 上做 gene-set
   direction，而不是追求精确 gene-level magnitude。
3. **寻找 Phase-I-positive、Phase-II-consistent 的 drug families。** 只把两个
   Phase 与 Tahoe 同向的 compounds 作为候选假设，并进行 multiple-testing
   控制；不要根据 Phase I 单独挑最好看的 drugs。
4. **若要测试 L1000 auxiliary prior，先做最小模型。** 对每个 held-out Tahoe
   line，完全排除该 line 的 L1000 treatment signatures，只用其他 L1000 lines
   构建 drug-level prior，再在 inner line-CV 中测试其是否改善 frozen
   Rhaister residual。必须重新设置 bootstrap/permutation gate。
5. **暂不做 multimodal MLP。** DepMap correction gate 已失败，L1000 也只显示
   很弱且 phase-dependent 的 concordance。先证明一个低容量 linear/stacked
   prior 有稳定增益，再考虑 CNV/dependency/MLP。
6. **若目标是论文/课程项目，当前结果已能形成完整故事。** 贡献可以是严格
   unseen-line reproduction、response-component audit、external-context negative
   gate，以及 L1000 cross-platform reproducibility analysis，而不必强行得到
   一个更复杂且更高分的模型。

## 产物与可复现性

| 内容 | 位置 |
|---|---|
| Rhaister/DepMap ordered report | [`ORDERED_BASELINE_EXPERIMENTS.md`](ORDERED_BASELINE_EXPERIMENTS.md) |
| L1000 runner | `scripts/run_lincs_tahoe_concordance.py` |
| L1000 Slurm wrapper | `slurm/run_lincs_tahoe_concordance.sbatch` |
| L1000 unit tests | `tests/test_lincs_tahoe_concordance.py` |
| Extracted compact signatures | `/nfs-stor/yuqi.lei/Therapeutic/derived/lincs_tahoe_concordance` |
| L1000 result tables and manifest | `/nfs-stor/yuqi.lei/Therapeutic/results/lincs_tahoe_concordance` |

Final L1000 job `218460` completed with exit `0:0` in 22 seconds using cached
extractions. Initial job `218448` performed the one-time GCTX extraction and was
cancelled after an inefficient repeated-NPZ read was detected; intermediate job
`218459` was superseded because its condition archives contained object-dtype
labels. The final four NPZ archives all load with `allow_pickle=False`.

Final L1000 summary SHA-256:
`f475f20679dbc9724cd2beccce353b7db99a668207e7760938009562c0036aaa`.

---

# English Version

## Executive summary

The central question is whether a model can predict the 24-hour transcriptional
response to a known drug and dose in a cancer cell line with no Tahoe treated
responses available during training.

We completed three complementary experimental tracks:

1. **Tahoe / Rhaister-O** as the strict held-out-cell-line benchmark;
2. **DepMap expression** as independent baseline molecular context; and
3. **LINCS L1000** as an external perturbation atlas for cross-platform
   direction and rank validation.

The combined conclusion is that Rhaister-O is strong and retains substantial
signal after removing the drug mean and a training-derived generic component.
DepMap expression contains genuine cell-context information but does not
stably correct Rhaister-O. L1000 has weak cross-platform concordance: the
Phase-I primary result is bootstrap- and permutation-supported, while the
Phase-II primary Pearson result does not replicate it. L1000 is therefore more
defensible as external direction/pathway evidence than as a direct Tahoe
response substitute.

## Dataset roles

| Dataset | Role in this project | Main information | Important limitation |
|---|---|---|---|
| Tahoe-100M | Supervised response training and strict unseen-line evaluation | 50 assayed lines, 379 compounds, three nonzero doses, 24-hour single-cell-derived deltas | The primary task is a complete line holdout, not random condition completion |
| DepMap Public 24Q4 | Independent baseline context | Bulk expression; CNV, mutation, and dependency are staged but gated off | It has no matched perturbation outcomes |
| LINCS L1000 Phase I/II | External perturbation validation | Level-5 signatures; 978 landmark transcripts are directly measured | Bulk assay, dose, platform, and cell coverage differ from Tahoe; non-landmarks are inferred |

The corresponding primary literature is indexed in
[`papers/README.md`](../papers/README.md).

## Information boundary and evaluation

The primary Tahoe task is strict zero-shot cell-line generalization. Every
treated response from an outer-test line is excluded from preprocessing,
fitting, and model selection. Target-line DMSO or an external baseline profile
may be used; the drug-dose is observed in reference lines. Pearson is computed
across genes for each condition and then macro-averaged by held-out line.
Bootstrap units are drugs or held-out lines, never individual cells.

The L1000 experiment is external validation, not training augmentation. No
target-line L1000 treatment response is supplied to a Tahoe model.

## Experiment 1: Rhaister-O reproduction

Using the pinned Rhaister commit, data revision, fixed 2,000-gene panel, and
five released outer splits, we evaluated 25 distinct held-out lines and 19,268
test conditions.

| Model | Full-delta Pearson |
|---|---:|
| Zero delta | 0.0000 |
| Global gene mean | 0.1880 |
| Drug-dose mean | 0.4434 |
| Shared gene slope | 0.4893 |
| **Rhaister-O treatment-specific slope** | **0.6315 ± 0.0680 across splits** |

This matches the released result of approximately 0.63. Rhaister-O improves
Pearson by about 0.188 over the drug-dose mean and reduces both RMSE and MAE by
approximately 15.4%.

## Experiment 2: response-component and neighbor controls

| Evaluation level | Drug mean | DMSO 1NN | DMSO 3NN | Rhaister-O |
|---|---:|---:|---:|---:|
| Full delta | 0.4434 | 0.2985 | 0.3405 | **0.6314** |
| Drug-mean residual | 0.0000 | 0.1472 | 0.1388 | **0.5380** |
| Reference-only PC1 removed | -0.0028 | 0.1282 | 0.1174 | **0.5259** |

Rhaister-O remains strong after removal of the transferable drug program. The
leading generic residual component explains only 4.90–5.49% of reference
residual variance, and removing it leaves a 0.5259 Pearson result. All 25
per-line drug-cluster intervals remain positive, although PC1-removed line
performance ranges from 0.0648 to 0.6850. Fixed DMSO 1NN/3NN transfer is
heterogeneous and does not approximate the treatment-specific mapping.

## Experiment 3: DepMap-expression residual models

The standalone reference-only PCA/ridge model predicts drug-mean residuals at
0.2087 Pearson on the 23 expression-covered held-out lines, with a line
bootstrap interval of [0.1591, 0.2601] and a permutation 95th percentile of
0.0304. DepMap expression therefore contains real context signal, but it is
well below Rhaister-O at 0.5394 on the same cohort.

The direct complementarity test predicts `truth - cross-fitted Rhaister-O`
from DepMap expression and adds that correction to frozen outer-test
Rhaister-O:

| Evaluation level | Frozen Rhaister-O | + DepMap correction | Improvement | 95% line bootstrap |
|---|---:|---:|---:|---:|
| Full delta | 0.634447 | 0.636036 | +0.001589 | [-0.000286, 0.003819] |
| Drug residual | 0.539430 | 0.539787 | **+0.000357** | **[-0.002679, 0.003610]** |
| PC1 removed | 0.527100 | 0.527491 | +0.000392 | [-0.002850, 0.003983] |

Only 10 of 23 point estimates improve and the primary interval crosses zero.
The predeclared expansion gate fails, so CNV, mutation, dependency, and MLP
experiments were not run.

## Experiment 4: synchronized LINCS L1000 external comparison

There was previously no L1000 expression experiment—only downloaded,
checksum-verified matrices and metadata crosswalks. The new comparison uses
exact curated shared cell lines, structure-validated compounds, exact 24-hour
Level-5 signatures, and the nearest dose within a two-fold ratio. Replicate
signatures are averaged within line/drug/dose.

The primary panel contains 79 directly measured L1000 landmarks that are also
in the Tahoe/Rhaister 2K panel. The 1,076 BING and 1,365 distributed overlaps
are secondary because many of them are inferred. The matched 24-hour cohort is
limited to A549, HT-29, and HS-578T.

| L1000 release | Conditions / drugs / cells | Drug-macro Pearson | 95% drug bootstrap | Drug-permutation 95th | p |
|---|---:|---:|---:|---:|---:|
| Phase I / GSE92742 | 171 / 97 / 3 | **0.0605** | **[0.0360, 0.0859]** | 0.0413 | 0.0020 |
| Phase II / GSE70138 | 304 / 78 / 3 | 0.0153 | [-0.0041, 0.0351] | 0.0172 | 0.1078 |

Phase-I Spearman is 0.0524 and sign agreement is 0.5191. Phase-II Spearman is
0.0285 and sign agreement is 0.5079. Phase I therefore contains a small
drug-specific signal beyond the within-cell/dose shuffled-drug null, but the
primary Phase-II Pearson result is not stable.

Across the 31 conditions and 13 drugs shared by both releases, Phase-I versus
Phase-II landmark Pearson averages 0.2025. The two L1000 releases are more
consistent with each other than either is with Tahoe, emphasizing assay,
batch, dose-formulation, and response-construction differences.

Only HS-578T overlaps the 24-hour exact-line cohort and the 25 frozen Rhaister
holdouts. On that one-line case study, evaluated on the same Tahoe truth and 79
genes:

| Release | Conditions / drugs | L1000 | Drug mean | Rhaister-O |
|---|---:|---:|---:|---:|
| Phase I | 33 / 11 | 0.0577 | 0.6814 | **0.8437** |
| Phase II | 16 / 6 | 0.0408 | 0.6822 | **0.8597** |

This is not a fair general model leaderboard—L1000 is an external bulk
signature and the subset contains only one held-out line. It does show that
direct cross-platform signature copying is not a useful Tahoe predictor.

## Integrated interpretation

The three tracks give a coherent result. Task-matched Tahoe DMSO context is
highly informative when coupled to a treatment-specific model. Static DepMap
expression contains context signal, but most of it is redundant with or weaker
than what Rhaister-O already extracts. External perturbation responses from
L1000 contain limited direction information, with material release-to-release
and platform dependence.

The scientifically defensible contribution is therefore not simply “another
model predicts unseen lines.” It is a strict reproduction plus a response-
component audit, a negative external-context gate, and a quantified
cross-platform validation result.

## Recommended next steps

1. Analyze frozen predictions first: contrast the lines helped and harmed by
   the DepMap correction across lineage, DMSO similarity, dose, response
   magnitude, and drug/MOA.
2. Use the extracted L1000 subset for dose-matching and pathway-direction
   sensitivity analyses. Separate tight matches from the two-fold boundary.
3. Identify compounds that are directionally consistent across both L1000
   phases and Tahoe, with multiple-testing control; do not select drugs from
   Phase I alone.
4. If L1000 is tested as an auxiliary prior, exclude all L1000 treatment
   signatures from each held-out Tahoe target line. Start with a low-capacity
   drug prior and require nested line-CV, bootstrap, and permutation evidence
   for improvement over frozen Rhaister-O.
5. Do not proceed directly to a multimodal MLP. Demonstrate stable benefit from
   a small linear or stacked prior before reopening CNV, dependency, or neural
   extensions.
6. For a paper or course project, the current rigorous negative/diagnostic
   result is already a complete and defensible story.

## Artifacts and provenance

| Artifact | Location |
|---|---|
| Ordered Tahoe/Rhaister/DepMap report | [`ORDERED_BASELINE_EXPERIMENTS.md`](ORDERED_BASELINE_EXPERIMENTS.md) |
| L1000 runner | `scripts/run_lincs_tahoe_concordance.py` |
| Slurm wrapper | `slurm/run_lincs_tahoe_concordance.sbatch` |
| Unit tests | `tests/test_lincs_tahoe_concordance.py` |
| Compact extracted signatures | `/nfs-stor/yuqi.lei/Therapeutic/derived/lincs_tahoe_concordance` |
| Result tables and manifest | `/nfs-stor/yuqi.lei/Therapeutic/results/lincs_tahoe_concordance` |

Final job `218460` completed with exit `0:0`. All four NPZ artifacts load with
`allow_pickle=False`, and the complete project test suite passes 20 tests. The
final L1000 summary SHA-256 is
`f475f20679dbc9724cd2beccce353b7db99a668207e7760938009562c0036aaa`.
