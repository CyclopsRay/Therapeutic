# Therapeutic — Weekly Record, week ending Friday 2026-09-18

Reporting interval: [2026-09-11T18:00:00+04:00, 2026-09-18T18:00:00+04:00) — Asia/Dubai.

Plans, discussion, decisions, reasoning, and corrections (Chinese or English).

## 2026-09-16 — Cross-project reporting policy update

按用户全局指令，改为 Asia/Dubai 每周五 18:00 自动换周，文件以结束周五命名。在每次写入前运行 `python3 ~/.local/bin/research-week --project /home/yuqi.lei/Therapeutic --ensure`，无需等待用户宣布。当前为 09_18 周；既有历史结果和来源链接保留原位。本次仅更新记录规则，没有新增科学实验或修改研究计划。

## 2026-09-17 — 项目目标确立 / Project aim: RECOMB submission 2026-11-04

用户确立了本项目的交付目标：**2026 年 11 月 4 日 RECOMB 投稿**。从 2026-09-17
起算约 7 周。人力：用户本人 + 最多 3 名 PhD 学生 + 1 位 advisor 参与讨论。
当前设想的科学目标是「更好的 cell line encoder」或「在 unseen cell line 上更好的
gene perturbation 系统」，并开放于其他选项。

### 可行性判断（讨论意见，非既成决定）

可行，但**不建议把论文主张押在「更好的 cell line encoder」上**。理由：(1) Welch
benchmark 在 n≈1,000 细胞系上比较了 10 种 cell-line encoding，raw expression 未被
超越；(2) 本项目自己的 DepMap 互补性 gate 已失败一次；(3) 即使取胜，幅度可能在
0.01 量级，不足以支撑 RECOMB 的贡献声明。

建议的论文结构为三部分组合：

- **A 跨基因结构（方法主体）** — Rhaister-O 是 per-gene 对角映射，不利用基因间
  相关结构，也不在药物之间共享强度。低秩 / bilinear / reduced-rank 响应模型是
  明确且未被占据的建模空间，baseline 清晰，现有 infra 可直接复用。这与
  `PLAN.md` Phase 4 Track A 一致，也与 2026-09-14 astra 第一轮意见
  ("the strongest opportunity is better sharing of response information across
  genes and drugs") 独立吻合。**正结果概率最高。**
- **C 可识别性 / 分解分析（框架层）** — 形式化 `response = drug mean +
  line template + line x drug interaction`，量化各分量可恢复比例，论证在有限
  细胞系数下 interaction 项的可识别性上界。本项目已完成大半（drug-mean removal、
  reference-only PC1 removal、per-line drug-cluster bootstrap）。
- **B DepMap-only deployment（延伸任务，放 discussion）** — 目标线连 Tahoe DMSO
  都不可得。novelty 高但无既有 baseline，风险大，不适合作为主线。

cell line encoder 在此结构中降级为一个 component（四臂比较实验），而非论文全部。

### 分工与时间线（建议）

| 角色 | 任务 |
|---|---|
| 用户 | 方法主体 A + 统筹 + 写作 |
| 学生 1 | encoder 四臂比较 + PC cap 修复 |
| 学生 2 | 评估框架、baseline ladder、bootstrap/permutation、图表 |
| 学生 3 | 分解 / 可识别性分析 C + ablation |
| advisor | 每周一个决策点 |

第 1 周冻结 infra 并跑完 encoder 比较；第 2-3 周方法开发与 inner validation；
第 4 周全 5-split 评估与统计检验；第 5 周 ablation 与分解分析；第 6 周写作；
第 7 周缓冲。缓冲不可省（HPC 排队与返工）。

风险评估（主观估计，非测量值）：取得显著超越 Rhaister-O 的正结果约 30-40%；
取得可辩护论文（改进温和但分析扎实）约 60%；RECOMB 自身接受率约 20-25%。
建议同时准备 arXiv preprint 与备选会议。

### 待核实的替代路径

Arc **Virtual Cell Challenge 2026** 的任务定义（6 条完全未见细胞系上的 CRISPRi
敲低响应预测，仅给未处理状态与基因列表）与「unseen cell line 上的 gene
perturbation」目标高度吻合，且有公开 benchmark 与对手。**其截止日期、数据可得性
与 11/04 的兼容性尚未核实**，核实前不应纳入计划。

## 2026-09-17 — Cell-line encoder direction: literature collection and design critique

Status: **literature review and design discussion only.** No training, no feature
extraction, no Slurm submission, no revision of the existing expansion gate.
The experiment proposed in the final section is **awaiting user approval**.

### 方向与来源 / Direction

用户与 advisor 讨论后决定：先从 **cell line encoder** 入手，尝试用 VAE 把 cell
line 映射到隐空间。初始设想的输入是 **baseline gene expression + 一段自然语言
描述**。用户要求先做文献收集，看别人怎么做，并问还有什么其他信息可用。

Three literature subagents (Claude general-purpose, WebSearch/WebFetch) covered
(1) VAE/autoencoder cell-line representation learning, (2) transcriptome-text
joint embedding, (3) all other candidate cell-line modalities. A second
gpt-6-astra consultation (`xhigh`, session `01a0abf3-10d6-7ea3-9e2b-21170af4fce0`;
round 1 was `01a09d5b-ca2b-7562-b3fe-aad9f2c60100`) then critiqued the
consolidated design.

### Literature findings

**VAE latents do not beat raw expression on the closest published task.** The
Welch-lab encoding benchmark (bioRxiv 2025.12.15.694331) compared ten cell-line
encodings for perturbation-effect prediction in **held-out cell lines** on
DepMap CRISPR (1,183 lines) and PRISM (919 lines): full expression, HVG subset,
a 500-dim scVI VAE latent, Geneformer, CancerFoundation, KEGG, Hallmark50,
hotspot and damaging mutations, and expression+mutation. Reported conclusion:
raw gene expression was not surpassed by any dimensionality-reduction or
foundation-model encoding, and mutation added nothing over expression.
Corroborating: arXiv 2410.13956 finds scVI ties rather than beats PCA;
Simidjievski 2019 (n=1,980) found VAE above PCA but not significantly;
Ahlmann-Eltze/Huber/Anders (Nat Methods 22:1657, 2025) and scPerturBench
(Nat Methods 2025) both report deep models failing to beat simple linear
baselines, with scPerturBench naming cellular-context embeddings as the
unsolved part.

**Scope limit on that evidence (astra correction):** the encoding benchmark's
endpoints are **viability and gene dependency**, not a 2,000-gene transcriptional
delta. It is highly relevant negative evidence but not our exact task. Claiming
"the cell-encoder axis is closed" overstates it.

**Verified VAE wins come from cross-domain transfer, not in-domain accuracy.**
VAEN (Nat Commun 2021; 100-dim latent, 1,100 CCLE lines) gained on CCLE->TCGA
transfer. MOSA (Nat Commun 2024) trains a 200-dim joint latent over 1,523
DepMap/Cell Model Passports lines from seven modalities and releases embeddings
and weights.

**Correction to an initial reading of MOSA:** its 272-line r=0.90 result concerns
lines **lacking transcriptomics during training**, i.e. missing-modality
reconstruction. It is not evidence of inductive response prediction for entirely
unseen lines, and was initially summarized as such in this session.

**MOSA is outside our declared information boundary.** Two of its seven input
modalities are drug response (810 features) and CRISPR dependency (17,931). A
released MOSA embedding of a held-out Tahoe line is partly built from that line's
own viability profile. The different endpoint (viability vs 24 h transcriptional
delta) does not remove the violation, and 200-dim compression does not either —
the question is whether forbidden information entered the procedure. Preferred
label: **transfer with target-line functional assays**, which is a stronger
statement than "transductive". Legitimate alternative: an encoder trained on
*other* lines' functional outcomes and applied to the target baseline only.

**Text branch: negative information-content prior.** No published work embeds
genuine descriptive cell-line text and evaluates held-out lines on a quantitative
endpoint. CellWhisperer (Nat Biotechnol 2025) is the closest architecture —
CLIP/InfoNCE over 1,082,413 expression-text pairs at sample/pseudobulk
granularity, text being LLM summaries of GEO metadata — but its endpoints are
retrieval, zero-shot annotation and chat, with no continuous regression anywhere;
the same holds for LangCell, GenePT/scELMo, BioTranslator and BioVERSE. CLDR
(arXiv 2312.10707) is **not** a precedent: its "natural language supervision" is
the IC50 label rendered as a sentence, with no external cell-line corpus, split
by drug rather than cell line. A gene-embedding decomposition
(bioRxiv 2025.12.19.695582) found **symbol-only** text retains 64% Hallmark /
72% C2 pathway coverage versus 0.00% for a shuffled control, i.e. most apparent
text information is the LM prior keyed on the entity **name**. "When Raw Data
Prevails" (EMNLP Findings 2024) found LLM embeddings of serialized numeric
features lose to raw features by 4-8 AUROC points. Available text is small:
~400-450 words for a well-studied line such as A549, mostly registry
bookkeeping, with genuinely line-specific biology nearer 50-150 words; DepMap
`Model.csv` is entirely categorical.

**Two text-branch arguments recorded here were judged too strong (astra):** a
2,000-d expression head versus a 768-d text vector does not by itself establish
modality dominance (likelihood variance, normalization and gradient geometry
matter; Javaloy ICML 2022 addresses gradient conflict, not a dimensionality-only
rule), and metadata-derived text is not necessarily a deterministic function of
*expression*, so the surjective-MoE result (arXiv 2204.05229) is not an exact
diagnosis of this model.

**Modality ranking for the cell side.** Expression #1 and tissue-of-origin #2
"closely following" pan-cancer, with genomics poor (Iorio 2016). Proteomics is
the one equal-but-different candidate: ProCan-DepMapSanger (949 lines, 8,498
proteins) reports predictive power "highly similar" to transcriptomics, better
per matched feature. Phosphoproteomics is null at matched n (1.84 vs 1.86 MSE,
n=38). Methylation (987 lines) has never been benchmarked against expression and
fails TCGA transfer. Chronos dependency is largely derivative of expression
(FORGE, Nat Commun 2026: dependency predictable at r=0.68 vs IC50 r=0.31) and has
our worst coverage (38 lines). PRISM/GDSC sensitivity fingerprints are response
data and are excluded by the zero-shot claim. Cell Painting is **ruled out**: the
JUMP-CP production dataset is essentially single-cell-line (U2OS); no baseline
morphology panel exists across cancer lines. Growth rate/doubling time is a
confounder (Hafner GR metrics), not a feature. Separately, "Understanding the
sources of performance in deep drug response models" (2025) reports a
cancer-blind drug-average baseline at Pearson 0.884 with genomic-profile models
underperforming it.

### 对我们既有结果的重要修正 / Correction to our own baseline claim

`KNOWLEDGE.md` records that the DepMap-expression ridge searched
**{0, 2, 5, 10} PCs and all five splits selected 10** — the top of the grid. The
capacity cap was **binding**. The standalone 0.2087 drug-mean-residual Pearson is
therefore a capacity-limited result, and **the raw/high-capacity expression
control has never been run on our task**. This materially weakens any claim that
our failed gate simply reproduces the field's "raw expression wins" consensus:
we never ran the arm that wins in the literature.

### Assessment

- "A VAE discovers a better general-purpose cell representation": insufficient
  motivation.
- "DepMap adds information beyond target DMSO": currently unsupported; the
  existing complementarity gate stays closed.
- "External baseline pretraining improves response prediction when target DMSO is
  unavailable": unresolved and testable — this is the live question.
- "A compact response-supervised representation improves cross-line transfer":
  defensible if it beats regularized linear controls.

The missing-modality justification for a VAE was judged largely a
rationalization **at this time**: expression covers 47 lines, CNV/mutation would
rescue at most two more, and the 38-line Chronos bottleneck disappears once
dependency is excluded from the strict input set. A modality can still be useful
specifically when *expression is missing*; that is a separate use case requiring
whole-modality masking experiments, not a reason to build the encoder now.

`nonlinearity is not identifiable at n=45` was also judged too absolute: 45
contexts severely constrain the function class, but many response outputs can
support a tightly constrained shared model. What is unjustified is assuming a
flexible encoder becomes identifiable because it reconstructs unrelated baseline
profiles well.

### 决定 / Decisions

1. **Drop the natural-language branch from the first experiment.** Its
   information source and hypothesized benefit are insufficiently specified, and
   the information-content prior is negative. It may return later as a narrow,
   controlled ablation: structured facts / name-stripped text of exactly those
   facts / name-only text / full text, on one fixed head.
2. **Do not retrain MOSA this week.** Removing its drug-response and CRISPR
   inputs would clean the boundary but establishes no downstream value.
3. **Do not add proteomics, methylation, CNV, mutation or dependency yet.** The
   expansion gate in `CLAUDE.md` remains in force and is not revised by this
   entry.

### 2026-09-17 执行结果与修正 / Executed: two experiments, both negative

两个实验已由用户批准并执行，结果见 `09_18_report.md` 第 1、2 节。

**修正一：PC cap 假设被自己的实验否定。** 本记录上文写道「capacity cap was
binding，raw/high-capacity expression 控制从未跑过，这削弱了我们的负结果」。
容量扫描（job 249462）证明这个说法**是错的**：把网格扩到 40 个主成分和
λ=10⁵ 之后，五个 split 仍然全部选择 10 个主成分和 λ=100，drug-residual
line-macro Pearson 逐位相同（0.208719）。LOO RMSE 曲面在 10 处有干净的内部
极小值，之后单调变差，整个曲面只跨越 1.8%。**原负结果站得住，且现在带有
容量负对照。** 保留的边界：未白化的 raw-feature ridge 是另一个先验，仍未运行。

**修正二：文本分支的判读。** 细胞表征消融（job 249475）给出预先声明的判定
`text_branch_justified = false`。关键数字是 `template_text` 减
`structured_metadata` = +0.0150，CI [−0.0250, +0.0602] 跨零：**用同一批结构化
字段生成的散文，与把这些字段做成 one-hot，在统计上无法区分。** 这正是
「表示形式改变、但没有新增实测信息」所预期的结果，与
bioRxiv 2025.12.19.695582（symbol-only 保留 64%/72% 通路覆盖）和
EMNLP Findings 2024 "When Raw Data Prevails" 的方向一致。

真实 Cellosaurus 文本显著**更差**（−0.0486 vs structured，−0.0636 vs
template）。但这一臂有一个必须声明的混淆：**49 条里 40 条超过 512 token 被
截断**，而 template 文本未被截断。因此对这一臂只能说「在 512-token 截断下没有
优势」，不能说「策展自由文本不含信息」。要分离这两种解释需要分块编码后平均。

所有四臂都超过各自的 line-label 置换零假设，但 20 次置换的最小可达 p 为
1/21 = 0.0476，四臂全部恰好取到该值，所以该检验只能分辨「高于零假设」。

**判断的方向没有改变，但理由变了。** 不做文本分支的理由现在是本项目自己的
配对实测结果，而不是文献先验。这对 advisor 的说服力更强，也更适合写进论文。

### 2026-09-17 Track A 第一轮：跨基因低秩，负结果

方向 A 的第一个实验已执行（job 249519），结果见 `09_18_report.md` 第 3 节。
输入只有 Rhaister-O 自己用的 plate-matched DMSO 谱，因此不涉及新模态，
不受 DepMap 扩展 gate 约束；25 条留出系全部可评估。

预先声明的六条 gate 两个 arm 全部不通过。关键数字：

- `dmso_lowrank_correction`（Rhaister-O + 秩 K 跨基因校正）相对 frozen
  Rhaister-O 是 **−0.0145**，CI [−0.0359, +0.0025]，点估计为负，5 个 split
  只有 2 个为正且幅度极小（+0.0026, +0.0033），而退化达 −0.0513。
- 置换检验给出最强的证据：该臂 p = 0.9901，**零假设均值 0.5334 高于实测
  0.5235**。置换只打乱校正项、保留冻结的 anchor，所以这个零假设的含义是
  「Rhaister-O 加一个随机配对的校正」。**随机校正比学到的校正更好**——学到的
  跨基因校正在拟合噪声。
- `dmso_lowrank_direct`（纯跨基因低秩，不用对角映射）达到 0.2531，置换
  p = 0.0099，**含有真实信号**，且高于 DepMap 表达 ridge 在同一端点的
  0.2087（符合 DMSO 是任务匹配基线的预期）——但远低于对角映射的 0.5380。

选中的秩在 5 个 split 上一致为 K=10、λ=100，网格含 1–40 且有 44 个可用主成分，
是内部最优，不是边界选择。

**科学解读。** 「基因 g 的响应依赖其他基因的 baseline」这一类结构，在这个
数据量下不可恢复：45 条参考系不足以识别跨基因耦合，低秩瓶颈学到的主要是噪声。
Rhaister-O 的对角限制不是它的弱点，而是在 45 条系的样本量下的一个**有效正则化**。

**必须区分、且尚未检验的另一件事。** 本实验测的是**跨基因条件化**
（cross-gene conditioning）。它**没有**测**系数共享**（coefficient sharing）：
把 Rhaister-O 自己的 per-(treatment, gene) 对角斜率在治疗之间、基因之间做收缩
或去噪。这正是 Rhaister 仓库文档描述、但 pinned executable path 中缺失的
SVD / empirical-Bayes 变体（`PLAN.md` Phase 4 Track A）。两者方向相反：本次是
给模型**增加**自由度（失败），系数共享是**减少**有效自由度。鉴于本次结论是
「对角限制起到了有益的正则化作用」，系数共享方向的先验反而因这个负结果而**变强**，
不应因本轮失败而一并放弃。

### 2026-09-18 共享对照耦合诊断：疑虑成立但不重要

在写本周结论时提出的疑虑：目标 `Δ = treated − plate-matched DMSO` 与预测特征
（同一条系的 DMSO）共用同一份测量，共享噪声会制造机械关联，且该项在系内跨药物
恒定、又长在 query 线上，因此 drug-mean removal 和 reference-fitted PC1 removal
都拆不掉它。若成立，0.538 的相当一部分可能是假象。

诊断（job 258115）不需要重算任何目标：50 条系在 14 个 plate 上都有 DMSO
centroid，且 56,827 个 condition 里有 49,186 个只属于单个 plate。把 plate 随机
对半分，condition 限制在只落于 A 半的那些（于是每个保留目标的对照都在 A 半），
`same` 臂特征取自 A 半、`cross` 臂取自 B 半，两臂都平均 7 个 plate 所以特征噪声
匹配、在相减时抵消。5 次随机劈分 × 5 个 outer split × 25 条留出系。

结果：drug-residual `same` 0.5468 vs `cross` 0.5410，**差 +0.0057，
CI [+0.0042, +0.0079]，占该臂分数的 1.0%**；full delta 差 +0.0033（0.5%）。

**结论：方向上我是对的，幅度上不重要。** 置信区间排除零，说明耦合确实存在、
可检测；但它最多解释约 1% 的表现。**Rhaister-O 的优势是真实的
baseline-to-response 关系，不是对照出现在等式两侧的假象。**

这是一次对抗性检验，而 headline 结果通过了。比起把疑虑搁置不提，现在我们有了
一个**量化上界**，这在论文里更有力。

### 对 RECOMB 论文结构的影响

这两个结果合起来构成一个可发表的负对照组合：容量不是限制、细胞身份类表征
（结构化、模板文本、策展文本）全部远低于表达、而表达本身又远低于
DMSO-conditioned 的 Rhaister-O（0.078/0.093/0.029 < 0.209 < 0.539）。这支持
把论文重心放在**跨基因结构（方向 A）**而非更好的 cell encoder：证据表明
瓶颈不在如何表示细胞，而在如何建模细胞语境对药物响应的调制。

### Proposed first experiment — awaiting user approval

Question: *does an externally pretrained expression VAE improve strict
held-out-line Tahoe delta prediction beyond uncompressed ridge and PCA trained on
the same external expression data?*

Setting: existing five outer splits, fixed 2,000-gene output panel, the same 23
expression-covered held-out lines. **DepMap-only inference** — target-line Tahoe
DMSO is used by the evaluator to construct ground-truth delta, never by the
predictor. Rhaister-O appears only as a richer-input reference, not an
equal-input competitor.

| Arm | Cell representation | Purpose |
|---|---|---|
| Raw | all standardized DepMap expression features + ridge | tests whether compression is needed at all |
| Reference PCA | PCA fitted on the response-training lines | connects to the existing 0.2087 result |
| External PCA | PCA fitted on the external baseline collection | **essential control** for pretraining data access |
| External VAE | frozen posterior-mean embedding, same collection | tests nonlinear unsupervised compression |

Design requirements: external PCA must receive **exactly** the VAE's pretraining
data and gene features, or architecture is confounded with data access. Exclude
all 50 Tahoe lines from the external pretraining collection. One shallow VAE
family, declared dimension grid (e.g. 8 and 32) matched for PCA, three fixed
seeds with a predetermined aggregation rule. Each arm gets the same regularized
linear head with coefficients differing by known drug-dose; a shared head on
concatenated cell and drug features **without interactions** is an inadequate
test. All selection by inner whole-line validation, frozen before outer
evaluation.

Proposed gate (**a proposal, not an approved project rule**): at least +0.01
absolute drug-mean-residual line-macro Pearson over the inner-validation-selected
linear comparator; paired 95% line-bootstrap improvement interval above zero;
at least +0.01 surviving reference-only PC1 removal; positive aggregate
improvement in at least four of five outer splits; no more than 1% line-macro
delta RMSE deterioration; and performance above a line-block permutation null
(shuffle whole reference expression profiles against response tensors with drugs
and doses kept together, >=100 permutations, head selection repeated).

Predeclared interpretation: VAE passes -> continue investigating an
external-expression encoder, which validates neither text nor multi-omics fusion
nor Rhaister complementarity. External PCA improves but VAE does not beat it ->
retain PCA; the gain came from broader pretraining. Raw expression wins -> stop
the compression program. No clear advantage with a wide interval -> inconclusive,
not proof that all cell encoders fail.

All training and feature extraction must run under monitored Slurm per
`CLAUDE.md`. Because these outer results have already been inspected, any success
is retrospective development evidence, not prospective confirmation.
