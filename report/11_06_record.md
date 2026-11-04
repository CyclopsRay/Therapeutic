# Therapeutic — Weekly Record, week ending Friday 2026-11-06

Reporting interval: [2026-10-30T18:00:00+04:00, 2026-11-06T18:00:00+04:00) — Asia/Dubai.

Plans, discussion, decisions, reasoning, and corrections (Chinese or English).

Carried over from `10_30_record.md` 2026-10-30 02:05–02:15 (context × compound sanity checks; results in
`10_30_report.md` §6).

### 2026-10-31 02:40 — User: implement O1–O3; "do the four compound views have no weights — are they just added?"

**Decision (user).** O1 (oracle reference + line-kernel paragraph), O2 (learned/neighbour-line coupling) and O3
(compound side) are all to be implemented before the experiment freeze (Mon 11-02 18:00).

**Answer to the weight question (checked in code and v3 outputs).** The views are weighted, but not learned for
prediction. `run_sdst_hybrid.py` sets w_v ∝ max(0, A_v), A_v = centred kernel–target alignment (Cortes 2012) between
K_v on the training compounds and the Gram matrix of their mean backbone residuals, normalised to sum 1, then divides
the fused kernel by √(K_ii K_jj). The alignments are tiny and similar (0.033–0.049), so the weights come out almost
uniform: per fold chem 0.27–0.31, tgt 0.24–0.26, moa 0.21–0.25, morph 0.23–0.25. In practice this is close to an
equal-weight sum. Two further issues: (i) A_v is computed over all training compounds with zero rows for compounds
lacking the view (tgt covers 36%, morph 23%), so coverage and informativeness are mixed; (ii) the view ablation
(`10_30_report.md` §5.9) says chem and Cell Painting carry unique information while targets and MOA are redundant with
each other, yet equal weights count the annotation signal twice. After the diagonal normalisation the weights only
matter among compounds that carry several views.

**Plan (new script `scripts/run_undetect_v2.py`, tag `v7_ext`; new file so running/queued jobs of
`run_sdst_hybrid.py` are untouched; same folds, backbone seed 1729, selection on valid, test once).**
- O3a weights: `pool_ml`/`nnk_ml` = view weights (softmax), amplitude and two noise terms (constant + 1/n_d) by the
  multi-output GP marginal likelihood of training compounds' mean residuals (genes z-scored); `pool_val` = 84-point
  simplex grid chosen on valid (does the weighting matter at all?).
- O2: `pool3_resp` / `pool3_ctl` = UnDetect + c · h_nbr, the other lines' per-line KRR predictions averaged with line
  weights from response similarity (train rows only) or from the course control kernel.
- O3b: `pool_sens` = shared term scaled per line by a shrunk sensitivity t_ℓ (least squares of the line's training
  residuals on leave-one-compound-out g); motivated by Szalai 2019's viability signature.
- `pool_all` = ML weights + response neighbour term + sensitivity.
- O1 after these results (text depends on what the model becomes): shrunk oracle (residual weight chosen on the
  neighbouring fold) to remove the "unshrunk reference" caveat, then the paragraph and appendix table.

**Reading rules (declared before any run).** PCC_plate contrasts vs `pool_kta` (`analyze_sdst_centred.py --ref`):
an extension is adopted into UnDetect if it is > 0 in 5/5 folds, CI > 0 in ≥ 4/5, mean ≥ +0.003, and its PCC_DEG is
not lower than `pool_kta` in more than 1 fold. Weights: if `pool_ml` and `pool_val` both fail this rule, the paper
states that view weighting does not matter on this benchmark (and reports the learned weights). O2's oracle ceiling is
+0.006 (`10_30_report.md` §6.4), so a pass there is not expected. Anything adopted must then be rerun on the
LINCS/Tahoe/DeepCE tables before it can enter the main model; otherwise it is an appendix result.

**2026-10-31 02:50 — Correction to `10_30_record.md` 2026-10-30 02:15 (O1 reference).** That entry said UnDetect-NN
is "above a single measurement of the compound in its most similar other line (0.131)". 0.131 was the unshrunk
oracle (residual weight 1). With the weight chosen on the neighbouring fold, one measured line scores 0.1542 vs
UnDetect-NN 0.1536 PCC_plate: equal, not above (`11_06_report.md` §1). The "≈ half of the way to the mean oracle"
statement stands (53% shrunk vs 54% unshrunk). Paper wording for O1: UnDetect's prediction for an unseen compound is
about as informative as one real measurement of that compound in the most similar other cell line, and recovers about
half of what the mean of all its measurements elsewhere would give.

**2026-10-31 03:20 — Adoption rule for O2/O3 changed by the user (before any O2/O3 result was seen).**
Jobs 356099–356101 crashed in the simplex-grid weight search (cusolver syevd on a degenerate fused kernel) after the
KTA and ML arms had finished; fold 3+4 job 356102 was cancelled. Fix: eigh retries with a 1e-8 jitter, then LAPACK on
CPU, and logs whether the matrix was finite. Resubmitted as 356153–356156 (script sha 02f7a986…). Already seen (no
O2/O3 arm): pool_ml − pool_kta PCC_DEG +0.0002 / 0.0000 / +0.0001 in folds 0–2; ML weights move from targets to Cell
Painting (fold 0: chem 0.29, tgt 0.15, moa 0.21, morph 0.36) without changing the predictions.
User: "对于O2和O3，我觉得只要效果没有更差就可以加上。毕竟我们应该相信每个drug的效用可以侧重于不同的方面，这在原理上说得通。"
New rule for O2 (neighbour-line term) and O3 (line sensitivity, learned weights), replacing the 02:40 rule for these
arms: adopt if, against pool_kta, the 5-fold mean PCC_plate and PCC_DEG differences are both ≥ 0 and no fold's
PCC_plate compound-bootstrap interval lies entirely below 0. To make this a fair non-inferiority test the extended
model now nests the current one: c = 0 (no neighbour term) was already in the grid; κ = ∞ (t ≡ 1) was added to the
sensitivity grid, so validation can always return the current UnDetect.
Agent notes recorded with the decision: (i) report per fold which settings validation chose; if it picks c = 0 or
t ≡ 1 in most folds the component is included but inactive and the paper must not credit it with any gain; (ii) paper
wording under this rule = "lets each compound's effect be shared across lines or line-specific without loss of
accuracy", not "improves"; (iii) anything adopted still needs the LINCS/Tahoe/DeepCE reruns before the main tables
change; (iv) the user's rationale (each compound's effect emphasises different aspects) is per-compound; the current
arms mix globally (one c, one t per line, one weight vector). A per-compound version (a gate over per-view features in
the NN readout) is a possible next variant, not started.

**2026-10-31 06:05 — GPU queue too slow for the freeze; CPU copy added as the decision run.** At 06:00 the four GPU
jobs were still pending (Priority) with an estimated start of 11-01 16:55 for all, which leaves too little time for
the LINCS/Tahoe/DeepCE reruns if something is adopted. Cutting the time limits from 4 h to 1:15 (a fold should take
≈ 20 min on a GPU) moved folds 0/1/2 to 10-31 19:39 / 20:54 / 11-01 03:22. The fold 3+4 job (356156) stayed at 11-01 16:55
and was cancelled. The same script now also runs on the CPU partition (`slurm/run_undetect_v2_cpu.sbatch`, 32 cores, tag
`v7_ext_cpu`, jobs 356254–356258 = folds 0–4). The CPU QOS allows 3 jobs at a time. The CPU backbone MLP is not
bit-identical to the GPU/v3 backbone, so `pool_kta` in this run will be close to v3 but not equal. The non-inferiority
rule (03:20) is applied within the CPU run (every arm vs that run's `pool_kta`), which is unaffected by that difference.
The GPU folds 0–2 stay queued in case something is adopted and the paper tables need the v3-identical backbone; if
nothing is adopted they are cancelled.

### 2026-10-31 12:10 — O2/O3 verdict under the non-inferiority rule (results `11_06_report.md` §2)

**Rule applied as declared at 03:20** (vs `pool_kta`: 5-fold mean Δ PCC_plate ≥ 0, mean Δ PCC_DEG ≥ 0, no fold with
the PCC_plate CI entirely below 0), on the CPU run `v7_ext_cpu`:

| arm | Δ PCC_plate | folds CI < 0 | Δ PCC_DEG | verdict |
|---|---|---|---|---|
| `pool_ml` (O3a learned view weights) | +0.0007 (5/5 > 0, 4/5 CI > 0) | 0 | +0.0001 | **passes** |
| `pool_all` (ML weights + neighbour + sensitivity) | +0.0008 (5/5 > 0) | 0 | +0.0008 (5/5) | **passes** |
| `pool_sens` (O3b line sensitivity) | −0.0002 | 0 | +0.0006 (5/5) | fails (mean PCC_plate < 0) |
| `pool_val` (simplex weights on valid) | +0.0004 | 1 | 0.0000 | fails (fold 1 CI < 0) |
| `pool3_resp` (O2, response similarity) | −0.0016 | 3 | +0.0002 | fails |
| `pool3_ctl` (O2, control similarity) | −0.0016 | 4 | +0.0004 | fails |

Under the stricter 02:40 rule (mean ≥ +0.003), nothing would pass. The weights question would then have been answered
"weighting does not matter". Under the user's rule, learned weights are a small, consistent gain, not a large one.

**Reading (agent).**
- **O2 is worse alone, although validation used it.** c > 0 in every fold. Validation traded the own-line term (h
  0.5 → 0.25) for the neighbour average. That raises plain PCC_DEG slightly (5/5) but removes compound-specific
  signal: PCC_plate falls, with the CI below 0 in 3–4 folds. This matches the 10-30 oracle ceiling (+0.006 even with
  the test compound's own responses). Neighbour lines carry line-level structure that the plate mean already covers.
- **O3b is active but helps only plain PCC_DEG.** κ is never ∞, and the line multipliers spread about 3× (10–90%
  0.6–1.9). So lines really differ in how strongly they express the shared compound response, but that scale is not
  compound-specific information beyond the plate mean.
- **The ML weights carry the PCC_plate gain.** They move weight from targets (and chem in folds 3–4) to Cell
  Painting. That agrees with the 10-30 view ablation: chem and Cell Painting carry unique signal, while targets and
  MOA are redundant. `pool_val` points the same way (Cell Painting ≥ 2/6 in all folds) but overfits validation in
  fold 1.
- **`pool_all` passes as a bundle.** Its PCC_plate gain is essentially that of `pool_ml` (+0.0008 vs +0.0007), and
  its extra PCC_DEG comes from the sensitivity term (+0.0008 vs +0.0006 for `pool_sens` alone). Its neighbour term is
  small (c = 0.25; 0 in fold 2), with h kept at 0.5. Two of its three parts fail the rule on their own.
- **The NN readout is fragile to backbone numerics.** On the CPU backbone, `nnk_kta` − `pool_kta` = +0.0003 PCC_plate
  (fold 1: −0.0061, CI < 0; early stop at epoch 19). On the GPU v3 backbone it was +0.0004 above the closed form.
  This is relevant if UnDetect-NN is presented as the main model; the closed form is the stable one.

**Recommendation (agent), pending the user's choice.**
- **(A) Adopt the ML view weights only** (`pool_ml`; `nnk_ml` for the NN variant). Cleanest attribution: one change,
  a principled fit (multi-output GP marginal likelihood on training compounds), 4/5 fold CIs above 0, and
  interpretable weights. Paper wording: "view weights learned by marginal likelihood; small but consistent gain"
  (+0.0007 PCC_plate).
- **(B) Adopt `pool_all`.** It also passes the declared rule and is better on PCC_DEG (+0.0008 vs +0.0001), but it
  carries two components that are not non-inferior alone, and it means more code in the three rerun pipelines.
- O2 and O3b alone are not adopted. They go to the appendix as tested extensions with the numbers above; the O2
  result also supports the O1 "context side is saturated" paragraph.
- The per-compound gate idea (03:20 note iv) is still not started.

**Paper numbers.** The paper's SDST rows are from v3 on the GPU backbone. GPU folds 0–2 (356153–356155) and folds 3+4
(356629, submitted 12:00, limit 1:30) are queued, but the estimated start slipped back to 11-01 16:37. If they do not
start in time, the fallback is to report the adopted model's SDST row together with `mlp` and `pool_kta` from the CPU
run (same backbone within the run). Either way, the LINCS/Tahoe/DeepCE reruns need `ml_weights` ported into
`run_l1000_xline.py` and `repro/deepce/xline_kernel.py`.

**2026-10-31 17:35 — GPU copy crashed again; fixed and resubmitted.** GPU jobs 356153–356155 and 356629 (folds 3+4)
started early, at 14:51–15:04 (backfill on gpu-08). All four crashed at the first degenerate simplex-grid kernel in
`pool_val`, after the KTA/ML arms. The 03:20 fallback did not help: once cusolver syevd fails, the CUDA context is
unusable ("illegal memory access" on the next GPU call, here `torch.isfinite`), so no in-process retry can succeed. No
outputs were saved. Fix (sha a0b02e17…): the simplex grid now does its eigh on the host via `shared(..., cpu=True)`;
the CPU run showed LAPACK handles all these kernels. Resubmitted with `--skip-val` (pool_val is not adopted and has
its CPU result), as 357045 / 357046 / 357047 (folds 0 / 1 / 2, limit 0:40) and 357048 (folds 3+4, limit 1:00). The
crashed logs already give GPU PCC_DEG for folds 0–3 (report §2.4).

### 2026-10-31 18:00 — User: try the view weights on Tahoe; also transfer the L1000-learned weights

User: "先试试tahoe吧。我估计很难有优势，不过可以跑一跑。或者我们用L1000学到的放在tahoe中看看呢". (Asked before: has any of
this been validated on Tahoe? No. O2/O3 and the context sanity checks were run on the L1000 drug-blind benchmark only.
EmeraldBay is unsuitable for unseen compounds: 19 single drugs, no plates, and 5-day vs Tahoe 24-h same line / same drug
concordance has a median r of 0.030 (`10_02_report.md` §1).)

**Design (declared before any run).** `scripts/run_l1000_xline.py` gets two new kernel sets that change only the view
weights of the consensus term g. The per-line f_L keeps D's per-line alignment weights, so `pool_X` − `pool_D` isolates
the weights.
- `Dml`: weights, amplitude and noise by GP marginal likelihood on the fold's training compounds. Code ported from
  `run_undetect_v2.ml_weights`: Adam 250 steps, genes z-scored, noise constant + 1/n_lines.
- `Dl1000`: fixed weights = 5-fold mean of the SDST marginal-likelihood weights (chem .265, tgt .228, moa .201, morph
  .306; `v7_ext_cpu`). These come from L1000 responses only, never Tahoe; four scalars, so no test-response leakage.
- Runs: `--source Tahoe --kernels T1 D Dml Dl1000 --tag _Dw` on T-0.05, T-0.5 and T-5 (ik14 split, seed 0, as §4.3).
  Sanity checks: `mkl_Dml` = `mkl_Dl1000` = `mkl_D`; `pool_D`/`mkl_D` reproduce the earlier `_D` runs (T-0.5, T-5).
- Analysis: `scripts/analyze_xline_weights.py`, ACTIVE rows. Line-macro differences vs `pool_D` on `pearson_full`
  (PCC_DEG analogue) and `pearson_resid` (prediction and truth minus the training line mean, the drug-specific part,
  analogue of PCC_plate), with a 2,000-replicate bootstrap over ik14 groups and lines positive.
- Reading rule (the user's non-inferiority rule, adapted): an arm is non-inferior on Tahoe if, at both T-0.5 and T-5, the
  line-macro Δ is ≥ 0 on both metrics. T-0.05 (no drug-specific signal for unseen drugs, `10_30_report.md` §4.3) is
  reported but not judged. Expectation (user and agent): no advantage. Tahoe has ~280 training compounds per fold,
  99/350 with Cell Painting, and set D was already −0.001/−0.003 vs T3 there.
- Consequence: if `Dml` is non-inferior on Tahoe (and later LINCS), option A stands for the main model. If only
  `Dl1000` passes, the paper would describe weights learned on the large L1000 benchmark and transferred, which is a
  different method claim and must be discussed. If neither passes, ML weights stay an SDST-only appendix result.

### 2026-10-31 18:30 — Tahoe verdict on view weights; a Tahoe residual-metric problem for the cross-line term

**Rule (18:00) applied** (`11_06_report.md` §3.1). Non-inferior requires Δ ≥ 0 vs `pool_D` on full and resid at both
T-0.5 and T-5.
- `pool_Dml`: **fails.** All four Δ are negative: full −0.00016 / −0.00019, resid −0.0024 / −0.0015. Lines positive on
  resid are only 2/50 and 8/50, though every interval includes 0. The fit is unstable on ~280 compounds: morph
  ranges .12–.86 across T-0.5 folds, and at T-0.05 (no signal) it puts .83–.99 on morph.
- `pool_Dl1000`: **fails by the strict rule, by a negligible amount.** Full −0.00004 / −0.00002, resid +0.00005 /
  +0.00032, all intervals within ±0.001 of 0. On Tahoe, L1000-learned weights and KTA weights give the same
  predictions to the fourth decimal.
- Declared consequence: neither passes, so ML weights stay an SDST-only appendix result and the main model keeps KTA
  weights. If the user counts `Dl1000`'s −0.00004 as "not worse", the agent still sees nothing to gain on Tahoe from
  it.

**Recommendation (agent):** keep KTA weights in the main model on all datasets (drop options A/B for the weights).
Report in the appendix: "weights learned by marginal likelihood move weight to Cell Painting and add +0.0007 PCC_plate
on the drug-blind benchmark (5/5 folds); on Tahoe (~280 training compounds) the fit is unstable and slightly worse;
L1000-learned weights transfer without loss but without gain. View weighting is not where the signal is." Same for
`pool_all` (option B): its PCC_plate gain was the ML weights.

**New finding, relevant to Table 4** (§3.2; not looked for, seen in the same outputs). On Tahoe the cross-line term
raises full Pearson (the reported gain, +0.004 to +0.007) but lowers the drug-specific residual Pearson at every dose:
`mkl_D` − `pool_D` resid +0.0031 / +0.0050 / +0.0022, intervals above 0, 35/44/36 of 50 lines. On LINCS P2-24 and P1-24
it raises both (7/7 lines). It is the same pattern as O2 on SDST: line-level structure helps the plain metric, not the
compound-specific one.
- Agent's hypothesis, untested: Tahoe's grid is complete (every training compound is measured in every line), so the
  consensus g adds no compound information that f_L lacks. Its gain on full Pearson comes from line-generic response
  structure; (a_L, b_L) are tuned on full Pearson, so they trade drug-specific signal for it. LINCS lines are sparse,
  so g carries compounds that f_L never saw.
- Consequence for the paper (user decides): the Tahoe row of Table 4 should state the metric. If the paper argues that
  PCC_plate-like metrics matter (the framing after the Codex review), a reviewer can ask for the residual metric on
  Tahoe, where UnDetect's cross-line term is negative.
- Options:
  - (i) Report both metrics and say so.
  - (ii) Test choosing (a_L, b_L) by inner-CV residual Pearson. This is a cheap Tahoe rerun (~5 min/slice) with a new
    tag; it must be declared before the run, and is not run yet.
  - (iii) Present Tahoe as the setting where per-line models already see every compound.

### 2026-10-31 18:28 — User: reproduce the published deep-learning models ourselves

User (after the agent offered to download Bai et al.'s released predictions to compute their PCC_plate):
"行我们自己复现一边这些论文". Read as: yes to the download, and in addition retrain the models ourselves with the
benchmark's own pipeline.

**Correction to `10_30_record.md` 2026-10-29 (Fig. 2 notes):** that entry said Bai et al. release fold-level metrics
only, so PCC_plate cannot be computed for the published models. Wrong for the current archive. Zenodo 22095049 v2.0.0
(MIT) contains `benchmark_predictions.zip` (9.35 GB), with per-run predictions for all seven models and the MLP; the
local `run_index.csv` / `asset_index.csv` list 40 primary-cohort Full/MLP prediction files (2.5 GB).

**Plan.**
- R0, tonight, CPU.
  - `slurm/dpb_setup.sbatch`: conda env `envs/dpb-benchmark` with the upstream pins (prebuilt wheels only); the seven
    repos at the pinned commits; and only the data main_benchmark needs. That is the XPert `processed_data.zip`
    (already in `raw/xpert_sdst`, same md5, linked), the Figshare UniMol array and HG_data, the CIGER/DeepCE bundles
    and the TranSiGen release.
  - `slurm/dpb_fetch_released.sbatch`: the released predictions archive → `raw/bai2026_archive/`.
  - Working copy of the benchmark: `/nfs-stor/yuqi.lei/Therapeutic/repro/dpb_bench` (data and results on NFS, not
    `$HOME`).
- R1: PCC_plate and PCC_DEG of all seven published models from their released predictions, on our SDST rows via
  `analyze_sdst_centred.py`. ID alignment is checked through the archive's per-fold identity files. No training.
- R2, GPU: `prepare.py`, a 1-epoch smoke run, then full 5-fold runs. Order: XPert and TranSiGen first (the only two
  above the plate-response baseline), then DeepCE and CIGER, then PRnet and MultiDCP. PertDiT last: it needs an 8.9 GB
  RAR from Tsinghua Cloud and MolT5/BioLinkBERT embeddings. Its upstream fetch disables TLS checks, which is not done
  here; it will be fetched with normal TLS if time allows.
- Reproduction check: per-fold PCC_DEG vs the published `pcc_saved` in `run_index.csv` (XPert seed 1729). Report the
  gaps; do not tune anything to close them.
- Not before the freeze: porting XPert/TranSiGen to Tahoe/LINCS.

### 2026-10-31 18:50 — Published models on PCC_plate: XPert matches UnDetect (`11_06_report.md` §4)

**Result.** From Bai et al.'s released predictions (exact match to their published PCC_DEG): XPert PCC_plate 0.1569 vs
UnDetect-NN 0.1541 and closed form 0.1537; TranSiGen 0.1440; drug-free MLP 0.099. XPert − UnDetect-NN = +0.0029, with 3/5
folds positive, 1 fold CI > 0 and none < 0. On PCC_DEG UnDetect leads by +0.018 (0.649 vs 0.6315).

**What it means (agent).**
- Strong support for the post-review framing ("evaluation obscures compound-specific learning; joint optimisation
  trades context accuracy against it"). XPert's drug-zeroing ablation changes PCC_DEG by +0.0002, yet XPert carries
  +0.058 compound-specific signal over the drug-free MLP. PCC_DEG hides it because XPert lost context accuracy
  (0.6315 < MLP 0.6372).
- It also removes a claim. UnDetect is **not** better than XPert at the compound-specific part. Its advantage is
  getting both: XPert-level PCC_plate without losing the context accuracy, so PCC_DEG is +0.018 over XPert and +0.012
  over the MLP. Any paper wording that says UnDetect extracts more compound signal than published models must go; it
  holds only against DeepCE, MultiDCP, CIGER and TranSiGen.
- The 10-29 statement that PCC_plate cannot be computed for published models is corrected above (18:28 entry).

**Options (user decides; none started).**
- (a) Report §4 as a new table/figure panel (Fig. 2c extended with XPert 0.157, TranSiGen 0.144, DeepCE 0.055, MultiDCP
  0.024, CIGER 0.011) and rewrite the claim as "matches the best published compound signal while keeping context
  accuracy".
- (b) Test whether the two are complementary. XPert's compound part could be added to UnDetect's decomposition, e.g.
  our backbone + a·(XPert − its plate mean) + UnDetect terms, with the weight chosen on the validation fold. That needs
  XPert validation-fold predictions, which only our own reproduction gives (the released archive has test predictions
  only). Our reproduction runs (357100 XPert fold 0, …) would serve this.
- (c) The reproduction continues as planned. Its value is now independent verification, XPert on Tahoe/LINCS after the
  freeze, and (b). PCC_plate for the published models no longer depends on it.

### 2026-10-31 19:05 — User: how does XPert work, can learning its way raise the ceiling, why is it better?

User: "1. XPert具体是这么做的？我们如果能学习XPert学习化合物的方式，能不能达到更高的上限？ 2. 确实可以更新论文。但是在此之前，我们先看看能不能比XPert做的更好，或者他们为什么比我们更好".

**XPert, from the code** (`repro/dpb_bench/models/XPert`, config `config_l1000.yaml`).
- Gene tokens: 978 genes, each a 128-d PPI gene vector plus an embedding of its binned (128-bin) control expression.
- Drug tokens: UniMol atom-level 3D representations (≤ 122 atoms) plus a 256-d drug embedding from a pretrained
  knowledge heterogeneous graph (HG: PPI, DTI, DDS edges; contrastive link prediction).
- Treatment branch: cross-attention between gene tokens and drug atom tokens, then self-attention among genes
  (CA+SA+SA+CA). Control branch: 4 self-attention layers on the gene tokens alone.
- Heads: post-treatment profile, control reconstruction, and delta = head(trt − ctl embedding). Auxiliary cell-line
  classification from a CLS token.
- Loss: 0.2·MSE(trt) + 0.003·MSE(ctl) + 0.2·MSE(delta) + 1·(1 − PCC(delta)). Up to 2,500 epochs, patience 50.
- No L1000 multi-dose pretraining in the benchmark config.
- Not verified: what the HG's DDS (drug–drug similarity, 287,834 weighted pairs) edges were built from. The paper
  (NMI 2026) is login-gated, and the graph files are not distributed (only the final drug embeddings). If DDS came
  from transcriptional similarity, the embeddings would carry test compounds' responses. Testable with our
  reproduction: XPert's own `config_l1000_ablation_zero_hg.yaml`.

**Results** (`11_06_report.md` §4.1):
- XPert's compound signal comes from its drug inputs: zeroing them drops PCC_plate 0.157 → 0.106 (PCC_DEG +0.0002).
- XPert and UnDetect are strongly complementary. Their centred predictions correlate 0.32. A blend with w = 0.4
  (chosen on the neighbouring fold in every fold) reaches PCC_plate 0.181, +0.027 over UnDetect and +0.024 over
  XPert, about halfway to the shrunk oracle 0.202. PCC_DEG on shared rows is 0.665 vs 0.658.
- UnDetect wins the compound-shared (cross-line) part; XPert wins the line-specific part, i.e. compound × context,
  the user's original 10-30 question.
- XPert's lead is largest for compounds with an identical-fingerprint training analog (+0.013), and absent for
  chemically novel compounds (Q1 +0.001, ns). This points at UnDetect under-using exact analogs (multi-view
  averaging, λ = 3 shrinkage, chem weight ~0.3) more than at a better general representation in XPert.

**Answer to the user's questions (agent).**
1. The ceiling is higher. Combining the two kinds of compound learning gives PCC_plate 0.181 without any new model;
   neither alone passes 0.157.
2. What XPert does better: (i) compound effects conditioned on each gene's control state (line-specific part);
   (ii) near-duplicate compounds. What UnDetect does better: the shared response, compounds with Cell Painting, and
   context accuracy (PCC_DEG).

**Candidate next steps, cheapest first (none started; user decides).**
- (A) Exact-analog term in UnDetect: a chem-only, low-λ (or kNN) KRR component added to the decomposition, weight
  chosen on validation. Targets (ii). One CPU/GPU run.
- (B) XPert's drug representations as extra UnDetect views (mean-pooled UniMol, HG embedding), chosen on validation.
  Tests whether the representation matters. The HG caveat above applies, so UniMol first.
- (C) Context-conditioned compound term, targets (i): let the compound effect depend on the control profile rather
  than on line identity alone, e.g. Nyström drug features × control-PC features in the NN readout (low-rank
  bilinear), compared with nnk on validation.
- (D) Stack UnDetect + XPert, with weights fitted on validation-fold predictions from our XPert reproduction.
  Strong numbers, weaker novelty; also a ceiling reference for (A)–(C).
- (E) XPert reproduction with zero_hg (HG ablation) to settle the DDS question.

**2026-10-31 19:15 — XPert paper read (`papers/xpert.pdf`, NMI 8:96–112, 2026); DDS question settled.** Methods,
"Biological token based on prior-knowledge HG": "For DDS, we compute the Tanimoto similarity between all pairs of drugs
using the RDKit package. Drug nodes with a Tanimoto similarity above 0.5 are connected". PPI is from STRING (score >
700), DTIs from databases (12,890 known interactions), and drug nodes are initialised with UniMol mol tokens. No
transcriptional data enter the HG, so there is no response leakage, and (E) is not needed to answer that.
Side note: DDS edges link every test compound to its structural near-duplicates in training, which fits XPert's
lead on identical-fingerprint compounds (§4.1).
Compute, from the paper: "The model was trained on an NVIDIA 4090 GPU … Training on the L1000_sdst dataset took
approximately 10 h". UnDetect per fold on one A100 (job 357045): backbone MLP 164 s; closed-form compound terms
(KTA weights, shared + per-line KRR, valid selection) +28 s; NN readout +19 s.

### 2026-10-31 19:25 — User: go ahead with cache → C-lite / A → XPert-as-backbone; compute cost becomes a paper point

User: "这个执行顺序可行。另外我意识到我们在论文中从来没有讨论过计算量，我们在这方面是无可置疑的最轻量化，这应该是我们可以说的一个点。"

**Compute claim: what is needed (agent).** Measured, same hardware, per fold: wall time of training, parameter count,
peak GPU memory. Sources: our reproduction runs (`slurm/dpb_train.sbatch` logs) and UnDetect's own logs (backbone 164 s,
closed-form compound terms ~30 s, NN readout ~20 s on one A100; job 357045). The XPert paper reports ~10 h on an RTX
4090. Precise wording: lightest among models that use the compound and beat the drug-free network. It is not lighter
than model-free baselines (plate-response lookup, ridge, kNN), and the backbone MLP is part of UnDetect's cost.

**Plan and reading rules (declared before any run).**
1. Cache run `scripts/cache_undetect.py` (CPU, tag `cache_v1`). Per fold, save for **all** rows: the backbone
   prediction B, the shared compound term G and the per-line term H. On training rows these are closed-form
   leave-one-compound-out values (shared) and leave-one-out within the line (per-line), so a second stage never fits on
   in-sample kernel fits. Also save the selected λ, (g, h) and KTA weights. Sanity: the cached `pool_kta` must equal
   the v7_ext_cpu `pool_kta` (same CPU backbone seed and code path).
2. Stage 2 `scripts/stage2_context.py` on the cache, everything closed form, selection on valid (PCC_DEG, as for every
   UnDetect arm), test once:
   - **C-lite (context-gated compound term):** per gene j, r_ij ≈ u_ij·(α_j + β_j·x̃_ij + Σ_k γ_jk·z_ik), with
     u = g·G + h·H, x̃ = z-scored control expression of gene j, z = top-K control PCs (fitted on training rows),
     ridge per gene on training rows. Variants: `ctx_scale` (α only, a per-gene rescaling control), `ctx_gene` (α,
     β), `ctx_pc` (α, γ), `ctx_both`. Grid: ridge strength and K on valid.
   - **A (exact-analog term):** n_i = mean residual of the most similar training compound(s) with Morgan Tanimoto
     ≥ τ (τ ∈ {1.0, 0.9, 0.8}), in the same line when measured there, else its compound mean. Added as UnDetect + c·n,
     with c and τ chosen on valid (c = 0 allowed).
3. Reading rules vs `pool_kta` (PCC_plate contrasts by `analyze_sdst_centred.py --ref pool_kta`):
   - Adoption: the user's non-inferiority rule (03:20).
   - "C works", i.e. strong enough to be the paper's main modelling contribution: the 02:40 improvement rule, Δ PCC_plate
     > 0 in 5/5 folds, CI > 0 in ≥ 4/5, mean ≥ +0.003, and PCC_DEG not lower in more than 1 fold.
   - `ctx_scale` is the control: gating counts only to the extent `ctx_gene` / `ctx_pc` beat `ctx_scale`.
4. After the XPert reproduction: XPert as the backbone f (UnDetect terms fitted on XPert residuals) for the
   complementarity paragraph, replacing the post-hoc blend.

**2026-10-31 20:45 — Stage 2 folds 0–2 seen (job 357222, PCC_plate 357237); v2 additions declared before running
them.** Interim, 3 of 5 folds, vs `pool_kta`:
- PCC_plate: `ctx_pc` +0.0055 (3/3 > 0, all CIs > 0), `ctx_both` +0.0052, `ctx_gene` +0.0036, `ctx_scale` +0.0031.
- A arms: c = 0 chosen in every fold, so inactive.
- PCC_DEG: `ctx_pc` +0.0014 / +0.0017 / +0.0012.
- `ctx_scale` (per-gene rescaling of the compound term, no context) already gives +0.003, so the context gate's own
  share is about +0.0024.
- Validation picked K = 20, the grid maximum, in 2/3 folds.

Additions (tag `stage2_v2`; v1 arms unchanged and finished as declared on folds 3–4):
- `ctx_pc_ext`: K grid extended to (5, 10, 20, 40, 80) because validation chose the grid edge (selection on valid
  only).
- `ctx_line`: control arm. A per (line, gene) rescaling δ_lj by ridge (same c grid), so gating by line identity
  rather than by the measured control state. If `ctx_pc` ≈ `ctx_line`, the gain is a line-specific scale. That is
  still a compound × line interaction, but not one that follows the control state.
- Paired PCC_plate contrasts also vs `ctx_scale` (`--ref ctx_scale`): the attributable context gain, per the 19:25 rule.

### 2026-10-31 21:45 — C-lite verdict (5 folds; `11_06_report.md` §5)

**Declared rules (19:25) applied.**
- `ctx_pc` vs `pool_kta`: PCC_plate +0.0050, > 0 in 5/5, CI > 0 in 5/5, mean ≥ +0.003; PCC_DEG higher in 5/5.
  → **passes the "C works" improvement rule** and the non-inferiority rule.
- `ctx_pc` vs the `ctx_scale` control: +0.0018 PCC_plate (5/5 > 0, 4/5 CI > 0). The context-attributable share is
  about 0.002; the other 0.003 is per-gene rescaling of the compound term.
- `ctx_line` (line identity, added 20:45) ≥ `ctx_pc`: +0.0023 vs `ctx_scale`, 5/5 CI > 0.
- A (exact-analog term): inactive (c = 0 in 5/5). Despite XPert's lead on identical-fingerprint compounds, adding
  their residuals does not help UnDetect on validation; the kernel already uses them.
- `ctx_pc_ext` (K up to 80) chose K = 80 in 5/5 on validation PCC_DEG, but its PCC_plate (0.1596) is below K = 20.
  Selection on PCC_DEG does not favour the compound-specific metric beyond K = 20.

**Reading (agent).**
1. C works, but what it captures is mainly a compound × line interaction in the form "each line scales each gene's
   compound response". Gating by the measured control state is no better than gating by line identity on this
   benchmark (all lines seen). Control-PC gating has one argument for it: it is defined for an unseen line (cold-cell
   settings, Tahoe-style transfer), while `ctx_line` is not.
2. Against XPert: from −0.002 (`pool_kta`, CPU backbone) to +0.003 PCC_plate (`ctx_pc`), not significant. PCC_DEG
   +0.018. Defensible wording: "matches the best published model's compound-specific signal while keeping its
   context-accuracy lead, at a few seconds of closed-form compute".
3. Complementarity with XPert survives C: blend 0.186 (w ≈ 0.36).
4. Novelty and story for the paper (user decides): a closed-form bilinear compound × context term
   u_ij·(δ_j + γ_j·z_i) on top of UnDetect's decomposition. Second-stage ridge over 978 genes, seconds on CPU. The
   honest framing is "context modulates the transferred compound response"; it is not proof that control expression
   adds information beyond line identity.

**Open before the paper can use it.**
- (a) Numbers on the v3-identical GPU backbone (a GPU cache run; ~5 min/fold once it starts), or report all UnDetect
  rows from the CPU run consistently.
- (b) Portability: the same gate on LINCS/Tahoe (`run_l1000_xline.py`: per line × gene rescaling and control-PC gating
  of the consensus/per-line terms, chosen by inner CV), judged by the non-inferiority rule with the residual
  metric reported.
- (c) Optionally, the gate on the NN readout (`nnk`).

### 2026-11-01 00:45 — CPU vs GPU backbone: what differs (answer to "why rerun on GPU?")

**Question (user).** Why rerun on GPU? How does the CPU run differ?

**Check (agent; no new job).** Same script path, seed 1729 and hyperparameters (λ = 3, g = h = 0.5 in 5/5 folds both
runs). The backbone MLP is not bit-reproducible across devices: early stopping lands on different epochs (CPU
315/324/397/437/374 vs GPU v3 406/406/408/345/351). Paired, per fold, on the same test rows
(`results/sdst_hybrid/v3/fold_*/pred_test.npz` `pool_multi_norm` vs `stage2_v2/fold_*/pred_test.npz` `pool_kta`):

| fold | PCC_plate GPU | PCC_plate CPU | Δ CPU − GPU | PCC_DEG GPU | PCC_DEG CPU |
|---|---|---|---|---|---|
| 0 | 0.1558 | 0.1549 | −0.0008 | 0.6505 | 0.6503 |
| 1 | 0.1528 | 0.1581 | +0.0052 | 0.6472 | 0.6469 |
| 2 | 0.1567 | 0.1589 | +0.0022 | 0.6467 | 0.6471 |
| 3 | 0.1549 | 0.1504 | −0.0046 | 0.6507 | 0.6508 |
| 4 | 0.1484 | 0.1530 | +0.0046 | 0.6472 | 0.6474 |
| mean | 0.1537 | 0.1551 | +0.0013 | 0.6485 | 0.6485 |

Correlation of the two runs' plate-centred predictions: 0.77 (raw predictions 0.98).

**Reading.**
1. Nothing differs in method; PCC_DEG agrees to four decimals. PCC_plate moves by up to ±0.005 per fold between two
   runs of the same model. This is backbone run-to-run noise: the plate-centred signal is a small residual and is
   sensitive to where the backbone stops.
2. The +0.0050 `ctx_pc` gain is paired on one backbone, so it is valid. Mixing runs is not: CPU `ctx_pc` (0.1600)
   against the paper's GPU `pool_multi_norm` (0.1537) would show +0.0063, of which +0.0013 is run noise.
3. The noise is the same size as the `ctx_pc` gain and as the UnDetect–XPert gap (±0.003). "On par with XPert" is the
   only defensible claim. A GPU rerun buys table consistency, not accuracy.
4. Proposed instead (user to decide): two more backbone seeds on CPU (cache + stage 2). Report UnDetect rows as a
   3-seed mean and check that the `ctx_pc` gain holds per seed. The CPU cache costs 46–63 min per fold (`seconds` in
   `cache_v1/fold_*/summary.json`), against ~5 min on GPU once it starts; 10 jobs with 3 running at a time ≈ 4 h.

### 2026-11-01 00:55 — Decision: run the UnDetect cache and C-lite stage 2 on GPU, three backbone seeds

**Decision (user, 00:50).** Run on GPU (“就GPU交任务吧”) rather than the CPU multi-seed run proposed at 00:45.

**Protocol (agent).**
- Hardware: every cscc-gpu-p node is an A100-SXM4-40GB, as for v3 (gpu-07, job 351008). Seed 1729 may therefore
  reproduce the v3 backbone. To be checked: `B[test]` against v3 `mlp` predictions, early-stop epochs against
  406/406/408/345/351.
- Seeds: 1729 (the paper's v3 seed), plus 2718 and 31415 (the seeds of the earlier `v1_seed*` runs). Same code,
  hyperparameter grids and selection rule as `cache_v1` and `stage2_v2`. One code change: in `cache_undetect.py`, both
  eigendecompositions (shared and per-line kernel) now run on LAPACK, with results moved back to the GPU. A cusolver
  failure would poison the CUDA context (00:45 not affected: on CPU the result is the same).
- Jobs:
  - GPU caches: 357721 (seed 1729, 1 h limit) and 357722 (seeds 2718 + 31415, 1.5 h).
  - CPU stage 2 `--v2` + `analyze_sdst_centred.py --ref pool_kta`, afterok: 357723 / 357724 / 357725.
- Reading rule (fixed now, before results):
  1. Paper rows use the seed-1729 GPU run if it matches v3. If it does not match, report the 3-seed mean with the
     seed range for every UnDetect row.
  2. The `ctx_pc` gain counts as robust only if it is > 0 for each seed (5-fold mean) and its 3-seed mean meets the
     02:40 "C works" thresholds on the seed-pooled folds.
  3. The XPert contrast is reported as the 3-seed mean with its seed range. "Better than XPert" is claimed only if
     every seed's CI is > 0; otherwise "on par".

### 2026-11-01 02:45 — Decision: finalise the drug-blind numbers on the CPU run; GPU caches cancelled

**Decisions (user, 02:30).**
- Whether XPert gets reproduced does not matter; a later reproduction only changes one number. The reproduction
  jobs 357100/357101 stay queued, but no decision waits on them.
- The drug-blind numbers are finalised on the CPU backbone run (`cache_v1` + `stage2_v2`).

**Actions (agent).**
- Cancelled the GPU caches 357721/357722 and their stage-2 jobs 357723–357725 before they started. The 00:55 plan is
  withdrawn, so its reading rule does not apply.
- `scripts/finalize_sdst.py` (srun 357876, cn-20, 11 min) scores every arm on the same test rows. Arms: backbone,
  UnDetect and its stage-2 arms, released published models, the v3 GPU arms as reference, and cross-fitted XPert
  blends. Results in report §6.

**Implications for `tab:drugblind` and §4.1 of the paper (for Yuqi; the paper text is not edited here).**
- Backbone PCC_plate 0.099 → 0.100. UnDetect closed form PCC_plate 0.154 → 0.155. Its Δ (+0.011 / +0.055) is
  unchanged.
- New candidate row: UnDetect + context gate, 0.650 / 0.160 (Δ +0.013 / +0.060 vs backbone, 5/5 CI > 0).
- The XPert row can now carry PCC_plate 0.157. TranSiGen (0.614 / 0.144) could be added.
- The UnDetect-NN row (0.649 / 0.154) comes from the GPU run. On the CPU run the gated closed form is better
  (0.650 / 0.160) and lighter. Options: drop the row, or keep it labelled as a separate run.
- The ablation rows (plate co-membership kernel, chemistry kernel, end-to-end, residual head) come from GPU runs.
  Their Δ are paired within their own run, so they stay valid. Their absolute values sit on the 0.099 backbone. The
  two kernel rows are closed form and could be recomputed on the CPU backbone in minutes. End-to-end does not use
  the backbone.
- "Improves 62–69% of test compounds" → 60–65% (gated) / 60–63% (closed form) on the CPU run.
- Unshared rows: 0.536 → 0.570 (closed form), 0.574 (gated); unchanged in substance.
- The three-seed sentence (0.649 each time, 15/15 > backbone) refers to UnDetect-NN on the GPU runs. There is no
  multi-seed run of the gated model.

### 2026-11-01 03:05 — Decision: drop UnDetect-NN from the paper table; recompute the ablation rows on the CPU backbone

**Decisions (user, 03:00).**
- Drop the UnDetect-NN row. The method is closed form from end to end, which also fits the compute paragraph.
- Run the ablation rows on the CPU backbone, then wrap up.

**Protocol (agent).** All arms below sit on the CPU backbone (`cache_v1`).
- New `run_sdst_hybrid.py --backbone-cache cache_v1`: B is loaded from the cache rather than trained.
  Same code, grids and selection on valid as v1/v3.
- Arms:
  - `pool_plate`: plate co-membership kernel.
  - `pool_chem`: chemistry kernel.
  - `nnf_multi`: raw-feature residual head, its 3-config grid (`--input-arms nnf_multi`).
  - `pool_multi_norm`: closed form again, as a check against `pool_kta`.
- `mlp_input` (end to end, same features) does not use the backbone. Its v4_input predictions are reused and
  re-paired against the CPU backbone.
- Jobs 357912–357916, one per fold. `finalize_sdst.py` now loads these arms and asserts that the ablation run's
  `mlp` equals the cache B.

### 2026-11-01 04:10 — Ablation rows done on the CPU backbone; drug-blind table final (report §6.3)

**Outcome.**
- Ablation jobs 357912–357916 finished cleanly (03:12–03:34, 14–25 min per fold).
- The first finalize attempt (srun 357976, 4 cores on cn-20) hit its 30-min limit with no output. The rerun as
  sbatch 358039 (8 cores, progress timing) took 1.5 min. The cause of the first stall is not identified; the
  outputs are unaffected.
- `pool_check` (closed form recomputed in the ablation run) equals `pool_kta`, so the whole table sits on one
  backbone.

**Table changes vs the paper's `tab:drugblind` (for Yuqi).**

| row | paper (GPU runs) | final (CPU run) |
|---|---|---|
| XPert | 0.632 / — | 0.632 (Δ −0.006) / 0.157 (Δ +0.057) |
| backbone | 0.637 / 0.099 | 0.637 / 0.100 |
| + plate kernel | 0.643 +0.006 / 0.100 +0.001 | 0.643 +0.006 / 0.101 +0.001 |
| + chemistry kernel | 0.643 +0.006 / 0.131 +0.032 | 0.643 +0.006 / 0.133 +0.033 |
| same features, end to end | 0.613 −0.024 / 0.151 +0.052 | 0.613 −0.024 / 0.151 +0.051 |
| + residual head | 0.648 +0.011 / 0.151 +0.052 | 0.648 +0.010 / 0.152 +0.052 |
| UnDetect closed form | 0.648 +0.011 / 0.154 +0.055 | 0.648 +0.011 / 0.155 +0.055 |
| UnDetect-NN | 0.649 +0.012 / 0.154 +0.055 | dropped (user 03:00) |
| UnDetect + context gate | — | 0.650 +0.013 / 0.160 +0.060 |

**Text affected.**
- Caption: the two same-feature rows now have compound-bootstrap intervals on PCC_DEG too (end to end 0/5 > 0, 5/5
  CI < 0; residual head 5/5 CI > 0). "Only fold signs are available" can go.
- §4.1 "within 0.003 of UnDetect": still true for the closed form. The gated model is +0.008 PCC_plate above the
  residual head (5/5 folds, 3/5 CI > 0).
- "Improves 62–69% of test compounds": now 60–65% (gated).
- The three-seed sentence: no multi-seed run of the final model.
- The context gate needs a sentence in Methods and one in Results.

**Status.** No further SDST jobs are planned. XPert/TranSiGen reproduction 357100/357101 is still queued and blocks
nothing.

### 2026-11-01 16:40 — Decisions on open items; drug × cell-line gate on LINCS/Tahoe; results-related paper update

**Decisions (user, 2026-11-01).**
1. Tahoe in Table 4: explained in text only (option iii); "not our main battlefield".
2. The gate on LINCS/Tahoe: run first, then look at the results.
3. Compute: report everything on GPU. Leave a placeholder in the paper; the timing run can be done later.
4. Team: we (Yuqi + agent) do all analysis, code and text. Hanzhe only redraws the figures and gets the captions
   plus pointers to the relevant paper sections, with freedom in design. Keqin's K1–K4 and Hanzhe's H1–H3 fall to us.
5. Paper: update the results-related parts now. Everything touched by the new drug × cell-line interaction must be
   rewritten and checked carefully.
6. View weights: the user asked what KTA is (answered in chat). The final model keeps the KTA weights.

**Gate on LINCS/Tahoe (agent).**
- New option `run_l1000_xline.py --gate D`. With u = a_L g + b_L (f_L − mu_L), the prediction is
  `mu_L + u (1 + delta)`.
  - `gate_line_D`: one delta per (line, gene).
  - `gate_scale_D`: one delta per gene shared by all lines (control).
- Fitting: delta is fitted on the inner-CV predictions of the training rows (the rows that choose a_L, b_L), with
  ridge lam = c · mean_j Σ u².
- Selection: c in {1e-3 … 10, ∞} by cross-fitting over the inner folds; c = ∞ gives `pool_D`. There is no
  per-sample control on Level-5/Tahoe, so the context is the cell line (the analogue of `ctx_line`).
- Jobs 358880–358885: P1-24 ik14 / platemap, P2-24 ik14 / platemap, Tahoe T-0.5 / T-5; tag `_Dgate`,
  kernel D only.
- Analysis: `analyze_xline_weights.py --tag _Dgate [--split platemap]`, now with `--split` and the gate arms.
- Check: `pool_D` / `mkl_D` reproduce the earlier `_D` runs exactly (max |diff| 0).
- Results so far, line-macro ACTIVE rows, Δ vs `pool_D` with the ik14-group bootstrap CI (P1-24 still running at
  16:40):

| slice | `gate_line_D` Δ full | Δ resid | lines up (full) |
|---|---|---|---|
| P2-24 ik14 | +0.0007 [+0.0003, +0.0011] | +0.0012 [+0.0009, +0.0015] | 6/7 |
| P2-24 plate map | +0.0005 [+0.0001, +0.0009] | +0.0004 [+0.0000, +0.0007] | 6/7 |
| Tahoe 0.5 µM | −0.0002 [−0.0010, +0.0005] | +0.0003 [−0.0005, +0.0010] | 11/50 |
| Tahoe 5 µM | −0.0006 [−0.0016, +0.0004] | −0.0010 [−0.0019, −0.0003] | 9/50 |

`gate_scale_D` (shared per-gene scale) chooses c = ∞ in every fold on P2-24 plate map, T-0.5 and T-5, so Δ = 0
there. On P2-24 ik14 it gives +0.0002 (full) / +0.0004 (resid). On Tahoe 0.5 µM, validation switched the per-line
gate off (c = ∞) in 215 of 250 line × fold cases.

**Paper (`paper/`, local edits, not pushed).**
- Method:
  - Eq. (1) now includes the interaction.
  - New subsection "Compound–context interaction": control-PC gate, per-gene ridge, leave-one-out compound terms,
    nested at c = ∞.
  - Neural readout removed.
  - Model selection lists c and K.
  - New "Computational cost" paragraph with a `\todo` for the GPU timing.
- Results §4.1 rewritten on the final numbers (report §6):
  - Table 2 replaced.
  - New paragraphs: XPert on PCC_plate and its zeroed input; the interaction (with its gene-scale and line-identity
    controls); same features vs UnDetect; complementarity with XPert.
  - Three-seed sentence removed.
- §4.3: one paragraph on the Tahoe residual metric (report §3.2); `\todo` for the Level-5 gate.
- Introduction:
  - "These models do not use the compound" → "By the standard score, these models barely use the compound".
  - New sentence: XPert learns the compound on the plate-centred score but falls below the drug-free network on
    PCC_DEG.
  - Model paragraph, Fig. 1 caption (Interaction module) and contributions updated.
- Discussion: same-input numbers updated; complementarity + compute paragraph (`\todo`); unseen-line limitation.
- Appendix:
  - Per-fold PCC_DEG and new PCC_plate tables on the final run.
  - Interaction-variant table.
  - Published Full/Zero with PCC_plate.
  - View-ablation caption notes the separate backbone run.
  - UnDetect-NN and the seed paragraph removed.
- Fig. 2: `scripts/figures/fig2_benchmark.py` now reads `final_cpu` (panel a UnDetect = `ctx_pc`; panel c adds XPert
  and TranSiGen with and without drug input). Preview and source CSV regenerated; captions synced.
- `paper/figures/BRIEF.md`: Hanzhe's figure brief.
- Local build:
  - tectonic in `/nfs-stor/yuqi.lei/envs/tex` (cache `/nfs-stor/yuqi.lei/cache/tectonic`), building a scratch copy
    with `\providecommand{\note}`; tectonic's llncs already defines `\note`.
  - 14 pages, no undefined references.
  - Fig. 2 and Fig. 3 float behind the start of the references (PDF pages 10–11). Check in Overleaf (pdfLaTeX).

### 2026-11-01 17:00 — Gate on LINCS/Tahoe finished (report §7); decision needed on how the paper uses it

- All six runs finished without errors.
- LINCS: small gains in all four settings (+0.0005 to +0.0014 full, CI > 0), and the non-inferiority rule holds.
- Tahoe: no gain at 0.5 µM; at 5 µM the residual interval is below zero, so the rule fails at both doses.
- Validation mostly switches the gate off on Tahoe.

**Options for the paper (agent's recommendation first; user decides).**
- (a) **Recommended.**
  - Define the interaction with the per-sample control context, i.e. the drug-blind benchmark.
  - Leave Table 4 on the transfer terms, as now.
  - Report the line-identity variant in one sentence: "+0.001 on all four LINCS settings, none on Tahoe-100M, where
    validation switches it off in most lines".
  - Fig. 1 caption: drop "elsewhere through the cell line".
- (b) Add a row "+ interaction (cell line)" to Table 4 with the §7 numbers.
- (c) Make the line-identity gate part of UnDetect on Level-5/Tahoe and replace the Table 4 row. This fails the
  non-inferiority rule on Tahoe.

### 2026-11-01 19:20 — Gate wording decided; audit of which paper results predate the final pipeline; reruns started

**User decision (on the 17:00 options):** "关于门控，就说在LINC上有提升就好了". The paper reports the line-identity
gate for LINCS only.
- Method, interaction subsection: one sentence on the LINCS form (one δ per line × gene, fitted on inner-CV compound
  terms, c cross-fitted).
- §4.3: "+0.0005 to +0.0014 on all four settings, intervals above zero, 6–7/7 lines".
- Table 4 stays on the transfer terms; its caption now says "without the interaction (reported in the text)".
- Fig. 1 caption: "on LINCS through the cell line".
- The Tahoe result stays in report §7 as recorded. It is not in the paper.

**Audit, user question "所有的结果都重新生成了吗？": paper results and their pipeline**

Already on the final pipeline (`cache_v1` backbone + transfer terms + interaction):
- Table 2.
- Appendix per-fold PCC_DEG and PCC_plate tables, interaction-variant table.
- Fig. 2 a–c.
- §4.1 text.

No model involved (unchanged by the pipeline):
- §2 plate-response baseline.
- Published Full/Zero tables.
- Fig. 2b.

Earlier pipeline, rerun now:
- **Fig. 3a and `tab:app-views` (view ablation).** Earlier GPU backbone, transfer terms only (v4_views, job 351008).
  - Rerun on `cache_v1` with the interaction: new `scripts/sdst_view_ablation.py`, jobs 359222–359226, tag
    `views_final`.
  - Each view set gets KTA weights over its views, the leave-one-out transfer terms (`cache_undetect.run_fold`, now
    with `views`/`B`/`save` parameters, default behaviour unchanged), then ctx_pc with c and K chosen on valid.
  - Check: the all-four set reproduces `stage2_v2` exactly in folds 0–2 (PCC_DEG 0.65169 / 0.64865 / 0.64832).
- **Fig. 3b–d and `tab:app-pairs` (compound analysis).**
  - Fused-kernel weights came from v3 KTA, and Part B used v3 predictions.
  - Rerun with `analyze_sdst_compounds.py --final`: cache_v1 KTA weights; final_cpu `pool_kta − pool_chem` for the
    per-compound gain. Job 359229, output `compound_analysis_final/`.

Earlier pipeline, not rerun:
- **DeepCE benchmark** (Table 3, Fig. 2d, ten resplits): transfer terms; the interaction was never run there.
- **LINCS/Tahoe Table 4:** transfer terms by decision (above).
- **§4.4 LINCS P2-24 target-permutation and STRING-rewiring tests** (10_09 §7): per-line kernel model. These test
  views, not the pipeline.

Figures:
- Fig. 1 preview: the generator now draws an "Interaction" × node between the compound effect and the sum, fed by
  the context lane.
- Fig. 3: the generator is switched to the new sources; it is regenerated when the jobs finish.

### 2026-11-01 19:30 — DeepCE interaction not run; commit

- User: "不跑了". The interaction is not run on the DeepCE benchmark. The Table 3 caption says "UnDetect: transfer
  terms, without the interaction".
- Fig. 1 dossier v4 and BRIEF updated: the preview now shows the Interaction node.
- User asked to commit:
  - Paper repo: everything under `paper/`.
  - Main repo: scripts, slurm, report, the papers registry and `.gitignore`.
  - Left out: `draft/recomb2027/` (superseded draft), `CODEX_review.md` and `RECOMB_requirements.md` (review and
    call text).
  - Not pushed.

### 2026-11-01 20:15 — Fig. 3 inputs regenerated on the final run; paper updated (report §8)

- Fold 2 of the view ablation failed: `syevd` did not converge on the only_morph per-line kernel, which has many zero
  rows.
  - Fix: `eigh_cpu` falls back to scipy `evr`, only on failure.
  - Rerun as 359337; the fallback fired once.
- Paper changes:
  - `tab:app-views`: new rows, and the caption now reads "UnDetect with the interaction, backbone of Table 2"; the
    "earlier run" sentence is removed.
  - §4.4: 0.027–0.038 alone; 0.013/0.015 removed; 0.022/0.015 permuted; 0.006–0.008 with an analogue.
  - Fig. 3(a) caption: "UnDetect".
  - `tab:app-pairs`: the BVD-523/TAK-733 rounding is corrected and the pair order fixed.
  - Fig. 3 regenerated; dossier v4; BRIEF data note updated.
- The scientific reading is unchanged: chemistry and Cell Painting carry unique information; targets and mechanism
  are redundant with each other.

### 2026-11-01 22:20 — Reference check; benchmarked models and interaction-technique papers cited (paper, not committed)

- User: check the references and where they are cited. Include every benchmarked tool and the papers on cell × gene
  (compound × context) interaction techniques.
- Before: 9 references cited in the body. 17 bib entries were marked VERIFY. Five of the seven models in Bai et al.'s
  benchmark were never cited (CIGER, MultiDCP, PRnet, PertDiT, TranSiGen), and neither were STRING, the rewiring null,
  fingerprints, annotation sources or Cell Painting.
- Check: every entry was compared with its Crossref record (DOI) or with arXiv (chemCPA, arXiv-only items). Model papers
  are from `repro/dpb/upstream/reference/PROVENANCE.md`; drug encoders were read from the benchmark's model code.
  Corrections to earlier bib data:
  - Systema is *Nat Biotechnol* 44(6):1050–1059 (2026).
  - XPert: Guo et al., 8(1):96–112.
  - MAP: 8(9):1478–1491.
  - Bai et al. has three authors (Bai, Prince, Nitschke).
  - "Is graph-based feature selection of genes better than random?" has Hashir as first author, not Bertin.
  - PDGrapher: 10(5):1008–1025.
- Placement (35 references now in the bibliography):
  - Intro: the drug-encoder sentence cites all seven models plus chemCPA and MAP, grouped by encoder type. The Bai
    sentence names the seven and adds Ahlmann-Eltze 2025 and Wong 2025 (simple baselines in genetic perturbation).
  - §2: the L1000 row of the benchmarks table.
  - §3.3: Morgan/ECFP; Repurposing Hub and Tahoe metadata for the annotations; Cell Painting (Bray 2016, Way 2022).
  - §3.4: pairwise and multitask kernels: Gönen & Margolin, Costello, pairwiseMKL, Stock 2018.
  - §3.5: Molina & Zhang (perturbation × line interaction). Context–compound couplings: TranSiGen/PRnet
    concatenation, XPert cross-attention, CPA/chemCPA decoder, Hodos tensor completion, Rhaister-O gene-wise control
    slopes. Varying-coefficient models (Hastie & Tibshirani) and FiLM for the gene-wise rescaling.
  - §4.4: STRING, Maslov–Sneppen rewiring, network priors (DeepCE, GEARS, GSNN, XPert).
  - Discussion: unseen cell lines (Rhaister-O, Molina & Zhang).
  - Appendix `tab:app-zero`: one citation per model row.
- Page budget, from a local tectonic build with relaxed float parameters so that the figures sit in the text: the
  conclusion moved from the bottom of PDF page 10 to line ~6 of page 11. That is within the limit (page 11). With the
  default float parameters, Figs. 2–3 still float after the references; check on Overleaf.

**2026-11-02 00:40 — Fig. 1 redesign: compound cards (user sketch).** 用户手绘了新版 Fig. 1，要求以三张化合物卡片为主视觉。
- User design: three cards (daunorubicin, mitoxantrone, vorinostat), one row per view. Each view kernel shown concretely
  over the three compounds. Fusion and normalization drawn as one step. Kernel transfer split into shared and per line.
  Mark what comes from DMSO and how the residual is formed. Draw the parts first, then assemble.
- Calibration against the code (agent), to be built into the new figure:
  - residual = (treated − DMSO) − backbone(DMSO), on profiled compounds only (`cache_undetect.py`: `R = Y − B`, `Y` =
    delta). The sketch's "treated ⊖ predicted Δ" omits the DMSO subtraction.
  - The interaction's context z = leading PCs of the control expression (`stage2_context.py` L79–81), not the
    backbone output. The sketch's DMSO → "context interaction" line is right; the current Fig. 1 (v4) draws the ×
    input from the backbone output, which is wrong.
  - The final sum adds the backbone output (context) of the query sample; the sketch has no such arrow.
- Real values for the three compounds (`drugs.tsv`, Cell Painting npz; mean KTA weights over the five `cache_v1`
  folds: chem 0.290, tgt 0.245, moa 0.226, morph 0.238). Morgan Tanimoto: dau–mito 0.112, dau–vor 0.080, mito–vor
  0.137 (chemistry alone puts mitoxantrone nearer vorinostat). Targets: dau–mito 1 (TOP2A). MoA: dau–mito 0.5.
  Cell Painting cosine: dau–mito 0.151; vorinostat has no profile. Normalised fused kernel: dau–mito 0.427, dau–vor
  0.026, mito–vor 0.046. Daunorubicin entry = BRD-A37630846 (most samples, 42 lines; four other BRD entries exist).
  Proposed roles: mitoxantrone = unprofiled query, the other two profiled (illustrative; in the real folds they are in
  groups 1 / 0 / 2).
- 01:30 draft v5 rendered (parts + assembly), not yet in the paper. Generators `scripts/figures/fig1_cards_data.py`
  (deepce-gpu env: kernels, weights, 2D structures → `fig1_cards_source.{json,csv}`) and `scripts/figures/fig1_cards.py`
  (gnn-gpu env; `--parts` writes each part separately). Draft outputs in the session scratchpad (`fig1v5/`).
  Layout: cards → 2×2 view kernels (real values) → fusion + normalization (real mean KTA weights as a stacked bar;
  query row highlighted) → kernel transfer drawn as the query's similarity weights on the profiled residuals, shared
  (all lines) and per line (line ℓ) → + → compound effect → × interaction (control PCs from DMSO) → + context →
  response. Dashed region marks the profiled-compound path (treated − DMSO = measured Δ; − context = residual).
  Kept the project view palette (green/gold/pink/violet), not the sketch's red/yellow/pink/blue, because blue = UnDetect
  and Figs 2–3 share the palette. Simplification to state in the caption: the transfer weights shown are K̃ entries;
  the model applies (K̃_DD + λI)⁻¹ (λ = 3 in all five folds). Pending user approval before replacing
  `fig1_pipeline_preview.*`, the caption and the dossier (whose "no numbers" rule changes: kernel values are now data).
- 02:30 v6 after user review (same generators; outputs in scratchpad `fig1v6/`). User asks, all done: profiled
  cards (D, V) then an ellipsis card slot, query M at the bottom; kernels as one front matrix (structure) with the
  other three views stacked behind; no numbers anywhere (shading only); ellipsis row/column in every matrix, M last so
  the fused kernel's bottom row goes straight down into kernel transfer; the four weight bars removed from fusion
  (user asked why they were there: they were the mean KTA view weights, 0.23–0.29, and needed explanation); cards
  redesigned as structure left + four view sections on view-coloured backgrounds (Morgan fingerprint with the real
  on-bits as a barcode, targets, mechanism, Cell Painting profile / none); syringe-on-plate icon before Treated; right
  column top-to-bottom and straight (fusion → transfer → + → compound effect → × → + → response), with control PCs and
  context entering from the left; one unavoidable crossing (context → residual over the control-PC line) drawn as a hop.
  - Palette change (user: Cell Painting green so it is not confused with the fusion blue): `style.py` VIEW is now
    chem #d33f49 (red, as in the sketch), tgt #eda100, moa #e87ba4, morph #1baf7a. Figs 2 and 3 pick this up on their
    next regeneration (Fig. 2's chemistry-kernel bar becomes red next to the orange plate baseline; check contrast).
- 02:25 Fig. 1 v6 placed in the paper (user: "先替换fig1，包括svg等"; not committed, not pushed).
  - `paper/figures/fig1_pipeline_preview.{pdf,svg,png}` and `fig1_pipeline_source.{json,csv}` regenerated by
    `fig1_cards_data.py` / `fig1_cards.py` (both now default to `paper/figures`). User (mid-turn): component SVGs too
    → `paper/figures/fig1_parts/fig1_part_{cards,kernels,fusion,transfer,context}_preview.svg` (`--parts`;
    `style.save` gained an `exts` argument, default unchanged).
  - v4 assets kept in `paper/legacy/fig1_v4/`; `scripts/figures/fig1_pipeline.py` marked superseded and now writes
    `fig1_pipeline_v4*` only.
  - Caption rewritten in `sections/01_introduction.tex` (worked example; `% SRC:` report §9, added today with the
    kernel values); dossier `figures/fig1_pipeline.md` rewritten for v6, caption mirror synced (`sync_captions.py
    --check`: all in sync); `figures/BRIEF.md` Fig. 1 section updated (layout fixed, Hanzhe polishes only).
  - Build check (tectonic, scratch copies): compiles, no undefined references. Page budget with relaxed float
    parameters: "Code availability" on page 11, line 24 of 54 (old v4 figure and caption: line 8). Within title + 10.
    With default float parameters the 4.42-in figure plus caption exceeds `\topfraction` (0.7) and Fig. 1 floats to
    after the references, joining Figs 2–3 (known since 11-01). Fix options for the user: float parameters in
    `main.tex` (fixes all three) or a shorter figure/caption.
- 02:55 v7 layout (user mock-up arranged from the part SVGs: "kernels 大一点，字小一点放在左边，kernel transfer 放在下面离
  plate 近一点，避免电路图式长线"). Deck 0.92 in with view names (4.8 pt) in its lower-left corner; kernel transfer in the
  bottom band right after the residual (straight into the per-line row, short branch to the shared row); fusion box
  centred over the transfer's similarity column (straight drop); output column at x = 6.16 with labels on the left;
  canvas 6.5 × 4.60 in. Written to `paper/figures/` with `fig1_parts/*.svg`. Page check (relaxed floats): page 11,
  line 24 of 54 (unchanged). Control-PC and context lines stay long straight horizontals (DMSO is far left, the
  interaction and sum far right); the residual elbow is gone.
- 03:20 Float placement and push (user: "确实 我们需要让fig都在尽量正确的位置。然后可以push了").
  - `main.tex`: `\topfraction` 0.9, `\bottomfraction` 0.8, `\textfraction` 0.07, `\floatpagefraction` 0.85,
    topnumber 3, totalnumber 5 (fonts, margins, spacing untouched). Default build (tectonic): 15 pages; Fig. 1 cited
    p2 → placed top of p3; Fig. 2 cited p4 → p5; Fig. 3 cited p9 → p10; text ends p11 line 24 of 54; no undefined
    references. Before: all three figures after the references (16 pages).
  - Pushed `paper/` eb0f124..d94f79b: 376ea72 citations (the 11-01 reference work, previously uncommitted; the intro's
    citation hunks staged separately from the Fig. 1 caption), ba3ee73 Fig. 1 v7, d94f79b float settings.
    `legacy/fig1_v4/*.pdf` stays local (repo ignores PDFs outside `figures/`); its v4 copy is in eb0f124.
  - Overleaf still needs Menu → GitHub → Pull GitHub changes into Overleaf.

### 2026-11-02 03:40 — User: Tahoe sparsity test, residual-selected cross-line weights, and the external reviewer's four questions (plans declared before any run)

User (03:30): run the Tahoe sparsity test; it can become a conclusion. Test choosing the cross-line weights by the
residual metric if the agent thinks it is sound; the user checks the result. Update the paper: the Tahoe residual
paragraph and the Molina & Zhang positioning (Codex's wording as a reference). Assess which of the external (GPT)
reviewer's four questions can be resolved: (1) how much compound-specific signal PCC_plate isolates, under a
compound-label randomisation that keeps plate, line and missing-view pattern; (2) whether the interaction's
leave-one-out training inputs are out-of-sample with respect to the KTA view weights; (3) the gain by the test
compound's view coverage; (4) the scope of the Tahoe cross-line claim (wording).

Fact checked for (2), `cache_undetect.run_fold`: the KTA view weights are computed from the mean residuals of **all**
training compounds, so the left-out compound did enter its own weights; λ and (g, h) are chosen on valid rows and are
not affected. The backbone B on training rows is also an in-sample prediction (drug-free MLP trained on those rows).
The analytic leave-one-out is exact for the fixed fused kernel only. Fact for (3): in `drugs.tsv` targets and MoA have
the same coverage (2,957 of 8,276 = 35.7%; corrected 04:00 from 2,955, a rounding slip); "pharmacology" below = targets/MoA present.

**E1 — Tahoe sparsity (`run_l1000_xline.py --keep-lines k`).** Each training compound keeps its rows in k randomly chosen
lines (fixed per compound across folds, seed 0); its other rows are dropped from every fit (line means, per-line KRR,
consensus g, inner CV). Test compounds keep all 50 lines, so the test rows are those of the complete-grid runs. k ∈
{5, 10, 25}; k = 50 is the existing `_Dw` run. Slices T-0.5 and T-5, ik14 split, seed 0, kernel D. Arms `mkl_D`
(per-line), `pool_D` (+ cross-line). Statistic: Δ = `pool_D` − `mkl_D`, line-macro, ACTIVE rows, ik14-group bootstrap,
full and residual Pearson. Hypothesis (record 2026-10-31 18:30): the cross-line term adds compound information only
when a line has not measured the compound. **Rule:** the conclusion "on Tahoe the cross-line term improves the
compound-specific deviation once compounds are profiled in few lines" is stated if at k = 5 the residual Δ has a CI
above 0 at both doses and the residual Δ at k = 5 exceeds the k = 50 value at both doses. Otherwise the paper keeps the
observed contrast without the complete-grid explanation. With k = 5 a line has ~28 training compounds (≥ 20, the
script's minimum).

**E2 — residual-selected cross-line weights (arm `poolr_D`, same runs and the k = 50 run).** In plain terms: the model
predicts line mean + a·(cross-line term) + b·(per-line term); a and b are now picked by inner CV on full Pearson, which
the line mean dominates on Tahoe (0.644 of 0.667). `poolr_D` picks (a, b) on the inner-CV residual Pearson instead
(prediction and truth minus the inner line mean); nothing else changes. **Rule (reported to the user, who checks it):**
if on the complete grid `poolr_D` − `mkl_D` residual Δ has a CI that includes or exceeds 0 at both doses, the residual
drop of `pool_D` is attributed to the selection criterion; its full-Pearson cost is reported alongside.

**E3 — joint compound-label permutation null for PCC_plate (`sdst_view_ablation.py --sets permjoint`).** All four view
kernels get the same permutation of compound labels, drawn within strata of the compound's view-availability pattern
(which of chem / tgt / moa / morph it has), over all 8,276 compounds. Rows, plates, lines, backbone (`cache_v1`) and
coverage are unchanged; only which compound's information a row receives is broken. Full pipeline (KTA weights,
leave-one-out terms, interaction) refitted. Permutation seeds 0–4, five folds each. Statistic: PCC_plate and PCC_DEG of
`transfer` and `undetect` minus the backbone, per fold, against the real run (`views_final` multi_norm: +0.0600
PCC_plate). **Rule:** "the PCC_plate gain requires compound identity" is stated if every permuted run is below the real
gain in every fold; the permuted gain (mean, max over seeds) is reported as the part not attributable to compound
information. Answers question (1) together with the existing `pool_plate` control (+0.0006).

**E4 — fully cross-fitted interaction inputs (new `sdst_crossfit_interaction.py`).** On the `cache_v1` backbone, per
fold: valid/test transfer terms as in `cache_undetect`; training-row G and H replaced by 5-fold cross-fitted values over
training compounds (seed 0), each fit recomputing the KTA weights and the fused kernel from the in-fold compounds only
(λ fixed at the cache value). Arms: `undetect_loo` (must reproduce 0.1600), `undetect_xfit` (weights refitted),
`undetect_xfit_fixw` (cross-fitted, weights fixed: separates fewer training compounds from the weight leak). The backbone
B stays in-sample on training rows (not addressed; stated as a limitation if relevant). **Rule:** the interaction gain
(`undetect` − transfer, PCC_plate) is reported as robust to full cross-fitting if under `undetect_xfit` it is > 0 with
compound-bootstrap CI > 0 in ≥ 4 of 5 folds; otherwise the paper reports the reduced gain.

**E5 — gain by view coverage (analysis only, `final_cpu` predictions).** Test compounds in four strata: structure only,
pharmacology without Cell Painting, Cell Painting without pharmacology, both. Per stratum: UnDetect (`ctx_pc`) −
backbone, multi-view transfer (`pool_kta`) − chemistry-only transfer (`pool_chem`), PCC_plate and PCC_DEG, compound
bootstrap. No decision rule: descriptive, defines the paper's scope statement. Already known (report §8.2): the gain of
multi-view over chemistry is −0.0026 for compounds without targets and without Cell Painting.

Paper: rewrite the Tahoe residual paragraph after E1/E2; add the Molina & Zhang positioning sentence now.

### 2026-11-02 04:05 — E1/E2/E5 read; an unplanned diagnostic for question (1); paper edits so far

- **E1 (report §10.1): rule met.** At 5 lines per training compound the cross-line term raises the residual Pearson by
  +0.015 at both doses (CI > 0). The gain falls with coverage, to −0.005 / −0.002 on the complete grid; the full-Pearson
  gain stays at +0.005 to +0.007 throughout. The complete-grid explanation of 2026-10-31 18:30 is therefore supported
  and becomes a stated conclusion in the paper.
- **E2 (report §10.2): met at 0.5 µM, missed at 5 µM by a hair** (upper bound −0.00005). Selected on residuals, the
  complete-grid model switches the cross-line term off in 235 / 216 of 250 cases. A design flaw found while reading it:
  residual Pearson is scale-invariant, so with a = 0 the scale b is not identified and the grid's first value
  (b = 0.25) is chosen, which is what costs the full Pearson. A cleaner variant (ratio a/b on residuals, overall scale on
  full) was not declared and is not run. **The user checks E2;** it is not in the paper.
- **E5 (report §10.3), descriptive.** For structure-only compounds (57%) multi-view is slightly below chemistry
  (−0.004) and XPert is ahead of UnDetect on PCC_plate (−0.013). With Cell Painting UnDetect leads XPert (+0.020 to
  +0.036). The blend leads XPert in every stratum. The paper's Discussion now states this scope.
- **Unplanned diagnostic for question (1)** (`scripts/analyze_plate_mate_count.py`, report §10.6). Reason: the
  reviewer's mechanism (error in m_i enters both sides) predicts that a drug-free model's PCC_plate falls with the
  number of training plate-mates, which is checkable from saved predictions. It does: backbone 0.536 at ≤ 2 plate-mates,
  0.035 at > 50 (47% of rows). The UnDetect − backbone gain is +0.023 to +0.087 in every bin. Not declared in advance;
  reported as a diagnostic. Seen in the same table: XPert leads UnDetect where m_i is noisy (≤ 20 plate-mates: −0.022,
  CI < 0), and UnDetect leads where it is precise (> 20 plate-mates, 70% of rows: +0.014 [+0.008, +0.020]). Why XPert
  gains most on small plates is not established.
- Paper (local, not pushed): intro sentence on Molina & Zhang (source-line responses vs static descriptors; our question:
  can pharmacology/morphology supply the prior for a compound with no transcriptional profile); Tahoe paragraph
  rewritten on E1; Discussion scope (E5) and the Tahoe limitation (question 4). The abstract is not changed: it is
  frozen, and its Tahoe sentence is true for the full correlation; the user decides whether a minor edit is wanted.
  Build: text ends on page 11, line 34 of 51 (title + 10).

### 2026-11-02 04:20 — User decisions on presentation; paper edits (not built, not pushed)

User asked: (1) is Cell Painting the only place we beat XPert, and does reporting the structure-only loss break the
launch-style (non-defensive) principle; (2) should the UnDetect + XPert ensemble be in Table 2; (3) is Tahoe a positive
result; (4) report the > 20 plate-mate lead over XPert? Agent's answers: (1) No. UnDetect beats XPert on PCC_DEG in every
coverage stratum; on PCC_plate it leads with Cell Painting, ties with pharmacology only, and trails only for
structure-only compounds. Main text: positive gradient + ensemble, structure-only comparison removed from the main text;
full table in the appendix (data kept, not hidden). (2) Yes, as a separated, clearly labelled ensemble row. (3) Yes:
positive where lines are sparsely profiled, with the complete grid as the boundary case; sparsity is a simulation and is
stated so. (4) Yes, with per-bin table in the appendix; the 20 cut-off is post hoc, so the main text points to the bins.
User (04:15): "这些先改了，然后不用检查别的，之后等结果到了再改他们的部分，然后再检查。"

Edits to `paper/` (local):
- Table 2: new row "UnDetect + XPert (ensemble)" 0.661 / +0.023 / 0.186 / +0.086 below a rule; caption: bold = best
  single model, ensemble definition.
- §4.1:
  - plate-mate diagnostic sentence (backbone 0.54 → 0.03, gain at every plate size; PCC_plate read as a gain over the
    drug-free backbone);
  - XPert tie split by plate-mate count (+0.014 above 20, XPert ahead below).
- Discussion: removed "for structure alone the end-to-end encoder is ahead by 0.013"; points to `tab:app-coverage`.
- Appendix: `tab:app-coverage` (report §10.3) and `tab:app-platemates` (report §10.6).
- Pending E3/E4: the permutation null in §4.1 and the interaction cross-fit statement, then a build check.

### 2026-11-02 04:35 — E3 and E4 verdicts (report §10.4, §10.5)

- **E3: rule met.** Under a joint compound-label permutation within view-availability strata, UnDetect falls to the
  backbone: PCC_plate 0.0984, gain −0.0016 with CI < 0 in all 25 runs; PCC_DEG gain +0.0002. Real gain +0.060. With the
  plate-mate diagnostic (§10.6) this answers the reviewer's question (1): the drug-free 0.10 comes from error in the plate
  mean; the gain above it needs compound identity.
- **E4: rule met.** With view weights refitted inside a 5-fold cross-fit, the interaction gain is +0.0050 (5/5 CI > 0),
  identical to leave-one-out. Answers question (2). Remaining caveat (not tested): in-sample backbone on training rows.
- Paper next: §4.1 permutation sentence; interaction paragraph and Method note on cross-fitting; then build and check.

### 2026-11-02 05:00 — External reviewer's follow-up: wording tightened; the two split questions checked

Reviewer (GPT, pasted by the user): keeps weak accept with higher confidence. Asks to tighten the claims: plate-mean error
is the *main* source of the backbone's PCC_plate, not provably all of it; E3 shows the gain needs the correct
compound–view correspondence, not that it is an isolated biological effect; the real result is not 25 independent
replicates. E4 should be called compound-stage cross-fitting (backbone fixed). Two sentences in the main text; protocol
in the appendix. Two remaining questions:
- **Sparse Tahoe holds out test compounds entirely: yes.** ik14 split, `train_cf = ~test_cf` per compound
  (`run_l1000_unseen_drug.make_folds`). `--keep-lines` thins only training compounds' rows (`tr_rows = train_cf & keep_row`),
  and inner CV runs over training compounds. Test compounds enter no fit in any line.
- **Ensemble weight vs the target fold's labels: not strictly.** It was chosen on fold f+1's test rows, from models
  trained with group f. The clean rule (fold f−1) was checked: w = 0.4 in every fold again, so the numbers are
  identical. Fixed w = 0.5: still above both in 5/5 (report §10.7).

Paper edits (local): "mainly because" for the plate-mean explanation; the permutation sentence replaced by the
reviewer's wording; Method sentence → "With the backbone fixed, five-way compound-stage cross-fitting, including
re-estimation of the view weights, retains the interaction gain …"; ensemble weight described as "chosen on a fold whose
models never saw the test compounds" (Table 2 caption and §4.1); new appendix subsection "Controls for the compound
terms" (permutation protocol: one mapping per compound applied to all its samples and all four kernels, refit scope,
backbone fixed; cross-fit protocol) with per-fold table `tab:app-controls`. Report §10.4: real-compound row relabelled
as one run per fold.

### 2026-11-02 05:20 — User: push; E2 revised with b fixed at 1; abstract unchanged; Fig. 1 lower-right routing

User (05:15): push first (done: paper 8602645, Therapeutic 3ddaf83). E2: fix b = 1 and search a, so that a = 0 reads as
"under complete training coverage the model finds no stable extra cross-line gain". Abstract: no change (true as
stated). Fig. 1: the lines in the lower-right corner bend outwards; they should turn inwards.

**E2b (declared before running).** New arm `poolb_D` in `run_l1000_xline.py`: b = 1 (per-line term at full weight), a in
GRID_A chosen on the inner-CV residual Pearson. At a = 0 it equals `mkl_D` exactly, which removes the scale ambiguity of
`poolr_D`. Runs: T-0.5 and T-5, complete grid and k = 5, 10, 25 (same tags, rerun). Report the share of line × fold cases
with a = 0, and `poolb_D` − `mkl_D` (full, resid) with the ik14 bootstrap. **Reading rule:** the paper may state that
"under complete training coverage the model finds no stable extra cross-line gain" if on the complete grid a = 0 in
most cases (> 50%) at both doses and the residual Δ CI includes 0 at both doses.

### 2026-11-02 05:45 — E2b verdict (report §10.8): rule met at 0.5 µM, missed at 5 µM; paper unchanged

Complete grid: a = 0 in 94% / 86% of line × fold cases. The residual Δ vs the per-line model is +0.0001
[−0.0000, +0.0003] at 0.5 µM and −0.0007 [−0.0014, −0.00003] at 5 µM. By the declared rule the sentence "under complete
training coverage the model finds no stable extra cross-line gain" is not licensed (5 µM CI excludes 0, by a hair).
Agent's option for the user: a descriptive sentence without the rule's claim, e.g. "chosen on deviations, the
cross-line weight is zero in 86–94% of line × fold cases on the complete grid", which is a fact; or leave E2b out.

### 2026-11-02 06:10 — User: E2b stays out of the paper; headings, wording and two figure fixes (pushed)

- E2b (report §10.8) is not used in the paper (user).
- Style (user): avoid "What X does" and "X is Y, not Z". §2 → "Benchmark definition" with 2.1 "Task", 2.2 "Plate
  structure", 2.3 "Plate-centred metric"; Results headings shortened to "Drug-blind benchmark", "DeepCE benchmark",
  "Cross-line transfer", "Compound views". 2.2, 2.3, 4.1 and 4.3 were not named by the user; they were changed for
  consistency. About a dozen sentences and two captions were rephrased.
- Fig. 2a: drug-free MLP line removed (user). UnDetect's drug-zeroed point is the drug-free backbone at the same level.
- Fig. 3d: compound names beside each strip and a colour key with an explanation. Fig. 3 picks up the view palette from
  the user's style.py change (chemistry red, Cell Painting green, as Fig. 1), now committed.
- Build: no undefined references; the main text ends on p. 11 with references starting on that page.
- Pushed: paper 4d5fd42, Therapeutic 0cf55a6.

### 2026-11-02 06:50 — Figures: Fig. 2 order corrected, DeepCE table to the appendix, joint-shuffle row, appendix figures

User: Fig. 2c/d highest on top (correction); less grey text in Fig. 3; Table 3 to the appendix (Fig. 2d covers it).
Visualisation ideas discussed; user chose: (1) "all shuffled" row in Fig. 3a; (2)+(3) Tahoe sparsity curve and
gain-by-coverage as an appendix figure; (4) plate-mate figure in the appendix; all cited from the main text. Generator
`scripts/figures/fig_appendix.py` (sources: report §10.1, §10.3, §10.6). Main text unchanged in length (ends p. 11,
line 33 of 52). Pushed: paper a523564 and the commit after it; Therapeutic 71a3044 and the commit after it.

### 2026-11-03 02:30 — Review of the user's hand edits (paper 166bc5d); agreed fixes applied locally, not pushed

User pushed wording changes, TODOs and margin comments (166bc5d) and asked for a justification of each. The agent's
item-by-item review (2026-11-02 evening) was accepted ("可以"). Applied in `paper/` (uncommitted, awaiting the user's
diff review):

- **Intro.** Grammar fixes kept ("Existing", serial comma, "taken into account"). Reverted or rewritten:
  - "The need for a model…": a need does not turn atlases into a screen.
  - "Upon that … also provides": now "We show that … can supply", an assertion that Results support.
  - "Using that information, Bai et al. retrained…": factually wrong, since Bai et al. used no such views. Topic
    sentence restored.
  - "However, if we predict…": no contrast, and the name "plate-response baseline" must be introduced here.
  - "For a first approximation" → "To".
  - **"only one model XPert can learn the compound": contradicts Results.** TranSiGen's PCC_plate drops from 0.144 to
    0.101 when zeroed. Now "the strongest published model, XPert, does learn the compound".
- **Fig. 1 caption.**
  - Restored "(hatched where a view is missing)": vorinostat has no Cell Painting profile
    (`fig1_pipeline_source.json`).
  - The kernel sentence had typos and a "which" that attached to daunorubicin; it was rewritten with the reason
    (shared target and MoA) restored.
  - The backbone predicts the compound-independent response (the edit had said "the compound response"), and the
    residual definition is restored (the next sentence refers to "these residuals").
  - The in-caption "TODO" (plain text, invisible to the drafting-macro grep) is replaced by "DMSO (vehicle)". DMSO is
    now also defined in §2.1.
- **§2.**
  - Dose settings stated: drug-blind and LINCS one dose and time (10 µM, 24 h); Tahoe per dose; only DeepCE resolves
    doses in one model.
  - Heading typo fixed.
  - The commented-out drug-free-network sentence is restored and reworded. It is the main-text support for the intro's
    "to a first approximation" claim.
  - Name kept as **plate-centred** ("agnostic" would say the metric ignores the plate; it subtracts it).
- **Fig. 2 caption.**
  - A mid-line `%` had deleted the attribution "models we did not run from Pham et al." and "the MLP is given the same
    compound views"; both are restored.
  - Panel (c) legend (large/small dots) restored; "replaced by zeros" kept.
  - The undefined "compound lane" is now "drug-free backbone"; the header now says (d) is one published split.
  - Fold dots for (a) and (b) are added to Hanzhe's todo and `figures/BRIEF.md`.
- **Level-5 paragraph (user: evidence?).**
  - "Removes *most* plate structure" was unquantified; it is now "remove the shared plate response before modelling".
  - The old text called Tahoe "Level-5"; it is now the per-plate DMSO subtraction.
  - The plate-map split now carries a SRC (10_09_record 2026-10-03 21:10; 10_30_report §4.1, §4.4).
- **§3.**
  - **"b fixed to 1" (user): not so in UnDetect.** `cache_undetect.py` searches (g, h) over SCALES², and g = h = 0.5 was
    chosen in all five folds (`cache_v1/fold_*/summary.json`). b = 1 was E2b (`poolb_D`, Tahoe; record 2026-11-02
    05:20), which is not in the paper.
  - The question exposed a **wrong Model-selection sentence**: "λ, a, b, c ∈ {10⁻³ … ∞}". The real grids are in the new
    appendix Table `tab:app-grids` (drug-blind LAMS/SCALES/CS/KS; LINCS/Tahoe GRID_LAMBDA/GRID_A/GRID_B/GATE_CS with
    inner CV; DeepCE LAMS/RHOS/SIGMAS).
  - Stage-separation sentence: the cross-reference to §2 was wrong (joint training is shown in §4). Rewritten; the
    user's "reason and novelty" phrasing was not adopted.
  - Backbone optimiser details moved to the new appendix §`app:impl`. The training-compounds-only, early-stop-on-
    validation and frozen facts stay in the main text.
  - Zero-filling sentence kept, with a reference to Prop. 1.
  - "In a sense … should inform" reverted to "so … informs", a consequence of the equation.
  - Dose variable s and σ are now defined.
  - Interaction subsection reordered: what the model is, then fitting/LOO, then related designs last. The Molina
    genetic-screen sentence is dropped (still cited in the intro). Hodos and Rhaister are kept.
- **§4.** "Chemistry gives about half of the gain (+0.006 of +0.011)" is removed. The table row and the views section
  carry it.
- **Build** (tectonic, scratch copy): no undefined references. The main text ends on p. 10 and Fig. 3 floats to p. 11
  (before: text ends on p. 11). Overleaf/pdfLaTeX placement still to be checked. `sync_captions.py` refreshed the
  fig1, fig2 and fig_views dossiers.

**Wall time (user: "now computable").** Existing numbers are not usable as they stand:
- `cache_v1` ran on CPU (46–63 min per fold, backbone and kernel stages lumped together).
- XPert fold 0 (job 357100) took 268.2 min on A100-SXM4-40GB (gpu-52), early stop at epoch 145.
- Backbone parameters, 978→2048×3→978: 14.40M, against XPert's 5.60M.

Submitted 364344 (`slurm/time_undetect_gpu.sbatch`): all five folds on one A100-SXM4-40GB, tag `timing_gpu`.
- Logged: backbone time, fold totals, stage-2 total, peak GPU allocation and host RSS.
- The stage-2 total is an upper bound, because `--v2` fits all interaction arms on CPU.
- A first submission with an `nvidia-smi -l` sampler was rejected by the cluster's idle/monitoring filter.
- Numbers go to the report before the paper.

### 2026-11-03 03:25 — User: §3.5 formula first; title page; Results rewritten with fewer numbers (local, not pushed)

User asked for three changes:
1. §3.5 was hard to follow: put a formula first, then explain it step by step.
2. Does the abstract page count towards the 10 pages? If it does, start the introduction right after the abstract.
3. Rewrite §4 with the press-release principle. Keep only key, intuitive and striking numbers; say "improves" and
   point to the table for differences in the third decimal.

- **Title page.** The CFP (`RECOMB_requirements.md`) allows "an additional title page containing the paper title,
  author list, and abstract" on top of 10 pages. The abstract page does not count. If the introduction started on it,
  the page would no longer be a pure title page and the limit would become 10 pages in total, losing room. The
  `\clearpage` is kept (agent's recommendation; told to the user).
- **§3.5.**
  - Opens with the per-gene ridge objective, eq. `eq:interaction`: ũ is the LOO compound effect, and the penalty
    κ·s, with s = mean over genes of Σ_i ũ², is applied to all K+1 coefficients, as in `stage2_context.gate_fit`.
  - Then labelled steps: Role, Context, Closed form, Leave-one-out inputs, Without a per-sample control, Related
    designs.
  - The penalty factor is renamed c → κ, because c_d (mean residual) appears in the same paragraph. Changed in Method,
    Model selection, appendix grids, appendix text (κ = 1 chosen) and the joint-shuffle caption.
- **§4.** Each paragraph now leads with its claim.
  - Numbers kept: 0.637 / 0.650 / 0.632; PCC_plate 0.10 → 0.16; no plate-mate 0.536 → 0.574; XPert 0.157 → 0.106
    when zeroed; end-to-end same features 0.613; ensemble 0.661 / 0.186; DeepCE 0.602 vs 0.491, 79/92, MLP 0.544;
    Tahoe sparsity +0.015; P1-24 coverage 73% / 91%; Cell Painting +0.05 to +0.09; pair similarity 0.29 vs 0.17;
    panobinostat–vorinostat r = 0.94; gain without analogue 0.023 vs 0.006–0.008.
  - Moved to words plus a table or figure reference, with the values kept in `% SRC`: plate-kernel +0.006/+0.001,
    joint-permutation 0.098, per-bin gains, XPert lead 0.014, TranSiGen values, interaction +0.005/+0.001/+0.003/+0.006,
    residual-head values, DeepCE bootstrap intervals, resplit ranges, per-setting LINCS/Tahoe gains, view ablation
    costs, pair bins.
  - Not in the paper any more: the 0.05 µM sentence (moved to the `tab:xline` caption); the complete-grid negative
    deviation numbers (−0.005 / −0.002). The text now says that on the complete grid the cross-line gain is
    shared-across-lines response. The facts stay in report §3.2 and §10.1.
  - New claim checked: TranSiGen PCC_DEG 0.614 is below the drug-free network (fig2 source CSV), so "both stay below"
    holds.
- **Build** (tectonic): no undefined references. The main text ends on p. 11 (= title page + 10). Discussion now
  starts on p. 9.

### 2026-11-03 04:45 — Method rebuilt around one closed-form block; Proposition demoted; Results fixes; Discussion rewritten; oracle recomputed (local, not pushed)

**User (03:40–04:10):**
- §3.3: is the Proposition needed?
- Re-read §3–§4 for readability, AI style, notation and agreement with experiments.
- E: "decreasing trend" is fine.
- F: simplify, few numbers.
- Write the closed-form solution in the Method overview and explain each part in its subsection, with all indices
  unified and disjoint.
- Discussion: five highlights.

The agent's answers were accepted ("ok"):
- Proposition: demoted, with the reasons given in chat.
- Discussion: the five points reframed (see below).

**Agent findings in §3–§4, now fixed:**
- A: "recover both metrics / kernel transfer remains ahead" was wrong. Transfer alone vs the residual head is +0.003 with
  no fold CI excluding 0. Now "keep the compound signal and recover the context; UnDetect remains ahead on PCC_plate"
  (+0.008, 5/5 folds > 0, CI 3/5).
- B: "mainly because" restored.
- C, D: Level-5 vs Tahoe wording.
- E: "decreases as coverage grows".
- F: one sentence. The [0.3, 0.5) bin (+0.015) had been omitted.
- G: "mostly well-studied ones" removed (no evidence); coverage 36% / 23% now has a SRC (§10.3, 2,957 and 1,937 of
  8,276).
- H: "gating" → rescaling.
- "28 of 28 LINCS evaluations" aligned with the frozen abstract.

**Notation, now disjoint:**
- Indices: i sample, j gene (G = 978), d compound, ℓ line, v view.
- Sets: 𝒯, D, D_ℓ.
- Number of principal components K → q; the kernel keeps K. This is also changed in the appendix grid table, the
  "κ = 1, q = 20" sentence and the controls caption.
- Interaction scale s → ν; s stays the dose.
- Proposition indices i, j, g, D → d, φ, Δ.
- LOO → leave-one-compound-out.

**Method:**
- §3.1 is an alignat block: backbone, kernel, transfer, interaction (β_j = (Φ_jᵀΦ_j + κνI)⁻¹Φ_jᵀ(r_·j − ũ_·j),
  Φ_j = diag(ũ_·j)[1, Z]), then the prediction eq:predict, with one sentence per stage.
- This matches `stage2_context.gate_fit` (ridge of r − u on u·[1, Z], penalty on all q + 1 coefficients) and
  `cache_undetect` (KRR without intercept, one λ, KTA weights normalised to sum 1, fused kernel normalised).
- The Proposition became a 4-sentence argument in §3.3.
- Intro contribution 2 no longer cites prop:coverage.

**Oracle (job 364531; report §11):**
- On rows with a plate-mate and n_d ≥ 2 (99.3% of plate-mate rows): backbone 0.099, UnDetect 0.160, XPert 0.156,
  ensemble 0.186, uniform oracle 0.201.
- Share of the backbone → oracle distance: UnDetect 59%, XPert 57%, ensemble 85%.
- The user's "theoretical upper bound 0.20" is not an upper bound. It uses test responses, and a control-weighted
  oracle was higher (§6.2). The paper calls it an oracle.
- The ensemble value is 0.186 (the user remembered 0.183).

**Discussion (user's five points, reframed by the agent and agreed):**
1. Only UnDetect beats the drug-free network on the drug-blind benchmark (scoped; DeepCE does beat the mean on its own
   benchmark). "Just a closed-form model" is phrased as "a closed-form model suffices". Oracle headroom as above.
2. Modalities are framed as "the gain grows with the information available about a compound" (§10.3), and Cell
   Painting as the measurable view for any new compound. The structure-only stratum, where XPert leads UnDetect by
   0.013, is not made a comparison.
3. Cell Painting conditions. Correction: the profiles are A549, 48 h, 10 µM, not 24 h. Framed as "one A549 profile
   transfers to 164 lines; condition-matched morphological profiles are the next input", not "higher resolution".
4. PPI: "only degree, stop trying" was not used (one experiment; graph = degree stays an open hypothesis). Now:
   calibrate network priors against rewired nulls, and what the PPI kernel adds does not depend on wiring beyond
   degree.
5. "Linear beats DL" is inaccurate: the backbone is a 14.4M-parameter MLP and KRR is nonlinear in compound features.
   Now "structure matters more than capacity", citing Ahlmann-Eltze et al. (bioRxiv 2024, Nat Methods 2025; the
   paper the user recalled, scGPT/scFoundation/GEARS vs linear) and Wong 2025.

**Build** (tectonic): no undefined references. The main text ends on p. 11 at about 42% of the text block (y = 450 pt
from the bottom), so there is about half a page of slack.

### 2026-11-03 05:05 — Discussion: items kept from the old version; pushed

User asked what to keep from the previous Discussion before pushing.

**Restored:**
- The plate explanation as the opening sentence. Contribution 1 was otherwise absent from the Discussion.
- A cross-line transfer paragraph (28/28, Tahoe sparsity). The abstract cites cross-line transfer, but the new
  Discussion had nothing on it.
- "Ensemble best in every coverage group" (§10.3: blend − XPert > ctx_pc − XPert in all four strata).
- The screening tie-in for Cell Painting.

**Left out:**
- The same-input numbers (they are in the table).
- The single-dose caveat (stated in §2).
- "No better than the line's identity" (self-weakening; Results states the similar gain).

**Build:** the main text ends at 52% of p. 11.

### 2026-11-03 05:10 — Oracle in Table 2; ridge penalties back in Method (local, not pushed)

User:
1. The oracle is not visible in the paper outside the Discussion. Add it and its "fulfilling ratio" to Table 2.
2. Method misses things, e.g. the ridge penalty.
3. Are all five Discussion points covered?

- **Table 2.**
  - New column "gap closed" = (arm − backbone) / (oracle − backbone) on PCC_plate (report §11.1, job 364547).
  - New oracle row: PCC_plate 0.201, Δ +0.101, 100%. PCC_DEG is shown as "--", and the caption gives 0.628 with the
    reason (full-weight measured residuals) and "no intervals".
  - Results §4.1 has one sentence: oracle 0.20, UnDetect closes 60%.
  - The Discussion now uses the all-plate-mate shares (60% / 86%) instead of n_d ≥ 2 (59% / 85%), so every place
    quotes the same number.
- **Method.**
  - The overview names λ (KRR penalty), a and b (scales), κν (interaction ridge penalty), and says that the
    hyperparameters are chosen on validation compounds.
  - §3.4 gives the KRR objective ‖C − K̃α‖² + λ tr(αᵀK̃α).
  - §3.5 gives the interaction objective with penalty κν‖β‖² before saying it is one linear system per gene.
  - Nothing else was found missing on re-check against `cache_undetect.py` and `stage2_context.py`.
- **Build:** no undefined references, no overfull boxes. The text ends at 68% of p. 11.

### 2026-11-03 05:20 — Acknowledgements and author contributions (local, not pushed)

User:
- Add acknowledgements and contributions.
- Acknowledgements: heading only, with a TODO for Jun.
- Contributions:
  - Yuqi: method design and implementation, benchmarks, writing, figures.
  - Hanzhe: figures, paper review, method discussion.
  - Keqin: downstream and result analysis.
  - Jun: overall project, discussions, PI role.

Agent's change:
- New `paper/sections/07_declarations.tex`, input after the Discussion and before the bibliography.
- Acknowledgements: `\todo{Jun Wen to write …}`.
- Contributions use initials.
- Jun's line was worded by the agent ("conceived and supervised the project, set its direction and guided the
  discussions"). Jun or the user to confirm.
- Main text now ends at 81% of p. 11. The acknowledgements count towards the 10 pages, which leaves about 10 lines.

### 2026-11-03 05:30 — Fig. 2a/b fold dots drawn; audit of this chat's revisions against the paper (pushed)

User:
- Why does Fig. 2a have no fold dots?
- Check that every revision discussed in this chat is in the paper.

**Cause.** The agent had only put the fold dots in Hanzhe's brief and the caption todo (02:30) and never drew them.

**Fig. 2.** `scripts/figures/fig2_benchmark.py` now draws them:
- (a): five folds as a strip below each row, filled for with-drug and open for drug-zeroed. Published values come
  from `primary_comparisons.csv` fold_k_full / fold_k_comparator; ours from `final_cpu/scores.csv`. The fold spread
  (about 0.007) is smaller than the markers, hence the strip.
- (b): fold dots at the bar ends (`plate_baseline/summary.json`).
- The source CSV now has per-fold rows.
- The caption says "(small dots below each row: folds)" and "(dots: folds)". The todo is back to "Hanzhe to redraw".
  BRIEF is updated.

**Audit** (grep over all sections for every agreed item from 2026-11-02 evening to now):
- All text revisions are present.
- Remaining slips, fixed: two "leave-one-out" in the appendix controls table → "leave-one-compound-out".
- No leftover user comments, "prop:coverage", old penalty c, PC count K or scale s.

**Build:** no undefined references. The text ends at 81% of p. 11.

### 2026-11-03 06:00 — Appendix completed (paper c15e71e); ONGOING cleared; three external reviews received

User: 先补附录；其他待办暂留；ONGOING 可以清了；讨论三份外部 review（另一个 Claude、GPT、DeepSeek）是否修订。
User comment on the Claude review: "我感觉 Cell Painting ablation 极大地减弱了我们的 claim。要不删掉？"

**Appendix (paper c15e71e, pushed; tectonic build: no undefined references, main text still ends on p. 11):**
- A.3 DeepCE: reproduction paragraph (commit 9b0d04f; seeds 0.481 / 0.456 / 0.428, mean 0.455 vs reported 0.4907;
  `10_09_report.md` §5.3). States that Fig. 2d "DeepCE" is the reproduction and kNN / NN / DeepCE-without-attention are
  Pham et al.'s reported values. Chemistry KRR 0.4911 vs reported 0.4907 are separate measurements (the near-equality
  that one reviewer flagged). Test-compound view coverage 92 / 62 / 62 / 72. `tab:deepce` gains "MLP, chemistry 0.511"
  and "DeepCE (reproduced)". New `tab:app-resplits` (§4.5; exact means from `cv5_seed{0,1}/folds.csv`: chem+tgt+moa norm
  cross-line 0.53754 / 0.54346, multi norm cross-line 0.57440 / 0.58306).
- A.11 Null tests (`tab:app-nulls`, `10_09_report.md` §7): targets vs permuted, STRING one-hop vs rewired, adding Cell
  Painting. Results now cite this table for the STRING claim. Left out: T2 − T2null (both nulls at once, not
  separable) and degree-feature − mean (unclear contrast).
- A.12 Data sources (`tab:app-data`): Zenodo 17182939 (MD5 4a58bc01), Zenodo 22095049, GEO GSE92742 / GSE70138
  (SHA-512 verified, job 212309), DeepCE 9b0d04f, tahoe-de-rhaister c7963cf (2,000-gene deltas), compoundinfo_beta
  (SHA-256 a71fca6d; Repurposing Hub, literature-curated, not L1000-derived), lincs-cell-painting A549 consensus
  (SHA-256 1e04d0be, downloaded 2026-10-03, branch master, no commit pinned), STRING v12.0.
- Dropped from the appendix: the GSNN / GEARS / DeepCE-gene-vector network audits (superseded line; CLAUDE.md:
  network priors are a minor control). The grids were already in `tab:app-grids`.
- Layout: `\usepackage{float}`; all appendix floats `[H]`, so every A.x header sits directly above its table (the
  Claude review had seen empty A.3/A.5/A.6/A.9/A.10 headers).
- Fig. 2d caption: "DeepCE is our reproduction … (A.3)"; `sync_captions.py` refreshed fig2.

**ONGOING cleared:** only 364344 (GPU timing, PD) remains. Unanalysed finished runs (TranSiGen fold 0 357101,
learning curve v5_lc*, 2K ladder) and unapproved items moved to `PLAN.md` Phase 8, "Parked 2026-11-03". K4 ticked.

**Review facts checked against the reports (no new runs):**
- XPert on test samples without a training plate-mate: plain PCC_DEG 0.5650 (`11_06_report.md` §4 table) vs UnDetect
  final 0.5738 (§6.2) and backbone 0.536. Not yet in the paper.
- Backbone seeds: three seeds on v1, sd ≤ 0.0002, UnDetect − MLP > 0 in 15/15 (`10_30_report.md` §5.5); not in the paper.
- Shrunk oracle (§1, job 356090, older rows and arms): shrinkage chosen on validation gives 0.2020 vs 0.2002
  unshrunk on PCC_plate. The paper's PCC_DEG oracle (0.628, unshrunk) is below the backbone.
- Plate co-membership kernel +0.0057 PCC_DEG (§6.3), the same as the chemistry kernel; this is already shown in Table 2.
- PLAN H1 (within-plate compound discrimination) and K2 (plate-disjoint split) were planned and never run.
- Annotation source: compoundinfo / Repurposing Hub (literature-curated). An InChIKey first-block duplicate audit of the
  drug-blind and DeepCE splits has not been done.

Triage proposal sent to the user; decisions pending (next entry).

### 2026-11-03 06:45 — Review round A applied (paper b9a7ed6, pushed)

User: A 档全做；Cell Painting 两句也改；B1 先问：如果要 norm PCC_plate 原始的 0.10，怎么做？

Edits (all numbers already in reports; `% SRC:` added where new):
- Results: rows without a training plate-mate now "0.536 → 0.574, above XPert at 0.565" (§4 table, released preds).
  Plate-mate sentence rewritten as complementarity: XPert stronger with few plate-mates (−0.016 / −0.040 / −0.029 /
  −0.013 for 1–20), UnDetect on the 70% with > 20 (§10.6). DeepCE: "(0.511 against 0.491)".
- Table 2: "oracle" → "reference", column "gap closed" → "share". Caption: weight chosen on validation raised it by
  0.002 in an earlier run (§1: 0.2002 → 0.2020). The caption no longer quotes the reference's PCC_DEG 0.628 (it is
  defined for PCC_plate only). Decision: no need to quote a number for a metric the reference is not defined for.
- Discussion: "The compound-specific signal is large" deleted; "closes 60% of this distance" → "attains 60% of this
  gain without any measurement of the compound". Cell Painting: "XPert is stronger with structure alone, UnDetect with
  Cell Painting, and their ensemble is best in every group" (§10.3: structure only ctx_pc − XPert −0.0134
  [−0.0176, −0.0093]); "can be measured for any new compound" → "can be measured for a new compound in one imaging assay".
- Method: the RKHS bound is now stated as a bound ("the largest effect a compound can receive shrinks with the views it
  lacks"; after normalisation "the bound is the same for every compound"). Related designs: Molina & Zhang and
  Svensson et al. transfer a compound's measured responses; for an unseen compound UnDetect predicts these terms from
  its views.
- Intro: backbone sentence split into the two variants; "not shrunk towards zero" → "keep unit self-similarity".
  Data: PCC_DEG "across all 978 genes" (checked: Bai et al.'s `evaluate_all.py` uses x_deg = treated − control,
  all 978 genes).
- Appendix: backbone-seed sentence (`10_30_report.md` §5.5, v1 configuration, labelled as earlier configuration);
  Table 10 caption: PertDiT −0.01248, quoted as 0.012.
- **Not changed:** the abstract's "at most 0.012". `primary_comparisons.csv` gives PertDiT Full − Zero = −0.012476,
  which rounds to 0.012; the reviewer compared it with the 4-decimal table value −0.0125.

Build (tectonic): no undefined references. The first pass ended the text on the last line of p. 11, so the new
sentences were shortened. Now the references start on p. 11 after the declarations, about 7 lines of slack left
for Jun's acknowledgements.

B1 (within-plate contrast metric): method question answered in chat; not run, awaiting the user's go.

### 2026-11-03 07:00 — B1 within-plate contrast metric: design and reading rule (declared before running)

User: 先 push 然后继续（main repo pushed; paper already at b9a7ed6).

**Question.** PCC_plate centres truth and prediction on the same estimate m_i of the plate mean, so its error −e is
shared by both sides and a compound-blind prediction scores > 0 (backbone 0.100; 0.536 with 1–2 plate-mates).
Is the compound-specific gain still there on a metric whose compound-blind null is exactly zero?

**Metric (PCC_within).** For each fold, group the test samples by control row (= plate). Average truth and
predictions per (plate, compound) if a compound occurs more than once on a plate. Keep plates with ≥ 2 test
compounds. Centre the truth on the plate's mean truth over these compounds and each prediction on its own plate mean.
Then take the Pearson over the 978 genes per (plate, compound), averaged over units. A prediction that is constant
within a plate (the drug-free backbone and the published MLP, whose input is the control alone) has a zero centred
vector and scores 0 by definition. This is checked from the predictions (max within-plate spread).
Training plate-mates are not needed, so rows without one are included.

**Arms** (`final_cpu/fold_*/pred_test.npz`): mlp, mlp_bai, pool_plate, pool_chem, mlp_input, nnf_multi, pool_kta,
ctx_pc (UnDetect), xpert, xpert_zero, transigen, deepce, ctx_pc+xpert. Bootstrap: 2,000 replicates over test compounds
per fold, as for PCC_plate. Coverage reported: plates and units kept, and the share of test rows.

**Reading rule.**
- Main-text sentence "the gain holds on a metric with an exact zero null" if UnDetect's PCC_within > 0 in 5/5 folds
  with the compound-bootstrap CI > 0 in ≥ 4/5. Otherwise appendix only, and PCC_plate is described as a proxy.
- UnDetect vs XPert, the ensemble vs each, and XPert vs XPert-zero are reported as they come out. If UnDetect − XPert
  has CIs on both sides of 0, the paper says "level", as for PCC_plate.
- If coverage is below 20% of test rows, the result goes to the appendix only, with coverage stated.

### 2026-11-03 13:55 — B1 result (job 366182, report §12): reading rule met; paper edit follows

- Job 366182 (cn-11, 4 CPU, 84 s, completed). An earlier submission 366181 was cancelled before it ran (pyarrow
  missing in `deepce-gpu`; outputs switched to csv.gz).
- Reading rule outcome: UnDetect PCC_within 0.117, > 0 in 5/5 folds with CI > 0 in 5/5, coverage 86% → main text.
  UnDetect − XPert +0.025, CI > 0 in 5/5, so the paper says "leads", not "level". Ensemble 0.126.
- Planned paper edits: define PCC_within in §2 next to PCC_plate (the shared-error reason in one sentence); add a
  PCC_within column to Table 2 (no extra lines); Results: one sentence in the XPert paragraph; Discussion: recommend a
  within-plate score; appendix A.x with the definition details and per-fold table. Page budget to be checked.

### 2026-11-03 14:20 — PCC_within in the paper (paper pushed; see commit below)

- §2.3: one sentence on the shared error of m_i plus the PCC_within definition, pointing to App. A.11 (new
  `app:within`, `tab:app-within`, per-fold values and four paired differences).
- Table 2: new PCC_within column, so no extra lines. Results: the backbone sentence notes that it scores exactly 0 on
  PCC_within; the XPert paragraph adds "On PCC_within UnDetect leads XPert, 0.117 against 0.092, in every fold with
  intervals above zero". Discussion recommends PCC_within (or a plate-keeping split).
- **Removed from the main text for space:** "On PCC_plate, XPert is stronger where the plate mean rests on few
  plate-mates, UnDetect on the 70% with > 20" (added 06:45 with the user's approval). Reason: it is superseded by the
  zero-null metric, where UnDetect leads overall, and the page overflowed. `tab:app-platemates` and the figure stay in
  the appendix. The user should confirm.
- Page budget: the first builds overflowed. Table 2's caption grew by one line, so Table 3 left p. 9 and Fig. 3
  moved to p. 11; the declarations spilled to p. 12. Fix: the caption is back to its old length; short tail lines
  were trimmed (intro "about it", "same-input neural network", "reaches 0.637:", ensemble sentence, P1-24 parenthesis,
  "five-way" in Method). Final: Tables 2 + 3 on p. 9, Fig. 3 on p. 10, references start at line 42/50 of p. 11
  (about 7 lines of slack), no overfull boxes, no undefined references.
- The Overleaf merge e64f13c (before these edits) only added an empty `sections/01_introduction_JW.tex`.

### 2026-11-04 05:30 — Jun Wen's comments on the PDF (`paper/editions/MKL_Therapeutic_JW.pdf`): restructure plan

14 annotations, all by Jun Wen, mostly structural. Extracted with pypdf (scratchpad script). Decisions with the user:

- **Intro, four paragraphs (we write it; `01_introduction_JW.tex` stays unused):** (1) background and significance
  (his note: "No background introduction and strong motivation"); (2) challenges and approaches; challenges are the
  dominance of plate/line context, structure predicting response poorly, and *partial compound information*, which
  now also covers incomplete atlases: L1000 profiles each compound in few cell lines and plates (drug-blind
  benchmark: 78,453 pairs over 8,276 compounds × 164 lines, about 6% of the grid); (3) limitations of existing
  methods: the highlighted "barely use the compound" sentences are **the motivation**; (4) our method and results.
  Contributions: method first (also the RECOMB scope rule in `paper/README.md`).
- **§2 Benchmark is dissolved:** Task definition → Method "Problem formulation" ("Put in method"); datasets, plate
  structure and metrics → Experiments "Datasets and setup" ("put at experiments").
- **Experiments:** 1 Datasets and setup, 2 Compared methods, 3 Results: (i) unseen-compound generalization,
  (ii) ablation study, (iii) compound case study (old "Compound views").
- **Wording:** Jun Wen asked to avoid "transfer" and use "generalization". Our test holds out compounds, all lines
  seen, so the results heading is "Unseen-compound generalization", not "cross-cell-line generalization". The frozen
  title keeps "Multi-View Transfer"; "transfer" goes from headings and results framing.
- **XPert row (0.632, Δ −0.006) highlighted without comment:** a result to highlight: the best published model is
  below the drug-free baseline on the standard score.
- **Table 2:** only the compared methods; ablation rows go to a new ablation table, in the main text if the page
  budget allows (user prefers main text, since ablation is now its own section).
- **Fig. 2 (Jun Wen):** horizontal bar charts with error bars; one colour per tool, consistent across panels;
  UnDetect light red, UnDetect + XPert purple, XPert light blue; other colours at our choice. (d): drop the
  ablation row (UnDetect, chem. + targets + MOA).
- Page budget: about 7 lines of slack before this change (record 2026-11-03 14:20), so the new text must be paid for
  by cuts (candidates: Fig. 1 caption, the intro's method paragraph, which repeats Method §3.1).

### 2026-11-04 07:00 — Restructure applied (local, not committed or pushed)

- Paper (`paper/`, on top of 5e3a223): `02_data.tex` dissolved (git rm). Method gains §2.1 "Problem formulation"
  (task + context/compound split); §2.5 renamed "Generalization across and within cell lines"; the stage label in the
  overview equation is "compound effect" instead of "transfer". New §3 Experiments: 3.1 Datasets and setup (Table 1,
  plate structure, PCC_plate / PCC_within), 3.2 Compared methods (new), 3.3 Unseen-compound generalization,
  3.4 Ablation study, 3.5 Compound case study. Flat subsections instead of 3.3.1-3.3.3 to save heading lines.
- Intro rewritten in the four-paragraph scheme; contributions now start with UnDetect. New coverage sentence:
  9 of 164 lines per compound, almost always one plate per line, < 6% of the grid (report/10_30 §6 "Structural fact":
  77,966 pairs, 458 with a second sample); targets 36%, Cell Painting 23% (11_06 §10.3). Fig. 1 caption shortened.
- Table 2 = compared methods only: plate-response baseline, CIGER, MultiDCP, PRnet, PertDiT, DeepCE, TranSiGen, XPert
  ("best published", Δ −0.006), published drug-free MLP, our backbone, UnDetect, ensemble, reference. Columns:
  PCC_DEG, Δ, no-mate PCC_DEG, PCC_plate, Δ, PCC_within. All values from final_cpu/scores.csv (published models =
  released predictions, report §4), PRnet/PertDiT PCC_DEG only (released on another scale), PCC_within from §12
  (CIGER/MultiDCP not scored there: "--"). Recomputed at full precision; MultiDCP PCC_DEG 0.25445 → 0.254.
- New Table 4 (ablation, main text): backbone → + plate kernel → + chemistry kernel → + multi-view kernel →
  + interaction (UnDetect); same features end to end / residual head. Rows unchanged from the old Table 2.
- "transfer" removed from headings and prose (Method, Results, Discussion, appendix captions/headings). Kept: the frozen
  title and abstract, the `fig:app-transfer` label and file name, and the "kernel transfer" text inside the Fig. 1
  artwork (needs a redraw to change).
- Fig. 2 redrawn (`scripts/figures/fig2_benchmark.py`): horizontal bars, s.d. error bars (folds; seeds in d), one
  colour per tool, UnDetect light red / ensemble purple / XPert light blue. (c) now shows compared methods only (plate
  and chemistry kernels moved to Table 4); (d) lost the "chem. + targets + MOA" row. Rendered with numpy/pandas/
  matplotlib from a scratchpad pip install because the /nfs-stor gnn-gpu env hung in NFS I/O (D state) for > 9 min.
- Build (tectonic): main text pp. 2-11, references start at line 46/50 of p. 11 (about 4 lines of slack), no overfull
  boxes, no undefined references.

### 2026-11-04 08:00 — Second round on the user's comments (local, not committed)

- **Fig. 2:** old (b) (plate-response baseline with / without plate-mate) dropped; the numbers stay in the text
  (0.643 / 0.239, cell-line mean 0.243). New (b) = PCC_plate, wider, with a dashed reference line at 0.201
  (`oracle_final/summary.json` all_shared oracle_uniform 0.2007) labelled "uses test responses". One legend shared by (a)
  and (b) ("with drug" / "drug input zeroed"). "Drug-free MLP" in (b) is now the published MLP (0.102) to match
  Table 2. (c) (DeepCE) label "MLP, same compound views" → "MLP" (caption explains). Values in (b) drawn inside the
  longer bars so they do not cross the reference line.
- **Table 2:** our backbone row removed; the published row is "drug-free MLP"; Δ recomputed against it (mlp_bai
  0.63720 / 0.10183): UnDetect +0.013 / +0.058, ensemble +0.023 / +0.084, XPert −0.006 / +0.055, reference +0.099.
  Caption notes that our backbone reproduces it (0.637, 0.100). Marking: **bold = best (the UnDetect + XPert
  ensemble)**, underlined = best single model (UnDetect). Before, UnDetect was bold although the ensemble is higher.
- **Method headings:** 2.1 Task definition; 2.2 UnDetect pipeline; the Context backbone subsection removed (Jun Wen:
  not our novelty). 2.2 now says in three sentences where it comes from (Bai et al.'s drug-free network, an MLP on the
  control, 3 × 2,048) and that it is the line mean where the control is removed; the Level-5 detail moved to App. A.1.
- **Computational cost** filled from report §13: about 5 min per fold for UnDetect (3.2 min backbone + kernel terms,
  2 min interaction, 5 GB GPU memory) against 268 min for XPert fold 0, same GPU type (A100-SXM4-40GB). Same sentence
  in the Discussion. Only XPert fold 0 was timed.
- Build: main text pp. 2–11, References at line 46/50 of p. 11, no overfull boxes, no undefined references.

### 2026-11-04 08:30 — Fig. 1 v7 and Fig. 3 v2 drafts (local; paper/figures overwritten, not committed)

- Pushed before this: paper 6978b63, Therapeutic bc1bf0d.
- **Fig. 1 v7** (`scripts/figures/fig1_cards.py`), from Jun Wen's comments and the user's GPT mock-up (style only; its
  content was wrong: backbone input (ℓ, p) instead of the DMSO control, "context PCs = plate, cell line, batch",
  residual = treated − control, no interaction, wrong daunorubicin MoA and mitoxantrone structure):
  - Module frames, one meaning per colour (legend top right): lavender compound information, sand dashed measured
    (profiled only), gray context, blue compound effect, light red prediction (UnDetect's colour in Fig. 2).
  - View icons in the view colours: fingerprint, protein with a pocket and ligand, pathway with an inhibition bar,
    stained cell. Compound titles carry their letter: Daunorubicin (D), Vorinostat (V), Mitoxantrone (M).
  - Kernel deck as tabbed cards (icon + name on each tab), front = structure with its matrix.
  - Separate boxes: compound effect, control PCs, interaction, response. Rounded line corners.
  - "Kernel transfer" renamed "Kernel regression" (Jun Wen: avoid "transfer"). Fusion box shows the eq. 2 formula.
- **Fig. 3 v2** (`scripts/figures/fig_views.py`): (a) view ablation as gain bars with fold s.d.; (b) bins relabelled,
  pair counts on bars (the Cell Painting ≥ 0.7 bin has only 99 pairs); old (c) (gain by nearest chemical neighbour,
  4 bins, not monotone) dropped; new (c) = the three Fig. 1 compounds, each scored in its held-out fold (user's
  suggestion); (d) adds a third strip per pair: the second compound as predicted by UnDetect while held out, with its
  correlation to the measured signature. Numbers: report §14 (job 367833).
- Paper captions and the case-study text still describe the old figures; update after the user reviews the drafts.
- Python env note: /nfs-stor envs hang on NFS I/O today; figures were rendered with numpy/pandas/matplotlib pip-installed
  into the session scratchpad.
- 08:50 (user, Fig. 1 v7.1): shared and per-line terms summed inside the Kernel regression module; Compound effect
  directly right of it, then down-left into a centred Interaction box; Control PCs moved left; + context and Response
  under / right of the interaction (shorter bottom lines). Fingerprint rows show the barcode only (no "Morgan
  fingerprint" text). "Compound views" kept: the paper says "compound view v" (Method §2.2) and "compound views"
  (Results, Fig. 3 caption); the kernel section is "Multi-view compound kernel".

### 2026-11-04 09:10 — Fig. 1 v7.2 (user's 8 points) and case-study scan

- Module title bands greyer with a divider line; module names: Compound information, Similarity kernels (Method §2.2:
  "similarity kernel K_v"), Fusion + normalization, Kernel regression, Measured responses (profiled compounds only),
  Context. One arrow per module pair (compound information → kernels → fusion; fusion query row → regression).
- Cell Painting row shows the profile vector: z-scores on the 40 highest-variance of 1,026 CellProfiler features
  (`scripts/figures/fig1_cp_profiles.py` adds `cp_profile` to `fig1_pipeline_source.json`).
- Fusion formula removed (eq. 2 is in Method; the box only needs to show the fused, normalised kernel).
- "Targets" → "Protein targets". Control PCs line straight into the × of the interaction; interaction centred between
  the control PCs and the compound-effect column; box titles shrink to fit.
- Case study: scan (report §14.1) puts mitoxantrone 3rd of 337; R-547 and alvocidib rank higher but R-547 has no
  annotation and alvocidib fewer rows. Mitoxantrone kept.

### 2026-11-04 (session 2) — Fig. 1: thinner frames, fusion matrix at kernel size; font gate

- Module frames 0.8 → 0.45 pt, title divider 0.35 pt, small boxes 0.5 pt, query card 0.8 pt.
- Fusion module now spans the same height as the Similarity kernels module; its matrix has the kernel size (KS) and sits
  level with the front kernel; above it the KTA view weights w_v (mean of five folds, `fig1_pipeline_source.json`
  "weights": chem 0.290, targets 0.245, MoA 0.226, Cell Painting 0.238) as one stacked bar.
- Font: no Arial on the cluster (fc-list: DejaVu only); the figure standard rejects Liberation/Nimbus. `style.py` now
  loads every face of the FIG_FONT family from its folder (bold, italic) and sets mathtext to that family. Needs the
  user's licensed Arial files (macOS: /System/Library/Fonts/Supplemental/Arial*.ttf).
- Scratchpad (pip-installed numpy/pandas/matplotlib) was cleared at the session restart and reinstalled.
- Arial supplied by the user (~/.fonts/arial, macOS faces). `style.py` loads the Arial faces of that folder plus
  "Arial Unicode MS" as a glyph fallback; all five figures re-rendered with FIG_FONT=~/.fonts/arial/Arial.ttf, font
  gate passed, embedded fonts ArialMT / Arial-BoldMT / Arial-ItalicMT only (checked with pypdf). `\includegraphics`
  switched from `*_preview` to the final files. Build: main text pp. 2–11, References at line 46/50 of p. 11.
- Fig. 1: residual → per-line arrow now straight (Kernel regression placed so its per-line row is at the residual's
  height; only the branch to the shared row bends); orange module "Measured responses for profiled compounds";
  module legend removed (caption and module titles carry it); Fusion widened to the right edge, view weights as four
  short bars with icons and values beside the matrix.
- Fig. 1 fusion step drawn out without words (user: use the free space, no "view weights" text): left, the weighted
  sum Σ w_v K_v with each cell split into one stripe per view (width ∝ w_v, shade = that view's similarity, hatched
  where the pair lacks the view, so V's diagonal is visibly weaker: no Cell Painting); arrow; right, the normalised
  K̃ (unit diagonal). Both at kernel size (0.80 in), sharing row labels. Kernel regression now spans to the right edge;
  shared and per-line join at a + on the per-line row, which feeds Compound effect directly below; Compound effect →
  Interaction (left, straight) → + context → Response; Control PCs, Interaction and Response moved left
  (centres 2.78 / 3.82 / 4.82 in).
- Fig. 1 v8 candidate ("高级感", user asked; scientific-figure-making references are data-plot demos, so their
  discipline was applied rather than a template): module cards white with a soft shadow, hairline border in a light tint
  of the module colour, light grey title band, colour accent bar + capitalised title; the small boxes (control PCs,
  compound effect, interaction, response) in the same style; every expression-change strip (Δ, residual, context,
  shared/per-line sums, compound effect, response) on the blue–red diverging scale of Fig. 3, levels (DMSO, treated,
  control PCs) in grey; one arrow weight (0.6 pt, #4d4d4d; compound-effect lane blue); row names regular weight.
  v7 kept in the scratchpad for comparison (fig1_cards_v7.py, fig1_v7.png); v8 rendered to the scratchpad only.
- Fig. 1 final round (user on v8): kept capitals, lighter colours, white modules; removed the accent bars; fusion shows
  one matrix, the normalised K̃ (the striped weighted-sum matrix dropped); **vectors: hue = meaning, shade = value**
  (sand measured: treated, Δ, residual and the regression inputs; grey context: DMSO, context, control PCs; blue compound
  effect: shared / per-line sums, compound effect; red prediction: response; view colours for fingerprint and Cell
  Painting), low value light, high dark; 8 cells per vector; lighter kernel tabs; 0.08 in gutters and aligned left edges.
- Captions: Fig. 1 states the frame/vector colour rule; Fig. 3 rewritten for v2 (a–d) with the choice rule for (d).
  Case-study paragraph: mitoxantrone held out, UnDetect 0.58 PCC_plate vs chemistry −0.08 and XPert −0.01; pairs'
  predicted-vs-measured r 0.78–0.89 (report §14). Old sentence citing the dropped Fig. 3c (gain by nearest chemical
  neighbour) removed.
- Paper pushed 3d7e806: final Arial figures, `_preview` files and the stale v6 `fig1_parts/` removed.
  Build: main text pp. 2–11, References at line 46/50 of p. 11.
