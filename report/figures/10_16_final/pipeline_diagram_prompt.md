Write ONE Python script, `/home/yuqi.lei/Therapeutic/scripts/plot_pipeline_diagrams.py`, that uses matplotlib only (no graphviz, no network) to draw FOUR diagrams for presentation slides, then run it and look at every PNG it writes; fix any overlap or clipped text before finishing.

Environment: python3 with matplotlib + numpy. If missing, install outside $HOME, e.g.
`pip install --no-cache-dir --target /tmp/pylibs matplotlib numpy` and run with `PYTHONPATH=/tmp/pylibs`.
Output folder (create it): `/home/yuqi.lei/Therapeutic/report/figures/10_16_final/`
Files: `pipeline_rhaister_original.png`, `pipeline_rhaister_ours.png`, `pipeline_mkl.png`, `pipeline_e2e_model.png`.

## Common style
- figsize=(16, 9), dpi=200, background #f7f8fa, `ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")`; place everything on explicit coordinates.
- These are shown full-width on a 1920x1080 slide: ALL text >= 13 pt, box titles 16-17 pt bold. DejaVu Sans is fine. Wrap long lines manually with "\n".
- Colors: ink #16202b; secondary text #4a5a69; neutral boxes white, edge #c4ccd5; original method blue (fill #e6f0f6, edge #0070a6); OUR ADDITIONS orange (fill #fbeee6, edge #b5531f); evaluation teal (fill #e3eef0, edge #1f6f78); greyed (fill #eef1f4, edge #8a97a4).
- Rounded boxes (FancyBboxPatch), arrows (FancyArrowPatch, arrowstyle "-|>", mutation_scale 18). No figure title; the slide has the headline.
- After drawing, verify with the renderer that every text's window extent lies inside its box and no two boxes overlap; shorten wording (not font size) if needed.

## Figures 1 and 2 — the SAME pipeline, original vs. ours (must be purely additive)
Implement `draw_rhaister_base(ax)` and `draw_our_additions(ax)`.
`pipeline_rhaister_original.png` = base only. `pipeline_rhaister_ours.png` = base + additions.
**Hard rule:** the additions must not move, resize, recolour, re-word or remove anything drawn by the base. Reserve empty space in the base layout for them: put the base flow in the middle band (y ≈ 3.2–6.0), keep the top band (y ≈ 6.4–8.8) and bottom band (y ≈ 0.3–2.8) empty in the original figure, and draw all additions only in those bands, connected to base boxes by dashed orange arrows.

Base (blue), left → right, with solid arrows:
  B1 "Reference cell lines (45)" — "Tahoe-100M, 24 h, 2,000 genes" / "untreated profile x(c, g)" / "drug response Δ(c, t, g)"
  B2 "Fit, per drug-dose t and gene g" — "ridge across the 45 lines:" / "Δ(c,t,g) ≈ μ(t,g) + γ(t,g) · x(c,g)" / "all lines weighted equally"
  B3 "New cell line q (never treated)" — "input: its Tahoe DMSO profile x(q, g)"   (below-right of B2 inside the middle band, arrow into B4)
  B4 "Predict" — "Δ̂(q,t,g) = μ(t,g) + γ(t,g) · x(q,g)" / "1,137 drug-doses × 2,000 genes"
  B5 "Score" — "Pearson of predicted vs. measured Δ," / "per drug-dose, averaged over lines"
Small label at the top-left of the middle band: "Rhaister-O".

Additions (orange, numbered circle badges 1–3; teal for 4):
  1 (top band, above B2) "Similarity weights" — "weight reference line c by how close its" / "untreated profile is to q: w = exp(−d(q,c)/h)" / "→ μ, γ refit for each q: slopes depend on the line"  → dashed arrow to B2
  2 (top band, above B1) "Reliability weights" — "× q(c)^α, q = how well line c's own DMSO" / "reproduces across 14 plates (no drug data used)"  → dashed arrow to B2 (or B1)
  3 (bottom band, below B3) "DepMap bridge" — "no Tahoe data for q? per gene g:" / "Tahoe(g) = a(g) + b(g) · DepMap(g)," / "fitted on reference lines → pseudo x(q, g)"  → dashed arrow to B3
  4 teal (bottom band, below B5) "Evaluation" — "drug average removed · fresh never-scored lines" / "second atlas (Emerald Bay) · gate set before each run"  → dashed arrow to B5
Legend (bottom-right corner, small): blue = original Rhaister-O, orange = our addition, teal = evaluation.

## Figure 3 — pipeline_mkl.png: multiple-kernel model (unseen drug, LINCS L1000)
Top-left label: "Multiple-kernel ridge — task: unseen drug on LINCS L1000 (978 genes, 7 cell lines)".
Left column: seven "view" boxes, each producing a drug × drug similarity kernel, grouped by tier with a coloured bracket and tier label on the left:
  Tier 1 "chemistry":   "Chemical structure" — "Morgan fingerprint, Tanimoto"
  Tier 2 "knowledge":   "Drug targets" — "target-set overlap" ; "Mechanism of action" — "same MOA class" ; "Target degree" — "how connected the targets are" ; "One-hop PPI" — "targets expanded over STRING"
  Tier 3 "morphology":  "Cell Painting" — "image features, A549, 48 h"
  Tier 4 "measured response": "Response in other lines" — "the drug's L1000 signature elsewhere"
Middle: "Combine kernels" — "K = Σ_v w_v · K_v" / "w_v from kernel–target alignment" → "Kernel ridge, per cell line" — "λ chosen by grouped inner CV" → right: "Predicted signature of a new drug".
Tiers are cumulative (T1 = tier 1; T2 = T1 + tier 2; T3 = T2 + tier 3; T4 = T3 + tier 4); write that as one small note under the tier brackets.
Grey side box "Null controls": "targets permuted across drugs" / "PPI rewired, node degrees kept".

## Figure 4 — pipeline_e2e_model.png: integrated end-to-end model (exploratory; unseen drug on LINCS L1000)
Top-left label: "Integrated end-to-end model (in progress) — task: unseen drug, LINCS L1000".
Left inputs (white): "Compound" — "Morgan fingerprint, 2,048 bits"; "PrimeKG subgraph" — "32,024 nodes · 939,170 edges" / "PPI, drug–disease, disease–gene, drug–target"; "Disease text" — "SapBERT name + PubMedBERT definition"; "Landmark genes" — "978 genes, learned embedding"; "Cell line l" — "one decoder matrix W_l per line".
Middle (blue): "Drug encoder" — "MLP 2048 → 512 → 128, LayerNorm"; "RGCN over PrimeKG" — "2 layers, residual + LayerNorm" / "+ 0.1 × link-prediction loss". Outputs: "drug vector h_d", "gene vector e_g".
Right: "Bilinear decoder, per line" — "Δ̂(d,l,g) = ⟨W_l h_d, e_g⟩ / √128" (blue); "+ KRR anchor" (orange) — "kernel-ridge prediction as fixed offset;" / "network fits the remainder"; → "Predicted signature" — "MSE on Level-5 z-scores".
Bottom strip "Arms and controls": "skel: no KG" | "kg_a: PPI + drug–disease + disease–gene" | "kg_t: kg_a + drug–target" | "null: degree-preserving rewired KG".
No numeric results in any figure.

Keep the script readable: one function per figure plus small `box()` / `arrow()` helpers. Print each output path and its pixel size at the end. Do not modify any other file.
