# RECOMB 2027 draft — "Degree, not wiring"

Skeleton created 2026-10-11. Overleaf is the editing copy once uploaded; this folder is the source the
agent writes into. Sync rule: upload the whole folder; when Overleaf edits come back, replace this
folder with the Overleaf export before the agent edits again (one writer at a time).

- Abstract due **2026-10-28**, full paper **2026-11-04 23:59 AoE**; ≤ 10 pages + unlimited appendix
  (`PLAN.md` Phase 6). LNCS class assumed — verify against the RECOMB 2027 call.
- Authors: Yuqi Lei, Hanzhe Liang, Jun Wen (affiliations to fill).
- Drafting macros in `main.tex`: `\pending{}` (orange, result not yet computed), `\note{}` (blue,
  outline), `\citeneeded`. Remove all before submission.
- Every number carries a `% SRC:` comment pointing at `report/10_09_report.md` and the result file.
  Quote only from those sources; the weekly report is authoritative.

## Files

| file | content | state |
|---|---|---|
| `main.tex` | class, macros, title, authors | skeleton |
| `sections/00_abstract.tex` | abstract | full draft, GSNN/GEARS/PDGrapher pending |
| `sections/01_introduction.tex` | intro P1–P4 + contributions | outline |
| `sections/02_framework.tex` | data, networks, ladder, information test, theory | outline + definitions |
| `sections/03_results.tex` | §info (E1/E7), §audit (E3/E4/E5/E6/E2), §tiers (MKL/E8) | done results written, rest pending |
| `sections/04_discussion.tex` | discussion, conclusion, availability | outline |
| `sections/05_appendix.tex` | data, proof, full tables, reproductions, training | outline |
| `refs.bib` | seeded from `papers/README.md`; `VERIFY` notes | partial |
| `figures/` | Fig 1 ladder schematic; Fig 2 network heatmap; Fig 3 model audits; Fig 4 tiers | none yet |

## Claim → evidence status

| claim | experiment | status |
|---|---|---|
| target degree predicts response magnitude, survives publication correction | E1/E7, job 296788 | done (§6) |
| wiring beyond degree only in directed signalling / TF networks | E1, job 296788 | done (§6) |
| DeepCE: STRING vectors ≤ random ≤ one-hot; node2vec real ≈ rewired | E3, §5.3/§5.4 | done |
| our model families: real ≈ rewired | E2 partial (§4.5, §4.7, §7, §8); e2e round 2 + MKL `reg` (`10_23_report.md` §2–§3) | partial; KG arms = rewired-KG null; full ladder not run |
| GSNN +24% is degree/target identity, not wiring | E4, jobs 299345, 311593, 311595 | done, fold 0 (`10_23_report.md` §1): 4/4 rewired > 2/2 real; ER ≈ real; target-edge shuffle −0.09 |
| GEARS ladder | E5, job 297743 | done (`10_16_report.md` §2): empty GO graph within real-seed range |
| PDGrapher ladder | E6 | not submitted (GPU quota) |
| targets + degree, no wiring, matches network models | E8 | GSNN graph-free NN (338366, `10_30_report.md` §1): r_flat 0.303 vs GSNN real 0.313 (pre-declared tie threshold missed by 0.0001), mean_r equal; rewired GSNN best (0.335). L1000: krr_chem_target > e2e kg_t (`10_23_report.md` §2) |
| configuration-model proposition | E9 | not started (no compute) |
| information tiers for unseen drugs | MKL §7 | done (P2-24) |
