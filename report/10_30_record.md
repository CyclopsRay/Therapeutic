# Therapeutic — Weekly Record, week ending Friday 2026-10-30

Reporting interval: [2026-10-23T18:00:00+04:00, 2026-10-30T18:00:00+04:00) — Asia/Dubai.

Plans, discussion, decisions, reasoning, and corrections (Chinese or English).

**2026-10-24 20:05 — E8 part 1 read against its pre-declared rule (`10_30_report.md` §1).**
- 结果：无图 NN 三次 r_flat 均值 0.3034，阈值 0.30346（GSNN 真实图均值 0.3135 − 0.01），差 0.00008，按规则落入
  "NN < GSNN" 分支。记录原样，不改判。mean_r（次要指标）NN 0.2635 = GSNN 0.2630。
- Reading (agent): this is a tie at the boundary, not a separation. The NN-vs-real gap (−0.010) is the size of the
  replicate spread, and the NN was still improving at epoch 100. What is clear: (1) GSNN's real wiring is not
  better than either a rewired wiring or no graph at all by any margin larger than run noise; (2) GSNN on a rewired
  graph is the best model in the table (+0.032 over NN), so GSNN's sparse, target-routed architecture with a
  degree-matched graph does carry something beyond a plain MLP. Paper framing per the pre-declared second branch:
  "architecture / target routing, not wiring".
- Possible follow-up, not run (needs the user): NN with 200 epochs (removes the epoch-cap limitation; ~1.5 h CPU per
  run). Would only sharpen the NN-vs-real comparison; the rewired > real finding does not depend on it.
