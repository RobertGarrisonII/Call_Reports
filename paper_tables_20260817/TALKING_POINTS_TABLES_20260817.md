# Talking Points — Paper Tables, Run 20260817_015626 (corrected)

Companion to `TABLES_20260817`. For each table: the claim to make, the number to
say out loud, what NOT to say, and the likely pushback with its answer.
Prepared 2026-08-22.

**The arc in one breath:** (1) Tandem coupling is real and stress-ordered —
T4, T16, T2/T3. (2) It has an identifiable mechanism — T6, T11b, T12.
(3) Price discovery runs futures-first at machine timescales but the shares
are near parity — T7, T13, T14, T8. (4) The measurement discipline is the
differentiator — T15, T9, T10, T1.

---

## T1 — Sample composition and 10 ms estimability

- **Say:** A designed sample, not a convenience sample: 25 matched pairs plus
  MWCB days, screened on realized volatility with the control picker blinded
  to outcomes. Data quality is audited per session, and we publish the roster
  (A1).
- **Say:** ES ladder staleness at 10 ms is early-era and loads on the
  *benchmark* side (64% vs 24% of regime). We measured the bias direction by
  planting known truth through the real pipeline: it attenuates. So the 10 ms
  regime gaps we report are conservative, not inflated.
- **Don't say:** "the 10 ms record is clean." The run estimated on sessions
  its own QC flags; the clean-subsample rerun and the dose-response covariate
  are the P0 checks on the next run.
- **Pushback:** *"Why not just drop the 25 stale sessions?"* Because they are
  16/25 of the benchmark regime and almost all pre-2019 — exclusion destroys
  the era balance the pair design exists to protect, and power falls below
  0.8. Direction-of-bias plus dose-response is the better instrument.

## T2 — Table 5, crossed-book corner concentration

- **Say:** The corner concentration *orders* with stress: log OR 0.98 →
  1.12 → 1.47 from benchmark to volatile to MWCB. Positively co-moving flow
  corners are over-represented, negatively co-moving under-represented, and
  both margins steepen as markets degrade.
- **Say:** It is not a short-sale-restriction artifact — dropping the SSR
  member barely moves the MWCB panel (1.470 → 1.413) — and the corner
  asymmetry is flat (≈ −0.02) in every panel.
- **Don't say:** any z or p from this table. The Woolf z's are pooled-bar
  statistics (T15 is the demonstration of why that disqualifies them). The
  claim, for now, is the monotone ordering plus T3's structure.
- **Pushback:** *"So Table 5 has no inference?"* Correct on this run, by our
  own hand: the day-clustered machinery (day bootstrap, within-pair
  sign-flip, MWCB jackknife) is built and attaches to the next run. We struck
  our own stars rather than let a referee do it.

## T3 — Table 7, frequency-matched null

- **Say:** The published aggregation result reproduces on a fresh, designed
  sample: the negative-corner excess is 33.5x at 1 s bars, 1.32x at 10 ms,
  and 0.97x — nothing — in action time. Anti-correlated flow is a
  *clock artifact*; tandem (positively co-moving) flow is what survives at
  the transaction level.
- **Say:** This is the paper's cleanest replication-style exhibit and the
  best single argument for doing everything on two grids.
- **Don't say:** anything with a CI — none ships this run.

## T4 — Tier-1 tandem flow coupling  ★ the headline table

- **Say:** Flow coupling rises monotonically with stress on BOTH grids —
  benchmark 0.73 < volatile 0.82 < MWCB 0.87 at 1 s; 0.27 < 0.29 < 0.35 at
  10 ms — and the volatile−benchmark difference is significant at both
  (p = 0.0008 and 0.0327, day-level permutation).
- **Say:** This is the only flow exhibit that agrees across grids in both
  direction and significance; that cross-grid agreement is the reason it
  leads the results section.
- **Say (if pressed on staleness):** the measured bias is attenuation
  concentrated in the benchmark cells — the reported gap is a floor.
- **Don't say:** anything inferential about MWCB (4 days — describe, don't
  test).

## T5 — Directional asymmetry

- **Say:** In calm regimes coupling tightens on down-moves; in volatile
  regimes it reverses to up-moves (+0.047, p = 0.0045 vs −0.090, p < 5e-5 at
  1 s). Candidate interpretation: hedging-driven coupling in calm markets,
  short-covering/rebound-chasing coupling in stressed ones.
- **Frame it as scale-dependent, on purpose:** the asymmetry lives at 1 s and
  vanishes at 10 ms — like Table 7, the phenomenon *emerges under
  aggregation*. That framing is consistent with the paper's thesis; hiding
  the 10 ms null would not be.
- **Don't say:** it is established. The gap-of-gaps is untested and the
  SSR/direction-conditioning checks are outstanding (P1).
- **Pushback:** *"Is the volatile reversal just SSR days?"* Unknown on this
  record — say so, and point to the planned conditioning check.

## T6 — Mediation

- **Say:** The mediator link is the strongest coefficient in the run: a 1 SD
  innovation in tandem flow moves return correlation with t ≈ 18, coefficient
  67 (1 s) / 75 (10 ms). Whatever else is true, flow and correlation are one
  phenomenon at bar scale.
- **Say:** ES volatility suppresses flow innovations on both grids (Eq A
  negative twice). The full suppression decomposition — direct positive,
  indirect negative, total ≈ zero — holds at 1 s, where the indirect level CI
  [−41.9, −3.1] excludes zero.
- **Don't say:** the share statistic (605% of a near-zero total — we
  suppressed it ourselves), and don't present the decomposition as
  grid-robust.
- **Pushback:** *"499 vs 199 bootstrap draws?"* Acknowledged imbalance; next
  run is 2000 draws with BCa intervals on both grids.

## T7 — Lead-lag  ★ the price-discovery table

- **Say (lead with the co-jump test):** When both assets jump, ES moves first
  on 65 of 66 days (p = 5e-5); the HRY timing estimate agrees — ES leads by
  ~20 ms, one grid step, on 64 of 66 days (p = 0.0005). Two independent
  day-level tests, same verdict.
- **Say:** It is NOT a staleness artifact — the lead is *weakest* on stale
  sessions (Spearman +0.57 with ES update counts), the opposite of what
  carry-forward contamination would produce.
- **Don't say:** θ magnitudes that include 2017-06-26 or 2017-10-05 (the two
  stalest days), and don't cite the pre-averaged θ column — it reverses sign
  and the contradiction is unadjudicated. The co-jump test carries the claim.
- **Say:** At 1 s the tests are silent (p ≈ 0.5, 0.96) — the lead is a
  sub-100 ms phenomenon, which is exactly what T13 shows.

## T8 — Information-share regime tests

- **Say:** The *level* result: information shares sit near parity — ES ≈
  0.46–0.47 IS on average — against a literature prior of futures dominance.
  The ETF is a full partner in price discovery, not a satellite.
- **Say honestly:** the regime *shift* (ES gains share when volatile) is
  suggestive, not established: the one p < 0.05 cell (1 s free-permutation
  IS, p = 0.032) has no era-robust within-pair counterpart on this run — the
  within-pair test was run only on CS, where it dies. v0.9.87 adds the IS
  twin; the next run adjudicates.
- **Don't say:** "ES takes over in stress" — not on this record.
- **Pushback:** *"Why trust IS over CS?"* CS moves +22% across lag choices
  and swings 0.78 on one day under beta re-estimation; IS moves −0.014 and
  0.127 respectively. Fragility is measured, not asserted.

## T9 — Table 9, state-dependent IRFs

- **Say:** Measurement choice changes conclusions: under HY (Epps-free), the
  ES weighted-spread response flips sign vs Pearson (−0.048** vs +0.001),
  i.e. part of the published response is measurement artifact — the Δ column
  quantifies exactly how much, shock by shock.
- **Say:** RV shocks dominate the correlation response in both regimes.
- **Don't interpret:** DCC's near-zero responses (a+b = 0.9999, boundary
  estimate) or anything at 10 ms (that leg died at ~293 GiB and will return
  memory-bounded), and no cell is lag-band-certified this run.
- **Position:** this table is transitional — the lag-order-free local
  projection twin (T9_LP=1) supersedes it next run.

## T10 — FEVD/GFEVD

- **Say:** Roughly half of SPY-return forecast-error variance attributes to
  ES order flow (GFEVD: 46% benchmark, 50% volatile). For an instrument with
  its own $500B book, that is the tandem-market thesis in one number.
- **Say:** The diagonal-vs-generalized gap is itself evidence: orthogonal-
  shock FEVD fails precisely because flow innovations correlate at 0.70–0.79
  — you cannot even *decompose* variance while pretending the legs are
  independent.
- **Caveat to volunteer:** under tandem flow, common flow is counted toward
  both shocks — the GFEVD shares are upper bounds on unique attribution.
- **Note:** regime labels here were restored from the run report; the shipped
  CSVs were unlabeled (disclosed in the table note).

## T11a — Copula tail dependence by regime

- **Say:** Tail dependence rises from calm to stressed days (median λ ≈ 0.53
  → 0.69), and the winning copula family *changes* — symmetric t on calm
  days, asymmetric SJC/BB1 on volatile days. Stress does not just raise
  co-movement; it reshapes the joint tail.
- **Don't say:** benchmark tails are symmetric — the t family *forces*
  λ_L = λ_U and t wins 19/25 benchmark days, so that symmetry is partly
  assumption. The nonparametric estimator (v0.9.87) adjudicates next run.
- **Note:** the upper-tail LR column is omitted pending its definition fix.

## T11b — Tail dependence, thin vs deep books  ★ the mechanism table

- **Say:** Within volatile days, lower-tail dependence is significantly
  higher when the book is thin (+0.076, p = 5e-5). Liquidity state is not a
  backdrop — it is the transmission channel: thin books are when crashes
  travel together.
- **Say:** This is arguably the cleanest mechanism result in the paper —
  within-day, within-regime, day-level inference.

## T12 — Markov-switching flow regimes

- **Say:** Coupling is not a daily constant: 64 of 66 days reject a single
  within-day regime. Calm days are long low-coupling spells punctuated by
  short high-coupling bursts (median 156 vs 36 bars, asymmetry p = 0.0007);
  volatile days live in the coupled state.
- **Say:** Regime entries lean toward down-moves (p = 0.020 pooled) —
  coupling switches on when prices fall.
- **Phrase carefully:** "rejected at the bootstrap's resolution floor"
  (99 draws → p = 0.01 is a floor, not an estimate).

## T13 — Horizon profile

- **Say:** The cross-impact asymmetry survives every sampling rung — λ
  SPY←ES is an order of magnitude above λ ES←SPY from 10 ms to 1 s — while
  the ES lead share is visible only below 100 ms. The lead is fast; the
  asymmetry is structural.
- **Say (proactively):** we caught and fixed a units error here — the shipped
  half-life column was in bars; corrected, error-correction half-life runs
  0.05 s at 10 ms to ~64 s at 1 s. Volunteering this buys credibility for
  the rest.
- **Don't say:** level comparisons of λ across rungs as if they were
  economics — bar aggregation moves them mechanically; the *ordering within
  rung* is the content.

## T14 — ECM-SDE

- **Say:** Textbook error correction with the right signs on both grids: SPY
  adjusts toward the common curve, ES pulls it — consistent with T7's lead.
  Microprice-based IS exceeds mid-based (0.65 vs 0.57 at 10 ms): book
  imbalance carries information beyond the mid.
- **Say:** At 1 s the SPY adjustment coefficient is already insignificant
  (t = 0.30) — by one second, the correction is done. Same story as T13.

## T15 — iid vs day-clustered  ★ the referee-defense table

- **Say:** Same regression, same data: pooled-iid t = 386.9, day-clustered
  t = 5.56 — a factor of 69.5. This is the paper's inference policy in one
  row: nothing pooled gets stars, everything is day-level or day-clustered.
- **Use it** to preempt the "n = 1.5 million" objection: the effective n is
  66, and we act like it. It is also the stated reason T2's z column is
  struck — the discipline is applied to our own headline table first.

## T16 — Cross-impact summary

- **Say:** Cross-impact triples from benchmark to volatile (0.012 → 0.035)
  while own-impact barely moves (0.84 → 0.79). The tandem channel is what
  strengthens under stress — not markets generally becoming more impactful.
- **Note:** "volatile" here includes the MWCB days (folded taxonomy, marked).

## A1 — Per-session QC roster

- **Say:** Every session's staleness fraction and QC verdict is published;
  any reader can recompute any table excluding any session. The two known
  defects (2017-06-26 ladder monotonicity; 2020-12-11 contract-identity
  scoring) are disclosed in place.
- **Use it** as the transparency close: the paper's data-quality argument is
  auditable, not rhetorical.

---

## If you get one skeptical question per section

- *Results:* "Isn't the 10 ms benchmark cell garbage?" → T1: bias direction
  measured (attenuation), gaps conservative; dose-response next run.
- *Inference:* "Where are Table 5's stars?" → T15: we struck them ourselves;
  day-clustered replacements are built and attach to the next run.
- *Price discovery:* "20 ms lead with stale data?" → T7: the lead is weakest
  where data is stalest; artifact would run the other way.
- *Novelty:* "What's actually new?" → T4 + T11b + T3: stress-ordered coupling
  measured at flow level on a validated two-grid tape, with a liquidity-state
  mechanism and a clock-artifact decomposition nobody else has run.
