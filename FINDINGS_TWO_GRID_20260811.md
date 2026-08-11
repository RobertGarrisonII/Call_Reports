# Cross-Asset Price Discovery at Two Frequencies: Findings from the Corrected Replication

**Memo to co-authors — R. Garrison, August 11, 2026**

*Sample: 10 volatile + 10 paired baseline sessions (2022–2026) + the four March-2020 MWCB
sessions; SPY consolidated NBBO/ladder vs front-month ES from the CME venue ladder; one vendor
pull at 10 ms with the 1 s frames derived exactly from it. All estimates apply the halt-mask
policy (halt snapshots and reopen seams excluded from every estimator), the fixed-(1,−1) VECM
with per-day `ec_valid` screening, and day-clustered or permutation inference throughout.
Stack version v0.9.67.*

One housekeeping note before the findings: the four report files are two identical pairs —
the two 1 s reports match byte-for-byte (analysis lags auto-scale to 5 at 1 s) and the two
10 ms reports match (auto-scale to 60). The 15-vs-60 lag contrast you ran affects the Table 9
system, whose output lands in `table9_both_ways_*.csv` and the STAGE 5 console block rather
than these stage reports — send those and I will add the lag-sensitivity section. Everything
below is therefore the **two-frequency comparison**, which turns out to be the more
interesting axis anyway.

---

## 1. Summary of findings

1. **Futures leadership is conditional, not unconditional.** On the full 24-session panel the
   working paper's headline — ES dominates price discovery — does not survive as an
   unconditional claim: at 1 s the component share of ES is 0.38 and its information share
   0.48 (near parity, SPY slightly ahead); at 10 ms ES regains a modest lead (IS 0.54,
   CS 0.46). But **in the stressed regime ES is clearly ahead at both frequencies**, and the
   volatile-minus-benchmark gap is significant at the 5% level by day permutation at both
   grids. The futures market leads exactly when the paper's mechanism — tandem trading under
   extraordinary volatility — is operating. This is a sharper and, we think, more defensible
   claim than unconditional dominance.
2. **Tandem trading survives the corrected nulls at 1 s and is directly measurable at the
   innovation level — but the action-time claim does not survive.** Against independence
   *given the observed marginals*, the corner log-odds ratio rises monotonically from baseline
   (1.10) to volatile (1.39) to MWCB (1.47). The measured correlation of SPY and ES order-flow
   *innovations* is 0.74 on benchmark days and 0.83 on volatile days. However, at the
   frequency-matched binomial null, the observed off-corner mass is only 1.32× the null at
   10 ms and 0.97× in action time (vs 33.5× at 1 s): the published Table 7 comparison of all
   three rows against the per-second null overstates the fine-frequency evidence by orders of
   magnitude.
3. **The liquidity mechanism is a fine-grid phenomenon.** The state-dependent error-correction
   loading — the paper's "arbitrage weakens when the book thins" channel — is sharply
   identified at 10 ms (t = 5.2 on the mid, t = 11.7 on the microprice) and statistically
   invisible at 1 s (t = 0.25 / 1.35). Equilibrium half-life on stressed books is roughly
   three minutes.
4. **Discontinuities are where the futures lead is largest.** At 10 ms, ES's information share
   in the jump component (0.56) exceeds its continuous-component share (0.39), and on
   2020-03-09 the co-jump tape shows ES leading SPY into common jumps 7,610 times against
   1,669 (4.6:1) with 94.8% sign agreement.
5. **Two of the paper's identification devices need restating.** The Hasbrouck bounds at 1 s
   are nearly vacuous ([0.00, 1.00]-style) and tighten dramatically at 10 ms — the fine grid
   is the right place to quote IS. And Rigobon heteroskedasticity identification fails its own
   pre-test on this sample (variance-ratio spread 0.055 < 0.15): the regimes differ in scale,
   not in relative heteroskedasticity, so the Cholesky bracket — not het-ID — is what we can
   honestly report.

---

## 2. Price discovery across frequencies and regimes

**Full-panel means (24 sessions, all `ec_valid`):**

| | 1 s | 10 ms |
|---|---|---|
| mean IS_ES (midpoint of Hasbrouck bounds) | 0.484 | 0.542 |
| mean CS_ES (Gonzalo–Granger) | 0.384 | 0.455 |
| days with invalid error correction | 0 of 24 | 0 of 24 |

The 2014–2017 sample's futures dominance does not reappear unconditionally on 2022–2026 data.
Two things changed relative to the working paper's estimates: the halt windows no longer
contribute pseudo-returns (the four MWCB days previously entered with the 900-second halt and
its reopen seam included), and the ES leg is now the venue ladder with the price-level replay
validated against it. At 1 s, SPY carries slightly more than half of the common trend on the
average day.

**The regime split — the paper's central comparison — survives at both grids:**

| CS_ES | benchmark | volatile | difference | permutation p (day-level) |
|---|---|---|---|---|
| 1 s | 0.269 | 0.466 | +0.197 | 0.048 |
| 10 ms | 0.387 | 0.504 | +0.117 | 0.047 |

ES's share of the permanent component rises materially in the stressed regime at both
frequencies. The pooled fixed-effects panel VECM (lags built within-day, day fixed effects,
day-clustered SEs) shows the same sign — at 10 ms the volatile-interaction t-statistics are
−1.50 (SPY) and −1.60 (ES), p ≈ 0.13–0.15 — but with only 24 day-clusters the pooled
interaction does not clear conventional thresholds. With N = 24 days, the day-level
permutation test is the appropriately sized test, and it rejects at both grids. We recommend
reporting both: the permutation p as the headline, the clustered panel t as the honest pooled
counterpart.

**Why to quote the 10 ms information shares.** At 1 s the Hasbrouck bounds are close to
uninformative on most days — e.g., 2023-08-07 gives IS_ES ∈ [0.004, 0.861] — because at that
sampling interval nearly all SPY–ES adjustment is contemporaneous and the ordering assumption
does all the work. At 10 ms the same day's bounds are [0.244, 0.523]. Asynchrony at the fine
grid breaks the simultaneity; the bounds become estimates rather than restatements of the
ordering. This materially strengthens the paper's measurement section.

## 3. Tandem order flow against the corrected nulls

The published Table 5 rejected a Binomial(n, ½) null that bundles "each market is a fair coin"
with "the markets are independent"; the marginals alone reject it. Against independence
*conditional on the observed marginals* — the null that isolates cross-market trading — the
dependence is still decisively present and **increases with stress**:

| Panel | PCMOF/indep | log-odds ratio | corner asymmetry |
|---|---|---|---|
| A. Baseline | 1.27 | 1.098 | −0.014 |
| B. Volatile | 1.34 | 1.385 | −0.008 |
| C. MWCB | 1.37 | 1.470 | −0.019 |
| C′. MWCB ex-SSR | 1.36 | 1.413 | −0.021 |

The corner asymmetry (sell-side minus buy-side local log odds) is approximately zero in every
panel: no Rule 201 fingerprint at the pooled level, so the dependence is symmetric tandem
trading rather than mechanically constrained selling. (2020-03-16 was short-sale-restricted
all session and is reported both ways.)

**The frequency-matched null changes Table 7's message.** The per-second null (0.4% per
corner) is not portable across aggregations, because the null depends entirely on per-bar
order counts:

| Aggregation | observed off-corners | null at actual counts | ratio |
|---|---|---|---|
| 1 second | 11.9% | 0.35% | 33.5 |
| 10 ms | 30.1% | 22.9% | 1.32 |
| action time | 48.4% | 50.1% | 0.97 |

The 48.4% action-time figure that reads as overwhelming against 0.4% is almost exactly its
own null. The defensible statement is: tandem dependence is strong and highly significant at
the one-second aggregation, attenuates toward the null as bars shrink to the arrival scale,
and the *level* comparison across frequencies in the published table should be replaced by
per-frequency ratios.

**Tandem flow at the innovation level.** The correlation of SPY and ES OFI innovations is
0.742 (benchmark) and 0.835 (volatile) — direct, model-free evidence of common flow, rising
under stress. This has a knock-on consequence for the variance decomposition: the orthogonal
FEVD, which assumes uncorrelated flow shocks, attributes 95% of ES return variance to "ES
flow"; the generalized (Pesaran–Shin) decomposition at the measured correlation attributes
65% to ES flow and 35–43% to SPY flow, with the caveat that under tandem flow part of each
share is common flow counted toward both shocks. We suggest the GFEVD as the quotable
decomposition, presented explicitly as an upper bound on separability.

## 4. Liquidity-conditional price discovery and the ECM-SDE

The paper's mechanism — price discovery migrates and error correction weakens when liquidity
withdraws — shows up strongly at the fine grid and weakly at 1 s:

| State-dependent EC loading (a₁, SPY leg) | 1 s | 10 ms |
|---|---|---|
| t-statistic (mid) | 0.25 | 5.24 |
| t-statistic (microprice) | 1.35 | 11.70 |
| median IS_ES along the liquidity curve | 0.21 / 0.34 | 0.67 / 0.74 |
| implied equilibrium half-life (stressed states) | ~10 s | ~3 min |

At 10 ms, moving along the relative-liquidity state shifts the adjustment burden exactly as
the paper argues: κ falls and the half-life lengthens as the book thins, and the ES share of
discovery along the entire curve sits at 0.58–0.63 (0.62–0.63 on the microprice). The
depth-state and FPCA-state day-by-day splits are heterogeneous in sign day to day —
the pooled interacted estimator, not the per-day quantile splits, is the exhibit to lead with.

**Cross-impact confirms the direction of causation in stress.** Within each grid, the average
cross-impact coefficient roughly triples from benchmark to volatile days (10 ms: 0.020 → 0.060;
1 s: 0.198 → 0.268), and the asymmetry runs the right way for the paper's thesis: on the MWCB
days the impact of ES flow on SPY returns (0.13–0.25) is two to five times the impact of SPY
flow on ES returns (0.02–0.06). Futures flow moves the ETF; the reverse channel is an order of
magnitude weaker.

## 5. Jumps and co-jumps

At 10 ms with the Lee–Mykland local-volatility classifier, 58% of common-factor quadratic
variation on the average day is discontinuous (the truncation estimate at 1 s is 17%, and 6%
under LM — the fine grid is where jumps are measurable at all). Two results:

- **ES leads more at the discontinuities than in the diffusion**: mean ISj_ES = 0.563 against
  ISc_ES = 0.390. Price discovery's jump component is a futures phenomenon.
- **Co-jump timing** (2020-03-09): of 24,346 common jumps, ES moves first 7,610 times, SPY
  first 1,669 times (4.6:1), 15,067 simultaneous at 10 ms, with 94.8% sign agreement.

This is, in our view, the single sharpest exhibit for futures leadership the revision can
offer — it is model-free, it is at the events that matter, and it does not depend on a
Cholesky ordering.

## 6. Identification and inference upgrades the revision should adopt

1. **Day-clustered and permutation inference.** The legacy pooled-iid t on the SPY-return ~
   ES-flow regression is 253.3; day-clustered it is 4.62 (wild-cluster bootstrap p = 0.001).
   The result survives, the stars change; a referee will insist, and we now do it everywhere
   (24 clusters; Webb weights at the MWCB G=4).
2. **Rigobon het-ID fails its pre-test on this sample.** The variance-ratio spread across
   regimes is 0.055 (threshold 0.15): the regimes scale both legs' variances nearly equally,
   so the rotation is unidentified and the het-ID point estimates are numerical noise. Report
   the two Cholesky orderings as a bracket and the het-ID verdict row; do not quote the het-ID
   coefficients. (With MWCB days as their own third regime the over-identification statistic
   is available as a specification test — worth one row in the appendix.)
3. **The realized SPY–ES correlation is 0.932; the DCC persistence is a + b = 0.9999.** The
   conditional-correlation path is near-integrated on this sample; level statements about
   "correlation rising in stress" are safer made with per-bar realized correlation
   (non-overlapping bars) than with the DCC path.
4. **Lag robustness at the analysis layer.** The VECM-based shares barely move across lag
   choices at 1 s (CS_ES 0.384–0.412 for p ∈ {3, 5, 10, 20}) — the lag sensitivity that
   plagues the correlation system (the Eq. (5) SVAR, where BIC chases the correlation window's
   own MA structure to the search bound, our mechanical restatement of footnote 17) does not
   afflict the price-discovery estimates. The Table 9 lag question is exactly where your
   15-vs-60 comparison belongs; the RealBar and DCC dependent variables are the window-free
   columns to quote there.
5. **Book-state beats the quoted spread as the liquidity conditioner**: R² for |SPY returns|
   0.318 vs 0.138, with partial R² 0.179 for the state given the spread — the paper's
   liquidity narrative strengthens under the richer state variable.

## 7. Sample and data caveats for the appendix

- **Rolls.** 2020-03-18: calendar pick ESM0 carries 69.5% of two-contract volume; 2024-12-18:
  ESH5 carries 64.2%. Report the measured shares; do not splice (10–12 point calendar-spread
  seams). **2025-06-13: the calendar rule picked the minority contract** (ESU5, 13.3% of
  volume) — we must either re-extract pinned to ESM5 and say so, or keep the rule and report
  the share; it cannot pass silently.
- **2020-03-16** is Rule-201 restricted all session (reported in and out of the MWCB panel).
- **Halt masking** excludes 900 halt seconds per MWCB session plus the reopen seams from every
  estimator; the four 2020 sessions' estimates are on the ~22.5k (1 s) / 2.25M (10 ms)
  tradable observations.
- **Staleness at 10 ms** (79.6% zero-return snapshots for SPY, 89.2% for ES): second moments
  at the fine grid come from noise-robust estimators and the fleeting-quote filter is engaged
  on flow inputs; per-bar realized correlation, not tick-by-tick Pearson, is the fine-grid
  correlation object.
- One open validation item: the independent message-replay vs venue-ladder cross-check on
  ESH5 2024-12-18 recorded a disagreement we have not yet diagnosed; until then the ladder
  validation exhibit should stay out of the draft.

## 8. What we would change in the paper, concretely

1. Recast "futures dominate price discovery" as **"futures leadership is a stress
   phenomenon"**: near parity (1 s) to modest ES lead (10 ms) unconditionally; +12 to +20
   points of common-trend share in the volatile regime (permutation p < 0.05 at both grids);
   largest at the discontinuities (jump IS 0.56; co-jump lead 4.6:1).
2. Keep Table 5 with the marginal-preserving null and the log-odds ratio (monotone in stress);
   **replace Table 7's cross-frequency level comparison with per-frequency ratios** and let
   the action-time row say what it now says.
3. Quote information shares at 10 ms (tight bounds), with 1 s as the robustness column.
4. Lead the mechanism section with the interacted ECM-SDE at 10 ms (t = 5.2/11.7, three-minute
   stressed half-lives) and the cross-impact asymmetry.
5. Adopt day-cluster/wild-cluster/permutation inference throughout; retire the pooled-iid
   stars; report the Rigobon verdict row and the Cholesky bracket.
6. Add the innovation-level tandem correlation (0.74 → 0.83 in stress) as the direct
   measurement of the paper's title phenomenon, and the GFEVD as its variance-accounting
   consequence.

*Next: the 15-vs-60 lag Table 9 comparison from your two STAGE 5 outputs, and the ESH5
validation diagnosis.*
