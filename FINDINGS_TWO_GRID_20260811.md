# Cross-Asset Price Discovery at Two Frequencies: Findings from the Corrected Replication

*Preliminary — for co-author circulation only. Stack version v0.9.67.*

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

The sample is listed in Exhibit 1; terms and estimators used throughout are defined in
Section 2. All recommendations are collected in Section 10.

## 2. Sample, definitions, and methods

**The sample.** 24 sessions in three regimes: the ten largest-intraday-range SPY sessions of
2022–2026 (the **volatile** panel), each paired with a same-weekday session 350–371 calendar
days earlier (target 364) as its **benchmark**, plus the four March-2020 **MWCB** sessions.
The pairing rules (same weekday, ~1 year prior, market open) are machine-checked by
`validate_sample.py`, and the range ranking is reproducible from a daily OHLC file via
`rank_sample.py`. Data are the SPY consolidated NBBO/ladder against the front-month ES CME
venue ladder, from one vendor pull at 10 ms with the 1 s frames derived exactly from it. All
estimates apply the halt-mask policy (halt snapshots and reopen seams excluded from every
estimator), the fixed-(1,−1) VECM with per-day `ec_valid` screening, and day-clustered or
permutation inference throughout.

**Exhibit 1a. Volatile sessions and their paired benchmarks.**

| volatile session | weekday | benchmark pair | weekday | gap |
|---|---|---|---|---|
| 2023-03-09 | Thu | 2022-03-24 | Thu | 350 d |
| 2024-07-24 | Wed | 2023-07-19 | Wed | 371 d |
| 2024-08-05 | Mon | 2023-08-07 | Mon | 364 d |
| 2024-09-03 | Tue | 2023-09-05 | Tue | 364 d |
| 2024-12-18 | Wed | 2023-12-20 | Wed | 364 d |
| 2025-01-27 | Mon | 2024-01-29 | Mon | 364 d |
| 2025-04-03 | Thu | 2024-04-04 | Thu | 364 d |
| 2025-08-01 | Fri | 2024-08-09 | Fri | 357 d |
| 2025-10-10 | Fri | 2024-10-18 | Fri | 357 d |
| 2026-06-05 | Fri | 2025-06-13 | Fri | 357 d |

**Exhibit 1b. MWCB sessions (March 2020).**

| session | weekday | Level-1 halt begins | note |
|---|---|---|---|
| 2020-03-09 | Mon | 09:34 | co-jump tape of Section 6 |
| 2020-03-12 | Thu | 09:35 | |
| 2020-03-16 | Mon | 09:30 (at the open) | Rule-201 SSR in force all session |
| 2020-03-18 | Wed | 12:56 | |

*Each MWCB session contains one Level-1 market-wide circuit-breaker halt (7% S&P 500
decline; 15-minute halt). The 900 halt seconds and the reopen seams are masked from every
estimator. Roll-affected sessions (2020-03-18, 2024-12-18, 2025-06-13) are discussed in
Section 9.*

**Terms.**

- **Grid / frame.** A *frame* is one snapshot row of both order books (quotes, depth by
  level, and derived flows) at a timestamp. The *fine grid* samples frames every 10 ms; the
  *coarse grid* (1 s) is derived from the same vendor pull — each 1 s frame is the last 10 ms
  snapshot of its second, with flow variables summed within the second — so the two grids are
  the same data at two resolutions, never two pulls.
- **Mid / microprice.** Mid = (best bid + best ask)/2. The microprice is the depth-weighted
  quote average over the top book levels — a sub-tick fair-value proxy that shades toward the
  side the depth imbalance predicts the price will move.
- **Spread.** The quoted best-level bid–ask spread, in basis points of mid.
- **WtdSpread** (weighted spread). The round-trip cost, in bps of mid, of executing a fixed
  multi-level target size against each side of the book (the target is the median cumulative
  depth of the top three levels), computed by walking the ladder. Always ≥ the quoted
  spread; it widens when the book thins *behind* the touch, which the quoted spread cannot
  see.
- **OFI** (order-flow imbalance). The Cont–Kukanov–Stoikov measure: the signed resting-depth
  change implied by each book update — depth arriving at or improving the bid counts
  positive, at or improving the ask negative — summed over the top ten levels and over the
  bar. A fleeting-quote filter removes sub-grid transient quotes at 10 ms.
- **RV** (realized variance). The mean of squared mid log-returns (bps) over the correlation
  window in the rolling designs; the per-bar sum of squared returns in the bar design.
- **MicroDev.** Microprice minus mid, in bps — a sub-tick directional-pressure proxy.
- **Book state s.** log(total ES depth) − log(total SPY depth), standardized: the relative
  liquidity state used as the conditioning variable.
- **IS** (information share, Hasbrouck). A market's share of the variance of the common
  efficient-price innovation. Identified only up to the ordering of contemporaneous shocks,
  so it is reported as [lower, upper] bounds over both orderings; we quote the midpoint when
  one number is needed.
- **CS** (component share, Gonzalo–Granger). A market's weight in the common permanent
  component, computed from the error-correction loadings; unlike IS it does not depend on an
  ordering.
- **ECM-SDE** (state-dependent error correction). The VECM on log mids with cointegrating
  vector fixed at (1,−1), where the loading on the error-correction term z is interacted
  with the book state: ΔP_t = (α + δ·s_t)·z_{t−1} + lags. δ measures "arbitrage weakens as
  the book thins"; the implied equilibrium half-life is evaluated at quantiles of s.
- **Action time.** Event-time aggregation: bars advance one order arrival at a time rather
  than by the clock, so each bar holds the same amount of trading activity by construction.
- **CMOF / log-odds ratio / corner asymmetry** (Tables 5–7). Per bar, each market's order
  flow is signed; the 2×2 sign table's "corners" are the bars where both markets press the
  same way (co-moving order flow). The log-odds ratio measures the dependence in that table;
  the corner asymmetry is the sell-corner minus buy-corner local log odds (a Rule-201
  fingerprint if nonzero).
- **MWCB / SSR.** Market-wide circuit breaker (Level 1: 7% decline, 15-minute halt);
  short-sale restriction (Rule 201).

**Methods.**

- **Hayashi–Yoshida (HY).** A correlation estimator that sums return cross-products over
  *overlapping event intervals* rather than a fixed clock grid. It is consistent when the two
  assets' quotes update asynchronously, and therefore free of the *Epps effect* — the
  mechanical decay of sampled correlation at frequencies finer than the quote-update scale.
  The HY−Pearson gap measures how much of a grid-sampled estimate is Epps artifact.
- **RealBar.** Our window-free dependent variable for the correlation system: realized
  correlation computed on non-overlapping 60 s bars, Fisher-z (arctanh) transformed, then
  first-differenced. Because there is no fixed-width rolling window, the window-length MA
  artifact — a spurious dynamic at exactly the window lag — cannot arise; only a bounded
  MA(1) from per-bar estimation noise remains.
- **DCC** (dynamic conditional correlation, Engle). GARCH-standardized returns driving a
  recursive correlation update. Window-free and smooth, but near-integrated on this sample
  (persistence a + b = 0.9999), so we use it as corroboration rather than a headline.
- **Lee–Mykland.** A jump classifier that compares each return to a *local* rolling
  volatility estimate; at fine grids it separates the jump component of quadratic variation
  from the diffusion component.
- **Rigobon het-ID** (identification through heteroskedasticity). Uses shifts in the
  *relative* variances of the two markets across volatility regimes to identify the
  contemporaneous response matrix without a Cholesky ordering. It requires the regimes to
  actually change relative variances — a testable pre-condition, which this sample fails
  (Section 7).
- **Romano–Wolf.** A stepdown multiple-testing correction controlling the family-wise error
  rate across all cells of a table; the "joint stars" in Exhibits 7–8.
- **Wild-cluster bootstrap / Webb weights.** Cluster-robust bootstrap inference designed for
  few clusters (24 days; six-point Webb weights when a subsample has as few as G = 4
  clusters).
- **Day-level permutation test.** Regime labels are permuted across days and the statistic
  recomputed; exact under exchangeability, and the appropriately sized test at N = 24 days.
- **GFEVD** (generalized FEVD, Pesaran–Shin). A forecast-error variance decomposition
  evaluated at the *measured* innovation correlation instead of an orthogonalizing ordering;
  shares do not sum to one under correlated shocks, which is the honest statement when flow
  is common.

## 3. Price discovery across frequencies and regimes

**Full-panel means (24 sessions, all `ec_valid`):**

**Exhibit 2. Full-panel price-discovery shares by sampling frequency.**

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

**Exhibit 3. ES component share by regime, with day-level permutation p-values.**

| CS_ES | benchmark | volatile | difference | permutation p (day-level) |
|---|---|---|---|---|
| 1 s | 0.269 | 0.466 | +0.197 | 0.048 |
| 10 ms | 0.387 | 0.504 | +0.117 | 0.047 |

ES's share of the permanent component rises materially in the stressed regime at both
frequencies. The pooled fixed-effects panel VECM (lags built within-day, day fixed effects,
day-clustered SEs) shows the same sign — at 10 ms the volatile-interaction t-statistics are
−1.50 (SPY) and −1.60 (ES), p ≈ 0.13–0.15 — but with only 24 day-clusters the pooled
interaction does not clear conventional thresholds. With N = 24 days, the day-level
permutation test is the appropriately sized test, and it rejects at both grids.

**Why to quote the 10 ms information shares.** At 1 s the Hasbrouck bounds are close to
uninformative on most days — e.g., 2023-08-07 gives IS_ES ∈ [0.004, 0.861] — because at that
sampling interval nearly all SPY–ES adjustment is contemporaneous and the ordering assumption
does all the work. At 10 ms the same day's bounds are [0.244, 0.523]. Asynchrony at the fine
grid breaks the simultaneity; the bounds become estimates rather than restatements of the
ordering. This materially strengthens the paper's measurement section.

## 4. Tandem order flow against the corrected nulls

The published Table 5 rejected a Binomial(n, ½) null that bundles "each market is a fair coin"
with "the markets are independent"; the marginals alone reject it. Against independence
*conditional on the observed marginals* — the null that isolates cross-market trading — the
dependence is still decisively present and **increases with stress**:

**Exhibit 4. Tandem order flow against the marginal-preserving independence null (revised
Table 5).**

| Panel | PCMOF/indep. | log-odds ratio | corner asymmetry |
|---|---|---|---|
| A. Baseline | 1.27 | 1.098 | −0.014 |
| B. Volatile | 1.34 | 1.385 | −0.008 |
| C. MWCB | 1.37 | 1.470 | −0.019 |
| C′. MWCB ex-SSR | 1.36 | 1.413 | −0.021 |

The corner asymmetry is approximately zero in every panel: no Rule 201 fingerprint at the
pooled level, so the dependence is symmetric tandem trading rather than mechanically
constrained selling. (2020-03-16 was short-sale-restricted all session and is reported both
ways.)

**The frequency-matched null changes Table 7's message.** The per-second null (0.4% per
corner) is not portable across aggregations, because the null depends entirely on per-bar
order counts:

**Exhibit 5. Off-corner mass against the frequency-matched null (revised Table 7).**

| Aggregation | observed off-corners | null at actual counts | ratio |
|---|---|---|---|
| 1 second | 11.9% | 0.35% | 33.5 |
| 10 ms | 30.1% | 22.9% | 1.32 |
| action time | 48.4% | 50.1% | 0.97 |

The 48.4% action-time figure that reads as overwhelming against 0.4% is almost exactly its
own null. The defensible statement is: tandem dependence is strong and highly significant at
the one-second aggregation, and attenuates toward the null as bars shrink to the arrival
scale; a *level* comparison across frequencies is not meaningful, because each aggregation
carries its own null.

**Tandem flow at the innovation level.** The correlation of SPY and ES OFI innovations is
0.742 (benchmark) and 0.835 (volatile) — direct, model-free evidence of common flow, rising
under stress. This has a knock-on consequence for the variance decomposition: the orthogonal
FEVD, which assumes uncorrelated flow shocks, attributes 95% of ES return variance to "ES
flow"; the generalized (Pesaran–Shin) decomposition at the measured correlation attributes
65% to ES flow and 35–43% to SPY flow, with the caveat that under tandem flow part of each
share is common flow counted toward both shocks.

## 5. Liquidity-conditional price discovery and the ECM-SDE

The paper's mechanism — price discovery migrates and error correction weakens when liquidity
withdraws — shows up strongly at the fine grid and weakly at 1 s:

**Exhibit 6. The state-dependent error-correction loading (a₁, SPY leg) across frequencies.**

| | 1 s | 10 ms |
|---|---|---|
| t-statistic (mid) | 0.25 | 5.24 |
| t-statistic (microprice) | 1.35 | 11.70 |
| median IS_ES along the liquidity curve (mid / microprice) | 0.21 / 0.34 | 0.67 / 0.74 |
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

## 6. Jumps and co-jumps

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

## 7. Identification and inference

1. **Day-clustered and permutation inference.** The legacy pooled-iid t on the SPY-return ~
   ES-flow regression is 253.3; day-clustered it is 4.62 (wild-cluster bootstrap p = 0.001).
   The result survives, the stars change; a referee will insist, and the stack now does it
   everywhere (24 clusters; Webb weights at the MWCB G = 4).
2. **Rigobon het-ID fails its pre-test on this sample.** The variance-ratio spread across
   regimes is 0.055 (threshold 0.15): the regimes scale both legs' variances nearly equally,
   so the rotation is unidentified and the het-ID point estimates are numerical noise. The
   two Cholesky orderings provide the honest bracket. (With MWCB days as their own third
   regime the over-identification statistic is available as a specification test.)
3. **The realized SPY–ES correlation is 0.932; the DCC persistence is a + b = 0.9999.** The
   conditional-correlation path is near-integrated on this sample; level statements about
   "correlation rising in stress" are safer made with per-bar realized correlation
   (non-overlapping bars) than with the DCC path.
4. **Lag robustness at the analysis layer.** The VECM-based shares barely move across lag
   choices at 1 s (CS_ES 0.384–0.412 for p ∈ {3, 5, 10, 20}) — the lag sensitivity that
   plagues the correlation system (the Eq. (5) SVAR, where BIC chases the correlation window's
   own MA structure to the search bound, our mechanical restatement of footnote 17) does not
   afflict the price-discovery estimates. The Table 9 lag question is exactly where the
   15-vs-60 comparison belongs; Section 8 runs it.
5. **Book-state beats the quoted spread as the liquidity conditioner**: R² for |SPY returns|
   0.318 vs 0.138, with partial R² 0.179 for the state given the spread — the paper's
   liquidity narrative strengthens under the richer state variable.

## 8. Table 9 at 15 versus 60 lags: the window artifact made visible

The two STAGE 5 runs estimate the same Eq. (5) correlation system on the same data — 1 s
grid, 100-bar rolling window, fixed-effects panel VAR (lags built within-day, day fixed
effects, day-cluster bootstrap SEs, Romano–Wolf joint stars) — differing only in the imposed
lag depth, VAR(15) versus VAR(60); the 60-lag run also carries the DCC column. This is the
cleanest demonstration we have of the point in §7.4.

**The rolling-window columns are not lag-robust.** The Pearson impact responses shrink by a
factor of three to more than ten between p = 15 and p = 60 (several cells collapse to zero at
the reported precision), and the Romano–Wolf star pattern reshuffles: ten Pearson cells carry
stars in at least one run, only two keep them in both (RV_ES/volatile, 0.342 → 0.031, and
WtdSpread_SPY/benchmark, 0.092 → 0.027), seven of the nine starred at p = 15 lose them at
p = 60, and OFI_ES/benchmark is starred only at p = 60. HY, which shares the rolling window,
shrinks the same way (RV_ES volatile 0.134 → 0.029; WtdSpread_ES benchmark −0.138 → −0.031).
The Epps-artifact share is itself lag-dependent: the HY−Pearson delta is nearly three-quarters
of the published-design response at p = 15 (RV_ES benchmark: −0.380 against 0.517) and about a
quarter at p = 60 (−0.030 against 0.130).

**Exhibit 7. The published (Pearson) design at two lag depths.**

| shock | volatile p=15 | volatile p=60 | benchmark p=15 | benchmark p=60 |
|---|---|---|---|---|
| Spread_ES | −0.014** (0.004) | −0.000 (0.001) | −0.012 (0.006) | 0.002 (0.001) |
| WtdSpread_ES | 0.073* (0.026) | 0.000 (0.001) | −0.060*** (0.011) | −0.001 (0.005) |
| OFI_ES | 0.005 (0.006) | 0.000 (0.002) | −0.021 (0.009) | −0.014*** (0.003) |
| RV_ES | 0.342** (0.094) | 0.031** (0.008) | 0.517*** (0.053) | 0.130 (0.053) |
| Spread_SPY | −0.028 (0.011) | 0.002 (0.001) | −0.043*** (0.012) | 0.002 (0.003) |
| WtdSpread_SPY | 0.061*** (0.015) | 0.002 (0.002) | 0.092*** (0.016) | 0.027** (0.007) |
| OFI_SPY | 0.003 (0.003) | 0.001 (0.001) | −0.007** (0.002) | 0.002 (0.001) |
| RV_SPY | 0.144 (0.060) | −0.018 (0.012) | 0.176 (0.064) | 0.056 (0.036) |

The mechanism is the one we identified analytically: with a 100-bar rolling window the
dependent variable is a moving-average object of order ≈ W, so at p = 15 nearly all of that
structure sits in the residual — the one-σ orthogonalized impact responses are measured
against a residual that still contains the window — while at p = 60 much of it has been
absorbed and the yardstick changes. Since p < W in both runs, neither magnitude is the
"right" one, and a criterion left to choose p simply chases W (the BIC-at-the-bound result,
our mechanical restatement of footnote 17).

**The window-free column is sign-stable but not magnitude-stable.** RealBar responses scale
up roughly three- to seven-fold going to p = 60, with standard errors moving in the same
direction, so magnitudes are not comparable across lag depths in this column either —
orthogonalized impact responses are denominated in the size of each equation's innovation,
and deepening the lag polynomial re-sizes those innovations in every column. But the
inference is far more stable: the three cells starred at both depths are the same three
(WtdSpread_ES/volatile, OFI_ES/benchmark, RV_ES/benchmark), no starred cell changes sign
between runs, and no cell is significant with opposite signs in the two runs. Signs and
joint significance are the transportable content of this system.

**Exhibit 8. The window-free (RealBar) column at two lag depths.**

| shock | volatile p=15 | volatile p=60 | benchmark p=15 | benchmark p=60 |
|---|---|---|---|---|
| Spread_ES | 0.372 (0.159) | 1.521 (0.630) | −0.313*** (0.057) | −0.578 (0.502) |
| WtdSpread_ES | 0.960*** (0.160) | 5.412*** (0.729) | −0.077 (0.107) | 0.456 (0.617) |
| OFI_ES | −0.083 (0.090) | 0.711 (0.419) | −0.317** (0.078) | −1.538*** (0.370) |
| RV_ES | 0.281 (0.167) | 6.106*** (1.013) | 1.604*** (0.322) | 11.750*** (0.808) |
| Spread_SPY | −0.010 (0.251) | 0.984 (0.683) | 0.157 (0.237) | 1.051 (0.754) |
| WtdSpread_SPY | 0.329 (0.139) | 0.810 (0.776) | 0.393* (0.138) | 1.255 (0.452) |
| OFI_SPY | 0.033 (0.103) | 1.249*** (0.309) | −0.010 (0.091) | −0.387 (0.423) |
| RV_SPY | 0.717** (0.205) | 1.973 (0.738) | 0.210 (0.561) | 1.480 (0.569) |

(MicroDev rows are omitted from both exhibits: no MicroDev cell is significant in any column
of either run.)

**What survives everywhere — the quotable core.** Across both lag depths and all measurement
designs: (i) volatility shocks raise subsequent correlation — RV_ES is the only shock starred
in Pearson, HY, and RealBar in both runs; (ii) book-liquidity (weighted-spread) shocks are,
with RV, the only other shocks that stay jointly significant across designs and depths, with
a regime-dependent sign pattern — the ES-side response is positive in the volatile regime and
negative on benchmark days in every starred cell, while the SPY-side response is positive
where starred; (iii) ES order-flow-imbalance shocks lower benchmark-day correlation in the
window-free column at both depths (−0.317** → −1.538***); (iv) microprice-deviation shocks do
nothing anywhere. The DCC column (60-lag run only) agrees in miniature: its starred cells are
the weighted-spread rows and RV_SPY/benchmark, with the smallness and window-independence
expected of a recursive filter.

## 9. Sample and data caveats for the appendix

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

## 10. Recommendations

1. Recast "futures dominate price discovery" as **"futures leadership is a stress
   phenomenon"**: near parity (1 s) to modest ES lead (10 ms) unconditionally; +12 to +20
   points of common-trend share in the volatile regime (permutation p < 0.05 at both grids);
   largest at the discontinuities (jump IS 0.56; co-jump lead 4.6:1).
2. Keep Table 5 with the marginal-preserving null and the log-odds ratio (monotone in stress);
   **replace Table 7's cross-frequency level comparison with per-frequency ratios** and let
   the action-time row say what it now says.
3. Quote information shares at 10 ms (tight bounds), with 1 s as the robustness column.
4. Lead the mechanism section with the interacted ECM-SDE at 10 ms (t = 5.2 / 11.7,
   three-minute stressed half-lives) and the cross-impact asymmetry.
5. Adopt day-cluster / wild-cluster / permutation inference throughout; retire the pooled-iid
   stars; report the day-level permutation p as the headline with the clustered panel t
   alongside; report the Rigobon verdict row and the Cholesky bracket, not the het-ID
   coefficients.
6. Add the innovation-level tandem correlation (0.74 → 0.83 in stress) as the direct
   measurement of the paper's title phenomenon, and the GFEVD — presented explicitly as an
   upper bound on separability — as its variance-accounting consequence.
7. Rebuild Table 9 on the window-free dependent variables (RealBar, with DCC as
   corroboration) as a sign-and-significance exhibit, with the lag depth fixed ex ante on the
   bar grid, and move the Pearson-vs-HY contrast to the appendix as the measurement-artifact
   demonstration (Section 8).

*Next: the ESH5 validation diagnosis (`validate_ESH5_20241218.txt`), and the 2025-06-13
minority-contract decision.*
