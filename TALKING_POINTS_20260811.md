# Technical Talking Points: Specifications, Equations, and Anticipated Questions

*Companion to "The Methodology Revisions and Their Results" — for the presenting author's use.
Stack v0.9.67 — August 11, 2026.*

## 0. The one-slide version

1. Price discovery: fixed-β VECM, shares near parity unconditionally (CS_ES 0.38 at 1 s,
   0.46 at 10 ms); volatile-minus-benchmark gap +0.20 / +0.12, permutation p = 0.048 / 0.047.
2. Tandem: marginal-preserving null keeps Table 5 (log-OR 1.10 → 1.39 → 1.47, monotone);
   frequency-matched null kills the action-time comparison (33.5× → 1.32× → 0.97×);
   OFI-innovation correlation 0.742 → 0.835.
3. Mechanism: interacted ECM-SDE, δ-term t = 5.24 (mid) / 11.70 (microprice) at 10 ms vs
   0.25 / 1.35 at 1 s; stressed half-life ≈ 3 min.
4. Table 9: rolling-window MA(W) artifact demonstrated by the 15-vs-60 lag experiment;
   RealBar (window-free) keeps the same three Romano–Wolf-significant cells at both depths.
5. Inference: pooled-iid t = 253 → day-clustered 4.62 (wild-cluster p = 0.001); Rigobon
   pre-test fails (variance-ratio spread 0.055 < 0.15) → Cholesky bracket reported.

---

## 1. Sample and data engineering

**Selection and pairing.** Volatile days = the ten largest SPY daily ranges
(high−low)/close in 2022–2026, ranked from daily OHLC. Each is paired to the same weekday
350–371 calendar days earlier (target 364 = exactly 52 weeks), which controls weekday
seasonality in volume and liquidity while staying outside the crisis episode. The ranking is
re-derivable by anyone from a public daily file (`rank_sample.py --csv spy_daily.csv
--compare` prints the ranking, the suggested pairs, and where each shipped day ranks under
the user's own data); `validate_sample.py` machine-checks the pairing rules.

- *Q: Why intraday range rather than realized volatility for selection?* Range is computable
  from public daily OHLC — vendor-independent and referee-reproducible — and is monotone in
  RV for these sessions. The selection criterion must not depend on the microstructure data
  being analyzed.
- *Q: Why a 350–371-day window instead of exactly 364?* Holiday and closure adjustments;
  the window is the tightest that leaves every volatile day pairable on the same weekday.

**Two grids, one pull.** Extraction happens once, at 10 ms (full ladder, both legs). The 1 s
frames are *derived*: last 10 ms snapshot of each second for state variables (prices, depth,
spreads), within-second sums for flow variables (OFI, volume), with all-missing seconds left
as NaN rather than zero-filled. Automated gate tests verify derived-equals-directly-extracted.

- *Q: How do we know a 1 s-vs-10 ms difference is resolution and not data?* Because there is
  no second dataset. Any coarse/fine difference is the aggregation operator by construction.

**ES leg.** Front-month ES rebuilt from the CME venue ladder (not the consolidated feed),
contract chosen by a calendar roll rule, with an independent message-feed replay
cross-checked against the ladder day by day. One open disagreement (2024-12-18, ESH5) is
quarantined until diagnosed.

**Halt masking.** Halt windows (900 s per MWCB session) and reopen seams are set to NaN at
the frame level; every estimator drops non-finite *rows of its own design matrix*. Crucially,
prices are differenced on the raw grid *before* dropping — compressing NaNs first would
splice the last pre-halt price to the reopen price and manufacture exactly the pseudo-return
the mask exists to remove.

- *Q: Doesn't masking throw away the most interesting data?* No trading occurs inside a
  halt; the mask removes a non-observation, not information. All stressed *trading* stays:
  the four 2020 sessions still contribute ≈22.5k (1 s) / 2.25M (10 ms) tradable observations
  each.

---

## 2. Price discovery: the VECM, CS, IS, and the regime split

**The system.** For log mids p = (p_SPY, p_ES)′:

> Δp_t = α·z_{t−1} + Σ_{i=1..k} Γ_i Δp_{t−i} + ε_t,  z_t = p_SPY,t − p_ES,t (demeaned)

with the cointegrating vector *fixed* at β = (1, −1) and per-day `ec_valid` screening (the
error-correction term must actually equilibrate: correct loading signs and a stationary z).

- *Q: Why fix β instead of estimating it?* The two legs are claims on the same index; the
  arbitrage relation *is* (1, −1). Estimating β on intraday data lets microstructure noise
  rotate the cointegrating vector, and every downstream share (CS, IS) inherits that
  rotation. Fixing β is a restriction that is true by construction, and `ec_valid` verifies
  it day by day (0 of 24 days fail, at both grids).

**Component share (Gonzalo–Granger).** From the loadings α = (α_SPY, α_ES)′, the common
permanent component weights are α_⊥ (the direction orthogonal to adjustment):

> CS_ES = |α_SPY| / (|α_SPY| + |α_ES|)  (with equilibrating signs)

— the leader is the leg that does *not* adjust. No ordering assumption enters. Full-panel
means: 0.384 (1 s), 0.455 (10 ms).

**Information share (Hasbrouck).** From the VMA long-run impact row ψ and residual
covariance Ω, with F = chol(Ω):

> IS_j = ([ψF]_j)² / (ψ Ω ψ′)

computed under both Cholesky orderings → [lower, upper] bounds; the midpoint is quoted when
one number is needed. Means: 0.484 (1 s), 0.542 (10 ms).

- *Q: Why are the 1 s bounds so wide (e.g., [0.004, 0.861] on 2023-08-07)?* At 1 s the
  residual cross-correlation is near one — nearly all adjustment is within-interval — so the
  ordering assumption performs the entire attribution. The bounds are the estimator honestly
  reporting that. At 10 ms the same day gives [0.244, 0.523]: genuine asynchrony breaks the
  simultaneity and the data, not the ordering, do the attribution. This is the technical
  argument for quoting IS at 10 ms.

**The regime split and its inference.** Day-level CS_ES means: 0.269 → 0.466 (1 s),
0.387 → 0.504 (10 ms). Two tests:

1. *Day-level permutation:* regime labels permuted across days, statistic = difference in
   regime means; exact under exchangeability; p = 0.048 / 0.047.
2. *Pooled FE panel VECM:* lags built strictly within-day (no overnight seam-crossing), day
   fixed effects, day-clustered SEs, volatile×EC interaction; t = −1.50 (SPY), −1.60 (ES),
   p ≈ 0.13–0.15.

- *Q: Why lead with permutation rather than the panel t?* With G = 24 clusters the clustered
  t relies on asymptotics in G; the permutation test is exactly sized at any N. Both are
  reported; they agree in sign, and the appropriately sized one rejects.
- *Q: Did the headline change because of the method or the sample?* Both, and the memo says
  which: the halt fix and the venue-ladder ES leg are method (they mechanically reduce
  measured futures dominance on MWCB days); the 2022–2026 panel is sample. The *regime*
  result — futures take over under stress — holds on the new sample at both grids and is
  the claim the revision should make.

**Jumps and co-jumps.** Lee–Mykland classifier: L_i = r_i / σ̂_i with σ̂ a local bipower
(jump-robust) volatility; threshold from the extreme-value limit of max|L| under the null.
At 10 ms, ≈58% of common-factor quadratic variation is discontinuous. IS recomputed
separately on jump and continuous components: IS_j,ES = 0.563 vs IS_c,ES = 0.390. Co-jump
timing on 2020-03-09: of 24,346 common jumps, ES first 7,610, SPY first 1,669 (4.6:1),
15,067 simultaneous at 10 ms, sign agreement 94.8%.

- *Q: Isn't jump detection threshold-sensitive?* The 1 s comparison is reported both ways
  (truncation 17%, LM 6%) precisely to show the sensitivity; the 10 ms LM classifier is the
  one operating in its intended asymptotic regime. The co-jump lead ratio is a count, not a
  model output — it needs no orthogonalization and no threshold beyond jump flagging.

---

## 3. Tandem order flow: the two corrected nulls

**Table 5 (marginal-preserving null).** Per bar, sign each leg's net order flow; build the
2×2 sign table. The published null was Binomial(n, ½)² — fair coins *and* independence. The
corrected null is independence *conditional on the observed marginals*: expected corner mass
= p̂₊^SPY·p̂₊^ES (and analogues), estimated per day. Reported statistics:

> log-OR = log(n₁₁n₀₀ / n₁₀n₀₁);  corner asymmetry = local log-odds(sell corner) − local log-odds(buy corner)

Results: observed/null 1.27 → 1.34 → 1.37 across baseline/volatile/MWCB; log-OR 1.098 →
1.385 → 1.470 (ex-SSR 1.413); corner asymmetry ≈ 0 in every panel (no Rule-201 fingerprint —
the dependence is symmetric two-way coordination). Pooled Woolf z-statistics exceed 60
everywhere; the quotable inference is day-clustered.

- *Q: Doesn't conditioning on marginals throw away signal?* It removes exactly the component
  a referee will attribute to common one-sided pressure without cross-market coordination.
  What survives is the paper's actual claim. That it survives *and* is monotone in stress is
  a stronger exhibit than the old table.

**Table 7 (frequency-matched null).** The null corner probability is a function of the
per-bar order counts, which collapse with bar size (mean orders per bar: 505/112 at 1 s,
5.05/1.12 at 10 ms, 1/1 in action time). Recomputing the null at each aggregation's actual
counts: observed vs null = 11.9% vs 0.35% (33.5×) at 1 s; 30.1% vs 22.9% (1.32×) at 10 ms;
48.4% vs 50.1% (0.97×) in action time.

- *Q: So is the fine-grid evidence gone?* The *level* evidence at action time is gone — it
  was benchmark arithmetic. The correct statement: coordination operates at the
  hundreds-of-milliseconds-to-seconds horizon and attenuates to chance at the arrival scale.
  That is economically sensible (latency, hedging horizon) and defensible.

**Innovation-level correlation.** Each leg's OFI is prefiltered by its own past (AR/VAR
prefilter per day); the residual cross-correlation is 0.742 (benchmark) vs 0.835 (volatile).
This is the title phenomenon measured without any null construction.

**GFEVD.** The orthogonal FEVD (diagonal-Σ assumption) attributes 95% of ES return variance
to ES flow. The Pesaran–Shin generalized decomposition evaluated at the measured innovation
covariance,

> θ_ij(H) = σ_jj⁻¹ Σ_h (e_i′A_hΣe_j)² / Σ_h (e_i′A_hΣA_h′e_i),

attributes 65% ES / 35–43% SPY, with shares not summing to one because common flow is
counted toward both shocks — quoted explicitly as an upper bound on separability.

---

## 4. The mechanism: interacted ECM-SDE and cross-impact

**Specification.** With s_t = standardized log(total ES depth) − log(total SPY depth):

> Δp_t = c + (α + δ·s_t)·z_{t−1} + Σ_{i=1..k} Γ_i Δp_{t−i} + ε_t

Pooled across days with day fixed effects; δ is the mechanism ("arbitrage weakens as the
book thins"). Lag order is selected on the *base* VECM — the EC×state interaction is
p-invariant, so lag choice cannot tune the headline term. The implied adjustment speed
κ(s) = −(α + δs) gives half-life ln 2 / κ(s) at quantiles of s.

Results: δ-term t = 0.25 (mid) / 1.35 (microprice) at 1 s; 5.24 / 11.70 at 10 ms; stressed
half-life ≈ 3 minutes at 10 ms; median IS_ES along the liquidity curve 0.67 / 0.74.

- *Q: Why did the day-by-day quantile splits disappear?* They are underpowered and flip sign
  day to day; the pooled interaction estimates one parameter with day FE absorbing levels.
  The splits were a discretized, noisy version of the same regression.
- *Q: Why does 1 s hide the effect?* The book-state innovations that move κ live at
  sub-second horizons; averaging to 1 s attenuates the regressor exactly where it varies.

**Cross-impact.** Return-on-flow regressions (own + cross): the cross coefficient roughly
triples benchmark → volatile within each grid (10 ms: 0.020 → 0.060; 1 s: 0.198 → 0.268);
on MWCB days ES-flow→SPY-return (0.13–0.25) is 2–5× SPY-flow→ES-return (0.02–0.06).

**Book state vs quoted spread.** R² for |SPY returns|: 0.318 (book state) vs 0.138 (quoted
spread); partial R² of state given spread = 0.179. The state variable is not a repackaged
spread.

---

## 5. Table 9: the correlation system and the lag experiment

**Design.** FE panel VAR(p) on [Δcorrelation-measure, Spread, WtdSpread, OFI, RV, MicroDev
× both legs], lags within-day, day FE, Cholesky identification (futures block first),
one-σ orthogonalized impact responses ×100, day-cluster bootstrap SEs, Romano–Wolf joint
stars.

**The artifact, precisely.** The published dependent variable is the change in a W=100-bar
rolling Pearson correlation. Adjacent windows share W−1 bars, so Δρ_t is a moving-average
object of order ≈W in the underlying returns: every shock echoes for exactly W bars. A lag
polynomial with p < W cannot whiten it; an information criterion therefore buys lags all the
way to its bound (the earlier run's BIC chose p* = pmax — the mechanical restatement of the
paper's footnote 17).

**The experiment.** Identical data, identical estimator, p = 15 vs p = 60:

- Pearson/HY columns: impact responses shrink 3–10× (several to zero at reported precision);
  of ten Romano–Wolf-starred cells, two survive at both depths; seven starred at 15 lose
  stars at 60; one appears only at 60. HY shrinks in parallel (it shares the window).
- The HY−Pearson delta (the Epps-artifact share of the published design) is itself
  lag-dependent: ≈74% of the response at p = 15 (RV_ES benchmark: −0.380 of 0.517), ≈23% at
  p = 60.
- RealBar column (Δ Fisher-z of non-overlapping 60 s-bar realized correlation): magnitudes
  scale 3–7× with SEs moving proportionally — one-σ impact responses are denominated in
  innovation size, which the lag depth re-sizes — but the *same three cells* are starred at
  both depths (WtdSpread_ES/volatile, OFI_ES/benchmark, RV_ES/benchmark), no starred cell
  changes sign, no cell is significant with opposite signs.

- *Q: Why not just run p ≥ W = 100?* It concedes the point: the model would be spending 100
  lags modeling the measurement window. The fix is a dependent variable with no window.
  RealBar's only residual structure is an MA(1) from adjacent bars sharing one noisy level
  estimate — bounded and at lag 1, unlike the W-lag artifact.
- *Q: Why 60 s bars?* Bias–variance: enough ticks per bar for a stable per-bar correlation
  estimate, short enough to leave ≈390 bars/day of resolution. The Fisher z (arctanh)
  transform stabilizes the variance of the per-bar estimates.
- *Q: If RealBar magnitudes aren't stable either, what is Table 9 worth?* Signs and joint
  significance — which are stable — plus the DCC corroboration (its starred cells:
  weighted-spread rows and RV_SPY/benchmark). The recommendation is explicitly a
  signs-and-significance exhibit with p fixed ex ante on the bar grid, where the criterion
  is not chasing a window.
- *Q: Why is DCC only corroboration?* GARCH(1,1) per leg with a cDCC recursion
  (Q_t = (1−a−b)Q̄ + a·u_{t−1}u′_{t−1} + b·Q_{t−1}); estimated persistence a+b = 0.9999 —
  near-integrated, so its level responses are hard to interpret even though its recursive
  (window-free) structure makes it lag-robust, which it is in the data.

**What survives everywhere (the quotable core).** RV_ES is the only shock starred in
Pearson, HY, and RealBar in both runs (volatility raises correlation); weighted-spread
shocks are the only other jointly robust family, ES-side positive/volatile and
negative/benchmark in every starred cell, SPY-side positive where starred; OFI_ES lowers
benchmark-day correlation in the window-free column at both depths; MicroDev is null
everywhere.

---

## 6. Inference and identification

**Clustering.** All pooled regressions: cluster-robust at the day level (G = 24). Flagship
example: SPY-return ~ ES-flow, pooled-iid t = 253.3 → clustered t = 4.62.

**Wild-cluster bootstrap.** Rademacher weights at G = 24; Webb six-point weights where a
subsample has very few clusters (MWCB G = 4, where Rademacher yields only 2⁴ distinct
resamples). Flagship p = 0.001.

**Permutation tests.** Regime-label permutation at the day level for all
volatile-vs-benchmark contrasts; exact under exchangeability; the headline evidence at
N = 24.

**Romano–Wolf.** Stepdown FWER control across all cells of each multi-cell table (the joint
stars in the Table 9 exhibits); degenerate cells (undefined t) are excluded from the
stepdown and reported unstarred.

**Rigobon het-ID.** Identification through heteroskedasticity requires the regime shift to
move the two legs' variances by different proportions. Pre-test: the cross-regime
variance-ratio spread is 0.055 against a 0.15 threshold (and the relative eigengap
condition also fails) — the stress regime scales both legs nearly equally, so the rotation
is unidentified and het-ID point estimates are noise. With the MWCB days as a third regime,
the over-identification statistic is available as a specification test. The draft should
report the verdict row and the two-Cholesky bracket, not het-ID coefficients.

- *Q: Where does the 0.15 threshold come from?* Calibration on simulated
  scale-vs-rotation regimes; the qualitative point is robust — at a 0.055 spread the
  identifying variation is an order of magnitude below where the estimator stabilizes, and
  the over-ID test is available as the formal check.

---

## 7. Rapid-fire: cross-cutting questions

- *"Is this reproducible?"* One driver script replicates the pipeline end to end; a gate
  suite (compile checks, estimator unit tests, frozen golden-number regressions,
  driver-text invariants) runs before any stage; CI runs the full suite on every push. The
  sample selection is re-derivable from public daily data.
- *"Which published numbers change and why?"* Three causes, cleanly separable: (i) halt
  masking (MWCB-day estimates), (ii) the corrected nulls (Tables 5/7), (iii) the
  window-free correlation measures (Table 9). Each is documented with the before/after in
  the corresponding memo section.
- *"What is genuinely new versus corrected?"* New: the 10 ms grid and everything only it
  can see (tight IS bounds, the ECM-SDE result, jump/co-jump leadership, the innovation
  correlation, GFEVD). Corrected: nulls, inference, halt handling, Table 9's dependent
  variable.
- *"What's still open?"* The ESH5 2024-12-18 replay-vs-ladder disagreement (quarantined),
  and the 2025-06-13 contract-month decision (calendar rule picked the 13.3%-volume
  contract; re-extract pinned to the majority contract, or keep the rule and report the
  share).
- *"Biggest referee risk?"* The unconditional-headline retreat. The preemptive answer: the
  regime-conditional claim is stronger evidence *for the paper's mechanism* than the
  unconditional claim ever was — leadership appears exactly when tandem trading under
  extraordinary volatility is operating, at both measurement frequencies, by an exactly
  sized test.
