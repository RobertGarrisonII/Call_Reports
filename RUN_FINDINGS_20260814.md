# Findings of Record: Full-Sample Two-Grid Analysis Run of August 14, 2026

**Re:** Garrison, Jain & Paddrik, "Cross-Asset Tandem Trading and Extraordinary Volatility"
**Subject:** Interpretation of the production analysis run (24 sessions, 1-second and 10-millisecond grids)

---

## 1. Scope of the record

1.1. This report interprets the output archive of the analysis run completed August 14, 2026 (1-second grid at 04:24; 10-millisecond grid at 05:44). The archive comprises two run reports, two machine summaries, and 52 exhibit tables (37 at 1s, 15 at 10ms).

1.2. The sample is 24 trading sessions: 14 classified volatile — including the four March 2020 market-wide circuit-breaker sessions (2020-03-09, 2020-03-12, 2020-03-16, 2020-03-18) — and 10 classified benchmark. Each session carries approximately 23,400 one-second observations and 2.34 million ten-millisecond observations. Session roster at Appendix A.

1.3. Every stage of both runs completed without error. The error-correction validity check passed on all 24 sessions at both grids (`n_ec_invalid = 0`); the halt mask was in effect throughout, so LULD/MWCB windows contribute no spliced pseudo-returns.

1.4. **Scope limitation.** This archive covers the core price-discovery stages (information shares, state-dependent ECM, cross-impact, DCC, IRF/FEVD, jumps, robustness, microstructure, legacy comparisons). It does not contain the Table 9 both-ways output, the order-flow-correlation tables (Tier 1/Tier 2/regimes), or the copula tables; those are produced by separate stages and are addressed in separate memoranda when their outputs are in hand.

---

## 2. Methodology in effect, with measured consequences

Each revision below is active in this run, and the run itself quantifies what the revision changes. The numbers are from this archive, not from simulation.

**M1 — Lag depth scaled to wall-clock, not step count.** The lag order resolved to 5 at the 1-second grid and 60 at the 10-millisecond grid, holding dynamic memory at roughly comparable wall-clock spans rather than fixing one step count across grids. Source: run headers, both grids.

**M2 — Day-clustered inference.** The same regression of SPY returns on ES order-flow imbalance carries t = 253.25 under the legacy pooled-iid treatment and t = 4.62 under day-clustered standard errors (wild-cluster bootstrap p = 0.001; 557,995 observations, 24 clusters). The coefficient is unchanged; the legacy precision was overstated by a factor of roughly 55. Source: `legacy__inference_iid_vs_clustered`, 1s.

**M3 — Book-stress state versus quoted spread.** The book-stress state explains 31.8% of the variation in |SPY returns| against 13.8% for the quoted spread alone, with a partial R² of 17.9% for the state conditional on the spread. The depth-side information is not redundant with the touch. Source: `legacy__liquidity_spread_vs_bookstate`, 1s.

**M4 — Generalized FEVD under correlated flow shocks.** The measured correlation of SPY/ES OFI innovations is 0.742 on benchmark days and 0.835 on volatile days. A diagonal (orthogonalized) FEVD is not valid at that correlation; the Pesaran–Shin generalized decomposition is the quoted object, with the stated caveat that common tandem flow is partially attributed to both shocks. Source: `irf` block, 1s.

**M5 — Epps control at the coarse grid.** The mean SPY/ES return correlation at 1s is 0.918478 under grid Pearson and 0.918478 under Hayashi–Yoshida. At one second the two estimators coincide to six decimal places; Epps attenuation is not a factor at this grid, and 1-second findings cannot be Epps artifacts. (The 10ms comparison, where the gap is expected, is the Table 9 deliverable.) Source: `legacy__comovement_Pearson_vs_HY`, 1s.

**M6 — Grid-appropriate jump detection and quote filtering.** The 1s run uses truncation; the 10ms run switches to Lee–Mykland and engages the fleeting-quote filter (minimum rest 5 steps = 50ms) on OFI inputs. The 10ms staleness report records the conditions requiring this: 79.6% (SPY) and 89.2% (ES) of 10ms returns are exact zeros. Source: `microstructure__staleness_report`, both grids.

**M7 — Identification honesty gate.** The Rigobon heteroskedasticity identification was estimated and **refused by its own gate**: the relative variance-ratio spread across regimes is 0.055 against a threshold of 0.10, and the output states "NO — do not quote the het-ID column." The run reports the Cholesky ordering as the identification actually used. A method that declines to identify is recorded as such rather than quoted. Source: `legacy__identification_Cholesky_vs_Rigobon`, 1s.

**M8 — Uncertainty on the headline share.** The cross-sectional mean of CS_ES carries a day-level bootstrap interval: 0.384, 95% CI [0.293, 0.478], bootstrap SE 0.048. Source: `robustness__bootstrap_CS_ES`, 1s.

---

## 3. Findings of fact

**F1 — The average division of price discovery is near parity, and the ES share rises as the grid refines.**
At 1 second: mean IS_mid_ES = 0.484, mean CS_ES = 0.384 [0.293, 0.478]. At 10 milliseconds: mean IS_mid_ES = 0.542, mean CS_ES = 0.455. Both estimators move toward ES by 6–7 points when the grid moves from 1s to 10ms. The futures leg's informational lead is concentrated at sub-second horizons; by one second the equity leg has partially caught up. Source: `information_shares`, both grids.

**F2 — Price discovery migrates toward the futures on volatile days, at both grids, at conventional significance.**
The day-level permutation test on CS_ES: volatile mean 0.466 versus benchmark mean 0.269 at 1s (difference +0.197, p = 0.048); volatile 0.504 versus benchmark 0.387 at 10ms (difference +0.117, p = 0.047). Two grids constructed from different aggregations of the same tape return the same direction at the same significance. The pooled panel-VECM interaction term is **not** significant at either grid (1s: t = 0.86 / −1.14; 10ms: t = −1.50 / −1.60, day-clustered): the migration is a day-level phenomenon with substantial cross-day heterogeneity, not a uniform pooled coefficient shift (see Q1). Source: `information_shares__regime_test`, `panel_vecm`, both grids.

**F3 — Tandem order flow is directly measured and intensifies under stress.**
The correlation of AR-prefiltered OFI innovations across the two legs is 0.742 on benchmark days and 0.835 on volatile days (1s). In the generalized FEVD, ES flow accounts for 47.6% (benchmark) and 51.2% (volatile) of SPY return variance; SPY flow accounts for 34.8% (benchmark) and 43.4% (volatile) of ES return variance. Under stress, each leg's returns are increasingly explained by the other leg's flow — in both directions. Source: `irf` block, 1s.

**F4 — Cross-impact is asymmetric at one second, becomes two-way under stress, and largely vanishes at ten milliseconds.**
At 1s, mean cross-impact of ES flow on SPY returns is λ = 0.391 (benchmark) and 0.429 (volatile); the reverse direction, SPY flow on ES returns, is λ = 0.004 (benchmark) and 0.106 (volatile). On benchmark days the asymmetry is roughly 100:1 — ES flow moves SPY, SPY flow does not move ES; on volatile days the reverse channel activates by a factor of ~25. Own-impact simultaneously falls from 0.665 to 0.491 while total cross rises from 0.198 to 0.268. At 10ms all four coefficients are an order of magnitude smaller (cross means 0.017–0.080). The cross-asset propagation horizon therefore lies between 10 milliseconds and 1 second. Source: `cross_impact__regime_summary`, `cross_impact__panel`, both grids.

**F5 — At the fine grid, the futures lead the co-jumps; at the coarse grid the lead is invisible.**
On 2020-03-09 at 1s: 14 co-jumps, all 14 simultaneous, sign agreement 1.00, zero leads either way. The same session at 10ms: 24,346 co-jumps, of which ES leads 7,610 and SPY leads 1,669 — a 4.6:1 ratio — with 15,067 simultaneous and sign agreement 0.948. The lead-lag structure of tandem jumps is a sub-second phenomenon that one-second sampling aggregates away. Source: `jumps__cojump_session0`, both grids.

**F6 — The state dependence of adjustment speeds is decisively established at 10ms and weak at 1s.**
In the ECM-SDE, the sensitivity of SPY's error-correction coefficient to the liquidity state carries t = 5.24 at 10ms (ES leg: t = −9.35); at 1s the SPY-leg sensitivity is t = 0.25 (ES: −3.01). Along the fitted 10ms state curve the median IS_ES is 0.669 using the mid and 0.741 using the microprice — the microprice reference shifts roughly 7 points of information share toward ES. Source: `ecm_sde`, both grids.

**F7 — Jump and continuous components split price discovery about evenly at 1s.**
Decomposing by the truncation split at 1s: mean ISj_ES = 0.505 (jump component) versus ISc_ES = 0.480 (continuous component). Jumps account for 17.4% of common-factor variation by truncation and 5.8% by Lee–Mykland at 1s. The corresponding 10ms fractions (69.5% / 58.1%) are recorded but flagged as unreliable in Q4. Source: `jumps`, both grids.

**F8 — The headline share is flat in the lag order.**
CS_ES at lag orders 3, 5, 10, 20 (1s): 0.398, 0.384, 0.390, 0.412. The regime finding does not depend on the lag choice — consistent with the position, adopted in the v0.9.72 stack revision, that findings should be demonstrated across a band of defensible lag depths rather than at a single selected one. Source: `robustness__lag_sensitivity`, 1s.

**F9 — Conditional correlation is high and near-integrated.**
DCC(1,1) on the 24 sessions: a = 0.0096, b = 0.9903, persistence 0.9999 (at the estimation boundary); mean conditional correlation 0.885 against a realized correlation of 0.932. The persistence estimate is a boundary solution and the level series should be treated as near-integrated (Q3). Source: `dcc`, 1s.

---

## 4. Qualifications

**Q1 — The regime migration is established at the day level, not the pooled level.** The permutation tests (p = 0.048, 0.047) operate on per-day shares; the pooled interaction terms do not reach significance. The correct statement for the paper is that the *distribution of days* shifts toward ES leadership under stress, with cross-day heterogeneity large relative to the mean shift — not that a pooled adjustment coefficient moves uniformly. The per-day 10ms IS_mid_ES ranges from 0.328 to 0.944.

**Q2 — CS is fragile on slow-adjustment days; IS is not.** Re-estimating the cointegrating β (rather than imposing (1, −1)) moves CS_ES by more than 0.30 on 6 of 24 days (mean absolute change 0.205), while IS_mid_ES moves by a mean of 0.024. The unstable days are those with the smallest net adjustment speed κ (2024-12-18: κ = 0.0003; 2024-04-04: 0.0025; 2025-10-10: 0.0040 at 1s): when both α's are near zero, their ratio is noise. Recommended reporting: IS (bounded, β-robust) as the primary share; CS quoted with κ alongside, or with low-κ days flagged. Source: `robustness__beta_fixed_vs_estimated`, `information_shares__per_day`.

**Q3 — DCC persistence is a boundary estimate.** a + b = 0.9999 is the optimizer's cap, not an interior optimum. The conditional-correlation *level* is effectively a random walk over the session and should not be quoted as a stable parameter; the realized (unconditional) correlation of 0.932 is the safer summary.

**Q4 — The 10ms jump fractions are not yet quotable.** Jump shares of 58–69% of common-factor variation at 10ms sit on top of 80–89% exact-zero returns; at that staleness, threshold-based detectors classify quote-refresh clustering as jumps. The 1s fractions (5.8–17.4%) are the defensible ones. The 10ms co-jump *timing* asymmetry (F5) is less exposed — it conditions on jumps in both legs within a window and its 4.6:1 ratio would require a staleness mechanism that is asymmetric between legs in the observed direction — but corroboration on the remaining 23 sessions is required before it is quoted (E2).

**Q5 — Heteroskedasticity identification failed honestly.** The Rigobon variance-ratio spread across day-level regimes (0.055) is below the identification threshold (0.10). Day-level regimes do not move the relative SPY/ES variance enough; the identification needs sharper contrast (E5).

**Q6 — One session remains under a data-quality flag.** 2024-12-18 (FOMC) is simultaneously the most extreme observation at 10ms (IS_mid_ES = 0.944), the lowest-κ day at both grids, and the session with the unresolved replay-versus-ladder discrepancy in extraction validation. No finding should lean on this day until that discrepancy is resolved; F2's permutation result does not depend on it (removing it lowers the volatile mean).

**Q7 — Two liquidity-state constructions disagree at 10ms.** The depth-based state and the FPCA state produce day-level t-statistics of opposite average sign (+1.11 versus −3.97) for the ES-share response. The state definition is doing work; the disagreement is documented, not resolved (E7).

---

## 5. Matters for further examination

**E1 — The propagation horizon.** F4 brackets the cross-impact horizon between 10ms and 1s. A profile of cross-λ against aggregation interval (10ms, 50ms, 100ms, 250ms, 1s) would locate the timescale at which futures flow is impounded into the equity price — a single exhibit, estimable from the existing frames, and arguably the sharpest characterization of "tandem" the data can produce.

**E2 — Co-jump lead-lag on all sessions.** F5 rests on one session in the summary output. Extending the co-jump lead/lag/simultaneous counts to all 24 sessions, with a day-level sign test on the lead ratio (and, if warranted, the Hoffmann–Rosenbaum–Yoshida lead-lag estimator as the formal version), would either establish ES jump leadership as a population finding or confine it to March 2020.

**E3 — κ-weighted and IS-primary reporting.** Per Q2: recompute the regime contrast with days weighted by κ (or on IS rather than CS) and confirm the migration survives. The permutation machinery already exists; this is a reporting change, not new estimation.

**E4 — Grid-dependence of the error-correction half-life.** The fitted half-life is 9–12 seconds on the 1s grid and 177–205 seconds on the 10ms grid — a twenty-fold disagreement for what is nominally the same adjustment process. Two candidate mechanisms: (i) 60 lags at 10ms spans only 0.6s of dynamics, truncating the correction; (ii) stale quotes mechanically attenuate the measured α at the fine grid — the same attenuation logic the paper applies to correlation (Epps) applied to adjustment speeds. Distinguishing these matters for any statement about *how fast* the basis closes; mechanism (ii), if confirmed, extends the paper's own measurement-error theme to a second estimator family.

**E5 — Identification through sharper variance regimes.** Q5's failure used whole days as regimes. Intraday windows with genuine relative-variance contrast — MWCB reopening windows against mid-session windows, or the 2020-03-16 open — are the natural candidates for a Rigobon identification that passes its own gate.

**E6 — Microprice-referenced discovery at the fine grid.** F6's 7-point IS shift under the microprice reference (0.669 → 0.741) suggests the mid systematically lags the book's information at 10ms. Adopting the microprice curve as the primary 10ms exhibit (mid as robustness) is a candidate revision; it also connects to the queue-imbalance discussion in the tick-constrained-regime memorandum.

**E7 — Reconciling the state constructions.** Q7's sign conflict between depth-state and FPCA-state conditioning should be resolved before either is quoted as "the" liquidity state — most plausibly by identifying which days drive the FPCA divergence and whether its leading component is loading on depth level or depth *shape*.

**E8 — Completion of the record.** The Table 9 both-ways/RealBar output (with the v0.9.72 lag-band stability verdict), the flow-correlation tables, and the copula tail-dependence tables are produced by stages not included in this archive. The findings above — particularly F3 — are expected to be sharpened by those exhibits; interpretation of the full record is deferred until they ship.

---

## Appendix A — Session roster

| Class | Sessions |
|---|---|
| Volatile, MWCB (4) | 2020-03-09, 2020-03-12, 2020-03-16, 2020-03-18 |
| Volatile (10) | 2023-03-09, 2024-07-24, 2024-08-05, 2024-09-03, 2024-12-18, 2025-01-27, 2025-04-03, 2025-08-01, 2025-10-10, 2026-06-05 |
| Benchmark (10) | 2022-03-24, 2023-07-19, 2023-08-07, 2023-09-05, 2023-12-20, 2024-01-29, 2024-04-04, 2024-08-09, 2024-10-18, 2025-06-13 |

Observations per session: ≈23,400 (1s grid); ≈2,340,000 (10ms grid). Halt masking active on all sessions; error-correction validity check passed 24/24 at both grids.

## Appendix B — Key figures at a glance

| Quantity | 1s | 10ms | Source |
|---|---|---|---|
| Mean IS_mid_ES | 0.484 | 0.542 | information_shares |
| Mean CS_ES | 0.384 [0.293, 0.478] | 0.455 | information_shares, bootstrap |
| CS_ES, volatile − benchmark | +0.197 (p=0.048) | +0.117 (p=0.047) | regime_test |
| OFI innovation correlation (bench / vol) | 0.742 / 0.835 | — | irf |
| Cross-λ SPY←ES (bench / vol) | 0.391 / 0.429 | 0.024 / 0.080 | cross_impact |
| Cross-λ ES←SPY (bench / vol) | 0.004 / 0.106 | 0.017 / 0.040 | cross_impact |
| Co-jump leads ES : SPY (2020-03-09) | 0 : 0 (14 simult.) | 7,610 : 1,669 | jumps |
| ECM state-dependence t (SPY / ES) | 0.25 / −3.01 | 5.24 / −9.35 | ecm_sde |
| Zero-return fraction (SPY / ES) | 0.066 / 0.188 | 0.796 / 0.892 | microstructure |
| Legacy vs clustered t (SPY ret ~ ES OFI) | 253.25 → 4.62 | — | legacy |
