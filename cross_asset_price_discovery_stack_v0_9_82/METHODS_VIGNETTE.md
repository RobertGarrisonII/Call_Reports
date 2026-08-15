# Methods Vignette — Re-Flooring *Cross-Asset Tandem Trading and Extraordinary Volatility*

*A guide to the estimators added in the cross-asset price-discovery stack (v0.2.0 → v0.9.7): what each indicator does, the methodological issue it addresses in Garrison, Jain & Paddrik (JFM), how it strengthens the paper, and how to read its output. Doubles as the stack's methods README. Every validation figure below comes from a synthetic known-answer guard (`test_*.py`) shipped with the code.*

---

## 0. Orientation

The paper has a real dataset and a real idea wearing field-journal methods. Four methodological choices stand between it and a top-tier bar:

- **(i) Comovement is a rolling Pearson correlation.** At tick frequency this is biased toward zero by the **Epps effect** and asynchronous trading, and it carries **zero tail dependence** for any ρ < 1 — it asserts SPY and ES become *independent* in the extreme, which is backwards for a crash.
- **(ii) The SVAR is identified by assuming the answer.** A Cholesky ordering with futures first imposes "futures lead." The marquee price-discovery finding is then partly an identifying assumption, not an estimate.
- **(iii) Inference pools ~23.4M observations with no clustering.** Essentially every coefficient is starred at 1%, but the effective number of independent units is ~20 days (or **4 events** for the MWCB analysis). Treating autocorrelated ticks as independent inflates *t*-statistics by roughly one to two orders of magnitude.
- **(iv) Liquidity is the BBO spread only.** For two of the deepest books on earth the touch spread is pinned at the one-tick minimum almost always — near-zero variation, near-zero information. The liquidity dynamics live in **depth and book shape**.

**Design principle: upgrade in place, not teardown.** The PCMOF/NCMOF apparatus and the SVAR/IRF structure survive. What changes: every measured input becomes consistent under asynchronicity + microstructure noise; every identification claim is *earned* rather than assumed; every standard error is recomputed under day/event clustering; and liquidity is measured in the book. Each indicator below is tagged **substitute** (replaces an estimator inside the existing structure) or **complement** (adds a robustness/identification layer the structure lacks).

### Master map

| Indicator | Issue | Paper table / claim re-floored | Module · function | Guard | S/C |
|---|---|---|---|---|---|
| Day-cluster / wild-cluster bootstrap / Ibragimov–Müller | (iii) | Tables 9, 11, 13, 14 — all inference | `inference` · `wild_cluster_bootstrap`, `ibragimov_muller`, `cluster_robust_ols` | `test_wild_cluster` | complement |
| HRY lead-lag | (ii) | SVAR identification (Eqs 5/6); "futures lead" | `noise_robust_cov` · `lead_lag` | `test_leadlag` | substitute |
| Hayashi–Yoshida + noise-robust RV/cov | (i),(iii) | Table 8; Eq. 5 dependent var; vol regimes | `noise_robust_cov` · `hayashi_yoshida`, `epps_curve`, `realized_variance`, `realized_kernel_cov` | self-test | substitute |
| Jump-robust variation | (i),(iii) | H3 / §6 MWCB | `noise_robust_cov` · `bipower_variation`, `threshold_variance`, `jump_test` | `test_jumps` | complement |
| Rigobon ID + identifying-variation diagnostic | (ii) | Tables 9, 11, 13 (IRF identification) | `rigobon_id` · `rigobon_identify`, `identification_diagnostic`, `overid_statistic` | `test_rigobon_id` | substitute |
| Copula tail dependence + Aielli cDCC | (i) | H3 "correlation breakdown" | `copula_garch` · `select_copula`, `t_copula_dcc`; `dcc_garch` · `cdcc_fit` | `test_cdcc` | complement |
| FPCA liquidity (level/slope) | (iv) | All liquidity claims; SVAR conditioning | `functional_liquidity` · `asset_fpca`, `interpret_components`, `fpc_state_series` | `test_fpca_interp` | substitute |
| Cross-impact matrix + symmetry test | benchmark | Eq. 6 / H2(b) | `cross_impact` · `cross_impact_matrix`, `cross_impact_symmetry`, `compare_impact_depth` | `test_xi_symmetry` | substitute |
| VECM / information shares | naming | Independent price-discovery cross-check | `price_discovery_shares` · `panel_vecm`, `hasbrouck_is`, `gonzalo_granger` | self-test | complement |
| **Onset stress-surface f(state)** + noise-robust transmission | (i),(ii),(iii),(iv) | The headline price-discovery-under-stress claim — *identified*, not assumed | `onset_response` · `event_onset_estimate`, `fit_stress_surface`, `interaction_identified`, `excess_over_surface`, `run_onset_surface` | `test_onset_response`, `test_stress_surface`, `test_noise_robust_surface`, `test_surface_excess` | substitute |

---

## 1. Cluster-robust inference — the single highest-return fix

**What it does.** Three procedures valid when the number of *independent* units is small: `cluster_robust_ols` (Liang–Zeger 1986 one-way cluster SEs), `wild_cluster_bootstrap` (Cameron–Gelbach–Miller 2008 restricted "WCR" bootstrap; Webb 2014 six-point weights when clusters ≤ 12), and `ibragimov_muller` (2010/2016 group *t*-statistic — estimate the coefficient separately per cluster, test the *G* estimates against a *t*₍G−1₎ reference).

**The issue (iii).** With strong within-day dependence, the iid *t* counts every 10-ms tick as a fresh draw. The effective sample is the number of days (~20) or MWCB events (4), and the reported 1% stars are essentially uninformative.

**How it enhances the paper.** It changes *no* point estimate — only the inference — so it slots into the existing tables with zero disruption. Report wild-cluster-bootstrap *p*-values for the day panel (Tables 9, 11) and Ibragimov–Müller / randomization inference for the 4-event MWCB analysis (Tables 13, 14), and retire the universal `***`. Publish a before/after table; this is the honest disclosure a referee requires.

**How to interpret.** The WCB *p*-value is the fraction of bootstrap *t**-statistics whose magnitude exceeds the observed *t*. A coefficient that was significant under iid but has WCB *p* > 0.05 was a within-day-dependence artifact — do not claim it. For the four MWCB events, the IM output gives `theta` (the average effect across events), its `se`, and a *t*₍3₎ *p*-value; with only four events the test is deliberately conservative, and "we cannot reject no change" is the honest reading of an insignificant post-halt coefficient — *not* "the channel broke down" (that interpretation needs a power analysis the four events cannot support).

**Validation.** Under the null with G = 10 clusters, the iid *t* rejected at **68.8%** against a nominal 5%; CRVE-*t* at 9.4% (the known few-cluster bias); the wild cluster bootstrap at **5.0%** and Ibragimov–Müller at **7.5%** — both with full power (87.5% / 100% under the alternative).

**Call.** `inference.wild_cluster_bootstrap(y, X, clusters=day_id, test_idx=j)` → `{t, p, se_cluster, beta, n_clusters, weights}`; `inference.ibragimov_muller(per_event_betas)` → `{theta, se, t, p, df, ci}`.

---

## 2. HRY lead-lag — who leads, *measured* not assumed

**What it does.** `lead_lag` implements the Hoffmann–Rosenbaum–Yoshida (2013) shifted-Hayashi–Yoshida contrast: it scans a candidate time-shift θ over a grid, computes the asynchronous cross-covariance of the two price series with one shifted by θ, and returns the θ that maximizes the contrast. **Convention: a positive `lead_lag` means the *first* series leads the second** by that many time units.

**The issue (ii).** This is the direct replacement for the Cholesky ordering. Instead of *assuming* futures lead, you *estimate* the lead and let the data confirm or refute it.

**How it enhances the paper.** "We take the futures market as the lead" becomes "the estimated lead of ES over SPY is θ̂ ms (95% band …)," reported per regime. The robustness note that ordering "doesn't matter" is replaced by a measured number — and a *change* in θ̂ between baseline and MWCB regimes is itself a finding about how price leadership shifts under stress.

**How to interpret.** θ̂ = +X ms with a sharp, well-separated contrast peak is evidence the first series leads by X ms. θ̂ ≈ 0 (peak at the grid center) means the two are effectively contemporaneous — which would *undercut* any leadership framing and should be reported as such. Always estimate within a session (never across the overnight gap), and under heavy noise combine with pre-averaging (the raw contrast is consistent but noisy at the finest scales).

**Validation.** Injected leads of +50 ms, 0 ms, and −50 ms were recovered as +50, 0, −50 ms with the correct `leader` label (`first` / `none` / `second`).

**Call.** `noise_robust_cov.lead_lag(t_es, lp_es, t_spy, lp_spy, max_lag=0.2, n_grid=81)` → `{lead_lag, thetas, contrast, peak_contrast, leader}`. Positive ⇒ ES (the first argument) leads SPY.

---

## 3. Hayashi–Yoshida correlation + noise-robust realized measures

**What it does.** `hayashi_yoshida` estimates the integrated covariance of two asynchronously observed price series **without synchronizing them** (so it is not subject to the Epps bias). `epps_curve` and `signature_plot` are the diagnostics that *show* the bias. `realized_variance(method="tsrv"|"kernel")` gives noise-robust integrated variance (two-scale RV, Zhang–Mykland–Aït-Sahalia 2005; realized kernel, Barndorff-Nielsen–Hansen–Lunde–Shephard 2008), and `realized_kernel_cov` gives the multivariate realized kernel on refresh-time-synced returns (BNHLS 2011) — jointly noise- and asynchronicity-robust, and positive semi-definite.

**The issue (i) + (iii).** The paper's 100-second rolling Pearson correlation (Table 8; the Eq. 5 dependent variable) is mechanically biased downward at high frequency, and naive realized variance is inflated by bid-ask bounce.

**How it enhances the paper.** It substitutes a consistent comovement measure for a biased one. Crucially, it lets you check whether an apparent "correlation breakdown" during MWCB is real or just an Epps artifact: if measured ρ falls when trade intensity falls (which worsens the Epps bias), the HY estimate — which controls for it — is the arbiter.

**How to interpret.** The signature plot shows naive ρ sliding toward zero as the sampling interval tightens; the HY value attached as `corr_HY` is the consistent target the naive curve should be compared against. If the gap between naive and HY widens precisely on the volatile/MWCB days, the paper's "breakdown" is partly the artifact. For variance, the TSRV/kernel value is flat across sampling frequencies where naive RV explodes.

**Validation.** On noisy, asynchronous synthetic data with true correlation 0.5, the noise-robust moments recovered variances ≈ 1.0/1.0 and correlation **0.506**; the naive RV signature diverges as dt → 0 while TSRV stays flat.

**Call.** `noise_robust_cov.hayashi_yoshida(t1, lp1, t2, lp2)`; `noise_robust_cov.epps_curve(t1, lp1, t2, lp2, intervals)` (carries `corr_HY` in `.attrs`); `noise_robust_cov.realized_variance(logp, method="tsrv")`.

---

## 4. Jump-robust variation — the MWCB days

**What it does.** `bipower_variation` (Barndorff-Nielsen–Shephard 2004) and `threshold_variance` (Mancini 2009 truncated RV) estimate the **continuous** integrated variance, discarding jumps; `jump_test` (BNS / Huang–Tauchen ratio statistic, standardized by `tripower_quarticity`) tests for a price discontinuity and returns the jump's share of quadratic variation.

**The issue.** It extends issues (i)/(iii) into the tails. The four MWCB days are exactly where discontinuous price moves are likely, and naive realized variance conflates continuous volatility with jumps.

**How it enhances the paper.** The continuous/jump decomposition is itself economic evidence for H3 and the policy argument. If extreme post-halt moves are dominated by *jumps*, that supports the paper's own reasoning that "extreme price movements carry little fundamental information" — the empirical backing for joint circuit breakers. The continuous part (BV/TRV) also feeds the correlation/HEAVY layer cleanly.

**How to interpret.** `jump_test` returns `z` (≈ N(0,1) under no jumps) and a one-sided `p_value`: `z` above ~3 (p < 0.01) flags a discontinuity. `jump_var` = max(RV − BV, 0) is the jump component; `rj` = (RV − BV)/RV is the jump *fraction*. A high jump fraction in the five minutes after a halt is direct evidence the move is a discontinuity rather than informative price discovery.

**Validation.** On a diffusion with integrated variance 5×10⁻⁴ plus five known jumps, naive RV inflated to **9.9×10⁻⁴** (≈ 2× IV) while bipower recovered **5.3×10⁻⁴** (1.05× IV) and threshold RV **4.8×10⁻⁴** (0.96× IV); the test fired at **z = 26** (p ≈ 0). On a no-jump control, RV/BV = 1.02 and z = 0.90 (p = 0.18) — no false positive.

**Call.** `noise_robust_cov.jump_test(logp)` → `{rv, bv, rj, z, p_value, jump_var, tq}`; `noise_robust_cov.bipower_variation(logp)`; `noise_robust_cov.realized_variance(logp, method="threshold")`.

---

## 5. Rigobon identification through heteroskedasticity (+ the identifying-variation diagnostic)

**What it does.** `rigobon_identify` recovers the contemporaneous structural matrix between SPY and ES **without a recursive ordering**, using the fact that if the structural relationship is stable across regimes but the shock variances differ, the regime-varying reduced-form covariances over-determine the structure (an eigenvalue problem). `overid_statistic` tests the constant-structure restriction when 3+ regimes are available. `identification_diagnostic` is the **weak-identification check**: it flags the failure mode where regimes differ only in *scale*. `forbes_rigobon_corr` is the contagion-vs-interdependence adjustment.

**The issue (ii).** This is the principled substitute for the assumed ordering. The contemporaneous SPY↔ES structure — and *whether it is asymmetric* — becomes an estimate.

**How it enhances the paper.** The Rigobon `contemp` matrix recovers *both* directions at once and is printed next to the two Cholesky orderings (each of which mechanically zeroes one direction), so the ordering dependence the paper assumes away is visible beside the order-free answer. Agreement of the Rigobon lead direction with the HRY lead-lag (§2) is exactly the cross-method triangulation a referee wants.

**How to interpret — read the diagnostic first.** `identification_diagnostic` returns per-asset `variance_ratios` across the most-separated regime pair, their `var_ratio_spread` (≈ 0 means the variances scaled by a common factor → eigenvalues coincide → the structure is **not identified**), a `rel_eig_gap`, and an `identified` verdict. **A near-zero spread means the structural point estimate is meaningless however clean the eigen-decomposition looks** — do not report the coefficients. Use **exogenous** regimes (the dated MWCB halts, time-of-day buckets, scheduled-announcement windows), *not* realized volatility of the same series, because regime endogeneity contaminates the moment conditions. With 3+ regimes, `overid` ≈ 0 supports the constant-structure model.

**Validation.** With relative heteroskedasticity (asset variances scaling by different factors), the diagnostic returned `identified = True` (spread 2.22) and Rigobon recovered the true contemporaneous coefficients (0.398 and 0.194 against truth 0.40/0.20). With common-scale regimes (both variances × 3), the diagnostic returned `identified = False` (spread 0.01, eigenvalue gap 0.01) — correctly refusing to certify identification.

**Call.** `rigobon_id.identification_diagnostic(regime_cov)` → `{variance_ratios, var_ratio_spread, rel_eig_gap, identified, pair}`; then `rigobon_id.rigobon_identify(U, regimes)` → `{contemp_rigobon, contemp_cholesky_fwd, contemp_cholesky_rev, var_ratios, eig_separation, overid?}`.

---

## 6. Copula tail dependence + Aielli cDCC — the correct H3 "breakdown"

**What it does.** `select_copula` fits constant copulas {Gaussian, t, Clayton, Gumbel, BB1, SJC} by BIC, each reporting its **lower/upper tail-dependence coefficients (λ_L, λ_U)**. `t_copula_dcc` and `cdcc_fit` give the *dynamic* correlation path — `cdcc_fit` is Aielli's (2013) **corrected** cDCC with consistent correlation targeting (Engle's original DCC targeting is statistically inconsistent). `liquidity_conditional_dependence` splits the dependence estimate by a liquidity state.

**The issue (i).** A Pearson correlation has **zero** tail dependence for any ρ < 1: it asserts that far enough into the tail, SPY and ES become independent — exactly backwards for a joint crash. H3 is fundamentally a statement about dependence *in the tails*, and Pearson ρ is the wrong functional to test it.

**How it enhances the paper.** It tells the H3 story correctly. The breakdown question is "does the lower-tail co-crashing change during MWCB?" — answered by λ_L and its asymmetry (λ_L vs λ_U), not by a fall in average ρ.

**How to interpret.** λ_L is the probability that ES is in its extreme lower tail *given* SPY is — the structural co-crash propensity a Gaussian misses entirely. The H3 test compares λ_L (and the λ_L/λ_U asymmetry) on MWCB days versus baseline: a **rise** in λ_L under stress is genuine tail contagion; a **fall** is decoupling (the markets pulling apart). The cDCC path `R_t` gives the consistent time-varying *linear* correlation as the baseline overlay; prefer it to Engle DCC because the latter's targeting is biased.

**Validation.** `cdcc_fit` recovered (a, b, S) from data generated by a true cDCC (a = 0.063 vs 0.06, b = 0.884 vs 0.90, S₁₂ = 0.522 vs 0.50); the copula tail coefficients are analytic functions of the fitted parameters.

**Call.** `copula_garch.select_copula(Z)` → best family with `lambda_L`, `lambda_U`; `dcc_garch.cdcc_fit(Z)` → `{a, b, S, R, loglik}`; `dcc_garch.dcc_garch_x(returns, dcc_method="cdcc")`.

---

## 7. FPCA liquidity — beyond the one-tick spread

**What it does.** `depth_profile` represents the limit-order book at each instant as a depth-by-level *curve*; `fpca` extracts its functional principal components; `interpret_components` labels each component **level / slope / curvature** by the sign-change pattern of its loading curve; `fpc_state_series` / `relative_fpc_state` return a scalar liquidity *state* (the leading FPC score).

**The issue (iv).** The BBO spread is pinned at one tick for SPY and ES almost always — it cannot move, so it cannot serve as a liquidity state variable. The real liquidity dynamics live in depth and book shape.

**How it enhances the paper.** It supplies a liquidity conditioning variable that *actually varies*, replacing the spread in the SVAR conditioning and the PCMOF/NCMOF regime splits — and it is the natural instrument for the MWCB depth-withdrawal story (a halted futures book reopening empty).

**How to interpret.** `interpret_components` returns, per component, a `label` and its `variance_explained`. **FPC1 (level)** — constant-sign loading — is aggregate depth moving up or down at all price levels together; a *falling* level score is depth withdrawal, the real liquidity event during stress. **FPC2 (slope)** — one sign change — is the book tilting/steepening, i.e. near-touch depth versus deep depth moving oppositely (resiliency). Use the leading FPC score as the liquidity state; check `explained_variance_ratio` to confirm the leading factors actually dominate.

**Validation.** Synthetic book curves built from a known level factor (80% of variance) and slope factor (20%) were recovered and labeled correctly: FPC1 = level (variance share 0.79, 0 sign changes), FPC2 = slope (0.20, 1 sign change).

**Call.** `functional_liquidity.asset_fpca(df, "SPY")` → `{components, scores, eigenvalues, explained_variance_ratio, ...}`; `functional_liquidity.interpret_components(res)` → per-component `{label, variance_explained, n_sign_changes}`; `functional_liquidity.fpc_state_series(df, "SPY")` → the liquidity state series.

---

## 8. Cross-impact matrix + symmetry test

**What it does.** `cross_impact_matrix` estimates the K×K matrix Λ in r = Λ·OFI + e (diagonal = own/Kyle-style impact; off-diagonal = cross-impact / spillover), with Newey–West HAC inference (Cont–Kukanov–Stoikov 2014; Cont–Cucuringu–Zhang 2023). `compare_impact_depth` exposes the CCZ result that apparent best-level cross-impact largely shrinks once multi-level OFI is used. `cross_impact_symmetry` tests the Schneider–Lillo (2019) no-dynamic-arbitrage restriction Λ₁₂ = Λ₂₁ using the **joint cross-equation HAC covariance** (the per-equation SEs cannot test it).

**The issue.** The Eq. 6 order-imbalance-interaction specification is, in substance, an OFI cross-impact statement and should be benchmarked against that literature rather than presented as sui generis.

**How it enhances the paper.** It replaces the ad hoc imbalance interaction with a properly specified cross-impact matrix; the symmetry test is a no-arbitrage check; and `compare_impact_depth` is a publishable robustness point in its own right.

**How to interpret.** Λ off-diagonals are spillovers (SPY-OFI → ES return and vice versa). If `cross_impact_symmetry` **rejects** (p < 0.05), the cross-impact is asymmetric — one direction dominates beyond what no-arbitrage permits — which is a genuine economic finding and a sharper statement than "futures matter more." If `compare_impact_depth` shows the cross terms shrinking as book depth enters, report it honestly: CCZ find that multi-asset cross-impact often adds little once multi-level OFI is included, so the cross-impact claim must be framed against that caveat rather than as settled.

**Validation.** A symmetric DGP (Λ₁₂ = Λ₂₁ = 0.30) was not rejected (p = 0.67); an asymmetric one (0.50 vs 0.10) was rejected (z = 18.3, p < 10⁻⁴), with coefficients recovered (0.516 / 0.114).

**Call.** `cross_impact.cross_impact_matrix(df, assets=("SPY","ES"))` → `{Lambda, SE, t, R2}`; `cross_impact.cross_impact_symmetry(R, O)` → `{lambda_12, lambda_21, asymmetry, se, z, p_value, symmetric}`.

---

## 9. VECM / information shares — the cointegration cross-check

**What it does.** `panel_vecm` / `estimate_sample` fit the bivariate VECM exploiting the (1, −1) cointegration (same underlying, stationary basis); `hasbrouck_is` returns Hasbrouck (1995) information-share bounds + midpoint; `gonzalo_granger` returns the Gonzalo–Granger (1995) component shares; `lien_shrestha_is` gives an order-invariant variant.

**The issue.** The paper labels itself "price discovery" but never measures it, and under-exploits the cointegration that makes SPY/ES a textbook price-discovery system.

**How it enhances the paper.** This is an **independent** cross-check of the leadership claim — *not* a replacement for the SVAR (which is the point of the upgrade-in-place mandate). Agreement of the cointegration-based shares with the HRY lead-lag (§2) and the Rigobon contemporaneous structure (§5) is triangulation from three different identifying assumptions.

**How to interpret.** The IS bounds and CS give the permanent-component leadership; if they point the same way as the measured lead and the heteroskedasticity-identified structure, the price-discovery claim is robust and *earned*. **One honest caveat to print:** Hasbrouck IS and Gonzalo–Granger CS are each biased when the two series have different microstructure *noise* levels — and SPY (13 venues, order-level) versus ES (CME snapshots) plainly do. That is precisely why the noise-robust Putniņš Information Leadership Share would be the right *headline* metric. It is **deliberately not implemented** here: the vanilla ILS is academically contested (Shrestha–Lee 2023; Shen–Zivot 2024; corrected by Shen–Zhang–Zivot 2025, *JEF*), and the corrected variant should be validated against source before use rather than coded from memory. So read IS/CS as *bracketing* the leadership and lean on the noise-robust lead-lag for the point claim.

**Call.** `price_discovery_shares.panel_vecm(sessions)`; `price_discovery_shares.hasbrouck_is(alpha, Omega)` → `(lo, hi, mid)`; `price_discovery_shares.gonzalo_granger(alpha)` → component shares.

---

## 10. How the suite triangulates "who leads"

The paper's weakest seam is that one assumption (the Cholesky ordering) carries its central claim. The upgraded stack answers "who leads" from **three independent directions**:

1. **Time domain** — the HRY lead-lag θ̂ (§2): a signed lead in milliseconds.
2. **Heteroskedasticity** — the Rigobon contemporaneous matrix (§5): the simultaneous structure, identified by regime variance shifts, with a diagnostic that refuses to certify weak cases.
3. **Cointegration** — the VECM information/component shares (§9): permanent-component leadership.

When the three agree, the price-discovery conclusion is robust to *how* it was identified — which is the tier-1 standard: identification from multiple angles, not a single ordering. When they disagree, that disagreement is the finding, and the leadership language must soften accordingly.

---

## 11. Reporting checklist for the revision

| Paper element | Replace with | Statistic to report | Inference |
|---|---|---|---|
| Rolling Pearson ρ (Table 8, Eq. 5) | Hayashi–Yoshida correlation + signature plot | `corr_HY` vs the naive Epps curve | day-clustered |
| "Futures lead" (SVAR ID) | HRY lead-lag + Rigobon contemporaneous | θ̂ (ms); `contemp_rigobon` vs both Cholesky orderings | report `identification_diagnostic` |
| IRF stars (Tables 9, 11) | same point estimates | coefficient + WCB *p*-value | wild cluster bootstrap (day) |
| MWCB claims (Tables 13, 14) | same design, honest inference | effect + IM `theta`/`p` | Ibragimov–Müller / randomization (4 events) |
| "Correlation breakdown" (H3) | tail-dependence copula | λ_L, λ_L/λ_U asymmetry, MWCB vs baseline | bootstrap |
| RV/vol on MWCB days | jump-robust split | BV, jump fraction `rj`, `jump_test` z | — |
| BBO-spread liquidity | FPCA level/slope state | FPC scores + `interpret_components` labels | — |
| Eq. 6 imbalance interaction | cross-impact matrix + symmetry | Λ; `cross_impact_symmetry` z; depth-shrinkage | Newey–West HAC |

---

## 12. The onset identification spine — f(state) as a surface

The re-flooring above fixes the paper's *inputs and inference*. This section is the *identification* upgrade (v0.8.7–v0.9.3): a cleanly identified estimate of the stress-response function f(state) that the original design only gestured at, built so the confounds that would fake it are demonstrated and removed rather than assumed away.

**Why a function, why the onset.** "Price discovery degrades under stress" is a statement about a *function* — how cross-asset transmission depends on the state of the arbitrage link — not a high-vs-low-vol mean difference. The state is the SPY–ES **basis dislocation** jointly with **book capacity**, not realized vol: vol is the symptom, the basis is the health of the law-of-one-price link itself. Since SPY and ES share one underlying, "contagion" between them is breakdown of that link, which keeps **Forbes–Rigobon** the live referee weapon — a genuine contagion claim is the SPY–ES complex ↔ the broader market, tested as vol-adjusted linkage rising super-linearly past a stress threshold. Identification sits at the **onset** of each event (the release boundary), pre-feedback, because the within-crisis cascade endogenizes the very slope being estimated — state and response co-evolve once the loop engages. Two tiers kept distinct: the **identified onset** (the headline, f(state) traced by cross-event onset transmission, Ibragimov–Müller across events) and the **described cascade** (`markov_switching_vecm` — shown, not causally fit).

**The surface is the mechanism.** Brunnermeier–Pedersen and limits-of-arbitrage say the spiral is *multiplicative*: the arb fails when mispricing is large *and* capacity is thin, each feeding the other. In the data that is the **basis × capacity** interaction b3 in `fit_stress_surface`. The additive specification cannot represent a spiral, so the interaction is the minimum mechanism-bearing form — dropping it quietly rewrites the paper's claim into a linear reduced form. The trap: b3 is identified off the *off-diagonal* events (a big dislocation a deep book absorbs; a thin book without a shock) — exactly the configuration stress does *not* naturally produce — so the most important coefficient is the one the events are least able to identify. `interaction_identified` is the explicit weak-ID gate (corr, centred-interaction VIF, off-diagonal corner coverage), and a fit on an unspanned plane returns NaN rather than an exploded least-norm coefficient.

**The bid-ask-bounce confound — the referee's objection, defused.** The transmission is a heteroskedasticity-identified coefficient off the pre→post variance shift. Rigobon cancels *time-constant* microstructure noise, but post-release the touch thins and the BBO flips faster, so bounce variance *shifts at the onset* — and a hollow pre-window predicts a larger shift, attenuating the transmission most exactly where the spiral predicts, faking a negative b3 that shares the hypothesis's sign. `transmission_robust` re-estimates each regime's covariance with TSRV variances + a Hayashi–Yoshida cross (the same noise-robust machinery §3 uses for the correlation), so the bounce cannot enter the variance shift. A constructed NO-spiral DGP with onset-scaled bounce confirms it: the naive surface manufactures a significant negative b3, the robust transmission returns it to zero, and a *pre-window* spread/noise control provably cannot rescue it — the fix has to be the covariance estimator, not a covariate. (Note the asymmetry of the result: because the confound shares the spiral's sign, this is not optional robustness — it is the difference between a finding and an artifact.)

**Capacity is two-legged.** The arb is a round trip, so capacity is aggregated across both books three ways — sum, bottleneck (max), product — and the diagnostics adjudicate rather than a prior. The product is a disguised three-way (basis × cost_SPY × cost_ES) that the event count cannot fund and that is fat-tailed by construction; on co-moving synthetic data it is *dominated* (more leverage-fragile than the sum with no compensating signal), the bottleneck gives the strongest-but-most-fragile estimate (the binding leg carries the information the average washes out), and the sum is the stable baseline. Report **sum and max as co-headline**: b3 stronger under max is itself a joint-withdrawal signature — the binding leg matters only if the two books thin together.

**The curated tail.** The marquee shocks are tested as *excess* over the impact-blind-fitted surface — transmission beyond what a scheduled release at the same dislocation and capacity predicts — so the scheduled backbone's exogeneity identifies the excess even though the tail is salience-selected. The bilinear surface extrapolates outside its support, and the marquee points sit precisely there (high basis, hollow book), so `within_support_frac` flags out-of-box excess as extrapolation, not interpolation.

## 12.5 The mid the surface runs on is reconstructed — and never spuriously crossed (v0.9.5–v0.9.7)

Everything in §12 — the transmission, the basis dislocation, the bounce-robust covariance — is computed on a **consolidated mid reconstructed from the raw message stream**, not a vendor snapshot, so its construction has to be auditable end to end. The first real-tape pass exposed a replay bug rather than a market state: the consolidated top was strictly crossed on nearly every snapshot. The cause is ordering. UDP multicast arrives out of packet order, so ordering a feed by any timestamp inverts ~10% of adjacent events — a Cancel slips ahead of the Modify it follows, the Modify (remove+add) resurrects the deleted order, and the phantom level pins the top crossed. **v0.9.5** orders intra-feed by the venue's authoritative `sequencenumber` (per feed, the clock made monotone along it, feeds merged by that clock), which clears the SPY leg to 0%. **v0.9.7** closes the CME residual: `sequencenumber` is *packet-level* there, so a Modify and the Cancel of its order can share one value, and the within-tie fallback resurrected the order on a single packet (ES crossed on ~88% of snapshots even after v0.9.5); ranking liquidity removals **last** within a tie makes the cancel the final word — a no-op on strictly-increasing-sequence equity feeds, biting only on genuine ties. This is a genuine data-cleaning contribution, not plumbing: a mid that is mechanically crossed corrupts every information share and the basis itself, so the fix sits upstream of every estimate in §12. `verify_crossing.py` (v0.9.6) prints the before/after crossed-fraction for the paper's data-cleaning appendix, and a hard never-crossed invariant now guards every reconstructed book. Verified on synthetic known-answer guards (`test_reconstruct_ordering`, `test_verify_crossing`, `test_crossed_regression`) and the full suite; the real-tape audit on the now-in-hand co-temporal 2025-04-03 window is the next step.

## 13. Honest limits

- **The code cannot manufacture the data — but the first real window has arrived.** The decisive validation is running this battery on the actual April-2025 MayStreet tape (the `smoke_test_crossed.py` acceptance gate). Everything verifiable in-sandbox is verified — including the §12.5 reconstruction / event-ordering fix on its full guard suite — and the first co-temporal stress window (SPY + ES, 2025-04-03 09:29–09:32 ET) is now in hand; the real-tape crossed audit and the multi-event onset extraction are the steps that remain.
- **The Putniņš ILS is deferred, on purpose** — contested formula; the Shen–Zhang–Zivot (2025) correction needs source verification before implementation.
- **DCC at tick frequency** is a known compromise; the mainstream view favors realized-measure / HEAVY dynamics on the noise-robust covariances. cDCC is the correct *return-based* choice with the noise caveats disclosed.
- **Cross-impact beyond multi-level OFI** is contested: CCZ (2023) find it often adds nothing once OFI is integrated across book levels. Frame the Eq. 6 cross-impact accordingly.
- **Regime exogeneity is an identifying assumption** the diagnostic cannot test — only the distinct-eigenvalue / relative-heteroskedasticity condition is testable. Use externally-timed regimes.
- **Onset vs cascade.** The identified object is the *onset* (pre-feedback) response; the within-crisis cascade is described, not causally identified. The onset b3 is the static complementarity the spiral predicts (its cross-sectional shadow), not a measurement of the self-reinforcing loop.
- **The plane must be spanned for the interaction.** b3 lives in the off-diagonal corners; diagonal-clustered events leave it unidentified regardless of fit quality (`interaction_identified` is the gate). The remedy is corner-loading impact-blind events — an extraction target, not an estimator change.
- **The robust transmission trades bias for variance.** TSRV de-attenuation removes the bounce bias but adds estimation noise, and on short onset windows it can over-correct. Report `transmission` and `transmission_robust` side by side, and treat the *magnitude* (not just the sign) as calibrated only on real data.

---

*Cross-references: every claim above is backed by a guard in the repo (`test_wild_cluster`, `test_leadlag`, `test_jumps`, `test_rigobon_id`, `test_cdcc`, `test_fpca_interp`, `test_xi_symmetry`, `test_onset_response`, `test_stress_surface`, `test_noise_robust_surface`, `test_surface_excess`, `test_cost_aggregation`, `test_reconstruct_ordering`, `test_verify_crossing`, `test_crossed_regression`) plus the per-module self-tests. See `CHANGELOG.md` for the version history (v0.2.0 → v0.9.8).*
