# Augmenting OFR Working Paper 19-04 toward JF/RFS — A Revision Plan

**To:** Mark Paddrik, Pankaj Jain
**From:** Robert Garrison
**Re:** Elevating *Cross-Asset Market Order Flow, Liquidity, and Price Discovery* to a top-3 finance journal
**Date:** June 2026
**Revised:** August 2026 — §§9–10 added after the first clean full-sample replication run

## Thesis

Our working paper documents a real and important regularity — order-flow spillover between SPY and ES that resolves within about a second — but it leaves the central word in its own title, *price discovery*, formally unmeasured, and it proxies *liquidity* by the bid–ask spread alone. The opportunity is to close both gaps with a single idea: **price discovery is liquidity-state-dependent — the market whose limit-order book is relatively deeper and flatter carries the larger information share, and that advantage widens under stress.** That one mechanism unifies the paper's three descriptive themes — order flow, liquidity, price discovery — into a single falsifiable claim, and to my knowledge it is novel. With MayStreet depth data and a fully built, self-tested code stack (appendix), this is a credible JF/RFS revision rather than an incremental one. What follows is the report a top-journal referee will write, the contribution, the methods, the identification, and a prioritized sequence.

## 1. Why now

Three things have changed. First, the methods bar for cross-market price-discovery work has moved: formal information shares, multi-level order-flow imbalance and cross-impact, noise-robust realized covariance, and credible (non-recursive) identification are now table stakes at JF/RFS. Second, we have the data to meet it — full visible book depth for both venues via MayStreet, and a decade of stress episodes (the 2014 SIP outage, 2020, 2022, the 2023 regional-bank episode) to refresh and extend the event sample. Third — disclosure — I have implemented the entire analytical apparatus as a modular, self-tested Python stack, so the marginal cost of the revision is data wiring and writing, not method development.

## 2. What a JF/RFS referee will flag

It is worth pre-empting the report we would receive.

**Gap 1 — Liquidity is the spread only.** The spread is one number; the book is a curve. Depth, slope, convexity, and the cost of walking the book are first-order to who absorbs information and who follows, and none of them is visible to a spread. A referee will ask why a paper about liquidity and price discovery measures liquidity with its least informative statistic.

**Gap 2 — The title promises price discovery; the paper reports no information share.** We characterize order-flow lead–lag with a structural VAR but never report Hasbrouck information shares or Gonzalo–Granger common-factor weights. This is the most direct and least contestable objection, and closing it is the minimum publishable upgrade.

**Gap 3 — Identification assumes the answer.** The recursive (Cholesky) ordering imposes that one venue moves contemporaneously before the other. At the SPY–ES horizon the contemporaneous link is genuinely simultaneous, so the ordering assumes exactly the object we are trying to estimate. A referee will not accept recursive identification as the basis for a price-discovery claim.

Two secondary points will also surface: the order-flow measure (message-count proportions) discards signed size and is coarser than modern order-flow imbalance; and the sample is dated and pre-COVID.

## 3. The contribution: liquidity-conditional price discovery

The headline is a VECM on the two log mid-prices — cointegrated with the no-arbitrage vector (1, −1) — in which the error-correction speed is allowed to depend on the relative book state s_t (e.g. the log-ratio of ES to SPY depth). The price-discovery shares then become a *function* of liquidity, and the central hypothesis is directly testable: as the book tilts deeper and flatter toward one venue, that venue's common-factor share rises. We estimate this per session and, for power, in a windowed panel — many intraday windows × days, with within-day fixed effects and day-clustered standard errors — and we sign the effect against the relative-depth, relative-slope, and relative-entropy state variables.

Why this clears the bar: it is not a fourth descriptive result bolted onto three others. It is a mechanism that *explains* the paper's existing findings — order flow leads where the book is deep enough to absorb it, and that lead manifests as a larger information share — and it yields an economically interpretable, falsifiable prediction the spread-only framework cannot express. It also speaks to live policy questions: ETF resilience, the locus of price discovery under stress, and — adjacent to my work at the Commission — what fragmented or tokenized venues inherit when depth migrates.

## 4. Method upgrades

Each maps to a built module.

- **Depth-curve liquidity** — functionals of the whole book (entropy, arc-length shape, slope, center of mass, cost-to-fill, cross-venue shape cosine), replacing the spread; and a functional-PCA layer that lets the data choose the dominant book-shape mode as the liquidity-state variable rather than imposing one.
- **Information shares** — Hasbrouck IS with the Lien–Shrestha order-invariant variant and the midpoint bound, plus Gonzalo–Granger CS. I deliberately defer Putniņš's information-leadership share — the measure is contested in the recent literature — and report IS / MIS / CS.
- **Multi-level OFI and cross-impact** — Cont–Kukanov–Stoikov order-flow imbalance summed over the book, and a cross-impact matrix separating own- from cross-venue price impact, including the Cont–Cucuringu–Zhang point that apparent cross-impact shrinks once multi-level OFI is used.
- **Time-varying comovement** — a DCC-GARCH-X for the SPY–ES conditional correlation, replacing the coarse volatile/benchmark split with an estimated ρ_t and exogenous variance drivers.
- **Dynamics** — state-dependent impulse responses (Jordà local projections with a book-state interaction) and structural VECM IRFs with a forecast-error variance decomposition, so the "decays within a second" claim acquires proper error bands and the information shares become transparent (they come from the same VMA representation).
- **Sub-second layer** — noise-robust realized variance (two-scale, realized kernel) and asynchronicity-robust covariance (Hayashi–Yoshida; refresh-time plus multivariate realized kernel), so the 100ms/10ms results are not contaminated by microstructure noise and the Epps effect, with staleness and tick-discreteness diagnostics that flag where the information shares stop being trustworthy.
- **Continuous vs jump price discovery** — quote prices are jump-diffusions, so a vanilla information share blends the diffusive and the jump parts of the common factor's quadratic variation. Using jump-robust realized measures (bipower / MedRV; Mancini truncation; Barndorff-Nielsen–Shephard and Lee–Mykland tests; threshold covariation) we split the information share into a continuous component and a jump component, holding the cointegration fixed so the total is unchanged. This separates *which venue leads diffusive price discovery* from *which reacts first to jumps* — and the thesis predicts both that the deeper book carries the larger continuous share and that it absorbs jumps with less impact, each conditional on the relative book state. The jump *fraction* is reported via Lee–Mykland classification (each return normalized by a local bipower volatility), which — unlike the integrated (RV−BV) or global-threshold estimate — does not load fat tails, microstructure noise, and tick discreteness into the jump bucket, and is run on 100ms returns. A full bivariate Hawkes model of the order flow (whose static linear shadow is the cross-impact matrix) is left as a separate paper.
- **The mechanism as a state-dependent error-correction SDE** — the liquidity-conditional VECM is the Euler-Maruyama discretization of `dY = α(S)(β′Y)dt + Σ(S)^½ dW`, in which the error-correction loadings α(S) — hence the price-discovery direction ψ(S) and the information share IS(S) — are functions of the book state. The coefficient on the z·S interaction, with day-clustered standard errors, is a single-number test of liquidity-conditioning and the continuous-time statement of the thesis; kernel-local estimation traces the full IS(S) curve. Because the information share is a scale-invariant ratio, the discrete estimate targets the continuous-time object directly. Crucially, the share is computed on a **noise-robust observable** — a depth-weighted microprice / area-under-curve mid rather than the top-of-book midpoint, whose bid-ask bounce and tick discreteness bias every share toward one-half — and on a coarser grid (sparse sampling / pre-averaging) for the measure itself. The coupled-book SPDE (the full depth profile with a moving free boundary) is noted as separate theory, not invoked for decoration.
- **Robustness battery** — lag length, estimated vs imposed cointegrating vector, AM/PM and subsample stability, window-size sensitivity, alternative state variables, and stationary-bootstrap confidence intervals.

## 5. Identification

This is where the revision is strongest, and where the referees are won or lost.

**Cross-impact identification.** Rather than a Cholesky ordering, we identify the contemporaneous structure from the estimated cross-impact matrix — the contemporaneous OFI→return map. We *estimate* the simultaneous response instead of assuming an order.

**Heteroskedasticity (Rigobon).** Our event design already supplies the variance regimes — volatile vs benchmark days — that identification-through-heteroskedasticity exploits. The shift in the shock covariance across regimes pins down the structural parameters without an ordering, and gives an independent cross-check on the cross-impact identification.

**The October 30, 2014 SIP outage as a natural experiment.** This is the causal exhibit. A documented disruption to the consolidated quote feed for affected symbols is an exogenous, transient degradation of SPY's quoted-price quality that the futures market did not share. The liquidity-conditional mechanism makes a sharp prediction — price discovery should shift toward ES during the outage and revert afterward — which we test in a difference-in-differences design around the event. That converts the central claim from a conditional correlation into a causal one, which is precisely what elevates the paper.

## 6. Data plan

MayStreet, full visible depth for both SPY (consolidated) and ES (CME), on a common grid. A baseline at one second; a practical high-frequency primary at 100ms; and 10ms reserved for the cross-impact and local-projection IRFs, where the latency structure is the point (the round-trip between the two matching engines is the relevant clock). The event set extends the original volatile/benchmark design through the post-2019 stress episodes and brackets the SIP-outage window. The one prerequisite is wiring the per-level book passthrough into the extraction so it emits the per-session depth frames the stack consumes. Two refinements since this plan was first drafted. First, the extraction path is now **message-level reconstruction**, not a vendor snapshot: SPY is rebuilt as the consolidated multi-venue book and ES as a single-venue CME order-by-order replay, both on one GPS-disciplined capture clock — the snapshot tool sits on a *different* clock and would silently corrupt the SPY↔ES lead-lag the paper rests on. Second, that reconstruction forced a genuine data-cleaning fix worth its own appendix: the first real-tape pass returned a consolidated top crossed on nearly every snapshot, traced to event-ordering (out-of-packet UDP arrival, compounded by CME's packet-level sequence numbers); ordering intra-feed by the venue sequence with liquidity-removals-last within a tie resolves it, verified on a synthetic guard battery, and `verify_crossing.py` produces the before/after a referee will want to see. The first co-temporal stress window — SPY and ES, 2025-04-03 (Liberation Day) 09:29–09:32 ET — is now in hand; the remaining prerequisite is scaling that same extraction across the full event sample.

## 7. Prioritized revision sequence

1. **Data.** Wire the depth passthrough; run the extraction on the refreshed, extended sample; produce the per-session book frames. *(Unblocks everything.)*
2. **Information shares.** Report IS / MIS / CS for the existing event set. This alone answers Gap 2 and is the minimum publishable delta.
3. **Depth-curve liquidity + the headline.** Compute the book functionals; estimate the liquidity-conditional VECM and the windowed panel; sign the effect across state variables. *(The contribution.)*
4. **Identification.** Cross-impact matrices; structural and state-dependent IRFs + FEVD; the Rigobon cross-check. *(Answers Gap 3.)*
5. **Natural experiment.** The SIP-outage difference-in-differences. *(The causal exhibit.)*
6. **Frequency robustness + comovement.** The 1s→100ms→10ms exhibit with noise-robust covariance and the staleness/discreteness diagnostics; the DCC-GARCH-X ρ_t.
7. **Robustness battery and write-up.** The full referee battery; assemble the draft.

Items 2–3 are the publishable core; 4–5 move it from a strong field paper to a top-3 submission; 6–7 are armor against the report.

## 8. Target and positioning

JF or RFS. The one-sentence pitch: *cross-asset price discovery is governed by the relative shape of the limit-order book — the deeper, flatter market leads — and we identify the channel causally off an exogenous quote-feed disruption.* It advances the ETF/futures price-discovery literature (Hasbrouck; Chan) by making the information share endogenous to book depth; it brings the modern order-flow-imbalance and cross-impact apparatus (Cont–Kukanov–Stoikov; Cont–Cucuringu–Zhang) to the cross-asset question; and it does the second-moment work to current standard (Barndorff-Nielsen–Hansen–Lunde–Shephard; Hayashi–Yoshida) — the realized-variance toolkit Kevin flagged in the original acknowledgments is now load-bearing.

## 9. August 2026 addendum — what the full-sample runs taught us

Since June the stack has gone from verified-on-synthetic to a completed extraction of the full
event sample — 24 sessions (10 volatile, 10 matched baseline, the four March-2020 MWCB days),
2020–2026, 561,624 frame-rows — and the first end-to-end replication run is clean: every gate
green, both legs present on every session including the circuit-breaker days. Getting there forced
a sequence of methodological corrections. I list each with its reason, because several change
numbers we currently publish, and one produces a result I think belongs in the paper in its own
right. Everything below is pinned by a regression test against verbatim tape rows, so none of it
can silently regress.

**9.1 The nulls in Tables 5 and 7 measure the wrong thing.** The published Panel A null is iid
Binomial(n, ½) per market. Real order flow is over-dispersed, autocorrelated, and seasonal within
the day, so that null rejects for reasons that have nothing to do with *cross-market* trading — a
referee who notices will discount the tables entirely. We now benchmark against independence *given
the observed marginals* (which isolates the cross-market component by construction), report the
corner log odds ratio (marginal-free, hence comparable across panels whose marginals differ
enormously), and add a permutation null that shuffles only the pairing between the markets. The
consequential correction is Table 7's: the published comparison applies the per-second null (0.4%)
to every aggregation, but at the actual per-bar order counts the independence null is 22.9% at ten
milliseconds and 50.1% at action time. Restated at matched nulls the observed corners are **33.5×
the null at one second, 1.32× at ten milliseconds, and 0.97× at action time**. The tandem effect is
a *one-second phenomenon* — sharper and more interesting than the uniform-null version, and we say
it before a referee computes it against us.

**9.2 Table 9's dependent variable carries its own measurement error — and its lag order was the
window.** Two problems, one specification. First, grid-sampled Pearson correlation is attenuated by
asynchronous quote updating (Epps), and the attenuation moves with trading intensity — which sits
on the right-hand side of Eq. (5). A response can be measurement error correlating with its own
regressor. We estimate the table both ways (Pearson and Hayashi–Yoshida, identical sessions, window
and identification) and report the difference. On the extracted sample this is not cosmetic: the
ES order-flow-imbalance response in the benchmark panel **flips sign and loses significance**
(−0.014*** Pearson → +0.009 n.s. HY), two effects are significant only under HY, and one
(WtdSpread_SPY) is stable across both — the only cell we should interpret as published. Second,
footnote 17's mystery — AIC pointing at 60 lags — is now explained: the first difference of a
W-bar rolling correlation carries an MA term at exactly lag W, and the criterion locates it. On
simulated data with a *constant* correlation and iid liquidity, BIC selects p\* = W exactly (8, 12,
20, 25 for W = 8, 12, 20, 25) and collapses to 0 for wide windows. Raising the search bound walks
toward the window, not toward the truth. The fix is the dependent variable, not the search: the DCC
conditional correlation has no fixed-width box to difference and returns p\* ≈ 1 on the same null
data. Table 9 now runs three ways (Pearson / HY / DCC), the caption states the lag diagnosis, and
DCC is the defensible basis for the lag order. The sixty lags were an estimator artifact; we can
now say so in print.

**9.3 A halted market has no midpoint — and the halt boundary must come from the tape.** The MWCB
halt windows are derived per venue from the status stream (`mt_product_status.haltreason`), not
from a hand-entered table. During a Level 1 halt the exchanges stop matching but do not cancel
resting orders, so a *correctly* reconstructed book is crossed for the full fifteen minutes — on
2020-03-09 the residual "defect" of 3.88% crossed snapshots is 901 halt seconds of 23,401, and
outside the halts the crossed rate is 0.02–0.17% on all four days. Halt snapshots are excluded from
every estimate — including them adds mechanical comovement, not price discovery — and the
boundaries are now cross-validated from the *other leg*: ES's first trade back lands +10 ms, +6 ms
and +12 ms after our recorded SPY halt ends on the three days both tapes exist. Two independent
feeds agreeing at the millisecond on boundaries that were hand-entered before either was checked.

**9.4 The futures do not trade through the equity halt — the sole-venue window is a new exhibit.**
The natural prior (and my own first write-up) was that flow migrates to ES when equities halt. The
tape says otherwise, twice over. First, CME's halt flag marks the *stop*, not the duration: it is
set milliseconds after the last trade and clears 5.8–7.3 s later while nothing trades for
**817–846 s** — reading the flag as the window understates the stop by two orders of magnitude
(116–140×), and the two readings support opposite claims about which market leads. Halt ends now
come from the resumption of trading. Second, with the windows measured correctly, every MWCB day
has the same anatomy: the equity market halts; ES keeps trading *alone* for 54–89 seconds — the
only price venue in the pair — and then halts too; both reopen together (within 6 s). And in its
window of solitude ES's trading rate *falls* — −95%, −73%, −58% on the three days with a valid
intraday baseline — before going to zero. The futures do not absorb the flow; they nearly stop,
then stop. For the event study this means two nested onsets per day (the equity halt, then the
coordinated futures stop ~1 minute in), union-excluded pair estimates, and a sharply bounded
54–89 s window in which one venue *is* price discovery — which I think is a publishable exhibit on
its own.

**9.5 The ES book is the venue's own ladder — one mechanism across the whole panel.** The
message-level ES replay hid an era dependence: CME's 2024 capture is order-by-order (MBO), its 2020
capture is price-level (`mt_modify_price_level`/`mt_delete_price_level`), and the one price-level
type we had checked is empty in *both* eras — so the four 2020 sessions extracted with no ES leg at
all, silently (a full-length frame of NaN, one log line). Rather than hardcode a second era
assumption one release after the first one failed, the futures leg is now built from
`mt_aggregated_price_update` — CME's own ten-level ladder, populated in every era we checked, on
the same lake and capture clock. The reasons: CME is a single venue, so its ladder *is* the book
and there is nothing to consolidate; and a panel spanning 2020–2026 should not construct its
futures book two different ways depending on the year. What is given up (order-level queue detail)
is nothing our measurements use; what is gained is uniformity and a validation we can state
cleanly: an *independent* message replay, its family selected from row counts at run time,
reproduces the ladder — evidence the ladder is a faithful book, i.e. validation of the source
choice, stated in that direction to avoid circularity. A measured ladder-cadence statistic
(updates per grid cell) decides rebuild-vs-take empirically rather than by argument.

**9.6 The roll is a week, not a switch — measured, not assumed.** Front-month volume shares
measured from the statistics stream across the March 2020 roll: **93.7%** three days before the
boundary, **72.8%** at it, **60.4%** four days after, **78.0%** by six. The calendar rule picks the
volume leader on all four days, but a single-contract ES leg misses 6–40% of futures volume inside
the roll week — and all four MWCB days sit inside it. We do not splice (the contracts carry a 10–12
point calendar spread; a stitched series jumps at the seam); we report the measured share in the
sample appendix, per session, via a tool that settles any date from forty vendor rows. One
criterion note for the appendix: open interest *disagrees* with volume on exactly the contested day
(34.6% vs 60.4% on 03-16) because OI is a day-stale stock of positions while volume is the flow
price discovery is made of — volume is the criterion, and the disagreement is reported rather than
assumed away.

**9.7 Regulatory state as confounds: Rule 201 and price limits.** Tables 5 and 7 count *signed*
order flow, and Regulation SHO Rule 201 is one-sided by construction — short sales cannot execute
at or below the NBB, suppressing aggressive ETF selling while leaving buying untouched. The status
tape shows exactly one restricted session in our sample: 2020-03-16, restricted from the opening
bell for 100% of the session. A dummy "control" is not identifiable there — one restricted day,
collinear with its own at-the-open circuit-breaker halt — so the treatment is (i) the MWCB panel
reported with and without that session, and (ii) a one-sided diagnostic, the sell-minus-buy
Neutral-referenced local log odds ratio, which is ~0 under symmetric dependence and decisively
negative under sell-side suppression. On our published matrices it is +0.005 / +0.011 / +0.066 —
symmetric, no SSR fingerprint at the pooled level, which is itself worth a sentence. Separately,
the venue status stream settles the band question: the direct equity feeds carry no LULD fields
(the bands are SIP-disseminated), but CME *does* publish ES price limits — including the
limit-down ratchet on 2020-03-12 (lower limit stepping 2594.00 → 2190.00 with the upper unbounded)
— so the futures leg has a usable band control today and the equity leg does not; the coverage is
measured and reported so an all-NaN column is never mistaken for a control.

**9.8 Reproducibility is now a referee exhibit, not a promise.** Every failure above was *silent* —
a full-length, correctly-shaped, wrong dataset. The replication is therefore a single command with
staged gates: a correctness gate of fifty test modules, each pinning a named correction to verbatim
tape rows; a per-session crossed-book invariant (with halt-aware exclusions, and ladder-integrity
checks — monotone depth, coverage, staleness — for the taken futures book, where "crossed" is
vacuous by construction); driver-flag and version-consistency gates; caches keyed by everything
that changes a frame, including the ES book source; and run logs that state which source produced
each table — including the distinction between the paper's order-submission counts and the
trade-signed counts the frames carry (executions, not submissions: a better-identified object, and
labelled as such rather than passed off as the published measure). This is what current JF/RFS
data-and-code policies ask for, and it is also, frankly, how the errors above were found.

## 10. What this changes in the draft

Concretely, before anything new is estimated:

1. **Table 7 restated at matched nulls.** The headline becomes: tandem trading is a one-second
   phenomenon (33.5× / 1.32× / 0.97×). The uniform-null version does not survive refereeing.
2. **Table 5 re-benchmarked** — independence given marginals, corner log OR with z, the MWCB panel
   with and without 2020-03-16, and the corner-asymmetry diagnostic in the notes.
3. **Table 9 three ways** (Pearson / HY / DCC), the Pearson→HY deltas reported as the Epps share of
   each response, DCC as the lag-order basis, and footnote 17 replaced by the artifact explanation.
4. **The MWCB event study redesigned** around two nested onsets, union-excluded pair estimates, the
   ±6 s coordinated reopening, and the 54–89 s sole-venue window as its own exhibit.
5. **A sample appendix** with, per session: the ES contract used and its measured roll share, SSR
   state (including *unknown*, never conflated with unrestricted), halt windows from the tape, and
   the QC line (crossed rate outside halts, both legs).

**Sequence update (June §7):** step 1 (data) is *done* — the 24-session extraction is clean and
cached; step 2 (information shares) is unblocked and next. Three items are open: the 10 ms
extraction (the Epps exhibit is largest there); one unresolved ladder-vs-replay disagreement on
ESH5 2024-12-18 to run down before we lean on the 2024 ES leg; and re-estimating Tables 5/7 from
the extracted trade tape (wired, labelled DATA(trades)) alongside the published order-count
versions.

## Appendix: the code stack

Fourteen analysis modules, each with a self-contained synthetic self-test, composing through a common per-session book-frame interface, plus a single driver (`run_analysis.py`) that runs the whole sequence end-to-end and a run guide (README):

- `market_analysis_fixed.py` — MayStreet/Athena extraction (the column-resolution fix applied).
- `liquidity_curve_metrics.py` — depth-curve functionals.
- `price_discovery_shares.py` — Hasbrouck IS/MIS, Gonzalo–Granger CS, panel VECM, permutation tests.
- `cross_asset_pd_liquidity.py` — integration layer: OFI, the liquidity-conditional VECM, the windowed panel, frequency scaling, the fleeting-quote filter.
- `cross_impact.py` — cross-impact matrices with Newey-West HAC inference.
- `dcc_garch.py` — DCC-GARCH-X.
- `irf.py` — local-projection and structural VECM IRFs + FEVD.
- `robustness.py` — the robustness battery.
- `noise_robust_cov.py` — two-scale / realized-kernel variance, Hayashi–Yoshida / refresh-time covariance, Epps and signature diagnostics.
- `microstructure_diagnostics.py` — staleness, tick-discreteness, and noise-to-signal sweeps.
- `functional_liquidity.py` — functional PCA on the depth curve; the leading book-shape eigen-mode as a data-driven liquidity-state variable.
- `jump_robust.py` — jump-robust realized measures (bipower / MedRV, Mancini truncation, Barndorff-Nielsen–Shephard and Lee–Mykland tests, threshold covariation, co-jump lead-lag) and the continuous-vs-jump information-share split, built on `price_discovery_shares`.
- `ecm_sde.py` — liquidity-conditional price discovery as a state-dependent error-correction SDE: the state-interacted VECM (the z·S loading-gradient test with day-clustered SEs), kernel-local varying-coefficient IS(S)/CS(S)/κ(S) curves, and a Euler-Maruyama pseudo-MLE with state-dependent diffusion. Accepts a `price_fn` so the SDE runs on any observable.
- `robust_prices.py` — noise-robust observables for the SDE: depth-weighted microprice, book-centroid and area-under-curve/cost-to-fill mids, curve-length-normalized book states, and sparse-sampling / pre-averaging for the measure; the `price_fn` / `state_fn` glue that feeds them into `ecm_sde`.
- `run_analysis.py` — the driver: one command loads/extracts the frames, runs every stage above, and writes the result tables and a summary report.

Everything is synthetic-tested against known data-generating processes. As of the August revision the stack stands at roughly sixty modules under a fifty-module test gate, each test pinning a named correction to the verbatim tape rows that motivated it. The stack has since grown well beyond the original fourteen modules — most materially, the onset identification spine (the heteroskedasticity-identified stress-response surface f(state)) and the message-level book reconstruction with the event-ordering / crossed-book fix noted in §6 — all held to the same synthetic-self-test discipline. The empirical results await the multi-event data run in step 1; as of this revision the reconstruction layer is fully verified and the first co-temporal stress window is in hand.
