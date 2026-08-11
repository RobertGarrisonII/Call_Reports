# Revisiting "Cross-Asset Tandem Trading and Extraordinary Volatility": What We Changed, Why, and What We Found

*Preliminary — for internal circulation. Plain-language edition. Stack version v0.9.67 —
August 11, 2026.*

## 1. What this report is

We rebuilt the paper's analysis pipeline from the ground up, re-examined its methodology end
to end, and re-ran everything on a fresh 2022–2026 sample plus the March-2020 circuit-breaker
days. This report explains, in plain terms: how the original paper measured things, where
those methods were fragile, what we changed, why the changes are better, and what the answers
look like now. Technical terms are explained the first time they appear and collected in a
short glossary at the end.

The short version: the paper's central ideas survive — the two markets really do trade in
tandem, and futures really do lead — but both claims come out sharper and more *conditional*
than the published versions. Futures leadership turns out to be a stress phenomenon, not a
general law. Tandem trading is real and intensifies under stress, but one of the paper's most
dramatic-looking numbers was an artifact of comparing against the wrong benchmark. And the
liquidity mechanism at the heart of the paper's story is invisible at the frequency the paper
measured — and unmistakable at the frequency we now measure.

## 2. The question and the data

**The question.** The S&P 500 trades in two major forms at once: SPY, an exchange-traded fund,
and ES, a futures contract. They track the same index, so their prices should move together —
and arbitrage traders make money keeping them together. The paper asks what happens on the
wildest days: Do the two markets trade *in tandem* — buying and selling in the same direction
at the same moments? Which market *sets* the price and which one follows (what economists call
"price discovery")? And does the answer change when liquidity dries up — when the standing
orders that normally cushion the market get pulled?

**The data.** 24 full trading days, in three groups: the ten most volatile SPY sessions of
2022–2026 (the largest high-to-low intraday ranges), a calm "twin" for each of those days,
and the four days in March 2020 when the market-wide circuit breaker halted all trading.
Each volatile day's twin is the same weekday roughly one year earlier — same day of the week,
similar market structure, no crisis — so every "wild day" result can be compared against a
matched "normal day" baseline rather than against an average.

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
| 2020-03-09 | Mon | 09:34 | the co-jump analysis of §6.1 |
| 2020-03-12 | Thu | 09:35 | |
| 2020-03-16 | Mon | 09:30 (at the open) | short-selling restricted all session |
| 2020-03-18 | Wed | 12:56 | |

*Each March-2020 session contains one Level-1 market-wide circuit-breaker (MWCB) halt: the
S&P 500 fell 7%, and all trading stopped for 15 minutes.*

For every one of these days we have the full *order book* for both markets — not just the
price, but the whole ladder of standing buy and sell orders with their sizes — snapshotted
every 10 milliseconds. That is roughly 2.3 million snapshots per market per day. A one-second
version of the same data is built directly from those snapshots, which matters more than it
sounds (Section 5).

## 3. How the original paper went about it

Six pillars of the original methodology, in plain terms:

1. **Sample and handling.** Ten volatile days from 2014–2017 with paired calm days, plus the
   2020 circuit-breaker days. The 15-minute trading halts on the circuit-breaker days stayed
   in the dataset.
2. **One-second measurement.** Prices and order flow were sampled once per second, and every
   estimate in the paper was computed on that one-second grid.
3. **Tandem-trading tests.** Time was cut into bars; in each bar, each market was scored as
   net buying or net selling. The paper counted how often the two markets pressed in the same
   direction and compared that count to a *coin-flip world*: each market buying or selling
   with 50/50 odds, independently of the other. It then re-used the same benchmark rate
   (about 0.4%) when it re-cut the data at finer time scales, including "action time" (bars
   that advance one order at a time).
4. **What moves the correlation.** The correlation between the two markets' returns was
   measured through a 100-second *rolling window* (each second, recompute the correlation of
   the last 100 seconds), and a statistical model (a VAR — a regression of current values on
   many past values) asked which order-book variables move that correlation. The number of
   past values ("lags") was chosen by an automatic criterion.
5. **The liquidity mechanism.** The paper's story — arbitrage weakens when the order book
   thins, letting prices drift apart — was tested on the one-second grid, largely by
   splitting each day into high-liquidity and low-liquidity buckets and comparing.
6. **Statistical confidence.** Significance was computed treating every second as an
   independent observation, and the direction-of-causality questions were resolved either by
   assuming an ordering ("futures move first within the second") or by a
   variance-regime-shift technique (Rigobon identification).

## 4. The weaknesses in that approach

Each weakness below is numbered to match the pillar above.

1. **Halted markets were counted as trading.** When trading halts for 15 minutes and reopens
   at a very different price, a dataset that keeps the halt turns the entire gap into one
   giant one-second "return" — like a heart-rate monitor recording a terrifying spike because
   it was unplugged and plugged back in. On circuit-breaker days, that one fake observation
   can dominate a whole day's statistics — in exactly the sessions the paper cares most
   about.
2. **One second is too slow to see who moves first.** These markets react to each other in
   milliseconds. Sampled once per second, nearly all of the adjustment between SPY and ES
   happens "within the same tick" — like photographing a hummingbird with a one-second
   exposure and asking which wing moved first. The standard leadership score (the Hasbrouck
   information share) senses this: rather than a number, it returns bounds, and at one second
   the bounds on a typical day are [0.004, 0.861] — a statistical shrug. Any point estimate
   quoted inside that range is really the *assumed ordering* talking, not the data.
3. **The coin-flip benchmark is too easy to beat.** The coin-flip world bundles two separate
   claims: each market is a fair coin, *and* the two are independent. On a crash day, both
   markets are mostly selling — each "coin" is heavily biased — so the two agree constantly
   even with no cross-market link at all. Rejecting the coin-flip null therefore does not
   demonstrate tandem trading; a one-sided day rejects it on its own. Worse, the benchmark
   rate is not portable across time scales: the chance that both markets are active and
   agree in a bar depends mechanically on how many orders land in that bar. Comparing a
   48% observed rate at action time against a 0.4% per-second benchmark is comparing
   apples to a yardstick built for oranges.
4. **A rolling window manufactures its own dynamics.** A 100-second moving average remembers
   every event for exactly 100 seconds. Feed that into a model that hunts for time patterns,
   and it will faithfully "discover" 100-second dynamics — an echo we ourselves added, like
   measuring a room's acoustics with a microphone that has its own reverb. The automatic
   lag-selection criterion then chases that echo (it always picks the maximum allowed), and —
   as we demonstrate directly in Section 6.4 — the headline coefficients change wholesale
   when the lag setting changes. Results that depend on a knob setting are measuring the
   knob.
5. **The mechanism was tested at the wrong speed, with a fragile design.** Arbitrage capital
   responds to a thinning book within seconds; averaged to one-second resolution the effect
   is faint, and the day-by-day liquidity-bucket splits flip sign from one day to the next.
6. **The stars were too generous, and one identification tool had no grip.** Treating 23,400
   seconds from the same trading day as independent observations is like polling one
   household a thousand times and reporting a thousand-person survey: the effective sample is
   24 *days*, not millions of seconds. This is how the paper's flagship regression got a
   t-statistic of 253. Separately, the Rigobon technique requires the volatile regime to
   change the two markets' variances by *different proportions* — and on this sample stress
   scales both markets almost identically, like trying to tell which speaker is louder when
   someone turned up the master volume. The technique has nothing to grab.

## 5. The improvements, and why they are better

1. **Halt masking, everywhere.** Every snapshot inside a halt window, and the messy reopening
   seam after it, is excluded from every estimator — no exceptions, enforced by automated
   tests. *Why better:* circuit-breaker-day estimates now describe actual trading, not the
   unplugged monitor.
2. **Two measurement speeds, one data pull.** Everything is now measured twice: on the
   original one-second grid and on a 10-millisecond grid, with the one-second data *derived
   from the same 10 ms pull*. At 10 ms the markets' reactions are genuinely spread over time,
   so "who moves first" becomes measurable rather than assumed — the leadership bounds
   tighten from [0.004, 0.861] to [0.244, 0.523] on the same day. *Why better:* the fine
   grid answers the question the one-second grid could only assume away; and because both
   grids come from one pull, any difference between them is a difference of resolution,
   never a difference of data. (The ES side was also rebuilt from the exchange's own order
   ladder, cross-checked by independently replaying the message feed.)
3. **Two corrected benchmarks for tandem trading.** First, an independence benchmark that
   *keeps each market's actual buy/sell mix* and only randomizes the link between them — so
   beating it demonstrates cross-market coordination, not just a one-sided day. Second, each
   time scale is judged against its own coincidence rate, computed from the actual number of
   orders per bar at that scale. *Why better:* the first isolates exactly the paper's claim;
   the second makes cross-frequency comparisons meaningful.
4. **Window-free correlation measures.** The correlation dynamics are re-estimated on
   dependent variables that have no rolling window to echo: *RealBar* (correlation computed
   fresh on non-overlapping 60-second bars) as the headline, a conditional-correlation model
   (DCC) as corroboration, and the Hayashi–Yoshida estimator — which is immune to the
   sampling distortion called the Epps effect — to measure how much of the published design
   was measurement artifact. We also ran the old design at two lag settings (15 and 60) as a
   controlled experiment. *Why better:* what survives on the window-free measures is market
   behavior; what changes with the knob was never real.
5. **The mechanism estimated where it operates.** The arbitrage-weakening story is now
   estimated at 10 ms, in a single pooled model across all days (with each day keeping its
   own baseline) in which the strength of error correction is allowed to depend continuously
   on the state of the order book. *Why better:* it measures the mechanism at the speed it
   actually works, and replaces twenty fragile day-by-day splits with one powerful test.
6. **Honest statistics.** Confidence is now computed by clustering at the day level (24
   days = 24 independent observations), by bootstrap methods built for few clusters, and by
   permutation tests — shuffle the "volatile"/"calm" labels across days thousands of times
   and ask how often chance beats the real gap. Tables with many cells get a joint
   correction so that one star in twenty isn't there by luck. And instead of pretending the
   Rigobon technique works here, we test its precondition, report that it fails, and give
   the honest bracket from the two possible orderings. *Why better:* every star that
   survives now means something, and a referee cannot take the headline results down by
   attacking the error bars.

## 6. The results

### 6.1 Who sets the price: futures leadership is a stress phenomenon

Two standard scores summarize who is doing the price-setting. The *information share* (IS)
asks: of the news that permanently moves the common price, what fraction shows up in ES
first? The *component share* (CS) asks: when the two prices drift apart, whose price does the
pair converge back toward? Both run from 0 to 1; above 0.5 means futures lead.

**Exhibit 2. Full-panel price-discovery shares by sampling frequency (24 sessions).**

| | 1 s | 10 ms |
|---|---|---|
| mean IS_ES (midpoint of bounds) | 0.484 | 0.542 |
| mean CS_ES | 0.384 | 0.455 |

*How to read it: at one second, the two markets are essentially at parity (SPY slightly ahead
on CS); at 10 ms, ES has a modest lead. Nowhere is there the published-era dominance.*

The picture changes when the days are split by regime:

**Exhibit 3. ES component share by regime.**

| CS_ES | benchmark | volatile | difference | permutation p |
|---|---|---|---|---|
| 1 s | 0.269 | 0.466 | +0.197 | 0.048 |
| 10 ms | 0.387 | 0.504 | +0.117 | 0.047 |

*How to read it: on calm days SPY leads; on volatile days the futures' share of price-setting
jumps by 12–20 points, at both measurement speeds. The "permutation p" answers "if we
shuffled the volatile/calm labels at random, how often would chance produce a gap this big?"
— about 5% — so the pattern is unlikely to be luck, even with only 24 days.*

The sharpest version of this result comes from the biggest moves. At 10 ms we can classify
each price change as ordinary wiggle or as a *jump* (a move too large and fast for the
prevailing volatility). ES's information share is higher in the jump component than in the
ordinary component (0.56 vs 0.39), and on 2020-03-09 — the first circuit-breaker day — when
both markets jumped together, ES moved first 7,610 times and SPY first 1,669 times: a 4.6-to-1
ratio, with the two markets' jumps agreeing in direction 94.8% of the time. This exhibit needs
no model assumptions at all: it is simply counting who moved first.

**Bottom line:** the paper's headline should change from "futures dominate price discovery" to
"futures *take over* price discovery under stress." That is a sharper claim, it is exactly the
paper's mechanism at work, and — unlike unconditional dominance — it is what the data show.

### 6.2 Tandem trading: real, rising with stress — and one headline number was the yardstick's fault

Against the corrected benchmark that keeps each market's actual buy/sell mix, coordinated
trading is still clearly present, and it strengthens with stress:

**Exhibit 4. Tandem order flow against the corrected independence benchmark (revised Table 5).**

| Panel | observed / benchmark | log-odds ratio | corner asymmetry |
|---|---|---|---|
| A. Baseline | 1.27 | 1.098 | −0.014 |
| B. Volatile | 1.34 | 1.385 | −0.008 |
| C. MWCB | 1.37 | 1.470 | −0.019 |
| C′. MWCB ex-restricted day | 1.36 | 1.413 | −0.021 |

*How to read it: "observed / benchmark" is how much more often the two markets press the same
way than independence predicts, given each market's own mix that day. The log-odds ratio is a
standard strength-of-association measure — the key fact is that it climbs steadily from calm
days to volatile days to circuit-breaker days. The near-zero "corner asymmetry" says buying
coordination and selling coordination are equally strong — so this is genuine two-way tandem
trading, not a side effect of short-selling rules.*

An even more direct measurement: after stripping each market's order flow of what is
predictable from its own past, the *surprises* in SPY flow and ES flow are correlated 0.74 on
calm days and 0.83 on volatile days. That is the paper's title phenomenon measured directly —
common trading pressure hitting both markets at once, rising under stress.

The correction cuts the other way at fine time scales. Judged against its own coincidence
rate, each aggregation tells a different story:

**Exhibit 5. Same-direction trading against the frequency-matched benchmark (revised Table 7).**

| Aggregation | observed | benchmark at actual counts | ratio |
|---|---|---|---|
| 1 second | 11.9% | 0.35% | 33.5 |
| 10 ms | 30.1% | 22.9% | 1.32 |
| action time | 48.4% | 50.1% | 0.97 |

*How to read it: at one second, coordinated trading is 33 times its chance rate —
overwhelming. At 10 ms it is 1.3 times chance. At action time it is exactly at chance. The
published table compared all three observed rates against the same 0.4% benchmark, which made
the action-time row look like the strongest evidence in the paper; correctly benchmarked, it
is no evidence at all.*

**Bottom line:** tandem trading is a one-second-scale phenomenon — the markets coordinate
over hundreds of milliseconds, not order by order. Table 5 survives its correction and gets a
cleaner story (dependence rising monotonically with stress); Table 7's cross-frequency
comparison should be replaced with per-frequency ratios.

### 6.3 The liquidity mechanism: invisible at 1 s, unmistakable at 10 ms

Think of arbitrage as a rubber band keeping SPY and ES prices tied together. The paper's
mechanism says: when the order book thins, the band slackens. Our test interacts the strength
of the "snap-back" (error correction) with the state of the order book, pooled across all
days:

**Exhibit 6. The state-dependent error-correction test across frequencies.**

| | 1 s | 10 ms |
|---|---|---|
| t-statistic (mid price) | 0.25 | 5.24 |
| t-statistic (microprice) | 1.35 | 11.70 |
| price-gap half-life on stressed books | — | ~3 min |

*How to read it: a t-statistic near zero means "no detectable effect"; above ~2 means real. At
one second the mechanism is undetectable. At 10 ms it is one of the strongest effects in the
whole study: when books thin, the snap-back visibly weakens, and a price gap that would
normally close in seconds takes about three minutes to close halfway.*

Two corroborating facts. First, *cross-impact*: order flow in ES moves SPY's price two to
five times more strongly than SPY's flow moves ES on the circuit-breaker days — the causal
arrow points from futures to ETF exactly when it matters. Second, the order-book state
predicts SPY's volatility far better than the usual liquidity measure, the quoted spread
(explaining 32% of the variation versus 14%) — the book-depth variables the paper introduced
are the right conditioning variables.

**Bottom line:** the paper's mechanism is confirmed — but only the fine grid can see it. The
revision should lead with the 10 ms pooled test and drop the fragile day-by-day bucket
splits.

### 6.4 What moves the correlation between the markets — and the experiment that settles the lag question

The last system asks which order-book shocks (spreads widening, depth thinning, order-flow
imbalance, volatility) drive *changes in the correlation* between the two markets' returns.
Here we ran a controlled experiment: the identical model, on identical data, with the lag
setting at 15 and then at 60.

**Exhibit 7. The published (rolling-window) design at two lag settings.** One-unit-shock
responses of the correlation measure, ×100; day-clustered standard errors in parentheses;
stars are jointly corrected (***/**/* = 1/5/10%).

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

*How to read it: compare any cell across the p=15 and p=60 columns. Coefficients shrink up to
tenfold and the pattern of stars scrambles — ten cells are starred in at least one run, and
only two keep their stars in both. Nothing about the market changed between these columns;
only a model setting did. This is the rolling window's echo being absorbed differently at
different lag depths — the smoking gun for the weakness described in Section 4.4.*

**Exhibit 8. The window-free (RealBar) measure at the same two lag settings.**

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

*How to read it: sizes still move with the lag setting (coefficient sizes in this kind of
model are denominated in model-dependent units, at either lag depth), but the* inference *is
stable: the same three cells are significant at both settings, no significant cell changes
sign, and no cell is significant with opposite signs in the two runs.*

**What survives everywhere** — across both lag settings and every measurement design:
volatility shocks raise the markets' correlation (ES volatility is the one shock significant
in every design at both settings); order-book-depth shocks matter, with a sign that depends
on regime (an ES-side book-thinning shock raises correlation on volatile days and lowers it
on calm days); ES order-flow imbalance lowers calm-day correlation; and the microprice
deviation variable does nothing anywhere.

**Bottom line:** the published Table 9's coefficient magnitudes were, to first order, echoes
of the measurement window. The revision should rebuild the table on the window-free measure,
present it as a signs-and-significance exhibit, fix the lag setting in advance, and move the
published-design comparison to an appendix as the demonstration of the artifact.

### 6.5 What the honesty upgrades did to the stars

The flagship regression's t-statistic falls from 253 (treating every second as independent)
to 4.6 with day-clustering — still decisively significant (p = 0.001), now honestly so. The
regime differences in Exhibit 3 survive the permutation test at both frequencies. And the
Rigobon identification technique, tested rather than assumed, turns out to have no grip on
this sample (the stress regime scales both markets' variances nearly equally) — so the
revision should report the assumption-based bracket and the failed precondition, not het-ID
point estimates. Every result quoted in this report carries the corrected inference.

## 7. What the paper should now say — recommendations

1. **Recast the headline**: "futures leadership is a stress phenomenon." Near parity on
   average; a 12–20-point futures takeover in the volatile regime (chance probability ~5%);
   largest at the discontinuities (jump share 0.56; co-jump lead 4.6:1).
2. **Keep Table 5** on the corrected independence benchmark (dependence rising monotonically
   with stress); **replace Table 7's** cross-frequency comparison with per-frequency ratios,
   and let the action-time row say what it now says: at the order-by-order scale,
   coordination is at chance.
3. **Quote leadership shares at 10 ms** (where they are measurements, not assumptions), with
   1 s as the robustness column.
4. **Lead the mechanism section with the pooled 10 ms error-correction test** (t = 5.2 and
   11.7; three-minute stressed half-lives) and the cross-impact asymmetry; drop the
   day-by-day liquidity-bucket splits.
5. **Adopt the honest inference everywhere**: day-clustering, few-cluster bootstrap,
   permutation tests as the headline evidence at 24 days, joint corrections on
   multi-cell tables; report the Rigobon precondition failure and the ordering bracket
   instead of het-ID estimates.
6. **Add the innovation-level tandem correlation** (0.74 calm → 0.83 volatile) as the direct
   measurement of the title phenomenon.
7. **Rebuild Table 9 window-free** (RealBar headline, DCC corroboration), signs and
   significance only, lag setting fixed in advance; move the rolling-window-vs-HY contrast
   to the appendix as the measurement-artifact demonstration.

## 8. Caveats and open items

- **Futures contract rolls.** On three sessions the calendar rule for choosing the futures
  contract month is debatable: 2020-03-18 and 2024-12-18 land mid-roll (the chosen contract
  carries 69.5% and 64.2% of volume), and on 2025-06-13 the rule picked the minority contract
  (13.3% of volume). We must either re-extract that day pinned to the majority contract and
  say so, or keep the rule and report the share — it cannot pass silently.
- **2020-03-16** had short-selling restricted all session; every pooled result is reported
  with and without it.
- **Quote staleness at 10 ms.** At that speed most snapshots show no price change (80% for
  SPY, 89% for ES); all fine-grid second-moment estimates use noise-robust methods, and
  correlation at the fine grid is always computed on bars, never tick by tick.
- **One open validation item.** Our independent replay of the ES message feed disagrees with
  the vendor's order ladder on one day (2024-12-18); until that is diagnosed, the ladder
  validation exhibit stays out of the draft.

## Appendix: glossary

- **SPY / ES.** The S&P 500 ETF (trades on stock exchanges) and the E-mini S&P 500 futures
  contract (trades on CME). Same index, two markets.
- **Order book / ladder.** The standing list of buy orders (bids) and sell orders (asks) at
  each price, with sizes. "Depth" is how much size is resting there.
- **Grid / frame.** A frame is one snapshot of both order books. The fine grid snapshots
  every 10 ms; the coarse grid (1 s) is built from the same snapshots.
- **Mid / microprice.** Mid = halfway between best bid and best ask. The microprice is a
  depth-weighted version that leans toward where the book imbalance says price is headed.
- **Spread / WtdSpread.** The quoted spread is the gap between best bid and best ask. The
  weighted spread is the *effective* round-trip cost of actually executing a realistic size,
  walking down the ladder — it sees book thinning that the quoted spread misses.
- **OFI (order-flow imbalance).** Net buying-vs-selling pressure read off order-book changes:
  depth arriving on the bid side counts positive, on the ask side negative.
- **RV (realized variance).** A running measure of how violently the price has been moving.
- **Price discovery.** Which market impounds new information into the price first.
- **IS / CS (information share / component share).** The two standard 0-to-1 scores of price
  discovery leadership; above 0.5 means the futures lead. IS is reported as bounds because
  it depends on an ordering assumption; CS does not.
- **Error correction / VECM.** The statistical model of two prices tied together: when they
  drift apart, the model measures how fast each snaps back. The "state-dependent" version
  (ECM-SDE) lets the snap-back strength depend on the order book's condition.
- **VAR / lags.** A regression of current values on past values; the "lag" setting is how far
  back the model looks.
- **Rolling window.** Recomputing a statistic each period over the last W periods; smooths,
  but also smears every event over exactly W periods.
- **RealBar.** Correlation computed fresh on non-overlapping 60-second bars — no rolling
  window, so no smearing.
- **Hayashi–Yoshida (HY).** A correlation estimator built for asynchronous data; immune to
  the *Epps effect*, the mechanical fade of measured correlation at very fine sampling.
- **DCC.** A model-based conditional correlation (GARCH family); window-free corroboration.
- **Lee–Mykland.** A statistical test that classifies each price move as ordinary or as a
  jump, relative to prevailing local volatility.
- **Rigobon identification.** A technique that uses volatility-regime shifts to establish
  causal direction; requires the regimes to change the two markets' variances by different
  proportions (testable — fails here).
- **Cholesky ordering / bracket.** The assumption that one market moves first within the
  sampling interval; computing results under both orderings gives an honest bracket.
- **Day-clustering / wild-cluster bootstrap / Webb weights.** Ways of computing error bars
  that treat each *day* (not each second) as one independent observation, built to work with
  as few as 4–24 clusters.
- **Permutation test.** Shuffle the group labels (volatile/calm) across days thousands of
  times; the p-value is the fraction of shuffles that beat the real gap.
- **Romano–Wolf correction.** Adjusts significance stars when a table has many cells, so a
  few stars can't appear by luck alone.
- **Action time.** Bars that advance one order arrival at a time instead of by the clock.
- **MWCB.** Market-wide circuit breaker: at a 7% S&P 500 decline, all trading halts for 15
  minutes.
- **GFEVD.** A variance decomposition that respects the measured correlation between shocks
  instead of assuming them uncorrelated.

*Open items: the ESH5 2024-12-18 validation diagnosis, and the 2025-06-13 contract-month
decision.*
