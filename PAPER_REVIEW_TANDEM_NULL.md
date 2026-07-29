# Validating the code stack against the paper

Scope of this pass: the order-flow half of the paper (§4, Tables 5–7) plus a health
sweep of the module self-tests. The finding below is the one that changes what the
paper can claim. It is derived entirely from the paper's **own published Table 5.II**,
so it can be checked without re-running anything.

---

## Finding: the Table 5 null is rejected by the marginals, not by cross-market trading

### What the null actually tests

Table 5.I Panel A benchmarks the observed corner cells against a Binomial(*n*, ½)
independence null, *n* = 505 new orders/second for the ETF and 112 for the future
(footnote 11). That null bundles **two** hypotheses:

1. each market's own order flow is a fair coin across *n* orders, **and**
2. the two markets are independent of one another.

Only (2) is tandem trading. Rejecting the joint null does not tell you which half failed —
and on this data it is (1) that fails, overwhelmingly.

### The paper's own marginals show it

From Table 5.II, the share of one-second bars in which each market is *directional*
(Sell or Buy), against what the binomial null predicts:

| Panel | ETF observed | ETF null | Future observed | Future null |
|---|---|---|---|---|
| A baseline | **68.7%** | 2.4% | **83.2%** | 29.0% |
| B volatile | **64.8%** | 2.4% | **80.2%** | 29.0% |
| C MWCB | **42.2%** | 2.4% | **69.5%** | 29.0% |

The ETF is directional roughly **28× more often** than a fair coin over 505 orders would
be. This is a *within-market* fact — order flow is clustered and autocorrelated (order
splitting, queue jockeying, momentum) — and it inflates **all four** corner cells
mechanically, with zero cross-market linkage required.

### Re-benchmarked against independence *given the observed marginals*

Holding each market's own directionality fixed and asking only whether the two line up
more often than chance:

| Panel | PCMOF obs | PCMOF indep | ratio | NCMOF obs | NCMOF indep | ratio | log OR |
|---|---|---|---|---|---|---|---|
| A baseline | 43.1% | 28.6% | **1.51×** | 15.7% | 28.6% | **0.55×** | +2.02 |
| B volatile | 46.5% | 26.0% | **1.79×** | 8.4% | 26.0% | **0.32×** | +3.45 |
| C MWCB | 21.1% | 14.5% | **1.46×** | 9.1% | 14.9% | **0.61×** | +1.82 |

Two consequences:

**PCMOF survives, at a much smaller magnitude.** Positive tandem trading is real — the
corner log odds ratio is strongly positive in every panel — but the effect is ~1.5×
independence, not the ~100× implied by comparing 43% against 0.4%.

**The NCMOF claim reverses.** The paper states (§4): *"Notably, NCMOF levels are much
greater than the predicted values under either form of the null hypothesis."* Against a
correctly specified null, NCMOF is **roughly half** of what independence predicts, in all
three panels. It is a deficit, not an excess.

### This is a demonstrated false positive, not a theoretical worry

`test_tandem_null.py` builds a session with **exactly zero cross-market dependence** and
realistic within-market clustering. Result:

```
(1) NO cross-market dependence, clustered within-market flow
    raw PCMOF=36.2% vs binomial null 0.36% -> looks like 101x tandem trading
    vs independence-given-marginals: PCMOF 0.99x, NCMOF 1.01x (want ~1.00x) : True
```

The paper's null calls a session with no tandem trading whatsoever "101× tandem trading."

### What survives — and it is the more interesting result

The *regime pattern*, which is what H1 and H3 are actually about, survives cleanly and
becomes sharper once the marginals are removed:

| | baseline | volatile | MWCB |
|---|---|---|---|
| PCMOF vs independence | 1.51× | 1.79× | 1.46× |
| NCMOF vs independence | 0.55× | **0.32×** | 0.61× |
| corner log OR | +2.02 | +3.45 | +1.82 |

Positive tandem trading *strengthens* into volatility and *falls back* at the MWCB;
opposing (arbitrage) flow is suppressed relative to chance always, most severely under
volatility, and partially recovers at the MWCB. That is exactly the H1/H3 nonlinearity —
arbitrage is safest in calm markets, riskiest in volatile ones, and re-emerges once
dislocations get large enough — now stated in a form a referee cannot attribute to
within-market clustering.

**Recommendation:** keep the H1/H3 narrative, restate it on the marginal-free scale, and
drop the "NCMOF exceeds the null" sentence. The log odds ratio is the natural headline
statistic: it is invariant to the table's margins, so it is comparable across the three
panels even though their marginals differ substantially (68.7% → 42.2% ETF directional).

### Separately: Table 7's null is the wrong frequency

The binomial null is entirely a function of orders-per-bar, so a per-second calibration
cannot be reused at a finer aggregation:

| Aggregation | orders/bar (ETF, fut) | independence null, NCMOF | paper's observed NCMOF |
|---|---|---|---|
| 1 second | 505, 112 | **0.4%** | 11.9% |
| 10 millisecond | ~5.1, ~1.1 | **22.9%** | 30.1% |
| action time | ~1, ~1 | **~50%** | 48.4% |

Table 7 reports all three rows and §4 describes them as *"much higher than the theoretical
probabilities of independent order flow."* At one second that is right (≈30×). At ten
milliseconds it is ~1.3×, and in action time the observed value is **at or slightly below**
independence. The fine-frequency rows of Table 7 do not support the claim made for them.

(The 10 ms and action-time nulls assume Poisson counts at the paper's own footnote-11 mean
rates. Real counts are over-dispersed, which shifts the null somewhat — which is precisely
why the null should be computed from the actual per-bar counts rather than a fixed *n*.)

---

## Code changes

### `tandem_order_flow.py`

- `independence_given_marginals(freq)` — expected cells under cross-market independence
  holding each market's observed marginal state distribution fixed.
- `dependence_summary(freq, n_bars)` — observed vs independence PCMOF/NCMOF, their ratios,
  and the corner **log odds ratio** with SE and *z*. The marginal-free statement of the
  paper's claim.
- `permutation_null(fut_state, etf_state)` — distribution-free null that shuffles the
  *pairing* between markets, preserving each market's exact marginal distribution
  (including over-dispersion, clustering, intraday seasonality) and destroying only the
  cross-market association. Automatically correct at any aggregation.
- `independence_null_from_counts(n_etf, n_fut)` — the binomial null evaluated at the
  **actual per-bar counts** instead of a fixed *n*; the frequency-matched replacement for
  `theoretical_null` whenever the aggregation is not one second.
- `table5_from_series` now returns `dependence`, `independence_expected` and
  `binomial_null_matched` alongside the existing outputs, so the corrected benchmark
  appears next to every Table 5 it builds.

`theoretical_null` is unchanged so the published Table 5.I still reproduces exactly.

### `test_tandem_null.py` (new)

Five known-answer checks, all passing:

```
(1) zero cross-market dependence + clustering -> binomial null says 101x, corrected null says 0.99x
(2) true dependence (rho=0.7) -> PCMOF 1.65x, NCMOF 0.38x, logOR +2.95 (z=171), detected
(3) same association, marginals re-raked -> raw PCMOF moves 19.7pp, logOR moves 9e-16
(4) independence null by frequency -> 0.34% at 1s, 22.8% at 10ms, 49.9% in action time
(5) permutation null -> p=0.70/0.56 under no dependence, p=0.003 under rho=0.7
```

(3) is exact by construction: iterative proportional fitting changes a table's margins
while leaving every odds ratio algebraically untouched.

---

## Module health sweep (complete: 46 modules)

36 of 46 exit clean. Of the 10 that do not, 6 are environmental and 4 are real. **All were
already failing on the unmodified stack** — none was introduced here.

| Module | Status |
|---|---|
| `autoscale`, `debug_crossing`, `release_template`, `validate_release_csv`, `validate_reconstruction`, `verify_crossing` | argparse CLIs — need arguments, no self-test entry point. Not defects. |
| `market_analysis_fixed` | needs the `maystreet_data` SDK, absent here |
| `paper_tables` | writes to a hard-coded `/mnt/user-data/outputs/`, `FileNotFoundError` anywhere else — the reporting layer cannot be smoke-tested on a normal checkout |
| `mean_variance` | does not finish in 400 s |
| `robust_prices`, `state_space_efficient_price` | **self-check reported False** — see below |

### The meta-defect: a failing self-test that exits 0

`robust_prices.py` printed `checks: False` and **exited 0**. The README's recommended smoke
loop is `for m in ...; do python "$m.py"; done`, which inspects nothing but the exit code —
so a module failing its own checks reads as green. Fixed: `_selftest()` now returns a bool
and `__main__` exits non-zero on failure. (`state_space_efficient_price` already exited 1.)

### `robust_prices` — silent NaN t-statistic (fixed)

The failure was `t(a1_SPY) = nan`. `a1_SPY` is the coefficient on `z·S` — per `ecm_sde`'s
own docstring, *"the single-number test of liquidity-conditioning."*

Cause: `ecm_sde._cluster_se` computes the Liang–Zeger small-cluster correction
`c = G/(G-1)·(N-1)/(N-K)`, and set `c = nan` when `G < 2`. With a single session there is
one cluster, so every SE came back NaN — and a NaN t-stat prints as a blank cell rather
than raising. Same silent-failure shape as the swallowed fetch error in the reconstruction
work.

Fixed: with one cluster, fall back to Newey–West HAC (which is exactly valid for a single
contiguous session) and log a warning naming the fallback. `t(a1_SPY)` now returns 1.35
instead of NaN. The warning also points out that with few clusters — the MWCB sample has
**G = 4** — a wild cluster bootstrap is preferable to cluster-robust SEs; `test_wild_cluster.py`
exists in the stack but is not wired into this path.

### `state_space_efficient_price` — open, not fixed

Self-check (1) asserts the Kalman smoother beats a naive average of the two quotes. It does
not: RMSE 0.366 vs 0.096, though correlation with the truth is 0.9991 and the error is 4%
of `sd(m_true) = 8.57`.

Two things I ruled out: it is not a level offset (removing the +0.003 mean changes RMSE by
nothing), and it is not purely an unfair benchmark. The test's DGP uses λ = [+1, −1], where
the cross-sectional average cancels the transitory component *exactly* and is therefore the
efficient estimator — no filter can beat it, so that assertion is unachievable as written.
But sweeping λ shows the smoother also loses at [1, −0.4] and [1, 0.3], only winning at
[1, 0.9]:

| λ | RMSE smoother | RMSE naive average | smoother wins |
|---|---|---|---|
| [1.0, −1.0] | 0.3662 | 0.0960 | no |
| [1.0, −0.4] | 0.3475 | 0.2922 | no |
| [1.0, 0.3] | 0.6715 | 0.6057 | no |
| [1.0, 0.9] | 0.7367 | 0.8794 | yes |

So the benchmark is mis-specified *and* the estimator underperforms a trivial alternative
over much of the parameter space. I did not chase this further. It needs the authors' eye
before this module is relied on — though note it is an extension module, not one the
published tables appear to depend on.

---

## Not yet examined

Being explicit about coverage, since "the entire stack" is ~55 modules:

- **The price-discovery half (§5, Tables 9–14)** — the SVAR/IRF machinery in
  `correlation_svar.py`, `price_discovery_shares.py`, `ecm_sde.py`, `irf.py` has been read
  but not audited to the depth applied to Table 5. Its self-tests pass.
- **The Epps/asynchronicity question.** The paper computes SPY–ES return correlations at
  10 ms and never mentions non-synchronous trading; the word "Epps" does not appear. At
  that frequency measured correlation is attenuated by asynchronous updating, and the
  attenuation depends on trading intensity — which is itself on the right-hand side of
  Eq. (5). The stack already has the fix (`noise_robust_cov.hayashi_yoshida`, and
  `legacy_tables.py` even contrasts Pearson vs HY), but `correlation_svar.build_svar_frame`
  builds Δρ from a plain Pearson rolling correlation with no HY option. Adding
  `corr_method="hy"` and reporting Table 9 both ways looks like the highest-value next
  step, and it is a natural referee question.
- One hypothesis I tested and **rejected**: that the default moving-block bootstrap length
  (~*n*^⅓) is too short for the overlapping 100-bar rolling correlation. Differencing kills
  the overlap persistence — the bootstrap SE (0.00008) matches the Monte Carlo truth
  (0.00009) and size is correct. No change needed.
