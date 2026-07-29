# The Epps effect in Eq. (5), and the Hayashi–Yoshida dependent variable

The paper's Δρ is the first difference of a **Pearson correlation of returns sampled on a
fixed grid**, at one second *and* at ten milliseconds. Grid-sampled Pearson is exactly the
estimator the Epps effect attacks. The word "Epps" does not appear in the paper, and
neither does any mention of non-synchronous trading.

## Why this is more than a precision issue for this paper

SPY and ES do not update at the same instants. Within a short bar one leg's mid is often
stale, the grid records a zero return for it, and measured correlation is pulled toward
zero. That much is standard.

The problem specific to Eq. (5) is the **direction** of the bias. How badly the correlation
is attenuated depends on how often each leg updates — i.e. on trading intensity. And
volume, message traffic and liquidity demand are all **regressors in the same system**. So
a measured "activity raises correlation" relationship can be the measurement error moving
with the regressor rather than a market mechanism, and the naive estimator cannot separate
the two.

Table 9's statement that *"changes in volume are consistently significant and positive in
their influence on correlation"* is the coefficient most exposed to this.

## How large is it? Measured, on a DGP where the true correlation never moves

`test_hy_correlation.py` simulates two books whose efficient returns have correlation
**exactly 0.60**, observed with staleness. The latent correlation is constant by
construction, so any movement in a measured correlation is measurement error.

| | Pearson (grid) | Hayashi–Yoshida | truth |
|---|---|---|---|
| every bar refreshes (synchronous) | 0.589 | 0.589 | 0.60 |
| 25% refresh per bar (Epps regime) | **0.072** | **0.580** | 0.60 |

At a 25% refresh rate, grid Pearson loses **88% of the true correlation**. HY recovers it.

Then the check that matters. Let activity cycle between 0.15 and 0.85 while the true
correlation stays pinned at 0.60:

```
corr(measured rho, activity):   Pearson = +0.774     HY = +0.026
```

**Pearson manufactures a strong activity→correlation relationship that is not in the
data-generating process at all.** HY does not. This is the confound, quantified.

## What was added

### `correlation_svar.py`

- **`corr_method="hy"`** in `build_svar_frame`, alongside the existing `"rolling"` (the
  paper) and `"dcc"`. Same window, same variable layout, same causal alignment — only the
  correlation estimator changes, so the two are directly comparable.

- **`_hy_rolling_corr(mid_spy, mid_es, window)`** — rolling HY correlation on the bar grid.
  Design points:
  - Each asset contributes only the bars where **its own mid actually moved**. On a grid,
    a bar in which an asset's mid did not move is precisely a bar in which it did not
    update, and that zero return is the artifact. Using each leg's own change-times
    restores the irregular, asynchronous sampling HY is built for.
  - Normalised as `HY_cov / sqrt(RV_SPY · RV_ES)` with each leg's own tick-by-tick realized
    variance over the same span.
  - Each interval-pair product is attributed to the **later** of the two interval ends, so
    the window ending at *t* uses nothing dated after *t*. Test (5) verifies this directly:
    perturbing bars after *t* leaves ρ_HY(≤t) bit-identical.
  - O(n + m + T) via cumulative sums, not O(T · window) — the same two-pointer sweep as
    `noise_robust_cov.hayashi_yoshida`, with per-bar attribution then a rolling cumsum.
  - One honest caveat, documented in the docstring: this is a realized correlation (no
    demeaning) whereas `rolling().corr()` demeans. Over 100 bars of near-zero-mean returns
    that difference is negligible next to the Epps correction.

- **`epps_comparison(data, methods=("rolling","hy"), **kw)`** — Table 9 computed both ways
  side by side, with `delta_<regime>` columns giving (hy − rolling). The delta is an
  estimate of how much of each response is measurement artifact. Coefficients that keep
  their sign, size and significance across both columns are safe to read as mechanism;
  ones that shrink toward zero under `hy` were partly the Epps channel.

### `test_hy_correlation.py` (new)

Six known-answer checks, all passing:

```
(1) synchronous -> Pearson 0.589, HY 0.589 (truth 0.60): HY changes nothing when nothing is wrong
(2) 25% refresh -> Pearson 0.072 attenuated, HY 0.580 recovered
(3) activity cycles, true rho constant -> corr(rho, activity): Pearson +0.774, HY +0.026
(4) build_svar_frame(corr_method='hy') -> same layout, different dependent series, all finite
(5) perturbing future bars leaves rho_HY(<=t) unchanged -> causal, no look-ahead
(6) epps_comparison max|hy - rolling|: 0.190 on stale books, 0.005 on synchronous books
```

(1) and (6) are the guards against over-correcting: the estimator must be inert where there
is no asynchronicity, and it is.

## How to use it

```python
import correlation_svar as cs

# Table 9 both ways, with the artifact estimate in the delta columns
tbl = cs.epps_comparison(sessions, spec="informational", n_lags=6, corr_window=100)

# or with bootstrap SEs and Romano-Wolf stars, one method at a time
res_pearson = cs.correlation_irf_inference(sessions, corr_method="rolling", n_boot=499)
res_hy      = cs.correlation_irf_inference(sessions, corr_method="hy",      n_boot=499)
```

The 10 ms specification is where the gap should be largest, since attenuation grows as the
bar shrinks. Running `epps_comparison` at both frequencies and reporting the pair would
close the issue pre-emptively rather than in a referee round.

## Verification

`correlation_svar.py` self-test passes; `legacy_tables.py`, `test_xi_symmetry.py`,
`test_tandem_null.py`, `test_crossed_root_cause.py` all pass. `theoretical_null`,
`corr_method="rolling"` and `corr_method="dcc"` are untouched, so every previously
published number reproduces exactly — `"hy"` is purely additive.
