"""
microstructure_diagnostics.py
=============================
Diagnostics that tell you WHEN the information-share section stops being
trustworthy as you push to 100ms / 10ms. Two failure modes the Gaussian VECM
is silent about:

  * PRICE STALENESS - the less-frequently-updating series looks like it "doesn't
    adjust", which mechanically inflates its Gonzalo-Granger component share. The
    zero-return fraction per series is the direct gauge (ES updates more often
    than SPY, so this bias has a known direction).
  * TICK DISCRETENESS - when price changes are mostly 0 / +-1 tick, the Gaussian
    VECM/Hasbrouck approximation degrades. The share of moves at <=1 tick and the
    mean absolute move in ticks measure the severity.

Plus the BNHLS noise-to-signal ratio xi^2 ~ omega^2 / IV as a microstructure-
noise severity measure, and a frequency sweep that turns all three into the
"this is the grid below which the shares are contaminated" exhibit.

Pairs with noise_robust_cov.py (used for the noise-robust IV in xi^2). Attach the
report to the information-share tables; treat a rising IS_ES at high frequency as
suggestive only until it survives these checks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import noise_robust_cov as nrc

EPS = 1e-15
TICK = {"SPY": 0.01, "ES": 0.25}


def _mid(df, asset):
    return ((df[f"{asset}_bidprice_1"] + df[f"{asset}_askprice_1"]) / 2.0).to_numpy()


# ── staleness ────────────────────────────────────────────────────────────────
def zero_return_fraction(logp):
    """Fraction of steps with no price change -> the price-staleness gauge."""
    r = np.diff(np.asarray(logp, float))
    return float(np.mean(r == 0.0))


# ── tick discreteness ────────────────────────────────────────────────────────
def tick_discreteness(price, tick_size):
    """How concentrated price changes are at the tick grid. High frac_le_1tick and
    small mean_abs_ticks_nz => the Gaussian VECM approximation is strained."""
    d = np.diff(np.asarray(price, float))
    d = d[np.isfinite(d)]                                    # drop NaN gaps (e.g. unaligned cross-asset rows)
    if d.size == 0:
        return {"frac_zero": np.nan, "frac_le_1tick": np.nan,
                "mean_abs_ticks": np.nan, "mean_abs_ticks_nz": np.nan}
    ticks = np.abs(np.round(d / tick_size))                 # keep float: an int cast overflows on a fat-finger tick
    nz = d != 0.0
    return {"frac_zero": float(np.mean(d == 0.0)),
            "frac_le_1tick": float(np.mean(ticks <= 1)),
            "mean_abs_ticks": float(np.mean(ticks)),
            "mean_abs_ticks_nz": float(np.mean(ticks[nz])) if nz.any() else np.nan}


# ── microstructure-noise severity ────────────────────────────────────────────
def noise_to_signal(logp):
    """BNHLS noise ratio xi^2 ~ omega^2 / IV (omega^2 from RV/(2n), IV from TSRV)."""
    logp = np.asarray(logp, float)
    logp = logp[np.isfinite(logp)]                          # drop NaN gaps before differencing
    r = np.diff(logp)
    if r.size == 0:
        return np.nan
    om2 = np.sum(r ** 2) / (2.0 * max(len(r), 1))
    iv = nrc.realized_variance(logp, "tsrv") if len(logp) > 50 else np.sum(r ** 2)
    return float(om2 / (iv + EPS))


# ── per-session report ───────────────────────────────────────────────────────
def staleness_report(df, assets=("SPY", "ES"), tick_sizes=None):
    """Per-asset staleness / discreteness / noise table for the current grid."""
    tick_sizes = tick_sizes or TICK
    rows = []
    for a in assets:
        mid = _mid(df, a); lp = np.log(mid)
        td = tick_discreteness(mid, tick_sizes[a])
        rows.append({"asset": a, "zero_return_frac": zero_return_fraction(lp),
                     "frac_le_1tick": td["frac_le_1tick"],
                     "mean_abs_ticks_nz": td["mean_abs_ticks_nz"],
                     "noise_to_signal": noise_to_signal(lp)})
    out = pd.DataFrame(rows)
    out.attrs["note"] = ("high zero_return_frac -> staleness -> CS biased toward the "
                         "staler (less-updating) asset; high frac_le_1tick -> discreteness")
    return out


# ── frequency sweep ──────────────────────────────────────────────────────────
def frequency_diagnostics(t, price, tick_size, intervals, omega2=None):
    """Staleness / discreteness / noise as a function of sampling interval. The
    grid below which these spike is the grid below which the shares are unsafe.
    `t` in seconds (irregular ok), `price` the level series, `intervals` in seconds.
    The microstructure-noise variance omega^2 is estimated ONCE at the finest
    resolution (where RV/(2n) is valid); the per-interval column is then the noise
    SHARE of naive RV, ~2*n*omega^2 / RV, which rises as dt -> 0."""
    t = np.asarray(t, float); price = np.asarray(price, float)
    if omega2 is None:
        lr = np.diff(np.log(price)); omega2 = np.sum(lr ** 2) / (2.0 * max(len(lr), 1))
    rows = []
    for dt in intervals:
        grid = np.arange(t[0], t[-1] + dt, dt)
        p = nrc._previous_tick(t, price, grid); lp = np.log(p)
        r = np.diff(lp); d = np.diff(p)
        ticks = np.abs(np.round(d / tick_size)).astype(int); nz = d != 0.0
        rv = float(np.sum(r ** 2))
        rows.append({"interval_s": dt, "n": len(r),
                     "zero_return_frac": float(np.mean(r == 0.0)),
                     "frac_le_1tick": float(np.mean(ticks <= 1)),
                     "mean_abs_ticks_nz": float(np.mean(ticks[nz])) if nz.any() else np.nan,
                     "rv_naive": rv,
                     "noise_share_of_rv": float(min(2.0 * len(r) * omega2 / (rv + EPS), 1.0))})
    return pd.DataFrame(rows)


def _selftest():
    rng = np.random.default_rng(21)
    M = 40000; times = np.linspace(0.0, 1.0, M)
    lp = np.cumsum(rng.normal(0, 3e-5, M)); mid_true = 100.0 * np.exp(lp)
    tick = 0.01
    price_disc = np.round((mid_true + rng.normal(0, 2e-3, M)) / tick) * tick  # discrete + noisy

    print("=== staleness / discreteness (native fine grid) ===")
    zr = zero_return_fraction(np.log(np.maximum(price_disc, EPS)))
    td = tick_discreteness(price_disc, tick)
    print("  zero_return_frac = %.3f  (price-staleness gauge)" % zr)
    print("  frac_le_1tick = %.3f   mean_abs_ticks_nz = %.2f  (~1 tick => severe discreteness)"
          % (td["frac_le_1tick"], td["mean_abs_ticks_nz"]))
    print("  checks: staleness present:", zr > 0.3,
          "| moves snap to ~1 tick:", abs(td["mean_abs_ticks_nz"] - 1.0) < 0.3)

    print("\n=== noise-to-signal (noise raises xi^2) ===")
    xi_clean = noise_to_signal(lp)
    xi_noisy = noise_to_signal(np.log(np.maximum(mid_true + rng.normal(0, 5e-3, M), EPS)))
    print("  xi^2 clean=%.2e  noisy=%.2e  (noisy larger): %s" % (xi_clean, xi_noisy, xi_noisy > xi_clean))

    print("\n=== frequency sweep (fine -> coarse) ===")
    fd = frequency_diagnostics(times, price_disc, tick, intervals=[1e-4, 5e-4, 2e-3, 1e-2, 5e-2])
    print(fd.round(4).to_string(index=False))
    fine, coarse = fd.iloc[0], fd.iloc[-1]
    print("  checks: staleness falls as dt grows:", fine.zero_return_frac > coarse.zero_return_frac,
          "| discreteness eases (frac_le_1tick falls):", fine.frac_le_1tick > coarse.frac_le_1tick,
          "| noise share falls:", fine.noise_share_of_rv > coarse.noise_share_of_rv)

if __name__ == "__main__":
    _selftest()
