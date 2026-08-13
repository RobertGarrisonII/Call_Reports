"""test_hy_correlation.py -- the Epps-robust dependent variable for Eq. (5).

Delta-rho in Eq. (5) is the first difference of a Pearson correlation of returns sampled on a
fixed grid, at one second and at ten milliseconds. Grid-sampled Pearson is precisely the
estimator the Epps effect attacks: SPY and ES do not update at the same instants, so in a short
bar one leg's mid is often stale, the grid records a zero return for it, and measured correlation
is pulled toward zero.

The reason this is more than a precision issue for THIS paper is the direction of the bias. How
badly the correlation is attenuated depends on how often each leg updates -- i.e. on trading
intensity. Volume, message traffic and liquidity demand are all regressors in the same system.
So a measured "activity raises correlation" relationship can be the measurement error moving
with the regressor rather than a market mechanism, and the naive estimator cannot tell the two
apart.

Hayashi-Yoshida is consistent for the covariance of asynchronously observed processes and imposes
no grid, so it breaks that link. These checks hold the TRUE correlation exactly constant and vary
only how often each leg updates:

  1. synchronous data           -> HY agrees with Pearson (it changes nothing when nothing is wrong)
  2. stale, asynchronous data   -> Pearson is attenuated toward zero; HY recovers the truth
  3. activity varies over time  -> Pearson tracks activity (spurious); HY does not
  4. the SVAR frame builds and the two dependent variables genuinely differ
"""
import sys
import warnings

import numpy as np
import pandas as pd

import correlation_svar as cs

TRUE_RHO = 0.6
NY = "America/New_York"


def _frame(n=20000, rho=TRUE_RHO, refresh=1.0, seed=0, activity=None):
    """Two books whose EFFICIENT returns have correlation `rho` exactly, observed with staleness.

    Each bar, each leg refreshes its quote with probability `refresh` (or `activity[t]` when
    given); an un-refreshed leg carries the previous mid forward, which is exactly the stale
    quote that injects a spurious zero return on the grid. The latent correlation never changes,
    so ANY movement in a measured correlation here is measurement error."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    e = rng.standard_normal((n, 2)) @ L.T                       # efficient returns, corr = rho
    lp = np.cumsum(e * 1e-4, axis=0)
    p = np.array([500.0, 5000.0]) * np.exp(lp)                  # efficient prices
    pr = np.asarray(activity, float) if activity is not None else np.full(n, float(refresh))
    obs = np.empty_like(p)
    for k in range(2):                                          # independent refresh per leg
        hit = rng.random(n) < pr
        hit[0] = True
        idx = np.maximum.accumulate(np.where(hit, np.arange(n), 0))
        obs[:, k] = p[idx, k]                                   # previous-tick (stale) observation
    cols = {}
    for a, j, tick in (("SPY", 0, 0.01), ("ES", 1, 0.25)):
        for l in range(1, 11):
            cols[f"{a}_bidprice_{l}"] = obs[:, j] - l * tick
            cols[f"{a}_askprice_{l}"] = obs[:, j] + l * tick
            cols[f"{a}_bidquantity_{l}"] = np.full(n, 200.0)
            cols[f"{a}_askquantity_{l}"] = np.full(n, 200.0)
    idx = pd.date_range("2024-08-05 09:30", periods=n, freq="s", tz=NY)
    return pd.DataFrame(cols, index=idx), pr


def _pearson_roll(df, w):
    r_s = cs._returns_bps(df, "SPY"); r_e = cs._returns_bps(df, "ES")
    return pd.Series(r_s).rolling(w).corr(pd.Series(r_e)).to_numpy()


def main():
    warnings.simplefilter("ignore")
    ok = True
    W = 100

    # ── 1. fully synchronous: HY must not change a correct answer ────────────────────
    df, _ = _frame(refresh=1.0, seed=1)
    pe = np.nanmean(_pearson_roll(df, W))
    hy = np.nanmean(cs._hy_rolling_corr(df["SPY_bidprice_1"] + 0.01, df["ES_bidprice_1"] + 0.25, W))
    a_ok = abs(pe - TRUE_RHO) < 0.05 and abs(hy - TRUE_RHO) < 0.05
    print("(1) synchronous (every bar refreshes), true rho=%.2f" % TRUE_RHO)
    print("    Pearson=%.3f   HY=%.3f   -> both correct : %s" % (pe, hy, a_ok))
    ok &= a_ok

    # ── 2. stale / asynchronous: Pearson attenuates, HY does not ────────────────────
    df, _ = _frame(refresh=0.25, seed=2)
    ms = df["SPY_bidprice_1"].to_numpy() + 0.01; me = df["ES_bidprice_1"].to_numpy() + 0.25
    pe = np.nanmean(_pearson_roll(df, W)); hy = np.nanmean(cs._hy_rolling_corr(ms, me, W))
    b_ok = pe < TRUE_RHO - 0.10 and abs(hy - TRUE_RHO) < 0.10
    print("(2) 25%% refresh per bar (Epps regime), true rho=%.2f" % TRUE_RHO)
    print("    Pearson=%.3f (attenuated)   HY=%.3f (recovered) : %s" % (pe, hy, b_ok))
    ok &= b_ok

    # ── 3. THE one that matters: activity varies, true rho does not ──────────────────
    n = 30000
    t = np.arange(n)
    act = 0.15 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 5000))       # slow activity cycle
    df, pr = _frame(n=n, refresh=None, seed=3, activity=act)
    ms = df["SPY_bidprice_1"].to_numpy() + 0.01; me = df["ES_bidprice_1"].to_numpy() + 0.25
    rho_pe = _pearson_roll(df, W); rho_hy = cs._hy_rolling_corr(ms, me, W)
    act_w = pd.Series(pr).rolling(W).mean().to_numpy()                   # activity over the window
    m = np.isfinite(rho_pe) & np.isfinite(rho_hy) & np.isfinite(act_w)
    c_pe = float(np.corrcoef(rho_pe[m], act_w[m])[0, 1])
    c_hy = float(np.corrcoef(rho_hy[m], act_w[m])[0, 1])
    c_ok = c_pe > 0.5 and abs(c_hy) < 0.5 * c_pe
    print("(3) activity cycles 0.15..0.85, TRUE rho constant at %.2f" % TRUE_RHO)
    print("    corr(measured rho, activity):  Pearson=%+.3f   HY=%+.3f" % (c_pe, c_hy))
    print("    Pearson manufactures an activity->correlation link that is not in the DGP : %s" % c_ok)
    ok &= c_ok

    # ── 4. the SVAR frame accepts corr_method='hy' and it is a different variable ────
    df2 = cs._demo_frame(n=4000, seed=7)
    Xr, nr, ir = cs.build_svar_frame(df2, spec="standard", corr_method="rolling", corr_window=W)
    Xh, nh, ih = cs.build_svar_frame(df2, spec="standard", corr_method="hy", corr_window=W)
    same_shape = (nr == nh) and (ir == ih)
    dr, dh = Xr[:, ir], Xh[:, ih]
    differs = np.nanmean(np.abs(dr[:min(len(dr), len(dh))] - dh[:min(len(dr), len(dh))])) > 0
    finite = np.all(np.isfinite(Xh))
    d_ok = same_shape and differs and finite and len(Xh) > 100
    print("(4) build_svar_frame(corr_method='hy'): rows=%d, dCorr column=%d, all finite=%s"
          % (len(Xh), ih, finite))
    print("    same variable layout as 'rolling', different dependent series : %s" % d_ok)
    ok &= d_ok

    # ── 5. no look-ahead: rho_HY(t) must not move when FUTURE bars change ───────────
    ms2 = ms.copy(); me2 = me.copy()
    cut = 20000
    ms2[cut:] *= 1.05; me2[cut:] *= 0.95                        # perturb only the future
    a = cs._hy_rolling_corr(ms, me, W); b = cs._hy_rolling_corr(ms2, me2, W)
    head = slice(W, cut)
    e_ok = np.allclose(a[head], b[head], equal_nan=True)
    print("(5) perturbing bars after t leaves rho_HY(<=t) unchanged (causal) : %s" % e_ok)
    ok &= e_ok

    # ── 6. epps_comparison surfaces the gap on stale data, and none when synchronous ──
    stale = [("d%d" % i, "benchmark", _frame(n=4000, refresh=0.25, seed=20 + i)[0]) for i in range(4)]
    sync = [("d%d" % i, "benchmark", _frame(n=4000, refresh=1.0, seed=30 + i)[0]) for i in range(4)]
    ts = cs.epps_comparison(stale, spec="standard", n_lags=3, corr_window=W)
    tq = cs.epps_comparison(sync, spec="standard", n_lags=3, corr_window=W)
    d_stale = float(np.nanmax(np.abs(ts["delta"].to_numpy())))
    d_sync = float(np.nanmax(np.abs(tq["delta"].to_numpy())))
    f_ok = d_stale > d_sync and np.isfinite(d_stale)
    print("(6) epps_comparison max|hy - rolling|: stale books=%.4f  synchronous books=%.4f"
          % (d_stale, d_sync))
    print("    the gap appears only where asynchronicity exists : %s" % f_ok)
    ok &= f_ok

    print("\nhy-correlation checks ->", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
