"""
robustness.py
=============
Referee-grade robustness battery for the cross-asset price-discovery / liquidity
results. Each check re-runs the estimators under a perturbation and returns a
tidy stability table.

  lag_sensitivity            shares across VECM lag lengths
  beta_fixed_vs_estimated    impose beta=(1,-1) vs Engle-Granger estimated beta
  subsample_am_pm            within-session AM vs PM stability
  window_size_sensitivity    panel regression coef across intraday window sizes
  alt_state_variable         liquidity-conditional VECM under alternative book-state
                             proxies (relative depth / slope / entropy)
  stationary_bootstrap_ci    Politis-Romano block-bootstrap CI for small-N means

Requires price_discovery_shares.py, liquidity_curve_metrics.py,
cross_asset_pd_liquidity.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import price_discovery_shares as pds
import liquidity_curve_metrics as lcm
import cross_asset_pd_liquidity as ca

EPS = 1e-12


# ── helpers ──────────────────────────────────────────────────────────────────
def _mid_sessions(sessions):
    return [(d, r, ca._mid(df, "SPY"), ca._mid(df, "ES")) for d, r, df in sessions]

def _shares_from_z(p1, p2, z, n_lags):
    """CS/IS from a supplied cointegration residual z (length len(p1))."""
    z = z - np.nanmean(z)
    dp1 = np.diff(p1); dp2 = np.diff(p2); T = len(dp1)
    cols = [np.ones(T - n_lags), z[n_lags:T]]
    for L in range(1, n_lags + 1):
        cols.append(dp1[n_lags - L:T - L]); cols.append(dp2[n_lags - L:T - L])
    dY = np.column_stack([dp1[n_lags:T], dp2[n_lags:T]]); Xd = np.column_stack(cols)
    ok = np.all(np.isfinite(dY), axis=1) & np.all(np.isfinite(Xd), axis=1)
    if int(ok.sum()) < Xd.shape[1] + 2:                      # too few finite rows
        return np.nan, np.nan
    B, resid = pds._ols(dY[ok], Xd[ok])
    alpha = B[1, :]; Om = np.cov(resid, rowvar=False, bias=True)
    _, _, ismid = pds.hasbrouck_is(alpha, Om)
    return pds.gonzalo_granger(alpha)[1], ismid[1]            # CS_ES, IS_mid_ES

def state_series(df, kind="rel_logdepth", n_levels=10):
    """1s book-state series, ES-minus-SPY, for the liquidity-conditional VECM."""
    if kind == "rel_logdepth":
        return ca.relative_depth_state(df, n_levels)
    col = {"rel_slope": "depth_slope", "rel_entropy": "entropy_norm"}[kind]
    def avg(a):
        b = lcm.side_curve_metrics(df, a, "bid", n_levels)[f"{a}_bid_{col}"].to_numpy()
        k = lcm.side_curve_metrics(df, a, "ask", n_levels)[f"{a}_ask_{col}"].to_numpy()
        return 0.5 * (b + k)
    return avg("ES") - avg("SPY")


# ── checks ───────────────────────────────────────────────────────────────────
def lag_sensitivity(sessions, lags=(3, 5, 10, 20)):
    ms = _mid_sessions(sessions); rows = []
    for L in lags:
        df = pds.estimate_sample(ms, n_lags=L)
        rows.append({"n_lags": L, "CS_ES_mean": df.CS_ES.mean(),
                     "IS_ES_mean": df.IS_mid_ES.mean(), "CS_ES_sd": df.CS_ES.std()})
    return pd.DataFrame(rows)

def beta_fixed_vs_estimated(sessions, n_lags=5):
    rows = []
    for d, r, df in sessions:
        p1 = np.log(ca._mid(df, "SPY")); p2 = np.log(ca._mid(df, "ES"))
        fin = np.isfinite(p1) & np.isfinite(p2)
        p1, p2 = p1[fin], p2[fin]
        if p1.size < n_lags + 4:                             # not enough clean data this day
            rows.append({"date": d, "beta_hat": np.nan, "CS_ES_fixed": np.nan,
                         "CS_ES_estbeta": np.nan, "IS_ES_fixed": np.nan, "IS_ES_estbeta": np.nan})
            continue
        cs_f, is_f = _shares_from_z(p1, p2, p1 - p2, n_lags)          # beta=(1,-1)
        Xc = np.column_stack([np.ones_like(p2), p2])                  # Engle-Granger
        c, bh = np.linalg.lstsq(Xc, p1, rcond=None)[0]
        cs_e, is_e = _shares_from_z(p1, p2, p1 - bh * p2 - c, n_lags)
        rows.append({"date": d, "beta_hat": bh, "CS_ES_fixed": cs_f,
                     "CS_ES_estbeta": cs_e, "IS_ES_fixed": is_f, "IS_ES_estbeta": is_e})
    out = pd.DataFrame(rows)
    out.attrs["note"] = "beta_hat ~ 1 validates the (1,-1) no-arbitrage imposition"
    return out

def subsample_am_pm(sessions, n_lags=5):
    rows = []
    for d, r, df in sessions:
        h = len(df) // 2
        for part, sub in (("AM", df.iloc[:h]), ("PM", df.iloc[h:])):
            res = pds.estimate_day(ca._mid(sub, "SPY"), ca._mid(sub, "ES"), n_lags=n_lags)
            rows.append({"date": d, "part": part, "CS_ES": res["CS_ES"], "IS_ES": res["IS_mid_ES"]})
    out = pd.DataFrame(rows)
    return out.groupby("part")[["CS_ES", "IS_ES"]].agg(["mean", "std"])

def window_size_sensitivity(sessions, windows=("10min", "20min", "30min"),
                            y="IS_mid_ES", x="rel_logdepth", n_levels=10, n_lags=5):
    rows = []
    for w in windows:
        panel = ca.build_window_panel(sessions, window=w, n_levels=n_levels, n_lags=n_lags)
        reg = ca.panel_regression(panel, y=y, X_cols=[x])
        rows.append({"window": w, "n_obs": len(panel),
                     "coef": reg.loc[x, "coef"], "t": reg.loc[x, "t"]})
    return pd.DataFrame(rows)

def alt_state_variable(sessions, kinds=("rel_logdepth", "rel_slope", "rel_entropy"),
                       n_levels=10, n_lags=5):
    rows = []
    for kind in kinds:
        spreads, tstats = [], []
        for d, r, df in sessions:
            s = state_series(df, kind, n_levels)
            lc = ca.liquidity_conditional_vecm(ca._mid(df, "SPY"), ca._mid(df, "ES"),
                                               s, n_lags=n_lags)
            sb = lc["shares_by_state"]
            spreads.append(sb[0.9]["CS_ES"] - sb[0.1]["CS_ES"])
            tstats.append(lc["t_delta_es"])
        rows.append({"state": kind, "mean_CS_spread_hi_lo": np.mean(spreads),
                     "mean_t_delta_ES": np.mean(tstats),
                     "frac_sign_consistent": float(np.mean(np.sign(spreads) == np.sign(np.mean(spreads))))})
    return pd.DataFrame(rows)

def stationary_bootstrap_ci(x, n_boot=5000, mean_block=20, alpha=0.05, seed=0):
    """Politis-Romano stationary bootstrap CI for the mean of a serially-dependent
    (or small-N) sample x."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n == 0:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "se_boot": np.nan}
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block; means = np.empty(n_boot)
    for bi in range(n_boot):
        idx = np.empty(n, dtype=int); t = rng.integers(n)
        for i in range(n):
            idx[i] = t
            t = rng.integers(n) if rng.random() < p else (t + 1) % n
        means[bi] = x[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(x.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "se_boot": float(means.std())}


def _selftest():
    rng = np.random.default_rng(5)

    def gen(label, regime, n=4000, N=10, g=0.5):
        s = np.zeros(n)
        for t in range(1, n):
            s[t] = 0.99 * s[t - 1] + rng.normal(0, 0.12)
        f = np.cumsum(rng.normal(0, 0.0008, n))
        a_spy = np.clip(0.05 * np.exp(+g * s), 0.002, 0.45)
        a_es = np.clip(0.03 * np.exp(-g * s), 0.002, 0.45)
        L = np.linalg.cholesky([[1, 0.3], [0.3, 1]]); e = rng.normal(0, 0.01, (n, 2)) @ L.T
        lp1 = np.zeros(n); lp2 = np.zeros(n)
        for t in range(1, n):
            z = lp1[t - 1] - lp2[t - 1]
            lp1[t] = lp1[t - 1] + (f[t] - f[t - 1]) - a_spy[t] * z + e[t, 0]
            lp2[t] = lp2[t - 1] + (f[t] - f[t - 1]) + a_es[t] * z + e[t, 1]
        mid_spy = 500 * np.exp(lp1); mid_es = 5000 * np.exp(lp2)
        lev = np.arange(1, N + 1); decw = np.exp(-0.3 * (lev - 1)); cols = {}
        for a, mid, tick in (("SPY", mid_spy, 0.01), ("ES", mid_es, 0.25)):
            depth = 300.0 * np.exp((0.5 if a == "ES" else -0.5) * s)
            for i, l in enumerate(lev):
                q = np.maximum(depth * decw[i] * (1 + rng.normal(0, 0.05, n)), 1.0)
                cols[f"{a}_bidprice_{l}"] = mid - l * tick
                cols[f"{a}_askprice_{l}"] = mid + l * tick
                cols[f"{a}_bidquantity_{l}"] = q
                cols[f"{a}_askquantity_{l}"] = q * (1 + rng.normal(0, 0.02, n))
        idx = pd.date_range("2024-08-05 09:30", periods=n, freq="s", tz="America/New_York")
        return (label, regime, pd.DataFrame(cols, index=idx))

    sessions = [gen(f"d{i}", "volatile" if i % 2 else "benchmark") for i in range(4)]

    print("lag_sensitivity:\n", lag_sensitivity(sessions).round(3).to_string(index=False))
    bv = beta_fixed_vs_estimated(sessions)
    print("\nbeta_fixed_vs_estimated (beta_hat~1, shares stable):\n",
          bv[["beta_hat", "CS_ES_fixed", "CS_ES_estbeta"]].round(3).to_string(index=False))
    print("\nwindow_size_sensitivity:\n",
          window_size_sensitivity(sessions, windows=("10min", "20min")).round(3).to_string(index=False))
    print("\nalt_state_variable:\n", alt_state_variable(sessions).round(3).to_string(index=False))
    cs_by_day = pds.estimate_sample(_mid_sessions(sessions)).CS_ES.to_numpy()
    print("\nstationary_bootstrap_ci(CS_ES across days):", {k: round(v, 3) for k, v in
          stationary_bootstrap_ci(cs_by_day, mean_block=2).items()})

    # non-finite robustness: zero mids (log -> -inf) and non-finite stats must not crash.
    z = sessions[0][2].copy()
    for c in ("SPY_bidprice_1", "SPY_askprice_1"):
        z.iloc[50:70, z.columns.get_loc(c)] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        bvn = beta_fixed_vs_estimated([("d0", "benchmark", z)])
    sbc = stationary_bootstrap_ci(np.array([0.6, np.nan, 0.7, np.inf, 0.65]), n_boot=200)
    print("\nnon-finite robustness:")
    print("  beta_fixed_vs_estimated no-crash on zero mids:", len(bvn) == 1)
    print("  stationary_bootstrap_ci drops non-finite:", bool(np.isfinite(sbc["mean"])))

if __name__ == "__main__":
    _selftest()
