"""
cross_asset_pd_liquidity.py
===========================
Integration layer that joins the liquidity-curve metrics (liquidity_curve_metrics)
with the price-discovery shares (price_discovery_shares) into a single research
pipeline aimed at a JF/RFS-grade extension of Garrison-Jain-Paddrik (OFR 19-04).

The central, novel object is LIQUIDITY-STATE-CONDITIONAL PRICE DISCOVERY: a VECM
whose error-correction speed alpha is allowed to depend on the relative book
state s_t (e.g. relative depth), so the price-discovery shares become a FUNCTION
of liquidity -- directly testing "does the deeper / flatter / higher-entropy book
lead price discovery?". This is the linkage the original paper lacks (it measures
liquidity by spread only and never computes information shares).

Components
  * order_flow_imbalance        multi-level OFI (Cont-Kukanov-Stoikov), replacing
                                the message-count proportions of the original.
  * realized_moments            RV / realized covariance / realized correlation,
                                a continuous conditioner vs the volatile/benchmark split.
  * relative_depth_state        the 1s liquidity-state series s_t = log(depth_ES)-log(depth_SPY).
  * liquidity_conditional_vecm  HEADLINE: alpha(s) = alpha + delta*s; shares evaluated
                                across the distribution of s.
  * window_features /           per intraday-window join of PD shares + liquidity-curve
    build_window_panel          features + OFI + RV -> the panel dataset (power: many
                                windows x days, not 14 day-level points).
  * panel_regression            within-day FE + day-CLUSTERED SEs: share ~ relative
                                liquidity state + controls.
  * state_split_shares          information shares within terciles of the book state.

Requires liquidity_curve_metrics.py and price_discovery_shares.py on the path.
Input: the pipeline's per-session wide frame on a 1s grid with columns
{ASSET}_{side}price_{i} / {ASSET}_{side}quantity_{i} for i=1..N and both assets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import price_discovery_shares as pds
import liquidity_curve_metrics as lcm

EPS = 1e-12


# ── frequency scaling & fleeting-quote filtering ─────────────────────────────
def infer_dt(df):
    """Median seconds between grid timestamps (1.0 if not a DatetimeIndex)."""
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        d = np.diff(idx.to_numpy()).astype("timedelta64[ns]").astype(np.int64) / 1e9
        return float(np.median(d))
    return 1.0

def frequency_defaults(df=None, dt=None, mem_seconds=5.0, irf_seconds=2.0,
                       fleeting_seconds=0.05, bounce_seconds=0.5,
                       max_lags=60, max_horizon=300):
    """Hold wall-clock roughly constant across sampling grids. Returns VECM lag
    count, IRF horizons, bootstrap block length, and the fleeting-quote filter
    width (in grid steps) appropriate to the inferred (or supplied) interval dt.
    At 1s the fleeting filter is inert; it engages only at sub-second grids."""
    if dt is None:
        dt = infer_dt(df) if df is not None else 1.0
    n_lags = int(np.clip(round(mem_seconds / dt), 5, max_lags))
    H = int(np.clip(round(irf_seconds / dt), 10, max_horizon))
    block = int(max(n_lags, round(bounce_seconds / dt), 10))
    min_rest_steps = int(max(0, round(fleeting_seconds / dt)))
    return {"dt": dt, "n_lags": n_lags, "horizons": range(0, H + 1),
            "block": block, "min_rest_steps": min_rest_steps}

def resting_time_filter(t, x, min_rest):
    """Event-level fleeting-quote (debounce) filter for a RAW best-quote stream:
    keep an update only once at least `min_rest` (same units as t) has elapsed
    since the last kept update; flickers are dropped and the prior state persists.
    Apply upstream of gridding. Returns (t_kept, x_kept, kept_index)."""
    t = np.asarray(t, float); x = np.asarray(x)
    keep = [0]; last = t[0]
    for i in range(1, len(t)):
        if t[i] - last >= min_rest:
            keep.append(i); last = t[i]
    keep = np.asarray(keep)
    return t[keep], x[keep], keep

def _despike(p, q, passes):
    """Grid proxy for the resting-time filter: flatten transient price excursions
    no longer than `passes` steps (and carry the size forward), so OFI does not
    register a post-and-pull round trip. passes=0 is the identity."""
    if passes <= 0:
        return p, q
    p = p.copy(); q = q.copy()
    for _ in range(passes):
        spike = (p[1:-1] != p[:-2]) & (p[2:] == p[:-2])
        if not spike.any():
            break
        idx = np.where(spike)[0] + 1
        p[idx] = p[idx - 1]; q[idx] = q[idx - 1]
    return p, q


# ── primitives ───────────────────────────────────────────────────────────────
def _mid(df, asset):
    return (df[f"{asset}_bidprice_1"].to_numpy() + df[f"{asset}_askprice_1"].to_numpy()) / 2.0

def order_flow_imbalance(df, asset, n_levels=10, min_rest_steps=0):
    """Cont-Kukanov-Stoikov OFI per snapshot, summed over levels 1..n. Returns an
    array of length len(df) (first element 0). With min_rest_steps > 0 a fleeting-
    quote filter flattens transient price excursions shorter than that many grid
    steps before the OFI is formed (use at sub-second grids; see frequency_defaults)."""
    ofi = np.zeros(len(df))
    for i in range(1, n_levels + 1):
        Pb = df[f"{asset}_bidprice_{i}"].to_numpy(); Qb = df[f"{asset}_bidquantity_{i}"].to_numpy()
        Pa = df[f"{asset}_askprice_{i}"].to_numpy(); Qa = df[f"{asset}_askquantity_{i}"].to_numpy()
        if min_rest_steps > 0:
            Pb, Qb = _despike(Pb, Qb, min_rest_steps)
            Pa, Qa = _despike(Pa, Qa, min_rest_steps)
        eb = np.where(Pb[1:] >= Pb[:-1], Qb[1:], 0.0) - np.where(Pb[1:] <= Pb[:-1], Qb[:-1], 0.0)
        ea = np.where(Pa[1:] <= Pa[:-1], Qa[1:], 0.0) - np.where(Pa[1:] >= Pa[:-1], Qa[:-1], 0.0)
        ofi[1:] += np.nan_to_num(eb - ea)
    return ofi

def realized_moments(mid_spy, mid_es):
    r1 = np.diff(np.log(mid_spy)); r2 = np.diff(np.log(mid_es))
    rv1 = float(np.nansum(r1 ** 2)); rv2 = float(np.nansum(r2 ** 2))
    rcov = float(np.nansum(r1 * r2))
    rcorr = rcov / (np.sqrt(rv1 * rv2) + EPS)
    return {"rv_spy": rv1, "rv_es": rv2, "rcov": rcov, "rcorr": rcorr}

def relative_depth_state(df, n_levels=10):
    """s_t = log(total ES depth) - log(total SPY depth) per 1s (book-state proxy)."""
    tot = {}
    for a in ("SPY", "ES"):
        q = np.zeros(len(df))
        for side in ("bid", "ask"):
            for i in range(1, n_levels + 1):
                q = q + np.nan_to_num(df[f"{a}_{side}quantity_{i}"].to_numpy())
        tot[a] = q
    return np.log(tot["ES"] + EPS) - np.log(tot["SPY"] + EPS)


# ── HEADLINE: liquidity-state-conditional VECM ───────────────────────────────
def liquidity_conditional_vecm(mid_spy, mid_es, state, n_lags=5,
                               quantiles=(0.1, 0.5, 0.9), names=("SPY", "ES"),
                               criterion=None, pmax=12):
    """
    VECM with beta=(1,-1) on log mids and an interaction of the EC term with the
    standardized book state s: dP = (alpha + delta*s_{t})*z_{t} + lags + e.
    Returns alpha, delta, their t-stats, and CS/IS evaluated at quantiles of s
    (shares as a function of the liquidity state). With ``criterion`` set the lag is chosen by that
    criterion on the base bivariate VECM (the EC*state interaction is p-invariant, so it does not
    enter lag-order selection); otherwise ``n_lags`` is used.
    """
    p1 = np.log(np.asarray(mid_spy, float)); p2 = np.log(np.asarray(mid_es, float))
    s = np.asarray(state, float)
    # no NaN compression: compressing (p1, p2, s) before differencing splices the last pre-halt
    # price to the reopen price -- the halt gap becomes one pseudo-return in exactly the estimator
    # the masking was built for. Diff on the raw grid; drop non-finite ROWS of the design instead.
    if criterion is not None:
        n_lags, _ = pds.select_lag(p1, p2, pmax=pmax, criterion=criterion)
    s = (s - np.nanmean(s)) / (np.nanstd(s) + EPS)
    z = (p1 - p2); z = z - np.nanmean(z)
    dp1 = np.diff(p1); dp2 = np.diff(p2); T = len(dp1)
    ec = z[n_lags:T]; sc = s[n_lags:T]
    cols = [np.ones(T - n_lags), ec, ec * sc]
    for L in range(1, n_lags + 1):
        cols.append(dp1[n_lags - L:T - L]); cols.append(dp2[n_lags - L:T - L])
    X = np.column_stack(cols)
    dY = np.column_stack([dp1[n_lags:T], dp2[n_lags:T]])
    ok = np.all(np.isfinite(dY), axis=1) & np.all(np.isfinite(X), axis=1)
    dY, X = dY[ok], X[ok]
    B, resid = pds._ols(dY, X)
    alpha = B[1, :].copy(); delta = B[2, :].copy()
    Omega = np.cov(resid, rowvar=False, bias=True)
    XtXinv = np.linalg.inv(X.T @ X)
    t_delta = [delta[k] / (np.sqrt((resid[:, k] ** 2).mean() * XtXinv[2, 2]) + EPS) for k in range(2)]

    sq = {q: float(np.nanquantile(s, q)) for q in quantiles}   # s carries NaN in masked windows
    by_state = {}
    for q, sv in sq.items():
        a = alpha + delta * sv
        _, _, ismid = pds.hasbrouck_is(a, Omega); cs = pds.gonzalo_granger(a)
        by_state[q] = {f"CS_{names[1]}": cs[1], f"IS_mid_{names[1]}": ismid[1], "s_value": sv}
    return {"alpha": alpha, "delta": delta, "n_obs": int(dY.shape[0]),
            "t_delta_spy": t_delta[0], "t_delta_es": t_delta[1],
            "shares_by_state": by_state}


# ── per-window feature join (PD shares + liquidity + OFI + RV) ────────────────
def _relative_liquidity(df, n_levels):
    agg = {}
    for a in ("SPY", "ES"):
        b = lcm.side_curve_metrics(df, a, "bid", n_levels).mean()
        k = lcm.side_curve_metrics(df, a, "ask", n_levels).mean()
        agg[a] = {
            "depth": b[f"{a}_bid_total_depth"] + k[f"{a}_ask_total_depth"],
            "entropy": 0.5 * (b[f"{a}_bid_entropy_norm"] + k[f"{a}_ask_entropy_norm"]),
            "slope": 0.5 * (b[f"{a}_bid_depth_slope"] + k[f"{a}_ask_depth_slope"]),
            "com": 0.5 * (b[f"{a}_bid_center_of_mass"] + k[f"{a}_ask_center_of_mass"]),
        }
    cb = lcm.cross_asset_curve_similarity(df, "SPY", "ES", "bid", n_levels).mean()
    ca = lcm.cross_asset_curve_similarity(df, "SPY", "ES", "ask", n_levels).mean()
    return {
        "rel_logdepth": float(np.log(agg["ES"]["depth"] + EPS) - np.log(agg["SPY"]["depth"] + EPS)),
        "rel_entropy": float(agg["ES"]["entropy"] - agg["SPY"]["entropy"]),
        "rel_slope": float(agg["ES"]["slope"] - agg["SPY"]["slope"]),
        "rel_com": float(agg["ES"]["com"] - agg["SPY"]["com"]),
        "shape_cosine": float(0.5 * (cb["SPY_ES_bid_shape_cosine"] + ca["SPY_ES_ask_shape_cosine"])),
    }

def window_features(win, n_levels=10, n_lags=5, min_rest_steps=0, criterion=None, pmax=12):
    ms, me = _mid(win, "SPY"), _mid(win, "ES")
    row = pds.estimate_day(ms, me, n_lags=n_lags, criterion=criterion, pmax=pmax)
    row.update(_relative_liquidity(win, n_levels))
    row.update(realized_moments(ms, me))
    row["ofi_spy"] = float(np.nansum(order_flow_imbalance(win, "SPY", n_levels, min_rest_steps)))
    row["ofi_es"] = float(np.nansum(order_flow_imbalance(win, "ES", n_levels, min_rest_steps)))
    return row

def build_window_panel(sessions, window="30min", n_levels=10, n_lags=None,
                       min_obs=300, min_rest_steps=0, criterion=None, pmax=12):
    """sessions: iterable of (date, regime, df). Splits each session into windows
    and computes the joined feature row per window -> panel DataFrame. With
    n_lags=None the VECM lag count is frequency-scaled from the first session's grid
    (frequency_defaults); pass min_rest_steps>0 to filter fleeting quotes sub-second.
    With ``criterion`` set, ONE lag order is chosen (pooled across sessions on the full-session log
    mids) and applied to every window, so the windowed shares share a specification."""
    sessions = list(sessions)
    if criterion is not None:
        pairs = [(np.log(_mid(df, "SPY")), np.log(_mid(df, "ES"))) for _d, _r, df in sessions]
        n_lags = pds.select_lag_pooled(pairs, pmax=pmax, criterion=criterion)[0]
    elif n_lags is None:
        n_lags = frequency_defaults(sessions[0][2])["n_lags"] if sessions else 5
    rows = []
    for date, regime, df in sessions:
        for wlabel, win in df.groupby(pd.Grouper(freq=window)):
            if len(win) < min_obs:
                continue
            try:
                r = window_features(win, n_levels, n_lags, min_rest_steps)   # fixed at the pooled lag
            except Exception:
                continue
            r.update({"date": str(date), "regime": regime, "window": str(wlabel)})
            rows.append(r)
    return pd.DataFrame(rows)


# ── panel regression: within-day FE + day-clustered SE ───────────────────────
def panel_regression(panel, y, X_cols, cluster="date", fe="date"):
    """OLS of y on X_cols with `fe` fixed effects (within transform) and cluster-
    robust SEs by `cluster`. Returns a coef/se/t DataFrame. Degrades to NaN (rather
    than raising) on a degenerate panel -- too few windows, missing columns, or no
    within-group variation in the regressors (a rank-deficient design)."""
    X_cols = list(X_cols); K = len(X_cols)
    cols = list(dict.fromkeys([y] + X_cols + [cluster, fe]))
    nan_out = pd.DataFrame({"coef": np.full(K, np.nan), "se": np.full(K, np.nan),
                            "t": np.full(K, np.nan)}, index=X_cols)
    if len(panel) == 0 or not set(cols).issubset(panel.columns):
        return nan_out
    d = panel[cols].dropna().reset_index(drop=True)
    if d.shape[0] < K + 2:
        return nan_out
    def _within(col):
        return (d[col] - d.groupby(fe)[col].transform("mean")).to_numpy()
    Y = _within(y)
    X = np.column_stack([_within(c) for c in X_cols])
    if not np.all(np.isfinite(X)) or np.linalg.matrix_rank(X) < K:   # no within-group variation
        return nan_out
    beta, resid = pds._ols(Y[:, None], X)
    beta = beta.ravel(); u = resid.ravel()
    XtXinv = np.linalg.pinv(X.T @ X)                                 # pinv: safe if near-singular
    meat = np.zeros((X.shape[1], X.shape[1]))
    for rows in d.groupby(cluster).indices.values():
        Xg = X[rows]; ug = u[rows]
        meat += np.outer(Xg.T @ ug, Xg.T @ ug)
    G = d[cluster].nunique(); N, K = X.shape
    scale = (G / (G - 1.0)) * ((N - 1.0) / (N - K)) if (G > 1 and N > K) else np.nan
    V = XtXinv @ meat @ XtXinv * scale
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    return pd.DataFrame({"coef": beta, "se": se, "t": beta / se}, index=X_cols)


def state_split_shares(panel, state="rel_logdepth", shares=("CS_ES", "IS_mid_ES"), n_bins=3):
    """Mean information shares within terciles of the liquidity-state variable."""
    p = panel.dropna(subset=[state]).copy()
    p["bin"] = pd.qcut(p[state], n_bins, labels=[f"Q{i+1}" for i in range(n_bins)])
    return p.groupby("bin", observed=True)[list(shares)].mean()


# ── self-test: synthetic DGP with a KNOWN liquidity -> leadership link ────────
def _selftest():
    rng = np.random.default_rng(7)

    def gen(label, regime, n=6000, N=10, g=0.55):
        s = np.zeros(n)
        for t in range(1, n):
            s[t] = 0.99 * s[t - 1] + rng.normal(0, 0.12)       # latent rel-depth state (stationary)
        f = np.cumsum(rng.normal(0, 0.0008, n))               # common efficient log-price
        a_spy = np.clip(0.05 * np.exp(+g * s), 0.002, 0.45)   # SPY adjusts MORE when s high
        a_es = np.clip(0.03 * np.exp(-g * s), 0.002, 0.45)    # ES adjusts LESS when s high
        L = np.linalg.cholesky([[1, 0.3], [0.3, 1]]); e = rng.normal(0, 0.01, (n, 2)) @ L.T
        lp1 = np.zeros(n); lp2 = np.zeros(n)
        for t in range(1, n):
            z = lp1[t - 1] - lp2[t - 1]
            lp1[t] = lp1[t - 1] + (f[t] - f[t - 1]) - a_spy[t] * z + e[t, 0]
            lp2[t] = lp2[t - 1] + (f[t] - f[t - 1]) + a_es[t] * z + e[t, 1]
        mid_spy = 500 * np.exp(lp1); mid_es = 5000 * np.exp(lp2)
        lev = np.arange(1, N + 1); decw = np.exp(-0.3 * (lev - 1)); cols = {}
        for a, mid, tick, sgn in (("SPY", mid_spy, 0.01, -1), ("ES", mid_es, 0.25, +1)):
            depth = 300.0 * np.exp((0.5 if a == "ES" else -0.5) * s)
            for i, l in enumerate(lev):
                q = np.maximum(depth * decw[i] * (1 + rng.normal(0, 0.05, n)), 1.0)
                cols[f"{a}_bidprice_{l}"] = mid - l * tick
                cols[f"{a}_askprice_{l}"] = mid + l * tick
                cols[f"{a}_bidquantity_{l}"] = q
                cols[f"{a}_askquantity_{l}"] = q * (1 + rng.normal(0, 0.02, n))
        idx = pd.date_range("2024-08-05 09:30", periods=n, freq="s", tz="America/New_York")
        df = pd.DataFrame(cols, index=idx)
        return (label, regime, df)

    sessions = [gen(f"d{i}", "volatile" if i % 2 else "benchmark") for i in range(6)]

    # 1) headline: shares as a function of the book state
    _, _, df0 = sessions[0]
    lc = liquidity_conditional_vecm(_mid(df0, "SPY"), _mid(df0, "ES"),
                                    relative_depth_state(df0), n_lags=5)
    sb = lc["shares_by_state"]
    print("liquidity-conditional VECM (session 0):")
    print("  CS_ES at s_low/mid/high = %.3f / %.3f / %.3f   (t on delta_ES = %.1f)"
          % (sb[0.1]["CS_ES"], sb[0.5]["CS_ES"], sb[0.9]["CS_ES"], lc["t_delta_es"]))
    print("  -> ES leadership rises with relative depth:",
          sb[0.9]["CS_ES"] > sb[0.1]["CS_ES"])

    # 2) windowed panel + clustered regression
    panel = build_window_panel(sessions, window="20min", n_levels=10, n_lags=5)
    print("\npanel: %d window-obs across %d sessions" % (len(panel), panel.date.nunique()))
    reg = panel_regression(panel, y="IS_mid_ES", X_cols=["rel_logdepth", "rcorr"])
    print(reg.round(4).to_string())
    print("  -> deeper ES book predicts higher ES information share:",
          bool(reg.loc["rel_logdepth", "t"] > 2))

    # 3) state-split shares
    print("\nshares by relative-depth tercile:")
    print(state_split_shares(panel).round(3).to_string())

if __name__ == "__main__":
    _selftest()
