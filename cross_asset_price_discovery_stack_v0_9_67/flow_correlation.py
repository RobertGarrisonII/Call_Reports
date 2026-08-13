"""flow_correlation.py -- tandem order flow as a TIME SERIES (v0.9.68).

The paper's title phenomenon is co-moving ORDER FLOW, but until now the stack measured it
only statically (one OFI-innovation correlation per regime) while the dynamic system
(Table 9) modelled the correlation of RETURNS -- one step removed. This module puts the
flow correlation itself on a bar clock and delivers three exhibits:

Tier 1 -- table_flow_corr_regimes():
    Per-bar correlation of the two legs' OFI INNOVATIONS on non-overlapping bars
    (Fisher-z), described by the sample's a-priori day labels: regime means with
    day-clustered SEs, the volatile-vs-benchmark permutation p, and the within-day
    relation to the relative-depth book state. Innovations, not raw OFI: raw flow is
    autocorrelated (order splitting), so raw-flow correlation partly measures common
    persistence; each leg is prefiltered by its own past first.

Tier 2 -- table_flow_corr_mediation():
    Does tandem flow MEDIATE the volatility -> return-correlation link? Two bar-level
    panel regressions with day fixed effects and day-clustered inference:
        Eq A:  d z_flow_t = FE + own lags + LAGGED {RV, book state, d WtdSpread} + e
        Eq B:  d z_ret_t  = FE + own lags + same lagged controls [+ gamma * d z_flow_t]
    reported total-vs-direct for the RV_ES coefficient with the indirect share and a
    day-level cluster-bootstrap CI. Signed OFI is EXCLUDED by construction: the dependent
    variable is built from the flows, so a same-bar signed-flow regressor would be
    arithmetic, not economics. The RV/state/spread treatments enter LAGGED for the same
    reason -- same-bar RV shares sub-returns with the same-bar correlation estimate.

Data-driven regimes -- table_flow_corr_ms_regimes():
    Per-day Markov-switching regression on the z_flow level series,
        z_t = mu_{s_t} + phi * z_{t-1} + eps_t,   eps_t ~ N(0, sigma^2_{s_t}),
    (switching intercept and variance, common AR(1)). BOTH the number of regimes and
    their levels come from the data: select_k_ms tests K vs K+1 sequentially from K=1 by
    PARAMETRIC BOOTSTRAP LR (the extra state's mixing weight is unidentified under H0 --
    Davies problem -- so chi-square calibration is invalid and the null LR is simulated),
    and the fitted means land wherever the data puts them (zero, weakly negative,
    strongly positive). The table also carries REGIME-DYNAMICS ASYMMETRY: top-vs-bottom
    persistence (log duration ratio, sign-flip p) and the direction of entry into the
    top regime (mean signed pre-entry bar return, demeaned per day).

Flow asymmetry -- table_flow_corr_asymmetry():
    Quadrant semicorrelations of the grid-level OFI innovations (down-down vs up-up).
    Truncation biases the two levels identically under point symmetry, so the per-day
    Fisher-z gap has a zero null; day-level sign-flip inference. The innovation-level
    analogue of Table 5's corner asymmetry.

Price-discovery link -- table_flow_pd_link():
    Within-day windows: CS/IS from the fixed-(1,-1) VECM on each window's mids regressed
    on the window's tandem-flow state (mean z_flow; smoothed top-regime occupancy), day
    FE + day-clustered SEs. The payoff exhibit: does price discovery migrate toward ES
    when tandem flow intensifies?

Every estimator masks on load (halt policy) through the caller; internally, bars touching
non-finite legs fail the coverage floor and drop out of every design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import correlation_svar as cs
import cross_asset_pd_liquidity as ca
import markov_switching_vecm as msv

EPS = 1e-12


# ══════════════════════════════════════════════════════════════════════════════
#  OFI innovations and the bar-level flow-correlation series
# ══════════════════════════════════════════════════════════════════════════════
def ar_innovations(x, ar_order=5):
    """Residuals of an AR(ar_order) OLS prefilter of ``x`` (1-D, NaN-aware).

    Rows whose window touches a NaN are NaN in the output, as is the warm-up. The filter
    removes own-persistence (order splitting, momentum) so a cross-correlation of two
    filtered series measures common SURPRISE, not common persistence."""
    x = np.asarray(x, float)
    T = len(x)
    p = int(ar_order)
    out = np.full(T, np.nan)
    if p <= 0 or T <= p + 10:
        return x - np.nanmean(x) if p <= 0 else out
    X = np.column_stack([np.ones(T - p)] + [x[p - i - 1:T - i - 1] for i in range(p)])
    y = x[p:]
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < p + 10:
        return out
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    resid = y - X @ beta
    out[p:][m] = resid[m]
    return out


def _masked_ofi(df, asset, n_levels=10, min_rest_steps=0):
    """Cont-Kukanov-Stoikov OFI, NaN'd wherever the leg's own mid is non-finite
    (order_flow_imbalance zero-fills masked rows, which would let halt bars pass the
    coverage floor as fake zeros)."""
    ofi = np.asarray(ca.order_flow_imbalance(df, asset, n_levels, min_rest_steps), float)
    a1 = df[f"{asset}_askprice_1"].to_numpy(float)
    b1 = df[f"{asset}_bidprice_1"].to_numpy(float)
    ofi[~(np.isfinite(a1) & np.isfinite(b1))] = np.nan
    return ofi


def ofi_innovations(df, asset, n_levels=10, min_rest_steps=0, ar_order=5):
    """OFI innovations for one leg: masked OFI, AR-prefiltered."""
    return ar_innovations(_masked_ofi(df, asset, n_levels, min_rest_steps),
                          ar_order=ar_order)


def _bar_corr(x, y, bar_ids, min_sub):
    """Per-bar Pearson correlation of two aligned series over ``bar_ids`` groups.
    A bar needs >= min_sub finite pairs and movement on both legs; else NaN."""
    s = pd.Series(np.asarray(x, float))
    t = pd.Series(np.asarray(y, float))
    out = {}
    for b, ii in s.groupby(np.asarray(bar_ids)).groups.items():
        xv = s.loc[ii].to_numpy()
        yv = t.loc[ii].to_numpy()
        m = np.isfinite(xv) & np.isfinite(yv)
        if m.sum() >= min_sub and xv[m].std() > EPS and yv[m].std() > EPS:
            out[b] = float(np.corrcoef(xv[m], yv[m])[0, 1])
        else:
            out[b] = np.nan
    return pd.Series(out)


def flow_corr_bars(df, bar_seconds=60, n_levels=10, min_rest_steps=0, ar_order=5,
                   wspread_kind="cost_to_fill"):
    """-> DataFrame indexed by bar start with the bar-level state of tandem trading.

    Columns
        z_flow   Fisher-z of the per-bar correlation of the two legs' OFI innovations
        z_ret    Fisher-z of the per-bar realized correlation of mid sub-returns
        rv_spy, rv_es    per-bar mean squared mid return (bps^2)
        state    per-bar mean relative-depth book state (log ES depth - log SPY depth)
        d_wspr_spy, d_wspr_es    change in the per-bar mean weighted spread
        ret_spy, ret_es          per-bar SUM of bps mid sub-returns (the bar return)
        net_ofi_spy, net_ofi_es  per-bar SUM of the (masked, unfiltered) signed OFI
        n_pairs  finite OFI-innovation pairs inside the bar

    Same coverage discipline as the RealBar dependent variable: a bar must contain at
    least max(10, 0.25 * capacity) finite pairs, so partially masked (halt-boundary) bars
    drop out rather than entering as attenuated fakes."""
    idx = df.index
    bar = idx.floor("%ds" % int(bar_seconds))
    dt = float(np.median(np.diff(idx.asi8))) / 1e9 if len(idx) > 1 else 1.0
    cap = max(1.0, float(bar_seconds) / max(dt, 1e-9))
    min_sub = max(10, int(0.25 * cap))

    o_spy = _masked_ofi(df, "SPY", n_levels, min_rest_steps)
    o_es = _masked_ofi(df, "ES", n_levels, min_rest_steps)
    u_spy = ar_innovations(o_spy, ar_order)
    u_es = ar_innovations(o_es, ar_order)
    r_spy = cs._returns_bps(df, "SPY")
    r_es = cs._returns_bps(df, "ES")

    B = pd.DataFrame(index=pd.Index(sorted(bar.unique()), name="bar"))
    rho_f = _bar_corr(u_spy, u_es, bar, min_sub).reindex(B.index)
    rho_r = _bar_corr(r_spy, r_es, bar, min_sub).reindex(B.index)
    B["z_flow"] = cs._fisher_z(rho_f)
    B["z_ret"] = cs._fisher_z(rho_r)

    aux = pd.DataFrame({
        "rv_spy": pd.Series(r_spy, index=idx) ** 2,
        "rv_es": pd.Series(r_es, index=idx) ** 2,
        "ret_spy": pd.Series(r_spy, index=idx),
        "ret_es": pd.Series(r_es, index=idx),
        "net_ofi_spy": pd.Series(o_spy, index=idx),
        "net_ofi_es": pd.Series(o_es, index=idx),
        "state": pd.Series(np.asarray(ca.relative_depth_state(df, n_levels), float), index=idx),
        "wspr_spy": pd.Series(cs.weighted_spread(df, "SPY", None, n_levels, wspread_kind), index=idx),
        "wspr_es": pd.Series(cs.weighted_spread(df, "ES", None, n_levels, wspread_kind), index=idx),
    })
    g = aux.groupby(bar)
    for c in ("rv_spy", "rv_es", "state"):
        B[c] = g[c].mean().reindex(B.index)
    for c in ("ret_spy", "ret_es", "net_ofi_spy", "net_ofi_es"):   # signed bar SUMS
        B[c] = g[c].sum(min_count=1).reindex(B.index)
    for a in ("spy", "es"):
        B[f"d_wspr_{a}"] = g[f"wspr_{a}"].mean().reindex(B.index).diff()
    fin = pd.Series(np.isfinite(u_spy) & np.isfinite(u_es), index=idx)
    B["n_pairs"] = fin.groupby(bar).sum().reindex(B.index).fillna(0).astype(int)
    return B


def per_day_bars(sessions, bar_seconds=60, n_levels=10, min_rest_steps=0, ar_order=5,
                 min_bars=20, verbose=False):
    """-> List[(date, regime, bars_df)] with days that yield < min_bars finite z_flow bars
    skipped (reported when verbose)."""
    out = []
    for date, regime, df in sessions:
        try:
            b = flow_corr_bars(df, bar_seconds, n_levels, min_rest_steps, ar_order)
        except Exception as e:                              # pragma: no cover - defensive
            if verbose:
                print(f"  [flow] {date}: SKIPPED ({type(e).__name__}: {e})")
            continue
        n_ok = int(np.isfinite(b["z_flow"].to_numpy(float)).sum())
        if n_ok < min_bars:
            if verbose:
                print(f"  [flow] {date}: {n_ok} finite z_flow bars < {min_bars} -- skipped")
            continue
        out.append((date, regime, b))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Asymmetry: downside vs upside semicorrelations
# ══════════════════════════════════════════════════════════════════════════════
def semicorrelations(u1, u2, min_n=50):
    """Quadrant semicorrelations of two innovation series: rho_dn on the observations
    where BOTH are negative (joint selling pressure), rho_up where both are positive.

    The quadrant truncation biases each LEVEL relative to the unconditional correlation,
    but the bias is identical for the two quadrants under any point-symmetric joint
    distribution -- so rho_dn - rho_up is a valid asymmetry statistic with a zero null,
    while neither level is comparable to the unconditional estimate. Returns
    (rho_dn, rho_up, n_dn, n_up); a quadrant with fewer than min_n pairs is NaN."""
    x = np.asarray(u1, float)
    y = np.asarray(u2, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    out = []
    for q in ((x < 0) & (y < 0), (x > 0) & (y > 0)):
        if q.sum() >= min_n and x[q].std() > EPS and y[q].std() > EPS:
            out.append(float(np.corrcoef(x[q], y[q])[0, 1]))
        else:
            out.append(np.nan)
        out.append(int(q.sum()))
    return out[0], out[2], out[1], out[3]


def sign_flip_p(vals, n_flip=20000, seed=0):
    """Exact-style sign-flip test that E[vals] = 0: under the symmetric null each day's
    statistic is equally likely to carry either sign, so flip signs at random and count
    |mean*| >= |mean|. The day is the independent unit (matches the clustering)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    obs = abs(v.mean())
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(int(n_flip), len(v)))
    cnt = int(((np.abs((S * v).mean(axis=1))) >= obs - EPS).sum())
    return (cnt + 1) / (int(n_flip) + 1)


def day_semicorr_asymmetry(sessions, n_levels=10, min_rest_steps=0, ar_order=5, min_n=50):
    """Per-day quadrant semicorrelations of the GRID-level OFI innovations (the whole
    day's finite pairs, not per bar -- the quadrants need mass). Returns a DataFrame with
    rho_dn, rho_up, dz = z(rho_dn) - z(rho_up) per day."""
    rows = []
    for date, regime, df in sessions:
        u1 = ofi_innovations(df, "SPY", n_levels, min_rest_steps, ar_order)
        u2 = ofi_innovations(df, "ES", n_levels, min_rest_steps, ar_order)
        r_dn, r_up, n_dn, n_up = semicorrelations(u1, u2, min_n=min_n)
        dz = (float(cs._fisher_z(r_dn) - cs._fisher_z(r_up))
              if np.isfinite(r_dn) and np.isfinite(r_up) else np.nan)
        rows.append({"date": str(date), "regime": str(regime), "rho_dn": r_dn,
                     "rho_up": r_up, "dz": dz, "n_dn": n_dn, "n_up": n_up})
    return pd.DataFrame(rows)


def table_flow_corr_asymmetry(sessions, n_levels=10, min_rest_steps=0, ar_order=5,
                              min_n=50, n_flip=20000, seed=0):
    """Asymmetry exhibit -> (DataFrame, notes). Rows: each a-priori regime plus all days;
    columns: median downside/upside semicorrelation, the mean Fisher-z gap with its
    day-clustered SE, and the day-level sign-flip p. This is the innovation-level
    analogue of Table 5's corner asymmetry: a positive gap says the two markets'
    flow surprises co-move more tightly under joint selling than joint buying."""
    per = day_semicorr_asymmetry(sessions, n_levels, min_rest_steps, ar_order, min_n)
    if per.empty:
        return pd.DataFrame(), "no usable sessions"
    out = {}
    for gname in _regime_order(per.regime) + ["all days"]:
        g = per if gname == "all days" else per[per.regime == gname]
        v = g.dz.to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        out[gname] = {
            "down corr (median)": float(g.rho_dn.median()),
            "up corr (median)": float(g.rho_up.median()),
            "mean dz (down - up)": float(v.mean()),
            "se(dz)": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
            "sign-flip p": sign_flip_p(v, n_flip=n_flip, seed=seed),
            "days": int(len(v))}
    df = pd.DataFrame(out).T
    df.index.name = "day group"
    notes = ("Quadrant semicorrelations of the grid-level OFI innovations: 'down' uses the "
             "observations where BOTH legs' flow surprises are negative, 'up' where both "
             "are positive. Quadrant truncation biases the LEVELS (they are not comparable "
             "to the unconditional correlation) but identically for the two quadrants under "
             "any point-symmetric distribution, so the Fisher-z gap has a zero null. The "
             "sign-flip p flips day-level gaps at random (%d draws; day = independent "
             "unit). Positive gap = tandem selling tighter than tandem buying -- the "
             "innovation-level analogue of Table 5's corner asymmetry." % n_flip)
    return df, notes


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 1: the a-priori-regime description
# ══════════════════════════════════════════════════════════════════════════════
def _regime_order(labels):
    pref = ["benchmark", "volatile", "mwcb"]
    seen = list(dict.fromkeys(labels))
    return [r for r in pref if r in seen] + sorted(r for r in seen if r not in pref)


def table_flow_corr_regimes(sessions, bar_seconds=60, n_levels=10, min_rest_steps=0,
                            ar_order=5, n_perm=20000, seed=0, bars=None):
    """Tier 1 -> (DataFrame, notes). Rows: each a-priori regime plus the
    volatile-minus-benchmark contrast; columns: mean flow correlation (tanh of the mean
    day-level z), mean z, day-clustered SE, within-day corr(z_flow, book state), days,
    bars. The contrast row carries the day-level permutation p."""
    bars = per_day_bars(sessions, bar_seconds, n_levels, min_rest_steps, ar_order) \
        if bars is None else bars
    rows = []
    for date, regime, b in bars:
        z = b["z_flow"].to_numpy(float)
        s = b["state"].to_numpy(float)
        m = np.isfinite(z)
        ms = m & np.isfinite(s)
        c_state = float(np.corrcoef(z[ms], s[ms])[0, 1]) if ms.sum() > 10 and \
            z[ms].std() > EPS and s[ms].std() > EPS else np.nan
        rows.append({"date": str(date), "regime": str(regime),
                     "z_flow": float(np.nanmean(z)), "n_bars": int(m.sum()),
                     "corr_state": c_state})
    per_day = pd.DataFrame(rows)
    if per_day.empty:
        return pd.DataFrame(), "no usable sessions"

    out = {}
    for rr in _regime_order(per_day.regime):
        g = per_day[per_day.regime == rr]
        zbar = float(g.z_flow.mean())
        se = float(g.z_flow.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan
        out[rr] = {"mean corr": float(np.tanh(zbar)), "mean z": zbar, "se(z)": se,
                   "corr(z, state)": float(g.corr_state.mean()),
                   "days": int(len(g)), "bars": int(g.n_bars.sum()), "perm p": np.nan}
    labs = _regime_order(per_day.regime)
    pair = None
    if {"volatile", "benchmark"} <= set(labs):
        pair = ("volatile", "benchmark")
    elif len(labs) == 2:                      # e.g. stress/calm fixtures: contrast the two
        pair = (labs[1], labs[0])
    if pair is not None:
        import price_discovery_shares as pds
        tmp = per_day[per_day.regime.isin(pair)].copy()
        tmp["regime"] = np.where(tmp.regime == pair[0], "volatile", "benchmark")
        res = pds.compare_regimes(tmp, metric="z_flow", n_perm=n_perm, seed=seed)
        out[f"{pair[0]} - {pair[1]}"] = {
            "mean corr": float(np.tanh(res["vol_mean"]) - np.tanh(res["ben_mean"])),
            "mean z": float(res["diff"]), "se(z)": np.nan, "corr(z, state)": np.nan,
            "days": int(len(tmp)), "bars": np.nan, "perm p": float(res["p_perm"])}
    df = pd.DataFrame(out).T
    df.index.name = "regime"
    notes = ("Per-bar correlation of OFI INNOVATIONS (AR(%d)-prefiltered per leg, per day) on "
             "non-overlapping %ds bars, Fisher-z. 'mean corr' is tanh of the mean day-level z; "
             "SE treats days as the independent unit; 'corr(z, state)' is the mean within-day "
             "correlation with the relative-depth book state; the contrast row's p is a "
             "day-level permutation test (%d draws)." % (ar_order, bar_seconds, n_perm))
    return df, notes


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 2: the mediation system
# ══════════════════════════════════════════════════════════════════════════════
_CONTROLS = ["rv_es_l1", "rv_spy_l1", "state_l1", "d_wspr_es_l1", "d_wspr_spy_l1"]


def _day_design(b, n_lags):
    """Per-day design block: within-day rows for the two equations. Returns a DataFrame
    with dz_flow, dz_ret, own-lag columns for each DV, and LAGGED standardizable controls;
    rows with any non-finite entry are dropped by the caller AFTER demeaning."""
    d = pd.DataFrame(index=b.index)
    d["dz_flow"] = b["z_flow"].diff()
    d["dz_ret"] = b["z_ret"].diff()
    for i in range(1, n_lags + 1):
        d[f"dz_flow_l{i}"] = d["dz_flow"].shift(i)
        d[f"dz_ret_l{i}"] = d["dz_ret"].shift(i)
    d["rv_es_l1"] = b["rv_es"].shift(1)
    d["rv_spy_l1"] = b["rv_spy"].shift(1)
    d["state_l1"] = b["state"].shift(1)
    d["d_wspr_es_l1"] = b["d_wspr_es"].shift(1)
    d["d_wspr_spy_l1"] = b["d_wspr_spy"].shift(1)
    return d


def _fe_panel(bars, n_lags):
    """Stack per-day designs: standardize the controls POOLED (per-1-SD coefficients),
    demean EVERY column within day (day FE), drop non-finite rows. Returns (frame, day)."""
    frames, days = [], []
    for date, _r, b in bars:
        d = _day_design(b, n_lags)
        d["__day"] = str(date)
        frames.append(d)
    if not frames:
        return pd.DataFrame(), np.array([])
    big = pd.concat(frames, axis=0, ignore_index=True)
    for c in _CONTROLS:
        v = big[c].to_numpy(float)
        sd = np.nanstd(v)
        big[c] = (v - np.nanmean(v)) / (sd if sd > EPS else 1.0)
    cols = [c for c in big.columns if c != "__day"]
    big[cols] = big.groupby("__day")[cols].transform(lambda s: s - s.mean())
    m = np.all(np.isfinite(big[cols].to_numpy(float)), axis=1)
    return big.loc[m, cols].reset_index(drop=True), big.loc[m, "__day"].to_numpy()


def _ols_cluster(y, X, day):
    import price_discovery_shares as pds
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtXinv = np.linalg.pinv(X.T @ X)
    beta = XtXinv @ (X.T @ y)
    u = y - X @ beta
    se, G = pds._cluster_se(X, u, day, XtXinv)
    return beta, se, G


def _norm_p(t):
    from inference import _norm_cdf
    return float(2.0 * (1.0 - _norm_cdf(abs(t)))) if np.isfinite(t) else np.nan


def mediation_from_bars(bars, n_lags=3, n_boot=499, seed=0):
    """Tier 2 core on prepared per-day bars -> dict of the three regressions plus the
    RV_ES mediation decomposition with a day-level cluster-bootstrap CI.

    Eq A : dz_flow ~ FE + dz_flow lags + controls(L1)
    Eq B0: dz_ret  ~ FE + dz_ret lags + controls(L1)            (total)
    Eq B1: dz_ret  ~ FE + dz_ret lags + controls(L1) + dz_flow  (direct + mediator)
    indirect(RV_ES) = beta_total - beta_direct; share = indirect / beta_total."""
    panel, day = _fe_panel(bars, n_lags)
    if panel.empty or len(panel) < 10 * (n_lags + len(_CONTROLS)):
        raise ValueError("not enough finite bar rows for the mediation panel")
    la_f = [f"dz_flow_l{i}" for i in range(1, n_lags + 1)]
    la_r = [f"dz_ret_l{i}" for i in range(1, n_lags + 1)]

    def _fit(frame, dd):
        Xa = frame[la_f + _CONTROLS].to_numpy(float)
        ba, sa, _ = _ols_cluster(frame["dz_flow"], Xa, dd)
        Xb0 = frame[la_r + _CONTROLS].to_numpy(float)
        b0, s0, _ = _ols_cluster(frame["dz_ret"], Xb0, dd)
        Xb1 = frame[la_r + _CONTROLS + ["dz_flow"]].to_numpy(float)
        b1, s1, _ = _ols_cluster(frame["dz_ret"], Xb1, dd)
        j = len(la_f) + _CONTROLS.index("rv_es_l1")          # rv_es_l1 slot (same in B0)
        tot = b0[j]
        dire = b1[len(la_r) + _CONTROLS.index("rv_es_l1")]
        return ba, sa, b0, s0, b1, s1, float(tot), float(dire)

    ba, sa, b0, s0, b1, s1, tot, dire = _fit(panel, day)
    indirect = tot - dire
    share = indirect / tot if abs(tot) > EPS else np.nan

    rng = np.random.default_rng(seed)
    uniq = np.unique(day)
    draws = []
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        parts, dd = [], []
        for gi, dsel in enumerate(pick):
            sub = panel[day == dsel]
            parts.append(sub)
            dd.append(np.full(len(sub), gi))
        try:
            *_x, t_b, d_b = _fit(pd.concat(parts, ignore_index=True), np.concatenate(dd))
            draws.append((t_b - d_b, (t_b - d_b) / t_b if abs(t_b) > EPS else np.nan))
        except Exception:
            continue
    draws = np.asarray(draws, float) if draws else np.empty((0, 2))
    ci = (np.nanpercentile(draws[:, 0], [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
    ci_share = (np.nanpercentile(draws[:, 1], [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
    return {"lags_flow": la_f, "lags_ret": la_r, "controls": _CONTROLS,
            "eqA": (ba, sa), "eqB_total": (b0, s0), "eqB_direct": (b1, s1),
            "rv_es_total": tot, "rv_es_direct": dire, "indirect": float(indirect),
            "share": float(share) if np.isfinite(share) else np.nan,
            "ci_indirect": (float(ci[0]), float(ci[1])),
            "ci_share": (float(ci_share[0]), float(ci_share[1])),
            "n_rows": int(len(panel)), "n_days": int(len(uniq)), "n_boot_ok": int(len(draws))}


def table_flow_corr_mediation(sessions, bar_seconds=60, n_levels=10, min_rest_steps=0,
                              ar_order=5, n_lags=3, n_boot=499, seed=0, bars=None):
    """Tier 2 -> (DataFrame, notes). Columns: Eq A (d z_flow), Eq B total (d z_ret without
    the mediator), Eq B direct (with it). Coefficients x100, per 1 SD of each control."""
    bars = per_day_bars(sessions, bar_seconds, n_levels, min_rest_steps, ar_order) \
        if bars is None else bars
    res = mediation_from_bars(bars, n_lags=n_lags, n_boot=n_boot, seed=seed)

    def _cell(bvec, svec, j):
        b, s = 100.0 * bvec[j], 100.0 * svec[j]
        p = _norm_p(bvec[j] / svec[j]) if svec[j] > EPS else np.nan
        star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{b:.3f}{star} ({s:.3f})"

    la = len(res["lags_flow"])
    rows = {}
    for i, c in enumerate(res["controls"]):
        rows[c] = {
            "Eq A: d z_flow": _cell(*res["eqA"], la + i),
            "Eq B: d z_ret (total)": _cell(*res["eqB_total"], la + i),
            "Eq B: d z_ret (direct)": _cell(*res["eqB_direct"], la + i)}
    rows["d z_flow (mediator)"] = {
        "Eq A: d z_flow": "--", "Eq B: d z_ret (total)": "--",
        "Eq B: d z_ret (direct)": _cell(*res["eqB_direct"], la + len(res["controls"]))}
    lo, hi = res["ci_indirect"]
    lo_s, hi_s = res["ci_share"]
    rows["indirect (RV_ES via z_flow)"] = {
        "Eq A: d z_flow": "--", "Eq B: d z_ret (total)": "--",
        "Eq B: d z_ret (direct)": "%.3f [%.3f, %.3f]" % (100 * res["indirect"], 100 * lo, 100 * hi)}
    rows["indirect share of RV_ES effect"] = {
        "Eq A: d z_flow": "--", "Eq B: d z_ret (total)": "--",
        "Eq B: d z_ret (direct)": ("%.2f [%.2f, %.2f]" % (res["share"], lo_s, hi_s)
                                   if np.isfinite(res["share"]) else "--")}
    df = pd.DataFrame(rows).T
    df.index.name = "regressor"
    notes = ("Bar-level FE panel (%d days, %d rows, %ds bars): every column demeaned within "
             "day, controls standardized pooled (coefficients x100 per 1 SD), day-clustered "
             "SEs in parentheses. Signed OFI is excluded by construction and all controls "
             "enter at lag 1 (same-bar RV shares sub-returns with the same-bar correlation "
             "estimate). Indirect effect and share via day-level cluster bootstrap "
             "(%d/%d draws usable), 95%% percentile CI."
             % (res["n_days"], res["n_rows"], bar_seconds, res["n_boot_ok"], n_boot))
    return df, notes


# ══════════════════════════════════════════════════════════════════════════════
#  Data-driven regimes: Markov-switching with a bootstrap LR
# ══════════════════════════════════════════════════════════════════════════════
def fit_ar1_gauss(y):
    """1-state null: y_t = mu + phi y_{t-1} + eps, eps ~ N(0, s2). Returns dict + loglik."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    b, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
    e = y[1:] - X @ b
    s2 = max(float(e @ e) / len(e), EPS)
    ll = float(-0.5 * len(e) * (np.log(2 * np.pi * s2) + 1.0))
    return {"mu": float(b[0]), "phi": float(b[1]), "s2": s2, "ll": ll, "n": int(len(e))}


def fit_ms_ar1(y, K=2, max_iter=120, tol=1e-5, n_init=2, seed=0):
    """K-state Markov-switching regression on a 1-D series: switching intercept and
    variance, COMMON AR(1) coefficient. EM with the Hamilton filter / Kim smoother from
    markov_switching_vecm. States are sorted by long-run mean mu_k/(1-phi) ascending.

    EM converges to LOCAL optima, and an asymmetry in optimization quality between the
    observed series and the bootstrap-simulated null series biases the LR test (a lucky
    optimum on the data against unlucky ones on the sims reads as evidence for more
    regimes). ``n_init`` runs the EM from the deterministic quantile init plus perturbed
    restarts (seeded, so results stay reproducible) and keeps the best likelihood --
    applied identically to data and sims.

    Returns dict(mu[K], phi, s2[K], P[KxK], ll, smoothed[T-1,K], means[K])."""
    y = np.asarray(y, float)
    ylag = y[:-1]
    yt = y[1:]
    T = len(yt)
    if T < 30:
        raise ValueError("series too short for MS fit")
    h0 = fit_ar1_gauss(y)
    rng = np.random.default_rng(seed + 7 * K)
    best = None
    for j in range(max(1, int(n_init))):
        f = _em_ms_ar1(yt, ylag, K, h0, max_iter, tol, rng if j else None)
        if best is None or f["ll"] > best["ll"]:
            best = f
    return best


def _em_ms_ar1(yt, ylag, K, h0, max_iter, tol, rng):
    """One EM run. ``rng`` None = deterministic quantile init; else perturbed restart."""
    T = len(yt)
    # init: split at the median level (quantiles for K > 2), common AR from the 1-state fit
    phi = h0["phi"]
    med = np.median(yt)
    mu = np.array([np.mean(yt[yt <= med]), np.mean(yt[yt > med])]) * (1 - phi)
    if K != 2:
        qs = np.quantile(yt, np.linspace(0, 1, K + 1))
        mu = np.array([(qs[k] + qs[k + 1]) / 2 for k in range(K)]) * (1 - phi)
    s2 = np.full(K, max(h0["s2"], EPS))
    P = np.full((K, K), 0.1 / max(K - 1, 1))
    np.fill_diagonal(P, 0.9)
    if rng is not None:
        mu = mu + rng.normal(scale=0.5 * float(np.std(yt)) * max(1 - phi, 1e-3), size=K)
        d = rng.uniform(0.85, 0.98, size=K)
        P = np.tile(((1 - d) / max(K - 1, 1))[:, None], (1, K))
        P[np.arange(K), np.arange(K)] = d
        s2 = s2 * rng.uniform(0.5, 2.0, size=K)
    ll_prev = -np.inf
    smooth = np.full((T, K), 1.0 / K)
    ll = ll_prev
    for _ in range(int(max_iter)):
        resid = yt[:, None] - (mu[None, :] + phi * ylag[:, None])
        logdens = -0.5 * (np.log(2 * np.pi * s2)[None, :] + resid ** 2 / s2[None, :])
        pi0 = msv._ergodic(P)
        filt, pred, ll = msv._hamilton(logdens, P, pi0)
        smooth, trans_num, _ssum = msv._kim(filt, pred, P)
        w = np.maximum(smooth, EPS)
        # M-step: joint weighted LS for (mu_1..mu_K, phi) -- normal equations over the
        # stacked per-regime weighted copies of [e_k, ylag].
        A = np.zeros((K + 1, K + 1))
        rhs = np.zeros(K + 1)
        for k in range(K):
            wk = w[:, k] / s2[k]
            A[k, k] = wk.sum()
            A[k, K] = A[K, k] = float(wk @ ylag)
            A[K, K] += float(wk @ (ylag * ylag))
            rhs[k] = float(wk @ yt)
            rhs[K] += float(wk @ (ylag * yt))
        try:
            sol = np.linalg.solve(A + np.eye(K + 1) * EPS, rhs)
        except np.linalg.LinAlgError:                        # pragma: no cover - defensive
            break
        mu = sol[:K]
        phi = float(np.clip(sol[K], -0.999, 0.999))
        resid = yt[:, None] - (mu[None, :] + phi * ylag[:, None])
        floor = max(EPS, 1e-8 * float(np.var(yt)))
        s2 = np.maximum((w * resid ** 2).sum(0) / np.maximum(w.sum(0), EPS), floor)
        P = trans_num / np.maximum(trans_num.sum(axis=1, keepdims=True), EPS)
        P = np.clip(P, 1e-6, 1 - 1e-6)
        P /= P.sum(axis=1, keepdims=True)
        if abs(ll - ll_prev) < tol * (1 + abs(ll_prev)):
            break
        ll_prev = ll
    means = mu / max(1.0 - phi, 1e-3)
    order = np.argsort(means)
    return {"mu": mu[order], "phi": phi, "s2": s2[order], "P": P[np.ix_(order, order)],
            "ll": float(ll), "smoothed": smooth[:, order], "means": means[order],
            "n": int(T)}


def _sim_ar1(f1, T, rng):
    """Simulate T observations of the fitted 1-state AR(1)."""
    sd = np.sqrt(f1["s2"])
    ys = np.empty(T)
    denom = max(1.0 - f1["phi"] ** 2, 1e-3)
    ys[0] = f1["mu"] / max(1.0 - f1["phi"], 1e-3) + rng.standard_normal() * sd / np.sqrt(denom)
    eps = rng.standard_normal(T - 1) * sd
    for t in range(1, T):
        ys[t] = f1["mu"] + f1["phi"] * ys[t - 1] + eps[t - 1]
    return ys


def simulate_ms(fit, T, rng):
    """Simulate T observations from a fitted MS-AR(1) fit dict (K >= 2 states)."""
    P = np.asarray(fit["P"], float)
    K = P.shape[0]
    mu = np.asarray(fit["mu"], float)
    phi = float(fit["phi"])
    sd = np.sqrt(np.asarray(fit["s2"], float))
    s = int(rng.choice(K, p=msv._ergodic(P)))
    y = np.empty(T)
    y[0] = mu[s] / max(1.0 - phi, 1e-3) + sd[s] * rng.standard_normal()
    u = rng.random(T)
    e = rng.standard_normal(T)
    cum = np.cumsum(P, axis=1)
    for t in range(1, T):
        s = int(np.searchsorted(cum[s], u[t]))
        s = min(s, K - 1)
        y[t] = mu[s] + phi * y[t - 1] + sd[s] * e[t]
    return y


def ms_lr_test(y, B=99, seed=0, max_iter=80, k0=1):
    """Bootstrap LR of MS(k0+1)-AR(1) against the k0-state null (k0=1 is plain AR(1)).
    The extra state's mixing weight is unidentified under H0 (Davies), so the null LR
    distribution is SIMULATED from the fitted null model rather than read off a
    chi-square table.

    Returns dict(lr, p_boot, fit1, fit2) -- fit1 the null-K fit, fit2 the K+1 fit."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    k0 = int(k0)
    f1 = fit_ar1_gauss(y) if k0 == 1 else fit_ms_ar1(y, K=k0, max_iter=max_iter)
    f2 = fit_ms_ar1(y, K=k0 + 1, max_iter=max_iter)
    lr = max(0.0, 2.0 * (f2["ll"] - f1["ll"]))
    rng = np.random.default_rng(seed)
    T = len(y)
    hits = 0
    for _ in range(int(B)):
        ys = _sim_ar1(f1, T, rng) if k0 == 1 else simulate_ms(f1, T, rng)
        try:
            g1 = fit_ar1_gauss(ys) if k0 == 1 else fit_ms_ar1(ys, K=k0, max_iter=max_iter)
            g2 = fit_ms_ar1(ys, K=k0 + 1, max_iter=max_iter)
            if max(0.0, 2.0 * (g2["ll"] - g1["ll"])) >= lr - EPS:
                hits += 1
        except Exception:                                    # pragma: no cover - defensive
            hits += 1                                        # conservative: count as >= lr
    return {"lr": lr, "p_boot": (hits + 1) / (int(B) + 1), "fit1": f1, "fit2": f2}


def select_k_ms(y, k_max=4, B=99, seed=0, alpha=0.05, max_iter=80):
    """Let the data choose the NUMBER of regimes: sequential bootstrap LR, testing K vs
    K+1 from K=1 upward and stopping at the first non-rejection (or at k_max). Nothing
    presupposes the levels either -- the fitted regime means land wherever the data puts
    them (zero, weakly negative, strongly positive, ...).

    Rejection is STRICT (p_boot < alpha), and the bootstrap must be able to reach it:
    with B draws the smallest possible p is 1/(B+1), so B >= 1/alpha is required for the
    test to have any power (B=99 at alpha=0.05 comfortably qualifies; B=19 cannot reject).

    Returns dict(k, fit, p_path): the chosen K, its fit normalized to the fit_ms_ar1
    shape (a 1-state fit gets means/P/smoothed stubs), and the per-stage bootstrap p."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    p_path = []
    chosen, fit = 1, None
    for k0 in range(1, int(k_max)):
        r = ms_lr_test(y, B=B, seed=seed + 101 * k0, max_iter=max_iter, k0=k0)
        p_path.append(float(r["p_boot"]))
        if not (r["p_boot"] < alpha):
            chosen = k0
            fit = r["fit1"] if k0 > 1 else None
            break
        chosen = k0 + 1
        fit = r["fit2"]
    if chosen == 1:
        f = fit_ar1_gauss(y)
        m = f["mu"] / max(1.0 - f["phi"], 1e-3)
        fit = {"mu": np.array([f["mu"]]), "phi": f["phi"], "s2": np.array([f["s2"]]),
               "P": np.array([[1.0]]), "ll": f["ll"], "means": np.array([m]),
               "smoothed": np.ones((max(len(y) - 1, 1), 1)), "n": max(len(y) - 1, 1)}
    return {"k": int(chosen), "fit": fit, "p_path": p_path}


def ms_flow_regimes(sessions, bar_seconds=60, n_levels=10, min_rest_steps=0, ar_order=5,
                    B=99, seed=0, min_bars=60, k_max=4, bars=None, verbose=False,
                    return_series=False):
    """Per-day Markov switching on the z_flow level series with the NUMBER of regimes
    chosen by sequential bootstrap LR (select_k_ms) -> per-day records (list of dict).
    With ``return_series`` also returns {date: Series} of the smoothed top-regime
    probability on the bar clock, for downstream conditioning (the PD link)."""
    bars = per_day_bars(sessions, bar_seconds, n_levels, min_rest_steps, ar_order,
                        min_bars=min_bars, verbose=verbose) if bars is None else bars
    recs, series = [], {}
    for i, (date, regime, b) in enumerate(bars):
        zfull = b["z_flow"].to_numpy(float)
        fin = np.isfinite(zfull)
        z = zfull[fin]
        if len(z) < min_bars:
            continue
        try:
            sel = select_k_ms(z, k_max=k_max, B=B, seed=seed + i)
        except Exception as e:                               # pragma: no cover - defensive
            if verbose:
                print(f"  [ms] {date}: SKIPPED ({type(e).__name__}: {e})")
            continue
        f = sel["fit"]
        k = sel["k"]
        p1 = sel["p_path"][0] if sel["p_path"] else np.nan   # the 1-vs-2 stage
        occ_hi = float(f["smoothed"][:, -1].mean()) if k >= 2 else np.nan
        # regime-dynamics asymmetry (K >= 2, top vs bottom state): smoothed[t] describes
        # finite-bar t+1 (the AR consumes one observation). Entries = upward 0.5-crossings
        # of the top-state probability; the entry-direction statistic is the mean signed
        # SPY bar return in the bar BEFORE each entry, demeaned by the day's own mean bar
        # return so a trending day does not masquerade as asymmetric entry.
        entry_ret, n_entries = np.nan, 0
        if k >= 2:
            p_hi = f["smoothed"][:, -1]
            ret = b["ret_spy"].to_numpy(float)[fin][1:]      # aligned with smoothed
            cross = np.where((p_hi[1:] >= 0.5) & (p_hi[:-1] < 0.5))[0] + 1
            pre = ret[cross - 1] if len(cross) else np.array([])
            pre = pre[np.isfinite(pre)]
            day_mu = float(np.nanmean(ret)) if np.isfinite(ret).any() else np.nan
            entry_ret = float(pre.mean() - day_mu) if len(pre) else np.nan
            n_entries = int(len(cross))
        means = [float(m) for m in np.atleast_1d(f["means"])]
        recs.append({"date": str(date), "regime": str(regime), "k": int(k),
                     "p_1v2": float(p1) if np.isfinite(p1) else np.nan,
                     "p_path": "/".join("%.3f" % p for p in sel["p_path"]),
                     "means": means,
                     "corr_levels": [float(np.tanh(m)) for m in means],
                     "corr_lo": float(np.tanh(means[0])),
                     "corr_hi": float(np.tanh(means[-1])),
                     "p_stay_lo": float(f["P"][0, 0]) if k >= 2 else np.nan,
                     "p_stay_hi": float(f["P"][-1, -1]) if k >= 2 else np.nan,
                     "occ_hi": occ_hi, "n_entries": n_entries,
                     "entry_ret": entry_ret, "n_bars": int(f["n"] + 1)})
        if return_series:
            bar_idx = b.index[fin][1:]
            series[str(date)] = pd.Series(f["smoothed"][:, -1] if k >= 2
                                          else np.full(len(bar_idx), np.nan), index=bar_idx)
    return (recs, series) if return_series else recs


def _modal_levels(g):
    """Median regime-correlation levels across the group's days at the group's MODAL K,
    rendered 'x / y / z'."""
    ks = g.k.to_numpy(int)
    if len(ks) == 0:
        return "--"
    km = int(pd.Series(ks).mode().iloc[0])
    levels = [r for r, k in zip(g.corr_levels, ks) if k == km]
    if not levels:
        return "--"
    M = np.array(levels, float)
    return " / ".join("%.2f" % v for v in np.median(M, axis=0))


def table_flow_corr_ms_regimes(sessions, bar_seconds=60, n_levels=10, min_rest_steps=0,
                               ar_order=5, B=99, seed=0, min_bars=60, k_max=4, bars=None,
                               recs=None):
    """Data-driven regimes -> (DataFrame, notes). One row per a-priori day group plus an
    all-days row. NOTHING is presupposed: the number of regimes per day comes from the
    sequential bootstrap LR (1 vs 2 vs 3 vs ...), and the regime levels land wherever the
    data puts them -- zero, weakly negative, or strongly positive."""
    if recs is None:
        recs = ms_flow_regimes(sessions, bar_seconds, n_levels, min_rest_steps, ar_order,
                               B=B, seed=seed, min_bars=min_bars, k_max=k_max, bars=bars)
    if not recs:
        return pd.DataFrame(), "no usable sessions"
    per = pd.DataFrame(recs)
    groups = _regime_order(per.regime) + ["all days"]
    out = {}
    for gname in groups:
        g = per if gname == "all days" else per[per.regime == gname]
        if g.empty:
            continue
        kc = pd.Series(g.k.to_numpy(int)).value_counts().sort_index()
        kdist = " ".join(f"K={k}:{n}" for k, n in kc.items())
        m2 = g.k >= 2
        dur_lo = 1.0 / np.maximum(1.0 - g.loc[m2, "p_stay_lo"], 1e-6)
        dur_hi = 1.0 / np.maximum(1.0 - g.loc[m2, "p_stay_hi"], 1e-6)
        ldur = np.log(dur_hi.to_numpy(float)) - np.log(dur_lo.to_numpy(float))
        er = g.entry_ret.to_numpy(float)
        er = er[np.isfinite(er)]
        out[gname] = {
            "days by chosen K": kdist,
            "regime corr levels (modal K)": _modal_levels(g),
            "share of bars in top regime": float(g.occ_hi.mean()) if m2.any() else np.nan,
            "median duration lo/hi (bars)": ("%.0f / %.0f" % (float(dur_lo.median()),
                                                              float(dur_hi.median()))
                                             if m2.any() else "--"),
            "duration asymmetry p": sign_flip_p(ldur, seed=1) if m2.any() else np.nan,
            "entry ret (bps, demeaned)": float(er.mean()) if len(er) else np.nan,
            "entry-direction p": sign_flip_p(er, seed=2) if len(er) else np.nan,
            "days multi-regime": "%d of %d" % (int(m2.sum()), len(g)),
            "median p (1 vs 2)": float(g.p_1v2.median())}
    df = pd.DataFrame(out).T
    df.index.name = "day group"
    notes = ("Per-day Markov-switching regression on the bar-level z_flow series (switching "
             "intercept and variance, common AR(1)); the NUMBER of regimes is chosen per day "
             "by sequential PARAMETRIC-BOOTSTRAP LR (K vs K+1 from K=1, %d draws per stage; "
             "the extra state's mixing weight is unidentified under H0, so chi-square "
             "calibration is invalid), capped at K=%d. States sorted by long-run mean; "
             "levels are reported exactly where the data puts them (zero / negative levels "
             "included). Occupancy is the mean smoothed probability of the TOP state; "
             "durations are 1/(1-P_kk). ASYMMETRY columns: 'duration asymmetry p' "
             "sign-flips the per-day log duration ratio (top vs bottom persistence); "
             "'entry ret' is the mean signed SPY bar return in the bar BEFORE each entry "
             "into the top regime, demeaned by the day's own mean bar return, with its "
             "day-level sign-flip p -- negative = the high-tandem regime is entered on "
             "selling." % (B, k_max))
    return df, notes


# ══════════════════════════════════════════════════════════════════════════════
#  Flow-correlation regimes -> price discovery (the payoff link)
# ══════════════════════════════════════════════════════════════════════════════
def window_pd_panel(sessions, bars_map, series_map=None, window_minutes=30, n_lags=5):
    """Within-day windows: price-discovery shares next to the tandem-flow state.

    For each ``window_minutes`` block of each session: CS_ES and IS_mid_ES from the
    fixed-(1,-1) VECM on the window's mids (price_discovery_shares.estimate_day, with its
    ec_valid flag), the window mean z_flow, and the window mean smoothed top-regime
    probability (when the day's chosen K >= 2). Returns a DataFrame, one row per window."""
    import price_discovery_shares as pds
    rows = []
    for date, regime, df in sessions:
        b = bars_map.get(str(date))
        if b is None:
            continue
        occ = (series_map or {}).get(str(date))
        m1 = np.asarray(ca._mid(df, "SPY"), float)
        m2 = np.asarray(ca._mid(df, "ES"), float)
        wid = df.index.floor("%dmin" % int(window_minutes))
        dt = float(np.median(np.diff(df.index.asi8))) / 1e9 if len(df) > 1 else 1.0
        need = 0.5 * window_minutes * 60.0 / max(dt, 1e-9)
        zb = b["z_flow"]
        wbar = zb.index.floor("%dmin" % int(window_minutes))
        for w in pd.unique(wid):
            sel = wid == w
            if sel.sum() < need:
                continue
            mm1, mm2 = m1[sel], m2[sel]
            if np.isfinite(mm1).sum() < need or np.isfinite(mm2).sum() < need:
                continue
            try:
                est = pds.estimate_day(mm1, mm2, n_lags=n_lags)
            except Exception:
                continue
            zv = zb[wbar == w].to_numpy(float)
            zmean = float(np.nanmean(zv)) if np.isfinite(zv).any() else np.nan
            ow = np.nan
            if occ is not None and len(occ):
                ov = occ[occ.index.floor("%dmin" % int(window_minutes)) == w].to_numpy(float)
                ow = float(np.nanmean(ov)) if np.isfinite(ov).any() else np.nan
            rows.append({"date": str(date), "regime": str(regime), "window": str(w),
                         "CS_ES": float(est["CS_ES"]), "IS_mid_ES": float(est["IS_mid_ES"]),
                         "ec_valid": bool(est["ec_valid"]), "z_flow": zmean,
                         "occ_top": ow})
    return pd.DataFrame(rows)


def _fe_reg_windows(panel, dv, xcol, standardize=True):
    """Within-day (FE) bivariate regression on the window panel with day-clustered SE.
    Returns (beta, se, p, n, G) -- beta per 1 SD of x when standardize else per unit."""
    d = panel[["date", dv, xcol]].copy()
    d = d[np.isfinite(d[dv].astype(float)) & np.isfinite(d[xcol].astype(float))]
    if len(d) < 12 or d.date.nunique() < 4:
        return (np.nan,) * 3 + (int(len(d)), int(d.date.nunique()))
    x = d[xcol].to_numpy(float)
    if standardize:
        sd = x.std()
        x = (x - x.mean()) / (sd if sd > EPS else 1.0)
    d["_x"] = x
    dm = d.groupby("date")[[dv, "_x"]].transform(lambda s: s - s.mean())
    y = dm[dv].to_numpy(float)
    X = dm["_x"].to_numpy(float)[:, None]
    if float(np.abs(X).sum()) < EPS:
        return (np.nan,) * 3 + (int(len(d)), int(d.date.nunique()))
    beta, se, G = _ols_cluster(y, X, d.date.to_numpy())
    t = beta[0] / se[0] if se[0] > EPS else np.nan
    return float(beta[0]), float(se[0]), _norm_p(t), int(len(d)), int(G)


def _pd_link_table(panel):
    """Assemble the PD-link table from a window panel (factored for the gate test)."""
    def _cell(b, s, p):
        if not np.isfinite(b):
            return "--"
        star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{b:.4f}{star} ({s:.4f})"

    cs_panel = panel[panel.ec_valid]
    rows = {}
    for label, xcol, std in (("z_flow (per 1 SD)", "z_flow", True),
                             ("top-regime occupancy (0 to 1)", "occ_top", False)):
        cells = {}
        for cname, pnl, dv in (("CS_ES (ec_valid windows)", cs_panel, "CS_ES"),
                               ("IS_mid_ES (all windows)", panel, "IS_mid_ES")):
            b, s, p, n, G = _fe_reg_windows(pnl, dv, xcol, standardize=std)
            cells[cname] = _cell(b, s, p)
        rows[label] = cells
    hi = panel.assign(_hi=(panel.occ_top.astype(float) > 0.5).astype(float))
    hi = hi[np.isfinite(hi.occ_top.astype(float))]
    cells = {}
    for cname, pnl, dv in (("CS_ES (ec_valid windows)", hi[hi.ec_valid], "CS_ES"),
                           ("IS_mid_ES (all windows)", hi, "IS_mid_ES")):
        b, s, p, n, G = _fe_reg_windows(pnl, dv, "_hi", standardize=False)
        cells[cname] = _cell(b, s, p)
    rows["top-regime majority window (vs rest)"] = cells
    n_cs = int(len(cs_panel))
    rows["windows / days"] = {
        "CS_ES (ec_valid windows)": "%d / %d" % (n_cs, cs_panel.date.nunique()),
        "IS_mid_ES (all windows)": "%d / %d" % (len(panel), panel.date.nunique())}
    df = pd.DataFrame(rows).T
    df.index.name = "regressor"
    return df


def table_flow_pd_link(sessions, bar_seconds=60, n_levels=10, min_rest_steps=0,
                       ar_order=5, window_minutes=30, n_lags=5, B=99, seed=0,
                       min_bars=60, k_max=4, bars=None, recs=None, series_map=None):
    """Does tandem flow move PRICE DISCOVERY? -> (DataFrame, notes). Window-level FE
    panel: within-day windows' CS/IS regressed on the window's mean z_flow and on the
    window's smoothed top-regime occupancy (from the chosen-K Markov fit), plus a
    top-regime-majority contrast. Day FE + day-clustered SEs throughout."""
    bars = per_day_bars(sessions, bar_seconds, n_levels, min_rest_steps, ar_order) \
        if bars is None else bars
    if series_map is None:
        _recs, series_map = ms_flow_regimes(sessions, bar_seconds, n_levels,
                                            min_rest_steps, ar_order, B=B, seed=seed,
                                            min_bars=min_bars, k_max=k_max, bars=bars,
                                            return_series=True)
    bars_map = {str(d): b for d, _r, b in bars}
    panel = window_pd_panel(sessions, bars_map, series_map, window_minutes, n_lags)
    if panel.empty or len(panel) < 12:
        return pd.DataFrame(), "not enough usable windows"
    df = _pd_link_table(panel)
    notes = ("Within-day %d-minute windows: CS_ES / IS_mid_ES from the fixed-(1,-1) VECM "
             "on each window's mids (lag %d), regressed on the window's tandem-flow state "
             "with day fixed effects (within-day demeaning) and day-clustered SEs. "
             "'occupancy' is the window mean smoothed probability of the day's TOP "
             "flow-correlation regime (chosen-K Markov fit; K=1 days carry no occupancy "
             "and drop from those rows). CS rows use ec_valid windows only. A positive "
             "coefficient = price discovery migrates toward ES when tandem flow "
             "intensifies." % (window_minutes, n_lags))
    return df, notes


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test (synthetic; no vendor data)
# ══════════════════════════════════════════════════════════════════════════════
def _selftest():                                             # pragma: no cover - CLI entry
    import test_flow_correlation as t
    rc = t.main()
    raise SystemExit(rc)


if __name__ == "__main__":                                   # pragma: no cover
    _selftest()
