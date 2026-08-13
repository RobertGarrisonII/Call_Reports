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
    Per-day 2-state Markov-switching regression on the z_flow level series,
        z_t = mu_{s_t} + phi * z_{t-1} + eps_t,   eps_t ~ N(0, sigma^2_{s_t}),
    (switching intercept and variance, common AR(1)), against the 1-state AR(1) null.
    The mixing weight is unidentified under H0 (Davies problem), so the LR is calibrated
    by PARAMETRIC BOOTSTRAP: simulate the fitted null, refit both models, count. Smoothed
    probabilities assign bars to regimes; the table reports the regimes the data chooses
    and how their occupancy lines up with the a-priori day labels -- no arbitrary ranges.

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


def ofi_innovations(df, asset, n_levels=10, min_rest_steps=0, ar_order=5):
    """OFI innovations for one leg: Cont-Kukanov-Stoikov OFI, NaN'd wherever the leg's own
    mid is non-finite (order_flow_imbalance zero-fills masked rows, which would let halt
    bars pass the coverage floor as fake zeros), then AR-prefiltered."""
    ofi = np.asarray(ca.order_flow_imbalance(df, asset, n_levels, min_rest_steps), float)
    a1 = df[f"{asset}_askprice_1"].to_numpy(float)
    b1 = df[f"{asset}_bidprice_1"].to_numpy(float)
    ofi[~(np.isfinite(a1) & np.isfinite(b1))] = np.nan
    return ar_innovations(ofi, ar_order=ar_order)


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
        n_pairs  finite OFI-innovation pairs inside the bar

    Same coverage discipline as the RealBar dependent variable: a bar must contain at
    least max(10, 0.25 * capacity) finite pairs, so partially masked (halt-boundary) bars
    drop out rather than entering as attenuated fakes."""
    idx = df.index
    bar = idx.floor("%ds" % int(bar_seconds))
    dt = float(np.median(np.diff(idx.asi8))) / 1e9 if len(idx) > 1 else 1.0
    cap = max(1.0, float(bar_seconds) / max(dt, 1e-9))
    min_sub = max(10, int(0.25 * cap))

    u_spy = ofi_innovations(df, "SPY", n_levels, min_rest_steps, ar_order)
    u_es = ofi_innovations(df, "ES", n_levels, min_rest_steps, ar_order)
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
        "state": pd.Series(np.asarray(ca.relative_depth_state(df, n_levels), float), index=idx),
        "wspr_spy": pd.Series(cs.weighted_spread(df, "SPY", None, n_levels, wspread_kind), index=idx),
        "wspr_es": pd.Series(cs.weighted_spread(df, "ES", None, n_levels, wspread_kind), index=idx),
    })
    g = aux.groupby(bar)
    for c in ("rv_spy", "rv_es", "state"):
        B[c] = g[c].mean().reindex(B.index)
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


def fit_ms_ar1(y, K=2, max_iter=200, tol=1e-6):
    """K-state Markov-switching regression on a 1-D series: switching intercept and
    variance, COMMON AR(1) coefficient. EM with the Hamilton filter / Kim smoother from
    markov_switching_vecm. States are sorted by long-run mean mu_k/(1-phi) ascending.

    Returns dict(mu[K], phi, s2[K], P[KxK], ll, smoothed[T-1,K], means[K])."""
    y = np.asarray(y, float)
    ylag = y[:-1]
    yt = y[1:]
    T = len(yt)
    if T < 30:
        raise ValueError("series too short for MS fit")
    # init: split at the median level, common AR from the 1-state fit
    h0 = fit_ar1_gauss(y)
    phi = h0["phi"]
    med = np.median(yt)
    mu = np.array([np.mean(yt[yt <= med]), np.mean(yt[yt > med])]) * (1 - phi)
    if K != 2:
        qs = np.quantile(yt, np.linspace(0, 1, K + 1))
        mu = np.array([(qs[k] + qs[k + 1]) / 2 for k in range(K)]) * (1 - phi)
    s2 = np.full(K, max(h0["s2"], EPS))
    P = np.full((K, K), 0.1 / max(K - 1, 1))
    np.fill_diagonal(P, 0.9)
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


def ms_lr_test(y, B=99, seed=0, max_iter=120):
    """Bootstrap LR of MS(2)-AR(1) against 1-state AR(1). The mixing weight is
    unidentified under H0 (Davies), so the null LR distribution is SIMULATED from the
    fitted AR(1) rather than read off a chi-square table.

    Returns dict(lr, p_boot, fit1, fit2)."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    f1 = fit_ar1_gauss(y)
    f2 = fit_ms_ar1(y, K=2, max_iter=max_iter)
    lr = max(0.0, 2.0 * (f2["ll"] - f1["ll"]))
    rng = np.random.default_rng(seed)
    T = len(y)
    sd = np.sqrt(f1["s2"])
    hits = 0
    for _ in range(int(B)):
        ys = np.empty(T)
        denom = max(1.0 - f1["phi"] ** 2, 1e-3)
        ys[0] = f1["mu"] / max(1.0 - f1["phi"], 1e-3) + rng.standard_normal() * sd / np.sqrt(denom)
        eps = rng.standard_normal(T - 1) * sd
        for t in range(1, T):
            ys[t] = f1["mu"] + f1["phi"] * ys[t - 1] + eps[t - 1]
        try:
            g1 = fit_ar1_gauss(ys)
            g2 = fit_ms_ar1(ys, K=2, max_iter=max_iter)
            if max(0.0, 2.0 * (g2["ll"] - g1["ll"])) >= lr - EPS:
                hits += 1
        except Exception:                                    # pragma: no cover - defensive
            hits += 1                                        # conservative: count as >= lr
    return {"lr": lr, "p_boot": (hits + 1) / (int(B) + 1), "fit1": f1, "fit2": f2}


def ms_flow_regimes(sessions, bar_seconds=60, n_levels=10, min_rest_steps=0, ar_order=5,
                    B=99, seed=0, min_bars=60, bars=None, verbose=False):
    """Per-day MS(2) on the z_flow level series -> per-day records (list of dict)."""
    bars = per_day_bars(sessions, bar_seconds, n_levels, min_rest_steps, ar_order,
                        min_bars=min_bars, verbose=verbose) if bars is None else bars
    recs = []
    for i, (date, regime, b) in enumerate(bars):
        z = b["z_flow"].to_numpy(float)
        z = z[np.isfinite(z)]
        if len(z) < min_bars:
            continue
        try:
            r = ms_lr_test(z, B=B, seed=seed + i)
        except Exception as e:                               # pragma: no cover - defensive
            if verbose:
                print(f"  [ms] {date}: SKIPPED ({type(e).__name__}: {e})")
            continue
        f2 = r["fit2"]
        occ_hi = float(f2["smoothed"][:, -1].mean())
        recs.append({"date": str(date), "regime": str(regime), "lr": r["lr"],
                     "p_boot": r["p_boot"], "mean_lo": float(f2["means"][0]),
                     "mean_hi": float(f2["means"][-1]),
                     "corr_lo": float(np.tanh(f2["means"][0])),
                     "corr_hi": float(np.tanh(f2["means"][-1])),
                     "p_stay_lo": float(f2["P"][0, 0]), "p_stay_hi": float(f2["P"][-1, -1]),
                     "occ_hi": occ_hi, "n_bars": int(f2["n"] + 1)})
    return recs


def table_flow_corr_ms_regimes(sessions, bar_seconds=60, n_levels=10, min_rest_steps=0,
                               ar_order=5, B=99, seed=0, min_bars=60, bars=None):
    """Data-driven regimes -> (DataFrame, notes). One row per a-priori day group plus an
    all-days row; the regimes INSIDE each day come from the Markov-switching fit, not from
    any constructed range."""
    recs = ms_flow_regimes(sessions, bar_seconds, n_levels, min_rest_steps, ar_order,
                           B=B, seed=seed, min_bars=min_bars, bars=bars)
    if not recs:
        return pd.DataFrame(), "no usable sessions"
    per = pd.DataFrame(recs)
    groups = _regime_order(per.regime) + ["all days"]
    out = {}
    for gname in groups:
        g = per if gname == "all days" else per[per.regime == gname]
        if g.empty:
            continue
        dur_lo = 1.0 / np.maximum(1.0 - g.p_stay_lo, 1e-6)
        dur_hi = 1.0 / np.maximum(1.0 - g.p_stay_hi, 1e-6)
        out[gname] = {
            "low-regime corr (median)": float(g.corr_lo.median()),
            "high-regime corr (median)": float(g.corr_hi.median()),
            "median duration lo/hi (bars)": "%.0f / %.0f" % (float(dur_lo.median()),
                                                             float(dur_hi.median())),
            "share of bars in high regime": float(g.occ_hi.mean()),
            "days 2-regime p<0.05": "%d of %d" % (int((g.p_boot < 0.05).sum()), len(g)),
            "median boot p": float(g.p_boot.median())}
    df = pd.DataFrame(out).T
    df.index.name = "day group"
    notes = ("Per-day 2-state Markov-switching regression on the bar-level z_flow series "
             "(switching intercept and variance, common AR(1)); states sorted by long-run "
             "mean. Significance is a PARAMETRIC BOOTSTRAP LR against the 1-state AR(1) "
             "null (%d draws per day; the mixing weight is unidentified under H0, so "
             "chi-square calibration is invalid). Occupancy is the mean smoothed "
             "probability of the high-tandem state; durations are 1/(1-P_kk)." % B)
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
