"""
correlation_svar.py
==================
Reproduce **Table 9** ("Impulse Response of Price Return Correlation") of Garrison,
Jain & Paddrik -- but in the spread-based, depth-frame-native framework of this stack,
and offer the structurally cleaner replacement.

Table 9 is the SVAR of Eq. (5): the response of the change in price-return correlation
to order-flow / liquidity shocks. The paper proxies liquidity with message-type
proportions (LS/LD/cancel). Here we instead use SPREADS computed from the book:

  * standard quoted spread   :  a1 - b1                          (bps of mid)
  * weighted spread          :  depth-weighted round-trip cost-to-fill
                                slippage_ask(Q) + slippage_bid(Q) (bps), or a
                                size-weighted half-spread average
  * informational proxies    :  multi-level OFI (book pressure, replacing the crude
                                NCMOF/PCMOF message states), realized variance, and the
                                microprice-minus-mid deviation

Two things matter methodologically (see the module functions):

  1. Dependent variable. The paper differences a 100-bar ROLLING return correlation.
     We support that (corr_method='rolling'), the Epps-robust Hayashi-Yoshida correlation
     over the same window (corr_method='hy'), and the DCC conditional correlation
     (corr_method='dcc', via dcc_garch) which has no look-ahead-window artifact.
     The paper samples at one second AND ten milliseconds; grid-sampled Pearson is
     attenuated by asynchronous quote updating at both, severely at the latter, and the
     attenuation moves with trading intensity -- which is on the right-hand side.

  2. Identification. The paper uses an identity shock covariance (no real
     orthogonalization) with correlation ordered first -- which would zero the
     contemporaneous effects it reports. We provide ident='cholesky' (correlation
     ordered LAST, recursive, futures-lead) and ident='identity' (the paper's shortcut,
     Theta_h = Psi_h scaled by own SD), so the gap is visible.

The *better* object is `spread_conditioned_is`: rather than regress d-correlation on
spread, it makes the (weighted) spread the conditioning state S in the information-share
curve IS(S) via ecm_sde -- the state-dependent price-discovery object this stack is built
around. Pure numpy/pandas (+ optional dcc_garch).
"""
import numpy as np
import pandas as pd

import liquidity_curve_metrics as lcm
import cross_asset_pd_liquidity as ca
import price_discovery_shares as pds
import robust_prices as rp
import ecm_sde as es

EPS = 1e-12


# ── spread observables (from the depth frame) ─────────────────────────────────
def quoted_spread(df, asset):
    """Standard quoted BBO spread (a1 - b1), in basis points of the mid."""
    a1 = df[f"{asset}_askprice_1"].to_numpy(); b1 = df[f"{asset}_bidprice_1"].to_numpy()
    mid = (a1 + b1) / 2.0
    return (a1 - b1) / np.where(mid > EPS, mid, np.nan) * 1e4


def _default_target(qa, qb):
    """A fill size that walks past level 1: median cumulative depth over the first
    min(3, N) levels (so the weighted spread is a genuine multi-level cost)."""
    k = min(3, qa.shape[1])
    cum = np.nanmedian(np.nansum(qa[:, :k], axis=1) + np.nansum(qb[:, :k], axis=1)) / 2.0
    return float(max(cum, 1.0))


def weighted_spread(df, asset, target_qty=None, n_levels=10, kind="cost_to_fill"):
    """Depth-weighted spread (bps of mid). kind='cost_to_fill' (default) is the
    round-trip effective spread to trade `target_qty` per side (walks the book via
    slippage_to_fill); kind='depth_weighted' is the size-weighted half-spread average
    a*q summed over levels. Both >= the quoted spread (deeper levels are worse)."""
    da, qa, mid = lcm.extract_book(df, asset, "ask", n_levels, unit="pct")    # pct of mid
    db, qb, _ = lcm.extract_book(df, asset, "bid", n_levels, unit="pct")
    if kind == "depth_weighted":
        wa = np.nansum(lcm._clean(qa) * da, axis=1) / np.maximum(np.nansum(lcm._clean(qa), axis=1), EPS)
        wb = np.nansum(lcm._clean(qb) * db, axis=1) / np.maximum(np.nansum(lcm._clean(qb), axis=1), EPS)
        return (wa + wb) * 1e4
    Q = _default_target(qa, qb) if target_qty is None else float(target_qty)
    sa = lcm.slippage_to_fill(da, qa, Q)                                       # avg dist to fill Q (pct)
    sb = lcm.slippage_to_fill(db, qb, Q)
    return (sa + sb) * 1e4


def microprice_dev(df, asset, n_levels=5):
    """Microprice minus mid, in bps -- a sub-tick directional-pressure / information proxy."""
    mp = rp.microprice(df, asset, n_levels)
    a1 = df[f"{asset}_askprice_1"].to_numpy(); b1 = df[f"{asset}_bidprice_1"].to_numpy()
    mid = (a1 + b1) / 2.0
    return (mp - mid) / np.where(mid > EPS, mid, np.nan) * 1e4


def _returns_bps(df, asset):
    mid = np.asarray(ca._mid(df, asset), float)
    r = np.diff(np.log(np.where(mid > EPS, mid, np.nan))) * 1e4
    return np.concatenate([[np.nan], r])                                       # align to df.index


# ── Epps-robust dependent variable (Hayashi-Yoshida) ──────────────────────────
# The paper's Delta-rho is the first difference of a Pearson correlation of returns sampled on
# a fixed grid, at one second AND at ten milliseconds. Pearson on a fixed grid is exactly the
# estimator the Epps effect attacks: SPY and ES do not update at the same instants, so within a
# short bar one leg's mid is often stale, the grid records a zero return for it, and the measured
# correlation is pulled toward zero. The bias grows as the bar shrinks -- which is why measured
# correlation rises with the sampling interval at all (that IS the Epps curve).
#
# This matters for Eq. (5) beyond precision. The severity of the attenuation depends on how
# often each leg updates, i.e. on trading intensity -- and volume, message traffic and liquidity
# demand are all REGRESSORS in the same system. So part of any estimated "activity raises
# correlation" relationship is the measurement error moving with the regressor, not a market
# mechanism. Hayashi-Yoshida is consistent for the covariance of asynchronously observed
# processes and uses no grid at all, so it removes that channel.
#
# On a bar grid the asynchronicity is recoverable: a bar in which an asset's mid did not move is
# precisely a bar in which that asset did not update, and its zero return is the artifact. Taking
# each asset's own mid-CHANGE times as its observation times restores the irregular, asynchronous
# sampling HY is designed for.
def _hy_rolling_corr(mid_spy, mid_es, window):
    """Rolling Hayashi-Yoshida correlation of two mid series observed on a shared bar grid.

    Each asset contributes only the bars where its own mid actually moved, so no synchronizing
    grid is imposed and stale-quote zero returns cannot attenuate the estimate. Returns a
    length-T array (leading NaN until `window` bars of history) of

        rho_HY(t) = HY_cov(t-window, t] / sqrt( RV_SPY * RV_ES )

    where RV is each leg's own tick-by-tick realized variance over the same span. Each
    HY interval-pair product is attributed to the LAST bar of the two intervals it spans, so
    the window ending at t uses no information dated after t (no look-ahead), and the rolling
    sums are exact cumulative sums -- O(n + m + T) overall, not O(T * window).

    Note this is a realized correlation (no demeaning), whereas `rolling().corr()` demeans.
    Over a 100-bar window of near-zero-mean returns the difference is negligible; the Epps
    correction is orders of magnitude larger."""
    T = len(mid_spy)
    W = int(window)
    out = np.full(T, np.nan)
    if T < W + 2 or W < 2:
        return out
    lp = {}
    for nm, m in (("SPY", mid_spy), ("ES", mid_es)):
        v = np.asarray(m, float)
        v = np.log(np.where(v > EPS, v, np.nan))
        i = np.flatnonzero(np.isfinite(v))
        if i.size < 3:
            return out
        # keep only bars where THIS asset's mid moved: its genuine observation times
        keep = np.concatenate([[i[0]], i[1:][np.abs(np.diff(v[i])) > 0]])
        lp[nm] = (keep.astype(np.int64), v[keep])
    (i1, p1), (i2, p2) = lp["SPY"], lp["ES"]
    if i1.size < 3 or i2.size < 3:
        return out
    r1 = np.diff(p1); a1, b1 = i1[:-1], i1[1:]
    r2 = np.diff(p2); a2, b2 = i2[:-1], i2[1:]

    cov_at = np.zeros(T)                       # HY cross products, attributed to a bar
    v1_at = np.zeros(T); v2_at = np.zeros(T)   # each leg's realized variance, per bar
    np.add.at(v1_at, b1, r1 * r1)
    np.add.at(v2_at, b2, r2 * r2)
    m, lo = len(r2), 0
    for k in range(len(r1)):                   # two-pointer HY sweep (O(n+m), as in noise_robust_cov)
        while lo < m and b2[lo] <= a1[k]:
            lo += 1
        j = lo
        bk, rk = b1[k], r1[k]
        while j < m and a2[j] < bk:
            cov_at[bk if bk >= b2[j] else b2[j]] += rk * r2[j]     # attribute to the later end
            j += 1

    def _roll(x):
        c = np.concatenate([[0.0], np.cumsum(x)])
        s = np.full(T, np.nan)
        s[W:] = c[W + 1:] - c[1:T - W + 1]
        return s
    C, V1, V2 = _roll(cov_at), _roll(v1_at), _roll(v2_at)
    den = np.sqrt(np.where((V1 > EPS) & (V2 > EPS), V1 * V2, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = C / den
    return np.clip(rho, -1.0, 1.0)


# ── variable assembly (spec progression: standard -> weighted -> informational) ─
PRESETS = {
    "standard":      ["qspr"],
    "weighted":      ["qspr", "wspr"],
    "informational": ["qspr", "wspr", "ofi", "rv", "mdev"],
}
_DIFF = {"qspr", "wspr"}                         # liquidity levels enter as changes (as in the paper)
_LABEL = {"qspr": "Spread", "wspr": "WtdSpread", "ofi": "OFI", "rv": "RV", "mdev": "MicroDev"}


def build_svar_frame(df, spec="informational", n_levels=10, target_qty=None,
                     corr_method="rolling", corr_window=100, wspread_kind="cost_to_fill",
                     dcc_returns=None, extra_fn=None, bar_seconds=60):
    """Assemble the per-bar SVAR variables in CAUSAL order (futures ES block, then
    equities SPY block, any `extra_fn(df)` columns, then the d-correlation LHS last).
    Liquidity levels are differenced; OFI/RV/MicroDev and extra columns enter as
    (stationary) levels. Returns (X[T', k], names, corr_idx) after dropping incomplete
    rows. `extra_fn(df) -> DataFrame` injects continuous cross-flow regressors (e.g.
    cross_flow.cross_flow_features(df)[["PCMOF","NCMOF"]]) right before the LHS.

    corr_method='rolling' -> d of the `corr_window`-bar return correlation (the paper).
    corr_method='hy'      -> d of the `corr_window`-bar HAYASHI-YOSHIDA correlation. Same
                             window, but asynchronicity-robust: immune to the Epps attenuation
                             that biases grid-sampled Pearson toward zero, and whose severity
                             moves with trading intensity -- itself a regressor in this system.
                             Report Table 9 both ways; the gap IS the measurement artifact.
    corr_method='dcc'     -> d of the DCC conditional correlation (pass dcc_returns or it
                             is fit here from the two return series).
    corr_method='bar'     -> the system RE-BARRED to ``bar_seconds`` bars with the dependent
                             variable the change in the Fisher-z of each bar's REALIZED
                             correlation (Pearson over the sub-returns INSIDE the bar).
                             Non-overlapping bars share no data, so differencing manufactures
                             no MA term at any lag -- the fixed-width-window artifact that made
                             the selected order track corr_window cannot exist here by
                             construction. Fisher-z because realized rho sits near 0.92: atanh
                             unbounds the support and makes a shock on a calm day commensurate
                             with one on a crash day. Regressors aggregate to the bar: OFI and
                             cross-flow counts SUM, RV becomes the bar's own realized variance
                             (mean squared sub-return -- also window-free, replacing the rolling
                             RV), spread levels bar-MEAN then differenced, MicroDev bar-MEAN.
                             On a 10ms frame the sub-returns are the fine grid: the two-grid
                             design exists exactly so this estimator has them."""
    feats = PRESETS[spec] if isinstance(spec, str) else list(spec)
    # Single-slot memo (v0.9.64): the day-cluster bootstrap resamples WHOLE sessions and
    # rebuilds every drawn session's frame, so with corr_method='dcc' the GARCH was re-fit
    # once per session per draw -- n_boot fits per estimator block, hours at 1s and weeks at
    # 10ms, all returning the identical series. The session frame OBJECTS are shared across
    # draws, so the last build is cached on df.attrs; one slot bounds extra residency at a
    # single X per session while every draw inside an estimator block hits. Callers treat the
    # returned X as read-only (all downstream ops -- X[:, keep], vstack, resampling -- copy).
    key = (tuple(feats), int(n_levels), target_qty, str(corr_method), int(corr_window),
           str(wspread_kind), int(bar_seconds), id(extra_fn) if extra_fn is not None else None)
    cacheable = dcc_returns is None
    if cacheable:
        hit = df.attrs.get("_svar_frame_memo")
        if hit is not None and hit[0] == key:
            return hit[1]
    idx = df.index
    cols = {}
    r = {a: _returns_bps(df, a) for a in ("SPY", "ES")}
    if corr_method == "bar":
        out = _build_svar_frame_bar(df, feats, n_levels, target_qty, wspread_kind,
                                    bar_seconds, extra_fn, r)
        if cacheable:
            df.attrs["_svar_frame_memo"] = (key, out)
        return out
    for a in ("ES", "SPY"):                                                    # futures first
        for f in feats:
            if f == "qspr":
                cols[(a, f)] = quoted_spread(df, a)
            elif f == "wspr":
                cols[(a, f)] = weighted_spread(df, a, target_qty, n_levels, wspread_kind)
            elif f == "ofi":
                cols[(a, f)] = ca.order_flow_imbalance(df, a, n_levels)
            elif f == "rv":
                rr = pd.Series(r[a], index=idx)
                cols[(a, f)] = (rr ** 2).rolling(corr_window, min_periods=corr_window // 2).mean().to_numpy()
            elif f == "mdev":
                cols[(a, f)] = microprice_dev(df, a, min(n_levels, 5))
    V = pd.DataFrame({f"{_LABEL[f]}_{a}": v for (a, f), v in cols.items()}, index=idx)
    for (a, f) in cols:                                                         # difference liquidity levels
        if f in _DIFF:
            name = f"{_LABEL[f]}_{a}"
            V[name] = V[name].diff()
    if extra_fn is not None:                                                    # inject cross-flow regressors
        E = extra_fn(df).reindex(idx)
        for c in E.columns:
            V[c] = np.asarray(E[c], float)
    # dependent: change in return correlation
    if corr_method == "dcc":
        rho = _dcc_corr(df, r, dcc_returns)
    elif corr_method == "hy":                                                   # Epps-robust
        rho = _hy_rolling_corr(ca._mid(df, "SPY"), ca._mid(df, "ES"), corr_window)
    else:
        rho = pd.Series(r["SPY"], index=idx).rolling(corr_window).corr(pd.Series(r["ES"], index=idx))
    V["dCorr"] = pd.Series(np.asarray(rho), index=idx).diff()
    names = list(V.columns)                                                    # dCorr is last
    X = V.to_numpy()
    mask = np.all(np.isfinite(X), axis=1)
    out = (X[mask], names, len(names) - 1)
    if cacheable:
        df.attrs["_svar_frame_memo"] = (key, out)
    return out


def _fisher_z(rho):
    """atanh with the boundary clamped: a bar where every sub-return pair agrees exactly
    (rho -> +-1, possible on a quiet bar of identical ticks) must map to a large finite z,
    not inf."""
    return np.arctanh(np.clip(np.asarray(rho, float), -1.0 + 1e-8, 1.0 - 1e-8))


def _build_svar_frame_bar(df, feats, n_levels, target_qty, wspread_kind, bar_seconds,
                          extra_fn, r):
    """The corr_method='bar' assembly: one row per NON-OVERLAPPING ``bar_seconds`` bar.
    Same causal ordering contract as the native-grid path (ES block, SPY block, extras,
    dependent last); returns (X, names, corr_idx)."""
    idx = df.index
    bar = idx.floor("%ds" % int(bar_seconds))
    V = pd.DataFrame(index=idx)
    for a in ("ES", "SPY"):                                                    # futures first
        for f in feats:
            name = f"{_LABEL[f]}_{a}"
            if f == "qspr":
                V[name] = quoted_spread(df, a)
            elif f == "wspr":
                V[name] = weighted_spread(df, a, target_qty, n_levels, wspread_kind)
            elif f == "ofi":
                V[name] = ca.order_flow_imbalance(df, a, n_levels)
            elif f == "rv":
                V[name] = pd.Series(r[a], index=idx) ** 2                      # per-bar mean below
            elif f == "mdev":
                V[name] = microprice_dev(df, a, min(n_levels, 5))
    bar_sum = set()                                    # extra_fn columns aggregated as per-bar sums
    if extra_fn is not None:
        E = extra_fn(df).reindex(idx)
        for c in E.columns:
            V[c] = np.asarray(E[c], float)
            bar_sum.add(c)                                                     # counts: sum per bar
    # Coverage floor scales with the GRID, not just the bar. bar_seconds//6 was calibrated on
    # the 1s grid (60 sub-intervals per 60s bar -> require 10); on a 10ms frame the same
    # constant admits bars with 10 of 6,000 sub-returns, where se(z) ~ 0.38 produces the
    # +-9 Fisher-z outliers. Require a quarter of the bar's sub-interval capacity, floor 10.
    dt = float(np.median(np.diff(idx.asi8))) / 1e9 if len(idx) > 1 else 1.0
    cap = max(1.0, float(bar_seconds) / max(dt, 1e-9))
    min_sub = max(10, int(0.25 * cap))
    g = V.groupby(bar)
    n_fin = V.notna().groupby(bar).sum()               # per-bar finite counts, per column
    B = pd.DataFrame(index=pd.Index(sorted(bar.unique()), name="bar"))
    for c in V.columns:
        base = c.split("_")[0]
        if c in bar_sum or base in ("OFI",):
            B[c] = g[c].sum(min_count=1)
            # A partially-masked bar (halt boundary) SUMS only its observed sub-intervals; for a
            # FLOW column that is structural attenuation correlated with exactly the stressed
            # sessions. Require at least half the bar's sub-interval capacity observed. This
            # floor is for SUM columns only: mean-aggregated state columns (spreads, RV, mdev)
            # are unbiased under within-bar sparsity -- WtdSpread is ~25%-finite at the native
            # grid by construction, and flooring it would NaN most bars outright.
            B.loc[n_fin[c].reindex(B.index).fillna(0) < 0.5 * cap, c] = np.nan
        else:                                                                  # RV / spreads / mdev
            B[c] = g[c].mean()
    for c in list(B.columns):                                                  # spread LEVELS -> changes
        if c.split("_")[0] in ("Spread", "WtdSpread"):
            B[c] = B[c].diff()
    # dependent: change in Fisher-z of the bar's realized correlation of the sub-returns.
    # A bar needs enough finite pairs and movement on both legs to yield a correlation at all.
    rs = pd.Series(r["SPY"], index=idx); re_ = pd.Series(r["ES"], index=idx)
    rho = []
    for b, ii in rs.groupby(bar).groups.items():
        x = rs.loc[ii].to_numpy(float); y = re_.loc[ii].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_sub and x[m].std() > EPS and y[m].std() > EPS:
            rho.append((b, float(np.corrcoef(x[m], y[m])[0, 1])))
        else:
            rho.append((b, np.nan))
    z = pd.Series(dict(rho)).reindex(B.index)
    B["dCorr"] = pd.Series(_fisher_z(z), index=B.index).diff() if z.notna().any() else np.nan
    names = list(B.columns)
    X = B.to_numpy(float)
    mask = np.all(np.isfinite(X), axis=1)
    return X[mask], names, len(names) - 1


def _dcc_corr(df, r, dcc_returns):
    """DCC conditional correlation series (length len(df), leading NaN). Falls back to
    NaN if dcc_garch is unavailable or the fit fails.

    The fitted rho depends ONLY on the frame's returns, not on corr_window or the regressor
    spec, so it memoizes on df.attrs independently of the frame memo above -- a spec or
    window change re-assembles the frame but must not re-fit the GARCH."""
    if dcc_returns is None:
        hit = df.attrs.get("_dcc_rho_memo")
        if hit is not None and len(hit) == len(df):
            return hit
    try:
        import dcc_garch as dg
        ret = dcc_returns if dcc_returns is not None else np.column_stack(
            [np.diff(np.log(np.asarray(ca._mid(df, "SPY"), float))), np.diff(np.log(np.asarray(ca._mid(df, "ES"), float)))])
        fit = dg.dcc_garch_x(ret)
        out = np.concatenate([[np.nan], np.asarray(fit["rho"], float)])
    except Exception:
        out = np.full(len(df), np.nan)
    if dcc_returns is None:
        df.attrs["_dcc_rho_memo"] = out
    return out


# ── panel VAR design: within-day lags + day fixed effects ─────────────────────
def _panel_var_design(Xs, p, fe=True, start=None):
    """Stack per-session VAR designs with lags built STRICTLY within-day and (with ``fe``)
    each session demeaned -- day fixed effects via the within transformation, which at
    T ~ 23,400 bars per day carries an O(1/T) Nickell bias of no consequence.

    The naive alternative this replaces -- np.vstack the sessions, then lag the stack --
    let the first p bars of every session borrow the PREVIOUS session's close as their
    "lags" (an overnight seam treated as one grid step) and pooled days around a single
    intercept, conflating within-day dynamics with between-day level shifts (a 2020 MWCB
    day and a 2023 baseline day differ in RV/spread LEVELS by orders of magnitude).

    ``start`` fixes the first usable row per session (>= p), so lag-order candidates can
    share a common sample. Returns (Y, Z, groups) with NO intercept column under fe."""
    st = int(p if start is None else max(start, p))
    Ys, Zs, Gs = [], [], []
    for g, X in enumerate(Xs):
        X = np.asarray(X, float); T = X.shape[0]
        if T <= st + 2:
            continue
        Xd = X - X.mean(axis=0) if fe else X
        cols = [] if fe else [np.ones(T - st)]
        for L in range(1, p + 1):
            cols.append(Xd[st - L:T - L])
        if not cols:
            cols = [np.ones(T - st)]                # p=0 with fe: pure-mean model, intercept-free
            Ys.append(Xd[st:]); Zs.append(np.zeros((T - st, 0))); Gs.append(np.full(T - st, g))
            continue
        Ys.append(Xd[st:]); Zs.append(np.column_stack(cols)); Gs.append(np.full(T - st, g))
    if not Ys:
        return None, None, None
    return np.vstack(Ys), np.vstack(Zs), np.concatenate(Gs)


def _panel_gram(Xs, p, fe=True, start=None, chunk=200_000):
    """Accumulated cross-products (ZtZ, ZtY, YtY, n, k, n_sessions) of the exact design
    ``_panel_var_design`` builds -- WITHOUT materializing it. The stacked design is rows x
    (k*p [+1]); at a 10ms grid that is tens of GB per fit (and every concurrent bootstrap
    draw built its own copy). The Grams are (k*p)^2 regardless of rows; the only O(rows)
    work is chunked matmuls over views of each session's array. Column layout matches
    _panel_var_design exactly ([intercept +] lag-1 block .. lag-p block), which
    select_lag_var_panel exploits: the design for p' < p is a leading-column subset."""
    st = int(p if start is None else max(start, p))
    ZtZ = ZtY = YtY = None
    n_rows = 0; n_sess = 0; k = None
    for X in Xs:
        X = np.asarray(X, float); T = X.shape[0]
        if T <= st + 2:
            continue
        Xd = X - X.mean(axis=0) if fe else X
        k = Xd.shape[1]
        m = (0 if fe else 1) + k * p
        if ZtZ is None:
            ZtZ = np.zeros((m, m)); ZtY = np.zeros((m, k)); YtY = np.zeros((k, k))
        n_g = T - st
        for c0 in range(0, n_g, chunk):
            c1 = min(c0 + chunk, n_g)
            Y = Xd[st + c0: st + c1]
            cols = [] if fe else [np.ones(c1 - c0)]
            for L in range(1, p + 1):
                cols.append(Xd[st - L + c0: st - L + c1])
            Z = np.column_stack(cols) if cols else np.zeros((c1 - c0, 0))
            ZtZ += Z.T @ Z
            ZtY += Z.T @ Y
            YtY += Y.T @ Y
        n_rows += n_g; n_sess += 1
    return ZtZ, ZtY, YtY, n_rows, k, n_sess


def _gram_solve(ZtZ, ZtY, YtY):
    """OLS slope + residual cross-product from Grams. -> (B, rss) with
    rss = (Y-ZB)'(Y-ZB), symmetrized (the full quadratic form, exact under pinv too).

    Solved under symmetric Jacobi equilibration: the Gram squares the design's condition
    number, and mixed-unit columns (price levels vs counts vs bps returns) push kappa toward
    the float64 cliff. Scaling by D = diag(sqrt(diag(ZtZ))) and solving
    (D^-1 ZtZ D^-1) x~ = D^-1 ZtY with x = D^-1 x~ is mathematically identical, is within a
    factor k of the optimal diagonal preconditioner (van der Sluis), and reproduces the
    unscaled solve exactly on well-conditioned designs."""
    if ZtZ is None or ZtZ.shape[0] == 0:
        return np.zeros((0, YtY.shape[0])), YtY.copy()
    d = np.sqrt(np.clip(np.diag(ZtZ), EPS, None))
    Zs = ZtZ / d[:, None] / d[None, :]
    Ys = ZtY / d[:, None]
    try:
        B = np.linalg.solve(Zs, Ys) / d[:, None]
    except np.linalg.LinAlgError:
        B = (np.linalg.pinv(Zs) @ Ys) / d[:, None]
    rss = YtY - ZtY.T @ B - B.T @ ZtY + B.T @ ZtZ @ B
    return B, (rss + rss.T) / 2.0


def panel_var_ols(Xs, p, fe=True, return_resid=False, chunk=200_000):
    """Pooled VAR(p) on the panel design (within-day lags, day FE). Returns
    (A list, Sigma, resid, groups) -- the same contract as ``var_ols`` plus the day id
    per residual row, so day-cluster machinery can consume it.

    v0.9.64: estimated from per-session accumulated Grams (see ``_panel_gram``) so the
    stacked rows x k*p design is never materialized. ``resid``/``groups`` are None unless
    ``return_resid=True`` (they are the only O(rows) output; every current caller discards
    them, and each concurrent bootstrap draw used to hold its own multi-GB design)."""
    ZtZ, ZtY, YtY, n, k, n_sess = _panel_gram(Xs, p, fe=fe, chunk=chunk)
    if n == 0 or k is None:
        raise ValueError("no session long enough for the requested lag order")
    B, rss = _gram_solve(ZtZ, ZtY, YtY)
    off = 0 if fe else 1
    A = [B[off + (i - 1) * k: off + i * k, :].T for i in range(1, p + 1)]
    m = ZtZ.shape[0] if ZtZ is not None else 0
    dof = max(n - m - (n_sess * k if fe else 0) // max(k, 1), 1)
    Sigma = rss / dof
    resid = groups = None
    if return_resid:
        st = p
        rs, gs = [], []
        for g, X in enumerate(Xs):
            X = np.asarray(X, float); T = X.shape[0]
            if T <= st + 2:
                continue
            Xd = X - X.mean(axis=0) if fe else X
            cols = [] if fe else [np.ones(T - st)]
            for L in range(1, p + 1):
                cols.append(Xd[st - L:T - L])
            Z = np.column_stack(cols) if cols else np.zeros((T - st, 0))
            rs.append(Xd[st:] - (Z @ B if Z.shape[1] else 0.0))
            gs.append(np.full(T - st, g))
        resid = np.vstack(rs); groups = np.concatenate(gs)
    return A, Sigma, resid, groups


def select_lag_var_panel(Xs, pmax=12, criterion="bic", fe=True, chunk=200_000):
    """Lag-order selection on the PANEL design: every candidate scored on the common
    sample (rows pmax.. per session), lags within-day, day FE. The stacked-selection it
    replaces scored candidates whose lag windows crossed the overnight seams.

    v0.9.64: the Grams are accumulated ONCE at pmax and every candidate p is scored by
    taking the leading ([intercept +] p*k)-column submatrices -- the common-sample design
    for p is exactly that subset. One pass over the data instead of pmax+1 stacked builds
    (which at a 10ms grid transiently held tens of GB per candidate)."""
    ZtZ, ZtY, YtY, n, k, _ = _panel_gram(Xs, pmax, fe=fe, start=pmax, chunk=chunk)
    rows = []
    off = 0 if fe else 1
    for p in range(0, pmax + 1):
        if n == 0 or k is None:
            rows.append((p, np.nan, np.nan, np.nan, np.nan)); continue
        m = off + k * p
        if n < m + 5:
            rows.append((p, np.nan, np.nan, np.nan, np.nan)); continue
        if m == 0:
            Sig = YtY / n
        else:
            _B, rss = _gram_solve(ZtZ[:m, :m], ZtY[:m], YtY)
            Sig = rss / n
        sign, ld = np.linalg.slogdet(Sig + np.eye(k) * EPS)
        if sign <= 0:
            rows.append((p, np.nan, np.nan, np.nan, np.nan)); continue
        K = m * k                                    # free slope params across equations
        aic = ld + 2.0 * K / n
        bic = ld + np.log(n) * K / n
        hq = ld + 2.0 * np.log(np.log(max(n, 3))) * K / n
        rows.append((p, ld, aic, bic, hq))
    tab = pd.DataFrame(rows, columns=["p", "logdet", "aic", "bic", "hqic"]).set_index("p")
    col = tab[{"aic": "aic", "hq": "hqic", "hqic": "hqic"}.get(str(criterion).lower(), "bic")]
    p_star = int(col.idxmin()) if col.notna().any() else None
    return p_star, tab


# ── reduced-form VAR(p) + orthogonalized IRF ──────────────────────────────────
def var_ols(X, p):
    """Reduced-form VAR(p): X_t = c + sum_{i=1}^p A_i X_{t-i} + u_t. Returns
    (A list of (k,k), Sigma (k,k) residual covariance, residuals)."""
    T, k = X.shape
    Y = X[p:]
    Z = np.column_stack([np.ones(T - p)] + [X[p - i:T - i] for i in range(1, p + 1)])
    B, resid = pds._ols(Y, Z)
    A = [B[1 + (i - 1) * k: 1 + i * k, :].T for i in range(1, p + 1)]
    dof = max(resid.shape[0] - Z.shape[1], 1)
    Sigma = (resid.T @ resid) / dof
    return A, Sigma, resid


def _ma_from_var(A, H):
    """Wold MA matrices Psi_0..Psi_H: Psi_0 = I, Psi_h = sum_{i=1}^{min(h,p)} A_i Psi_{h-i}."""
    k = A[0].shape[0]; p = len(A)
    Psi = [np.eye(k)]
    for h in range(1, H + 1):
        S = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            S += A[i - 1] @ Psi[h - i]
        Psi.append(S)
    return Psi


def orthogonalized_irf(A, Sigma, H, ident="cholesky"):
    """Orthogonalized IRF Theta_0..Theta_H (each (k,k)); Theta_h[j,m] = response of
    variable j to a 1-SD structural shock to variable m at horizon h, in the variable
    order of X. ident='cholesky' uses the recursive lower-triangular factor (so with the
    target ordered last, every shock hits it contemporaneously); ident='identity' is the
    paper's diagonal-covariance shortcut (own-SD scaling, no cross-orthogonalization)."""
    if ident == "identity":
        L = np.diag(np.sqrt(np.clip(np.diag(Sigma), 0.0, None)))
    else:
        S = Sigma + np.eye(Sigma.shape[0]) * EPS
        L = np.linalg.cholesky(S)
    Psi = _ma_from_var(A, H)
    return [Psi[h] @ L for h in range(H + 1)]


# ── Table 9: IRF of correlation to each shock, by regime ──────────────────────
def sieve_order(T) -> int:
    """The Lewis-Reinsel/Lütkepohl sieve rate ceil(T^(1/3)) -- the principled ceiling for 'let the
    lag order grow with the sample'. A sieve VAR approximates an infinite-order system consistently
    when p grows at this rate; a criterion proposing p far beyond it on a fixed sample is not
    finding structure, it is chasing something a finite VAR cannot whiten (in the Table 9 system:
    the corr_window MA spike). T=23,400 gives 29 -- context for the paper's 60."""
    return int(np.ceil(float(max(T, 1)) ** (1.0 / 3.0)))


def _lp_core(y, shock, Z, horizons, nw_extra=5):
    """Local projection of y_{t+h} on shock_t with controls Z_t, Newey-West SEs. -> rows of dicts.

    The property that matters here is LAG-ROBUSTNESS: theta_h is a direct projection, so it does
    not require the system to be whitened -- controls soak up predictable variation for efficiency,
    but omitting lags does not bias theta_h the way truncating a VAR biases every IRF step after
    the first (Jorda 2005; Plagborg-Moller & Wolf 2021 give the population equivalence). That is
    the answer to a boundary-constrained lag order: estimate the IRF by an object whose point
    estimate does not depend on getting p right."""
    y = np.asarray(y, float); shock = np.asarray(shock, float)
    T = len(y)
    out = []
    for h in horizons:
        yy = y[h:] if h else y.copy()
        n = len(yy)
        cols = [np.ones(n), shock[:T - h] if h else shock]
        if Z is not None and Z.shape[1]:
            cols.append(Z[:T - h] if h else Z)
        Xh = np.column_stack([np.column_stack([c]) if c.ndim == 1 else c for c in cols])
        ok = np.isfinite(yy) & np.all(np.isfinite(Xh), axis=1)
        yy, Xh = yy[ok], Xh[ok]
        if len(yy) <= Xh.shape[1] + 5:
            out.append({"horizon": int(h), "theta": np.nan, "se": np.nan})
            continue
        b, res, *_ = np.linalg.lstsq(Xh, yy, rcond=None)
        u = yy - Xh @ b
        L = int(h) + int(nw_extra)                       # overlap induces MA(h) errors by design
        XtXi = np.linalg.pinv(Xh.T @ Xh)
        w = Xh * u[:, None]
        S = w.T @ w
        for j in range(1, L + 1):
            G = w[j:].T @ w[:-j]
            S += (1 - j / (L + 1)) * (G + G.T)
        V = XtXi @ S @ XtXi
        out.append({"horizon": int(h), "theta": float(b[1]), "se": float(np.sqrt(max(V[1, 1], 0)))})
    return out


def correlation_irf_lp(sessions, spec="informational", n_levels=10, target_qty=None,
                       corr_method="rolling", corr_window=100, wspread_kind="cost_to_fill",
                       extra_fn=None, horizons=range(0, 7), ctrl_lags=2):
    """Table 9's IRF estimated by LOCAL PROJECTIONS -- the lag-robust alternative to the VAR.

    Same frame, same shocks, same dependent variable as `correlation_irf`; the difference is that
    each horizon's response is a direct HAC-inferenced projection, so the estimate does not depend
    on a lag order at all -- `ctrl_lags` only tunes efficiency, and moving it should not move the
    point estimates. That invariance is a CHECKABLE property (test_lag_informative pins it), unlike
    a boundary-constrained p*. Shocks are standardized per regime so theta is per 1 SD, matching
    the VAR table's units.

    -> tidy DataFrame [regime, shock, horizon, theta, se] (x100, as in Table 9)."""
    groups = {}
    for item in ([("all", "all", sessions)] if isinstance(sessions, pd.DataFrame) else sessions):
        d, r, df = item if not isinstance(sessions, pd.DataFrame) else item
        groups.setdefault(r, []).append(df)
    rows = []
    for regime, frames in groups.items():
        Xs, names, ci = [], None, None
        for df in frames:
            X, nm, c = build_svar_frame(df, spec, n_levels, target_qty, corr_method,
                                        corr_window, wspread_kind, extra_fn=extra_fn)
            if len(X) > 50:
                Xs.append(X); names, ci = nm, c
        if not Xs:
            continue
        X = np.vstack(Xs)
        y = X[:, ci]
        shocks = [(j, names[j]) for j in range(X.shape[1]) if j != ci]
        # controls: ctrl_lags lags of the full system (efficiency only -- see _lp_core)
        Zcols = []
        for L in range(1, int(ctrl_lags) + 1):
            Zcols.append(np.vstack([np.full((L, X.shape[1]), np.nan), X[:-L]]))
        Z = np.hstack(Zcols) if Zcols else None
        for j, nm in shocks:
            sd = np.nanstd(X[:, j])
            sh = (X[:, j] - np.nanmean(X[:, j])) / (sd if sd > 0 else 1.0)
            for r_ in _lp_core(y, sh, Z, horizons):
                rows.append({"regime": regime, "shock": nm, "horizon": r_["horizon"],
                             "theta_x100": 100 * r_["theta"] if np.isfinite(r_["theta"]) else np.nan,
                             "se_x100": 100 * r_["se"] if np.isfinite(r_["se"]) else np.nan})
    return pd.DataFrame(rows)


def lag_diagnosis(p, corr_window, pmax, corr_method="rolling", tol=1) -> dict:
    """Is the selected lag order an ANSWER, or an artifact of how the dependent variable is built?

    Two failure modes, both of which look like a number in a table note:

    ``window_artifact``  p* == corr_window (within ``tol``). `dCorr` is the first difference of a
                         W-bar rolling correlation, so it adds one observation and drops the one
                         from W bars ago: an MA term at lag W and nowhere else. The criterion
                         locates that spike precisely and reports it as the lag order. On simulated
                         data with a CONSTANT correlation and iid liquidity -- nothing for a VAR to
                         find -- BIC returns p* = 8, 12, 20, 25 for corr_window = 8, 12, 20, 25.
                         Exactly the window, every time (test_svar_lag_artifact.py).

    ``at_boundary``      p* == pmax. The criterion was still improving where the search stopped, so
                         the number is a bound. With the paper's 100-bar window the induced spike
                         sits at lag 100, so any search bounded below that climbs to its own edge --
                         which is what footnote 17's "AIC points at 60" is describing.

    Neither is fixed by a longer search or a different criterion, because neither is about the data.
    The fix is the dependent variable: ``corr_method='dcc'`` is a recursive conditional correlation
    with no fixed-width box to difference, and returns p* ~ 1 on that same simulated data whatever
    the window. ``corr_method='hy'`` removes the Epps attenuation but keeps the rolling box, so it
    does NOT remove this -- report it with the window in mind.

    -> {'window_artifact': bool, 'at_boundary': bool, 'ok': bool, 'text': str}"""
    rep = {"window_artifact": False, "at_boundary": False, "no_dynamics": False, "text": ""}
    if p is not None and int(p) == 0:
        rep["no_dynamics"] = True
        rep["ok"] = False
        rep["text"] = ("the criterion selected p=0 -- no dynamics at all. A VAR(0) has no impulse "
                       "response, so a caller must floor it at 1 and read the result as "
                       "impact-only. On a wide correlation window this usually means the criterion "
                       "could not see the window's own MA spike rather than that the system is "
                       "dynamically empty; corr_method='dcc' gives a lag order that means something.")
        return rep
    if p is None:
        rep["text"] = "no lag order could be selected (the IC table was empty or degenerate)"
        rep["ok"] = False
        return rep
    lines = []
    if corr_window and abs(int(p) - int(corr_window)) <= int(tol) and int(p) > 1:
        rep["window_artifact"] = True
        lines.append("SELECTED LAG p=%d EQUALS corr_window (%d). That is an artifact, "
                     "not a finding: d(rolling correlation) carries an MA term at exactly lag W, "
                     "and the criterion is locating it. On data with a constant correlation and iid "
                     "liquidity this reproduces exactly (p*=W for W=8,12,20,25). Re-run with "
                     "corr_method='dcc', whose conditional correlation has no fixed-width box to "
                     "difference and returns p*~1 on the same data; corr_method='hy' does NOT fix "
                     "it, because it keeps the rolling window." % (int(p), int(corr_window)))
    if pmax and int(p) >= int(pmax):
        rep["at_boundary"] = True
        if corr_window and int(corr_window) > int(pmax):
            # The artifact's REALISTIC form. p* == corr_window can only fire when the search can
            # reach the window -- but with the defaults (pmax=12, corr_window=100) it never can,
            # so gating the DCC remedy on that equality made the escape hatch unreachable in
            # exactly the configuration everyone runs. A boundary hit with the window's spike
            # sitting beyond the search IS the window artifact, observed from below: the
            # criterion climbs to its own edge walking toward lag W (the 2026-08-04 run: p=12=
            # pmax, BIC still falling monotonically). Flag it as such, not merely as a bound.
            rep["window_artifact"] = True
            lines.append("SELECTED LAG p=%d IS THE SEARCH BOUND (pmax=%d) WITH corr_window=%d "
                         "BEYOND IT -- the window artifact seen from below. d(rolling correlation) "
                         "puts an MA spike at exactly lag %d; a search capped at %d cannot reach "
                         "it, so the criterion improves all the way to its own edge. Raising pmax "
                         "walks toward the window, not toward the truth. Quote the DCC column, "
                         "whose recursive conditional correlation has no box to difference."
                         % (int(p), int(pmax), int(corr_window), int(corr_window), int(pmax)))
        lines.append("SELECTED LAG p=%d IS THE SEARCH BOUND (pmax=%d), so the criterion was still "
                     "improving where the search stopped -- a bound, not an optimum. If the "
                     "correlation window is %s, the induced spike sits at lag %s and any pmax below "
                     "it will end here; raising pmax will walk toward the window rather than "
                     "converge. Two lag-robust alternatives exist: correlation_irf_lp estimates the "
                     "same IRF by local projections, whose point estimates do not depend on a lag "
                     "order at all, and price_discovery_shares.lag_profile shows whether the "
                     "ESTIMANDS are flat across p while the criterion is still moving. For scale, "
                     "the T^(1/3) sieve ceiling is p~%d on a 23,400-bar session -- a criterion "
                     "proposing more is chasing the window."
                     % (int(p), int(pmax), corr_window or "?", corr_window or "?", sieve_order(23400)))
    rep["ok"] = not (rep["window_artifact"] or rep["at_boundary"])
    rep["text"] = " ".join(lines) if lines else (
        "p=%d: not at the search bound and not equal to the correlation window (%s), so the lag "
        "order is not obviously an artifact of how dCorr is built." % (int(p), corr_window))
    return rep


def select_svar_lag(data, spec="informational", n_levels=10, target_qty=None,
                    corr_method="rolling", corr_window=100, wspread_kind="cost_to_fill",
                    extra_fn=None, criterion="bic", pmax=12, bar_seconds=60, panel="fe"):
    """Choose the SVAR lag order from the data by information criterion. Returns (p*, IC table).

    Selection is on the SAME frame Eq. (5) is estimated on -- every session's built SVAR variables
    stacked -- so the criterion scores the model actually fitted, not a proxy. Candidates are all
    evaluated on the common sample of the largest one (``pds.select_lag_var``), which is required
    for the ICs to be comparable across p.

    Why this exists as a public function rather than a flag buried in the estimator: the paper's
    lag length is a stated compromise -- footnote 17 reports AIC pointing at 60 lags and 6 being
    used because the full model would not run at that length. That makes the lag a researcher
    degree of freedom, so it should be (a) chosen by a rule, (b) computed once, and (c) REPORTED.
    Every caller -- the Table 9 driver, the paired table, the replication script -- resolves the
    lag through this one function so the number in the table note is the number that was fitted.

    BIC is the default rather than AIC deliberately. AIC is not consistent for lag order and, on
    ~23k-bar intraday samples, it chases the enormous effective sample into lag lengths that make
    the SVAR unusable (the paper's own 60). BIC's log(T) penalty is what keeps the choice finite
    and is the standard choice for this kind of high-frequency VAR."""
    if isinstance(data, pd.DataFrame):
        frames = [data]
    else:
        frames = [item[-1] for item in data]
    Xs = []
    for df in frames:
        X, _nm, _ci = build_svar_frame(df, spec, n_levels, target_qty, corr_method,
                                       corr_window, wspread_kind, extra_fn=extra_fn,
                                       bar_seconds=bar_seconds)
        if len(X) > pmax + 5:
            Xs.append(X)
    if not Xs:
        return None, pd.DataFrame()
    X = np.vstack(Xs)
    # Drop exactly-constant columns before scoring. One degenerate regressor (a spread that never
    # moves on a quiet session, a microprice deviation that is identically zero on a symmetric
    # book) makes the VAR design singular, every candidate's log-determinant non-finite, and the
    # whole IC table NaN -- at which point pds.select_lag_var falls back to p=1 and returns it as
    # though it were a selection. Removing the constant column keeps the criterion computable on
    # the columns that carry information; it does not change the fitted model, only the scoring.
    keep = X.std(axis=0) > EPS
    if keep.any() and not keep.all():
        X = X[:, keep]
        Xs = [x[:, keep] for x in Xs]
    # panel="fe" (default): candidates scored with lags built WITHIN-day and day FE -- the
    # stacked scoring let every candidate's lag windows cross the overnight seams and pooled
    # days around one intercept. panel="stack" reproduces the pre-v0.9.63 behaviour.
    if len(Xs) > 1 and panel != "stack":
        p, tab = select_lag_var_panel(Xs, pmax=pmax, criterion=criterion)
    else:
        p, tab = pds.select_lag_var(X, pmax=pmax, criterion=criterion)
    col = tab[{"aic": "aic", "hq": "hqic", "hqic": "hqic"}.get(str(criterion).lower(), "bic")]
    if not col.notna().any():
        # Every candidate failed to score. Say so instead of handing back the p=1 fallback, which
        # is indistinguishable from a real selection at the call site.
        return None, tab
    return p, tab


def resolve_n_lags(data, n_lags, pmax=12, **kw):
    """Accept either a fixed integer lag or a criterion name ('bic'/'aic'/'hq'/'hqic').

    Returns (p, criterion_or_None, table). This is the adapter every CLI uses so `--n-lags 6` and
    `--n-lags bic` are both valid and resolve through the same code path."""
    if isinstance(n_lags, str) and not str(n_lags).strip().lstrip("+-").isdigit():
        crit = str(n_lags).strip().lower()
        p, tab = select_svar_lag(data, criterion=crit, pmax=pmax, **kw)
        return (int(p) if p is not None else None), crit, tab
    return int(n_lags), None, pd.DataFrame()


def correlation_irf(data, spec="informational", n_lags=6, horizon=10, ident="cholesky",
                    cumulative=False, n_levels=10, corr_method="rolling", corr_window=100,
                    target_qty=None, wspread_kind="cost_to_fill", min_obs=400, extra_fn=None,
                    criterion=None, pmax=12, lag_pooled=True, bar_seconds=60, panel="fe"):
    """Reproduce Table 9 in the spread framework. `data` is a (date, regime, df) session
    list (one VAR per regime, frames stacked) or a single df (regime 'all'). Returns a
    DataFrame: rows = shock variables, columns = regimes, values = the impact (h=0)
    orthogonalized response of d-correlation to a 1-SD shock, x100 (the paper's scaling).
    With cumulative=True the column is the summed response over `horizon`. `extra_fn`
    injects continuous cross-flow regressors (e.g. cross_flow.cross_flow_features) so
    Table 9 can be run on chi / PCMOF-NCMOF / common-divergent instead of just spreads."""
    if isinstance(data, pd.DataFrame):
        groups = {"all": [data]}
    else:
        groups = {}
        for item in data:
            df = item[-1]; regime = item[1] if len(item) == 3 else "all"
            groups.setdefault(regime, []).append(df)

    out = {}
    names_ref = None
    # Lag order. With a criterion set, select it ONCE over the pooled sample rather than per
    # regime: Table 9 compares the same shock's response ACROSS regime columns, so a benchmark
    # column fit at p=4 and a volatile column at p=9 would differ partly because the models
    # differ, not because the market does. `lag_pooled=False` restores per-regime selection for
    # the cases where each regime is read on its own.
    p_pooled = None
    if criterion is not None and lag_pooled:
        p_pooled = select_svar_lag(data, spec=spec, n_levels=n_levels, target_qty=target_qty,
                                   corr_method=corr_method, corr_window=corr_window,
                                   wspread_kind=wspread_kind, extra_fn=extra_fn,
                                   criterion=criterion, pmax=pmax, bar_seconds=bar_seconds,
                                   panel=panel)[0]
    keep_min = (pmax if criterion is not None else n_lags) + 5   # room for the LARGEST candidate
    # min_obs was calibrated to native-grid rows; a 60s bar frame has 1/60th of them. Scale it so
    # the bar column is not silently dropped for having exactly the row count re-barring implies.
    if corr_method == "bar":
        min_obs = max(40, int(min_obs) // max(int(bar_seconds), 1))
    for regime, frames in groups.items():
        Xs, names = [], None
        for df in frames:
            X, nm, ci = build_svar_frame(df, spec, n_levels, target_qty, corr_method,
                                         corr_window, wspread_kind, extra_fn=extra_fn,
                                         bar_seconds=bar_seconds)
            if len(X) > keep_min:
                Xs.append(X); names = nm; corr_idx = ci
        if not Xs or sum(len(x) for x in Xs) < min_obs:
            continue
        X = np.vstack(Xs); names_ref = names
        if p_pooled is not None:
            p_use = p_pooled
        elif criterion is not None:
            if len(Xs) > 1 and panel != "stack":
                p_use = select_lag_var_panel(Xs, pmax=pmax, criterion=criterion)[0]
            else:
                p_use = pds.select_lag_var(X, pmax=pmax, criterion=criterion)[0]
        else:
            p_use = n_lags
        if p_use is None:
            p_use = n_lags
        # panel="fe" (default): within-day lags + day fixed effects. "stack" reproduces the
        # pre-v0.9.63 estimation (seam-crossing lags, one common intercept) for comparison.
        if len(Xs) > 1 and panel != "stack":
            A, Sigma, _, _ = panel_var_ols(Xs, p_use, fe=True)
        else:
            A, Sigma, _ = var_ols(X, p_use)
        Theta = orthogonalized_irf(A, Sigma, horizon, ident)
        if cumulative:
            resp = np.sum([Theta[h][corr_idx, :] for h in range(horizon + 1)], axis=0)
        else:
            resp = Theta[0][corr_idx, :]
        out[regime] = resp * 100.0                                             # paper scales by 100

    if not out:
        return pd.DataFrame()
    tab = pd.DataFrame(out, index=names_ref)
    return tab.drop(index=names_ref[-1])                                       # drop the dCorr-own row


def correlation_irf_mean_group(data, spec="informational", n_lags=6, horizon=10,
                               ident="cholesky", cumulative=False, n_levels=10,
                               corr_method="rolling", corr_window=100, target_qty=None,
                               wspread_kind="cost_to_fill", extra_fn=None,
                               criterion=None, pmax=12, bar_seconds=60, min_obs_day=200):
    """Mean-group (Pesaran-Smith / Pedroni-style) Table 9: the SVAR is fitted and identified
    PER DAY -- T ~ 23,400 bars per day is ample -- and the regime column is the cross-day MEAN
    response with its dispersion and a day-level t (mean / (sd/sqrt(N_days))).

    This is the heterogeneity check the pooled fixed-effects column needs: the paper's own
    thesis is that dynamics differ by regime, so homogeneous pooled dynamics assume part of the
    conclusion. If mean-group ~ pooled-FE, pooling is validated and the table says so with
    numbers; if they diverge, the divergence is a finding about slope heterogeneity, not a
    nuisance. The lag order is resolved ONCE on the pooled sample (same p every day) so the
    cross-day average compares like with like.

    Returns {"point", "se", "tstat", "n_days" (DataFrames, shocks x regimes), "per_day"
    (tidy DataFrame), "n_lags"}."""
    if isinstance(data, pd.DataFrame):
        data = [("all", "all", data)]
    if criterion is not None:
        p_sel = select_svar_lag(data, spec=spec, n_levels=n_levels, target_qty=target_qty,
                                corr_method=corr_method, corr_window=corr_window,
                                wspread_kind=wspread_kind, extra_fn=extra_fn,
                                criterion=criterion, pmax=pmax, bar_seconds=bar_seconds)[0]
        if p_sel is not None:
            n_lags = int(p_sel)
    if corr_method == "bar":                         # bar rows are 1/bar_seconds of native rows
        min_obs_day = max(20, int(min_obs_day) // max(int(bar_seconds), 1))
    rows = []
    names_ref = None
    n_skip = {"short": 0, "frozen_dep": 0, "underdetermined": 0, "estimation_failed": 0}
    for item in data:
        date = item[0]; regime = item[1] if len(item) == 3 else "all"; df = item[-1]
        X, names, ci = build_svar_frame(df, spec, n_levels, target_qty, corr_method,
                                        corr_window, wspread_kind, extra_fn=extra_fn,
                                        bar_seconds=bar_seconds)
        if len(X) < max(min_obs_day, n_lags + 10):
            n_skip["short"] += 1
            continue
        keep = X.std(axis=0) > EPS
        if not keep[ci]:                             # a day with a frozen dependent variable
            n_skip["frozen_dep"] += 1
            continue
        if not keep.all():
            # A frozen regressor is unidentified on that day; leaving it in hands lstsq a
            # singular design whose minimum-norm answer looks like a coefficient. Drop it from
            # THIS day's system (the dependent stays last, so ci is re-derived, not shifted).
            X = X[:, keep]
            names = [nm for nm, k_ in zip(names, keep) if k_]
            ci = len(names) - 1
        # A per-day VAR(p) on k variables estimates 1 + k*p coefficients PER EQUATION, and
        # lstsq on an underdetermined design returns the minimum-norm solution without
        # complaint -- pseudo-responses that would be averaged into the mean group with full
        # weight. Require the effective sample to clear the parameter count with margin.
        if len(X) - n_lags < 1 + X.shape[1] * n_lags + 10:
            n_skip["underdetermined"] += 1
            continue
        try:
            A, Sigma, _ = var_ols(X, n_lags)         # per-day intercept = day FE, trivially
            Theta = orthogonalized_irf(A, Sigma, horizon, ident)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            n_skip["estimation_failed"] += 1
            continue
        resp = (np.sum([Theta[h][ci, :] for h in range(horizon + 1)], axis=0)
                if cumulative else Theta[0][ci, :])
        if names_ref is None or len(names) > len(names_ref):
            names_ref = names
        for j, nm in enumerate(names):
            if j == ci:
                continue
            rows.append({"date": str(date), "regime": regime, "shock": nm,
                         "resp_x100": 100.0 * resp[j], "n_obs": int(len(X))})
    per_day = pd.DataFrame(rows)
    if per_day.empty:
        return {"point": pd.DataFrame(), "point_weighted": pd.DataFrame(), "se": pd.DataFrame(),
                "tstat": pd.DataFrame(), "n_days": pd.DataFrame(), "per_day": per_day,
                "n_lags": int(n_lags), "n_skipped": n_skip}
    g = per_day.groupby(["shock", "regime"])["resp_x100"]
    mean = g.mean().unstack(); sd = g.std(ddof=1).unstack(); n = g.count().unstack()
    se = sd / np.sqrt(n)
    # Effective-sample-weighted mean alongside the unweighted MG: after halt masking, days
    # differ substantially in usable rows, and the pure Pesaran-Smith mean hands a truncated
    # MWCB day the same weight as a full baseline day. The weight is the day's usable design
    # rows (a precision PROXY -- per-day bootstrap SEs would be the exact weight but cost a
    # full bootstrap per day); agreement between the two columns is itself evidence the MG
    # mean is not driven by its shortest days.
    def _wmean(gg):
        w = gg["n_obs"].to_numpy(float); v = gg["resp_x100"].to_numpy(float)
        return float(np.sum(w * v) / max(np.sum(w), EPS))
    wmean = per_day.groupby(["shock", "regime"])[["n_obs", "resp_x100"]].apply(_wmean).unstack()
    order = [nm for nm in (names_ref or []) if nm in mean.index] or list(mean.index)
    mean, se, n = mean.reindex(order), se.reindex(order), n.reindex(order)
    wmean = wmean.reindex(order)
    return {"point": mean, "point_weighted": wmean, "se": se,
            "tstat": mean / se.replace(0.0, np.nan),
            "n_days": n, "per_day": per_day, "n_lags": int(n_lags), "n_skipped": n_skip}


# ── Table 9 with inference: bootstrap SE + Romano-Wolf joint significance ──────
def _stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _fmt_cell(coef, se=None, p=None, prec=3):
    if coef is None or (isinstance(coef, float) and not np.isfinite(coef)):
        return "--"
    s = f"{coef:.{prec}f}{_stars(p)}"
    if se is not None and np.isfinite(se):
        s += f" ({se:.{prec}f})"
    return s


def _irf_corr_response(X, corr_idx, n_lags, horizon, ident, cumulative):
    """Impact (or cumulative) orthogonalized response of d-correlation to each shock, x100."""
    A, Sigma, _ = var_ols(X, n_lags)
    Theta = orthogonalized_irf(A, Sigma, horizon, ident)
    if cumulative:
        return np.sum([Theta[h][corr_idx, :] for h in range(horizon + 1)], axis=0) * 100.0
    return Theta[0][corr_idx, :] * 100.0


def epps_comparison(data, methods=("rolling", "hy"), **kw):
    """Table 9 computed with each dependent variable in `methods`, side by side.

    The point of the pairing is not robustness box-ticking. Grid-sampled Pearson is attenuated
    by asynchronous quote updating, the attenuation is a function of trading intensity, and
    trading intensity is on the right-hand side of the same system -- so the naive estimator
    cannot separate "activity raises correlation" from "activity makes correlation easier to
    measure". Hayashi-Yoshida imposes no grid and breaks that link, so the DIFFERENCE between
    the two columns is an estimate of how much of each response is measurement artifact.

    Coefficients that keep their sign, size and significance across both are safe to interpret
    as market mechanism. Ones that shrink toward zero under 'hy' were partly the Epps channel;
    the volume and message-traffic responses are the ones to watch, since they drive the
    attenuation most directly.

    Returns a DataFrame indexed by shock with a column block per method, plus `delta_<regime>`
    columns giving (hy - rolling) for each regime."""
    out = {}
    for m in methods:
        tb = correlation_irf(data, corr_method=m, **kw)
        for c in tb.columns:
            out[(m, c)] = tb[c]
    res = pd.DataFrame(out)
    res.columns = pd.MultiIndex.from_tuples(res.columns, names=["corr_method", "regime"])
    if len(methods) == 2 and all(m in res.columns.get_level_values(0) for m in methods):
        a, b = methods
        for c in res[b].columns:
            if c in res[a].columns:
                res[("delta", c)] = res[b][c] - res[a][c]
    return res


def correlation_irf_inference(data, spec="informational", n_lags=6, horizon=10, ident="cholesky",
                              cumulative=False, n_levels=10, corr_method="rolling", corr_window=100,
                              target_qty=None, wspread_kind="cost_to_fill", min_obs=400, extra_fn=None,
                              n_boot=499, alpha=0.05, rw_by_regime=False, block_len=None, seed=0,
                              n_jobs=None, criterion=None, pmax=12, bar_seconds=60, panel="fe"):
    """`correlation_irf` with bootstrap standard errors and **Romano-Wolf joint significance**
    built in, so the IRF table comes out publication-ready: each cell is the impact (or
    cumulative) response x100 with a bootstrap SE and stars from the FWER-controlled
    (joint, cross-correlation-robust) p-value.

    Resampling unit:
      * a (date, regime, df) session list with >=2 sessions -> a DAY-CLUSTER bootstrap
        (resample whole sessions); because each replication rebuilds the SVAR frame, this
        also propagates the generated-regressor uncertainty in d-correlation.
      * a single df (or one session) -> a moving-block bootstrap of the built SVAR frame
        rows (d-correlation treated as fixed -- no first-stage propagation).

    Romano-Wolf scope: `rw_by_regime=False` controls FWER jointly over ALL shock x regime
    cells (the strongest joint claim); `True` controls within each regime column.

    Returns a dict: point, se, tstat, padj, reject, ci_lower, ci_upper (all DataFrames with
    the shock x regime shape of `correlation_irf`), a rendered `table` (coef*** (se)), a
    per-regime `joint` summary (n_shocks, n_reject, min_padj), plus alpha, n_boot, rw_scope.
    """
    from inference import romano_wolf_from_boot, moving_block_bootstrap, parallel_bootstrap

    # Resolve the lag ONCE on the full sample, then hold it fixed across bootstrap replicates.
    # Re-selecting inside each replicate would propagate model-selection uncertainty, which sounds
    # more honest but is not what the reported table is: the point estimate is a VAR(p*) fit, so
    # the resampling distribution must be of THAT estimator. Re-selecting also lets a replicate
    # land on a different p and silently change the number of shocks, breaking the Romano-Wolf
    # alignment across draws.
    if criterion is not None:
        p_sel = select_svar_lag(data, spec=spec, n_levels=n_levels, target_qty=target_qty,
                                corr_method=corr_method, corr_window=corr_window,
                                wspread_kind=wspread_kind, extra_fn=extra_fn,
                                criterion=criterion, pmax=pmax, bar_seconds=bar_seconds,
                                panel=panel)[0]
        if p_sel is not None:
            n_lags = int(p_sel)
    kw = dict(spec=spec, n_lags=n_lags, horizon=horizon, ident=ident, cumulative=cumulative,
              n_levels=n_levels, corr_method=corr_method, corr_window=corr_window,
              target_qty=target_qty, wspread_kind=wspread_kind, min_obs=min_obs, extra_fn=extra_fn,
              bar_seconds=bar_seconds, panel=panel)
    point = correlation_irf(data, **kw)
    empty = {"point": point, "se": pd.DataFrame(), "tstat": pd.DataFrame(), "padj": pd.DataFrame(),
             "reject": pd.DataFrame(), "ci_lower": pd.DataFrame(), "ci_upper": pd.DataFrame(),
             "table": pd.DataFrame(), "joint": pd.DataFrame(), "alpha": alpha, "n_boot": 0,
             "rw_scope": "by_regime" if rw_by_regime else "joint",
             "n_lags": int(n_lags), "lag_criterion": criterion}
    if point.empty:
        return empty
    rows, cols = list(point.index), list(point.columns)

    # ---- bootstrap draws (B, R, C) aligned to the point table ----
    draws = []
    if not isinstance(data, pd.DataFrame) and len(data) >= 2:
        sess = list(data); nse = len(sess)

        def _draw(child_seed):                                       # day-cluster: resample whole sessions
            r = np.random.default_rng(child_seed)
            idx = r.integers(0, nse, nse)
            tb = correlation_irf([sess[i] for i in idx], **kw)
            return tb.reindex(index=rows, columns=cols).to_numpy() if not tb.empty else None

        # threading: session frames shared read-only (no pickling), refit releases the GIL,
        # and the bootstrap is post-extraction so it is not memory-bound -> use all cores.
        draws = parallel_bootstrap(_draw, n_boot, n_jobs=n_jobs, backend="threading", seed=seed)
        B = np.array(draws) if draws else None
    else:                                                            # single frame: block-bootstrap its rows
        df = data if isinstance(data, pd.DataFrame) else data[0][-1]
        X, names, ci = build_svar_frame(df, spec, n_levels, target_qty, corr_method, corr_window,
                                        wspread_kind, extra_fn=extra_fn, bar_seconds=bar_seconds)
        nshock = len(names) - 1
        stat = lambda Xb: _irf_corr_response(Xb, ci, n_lags, horizon, ident, cumulative)[:nshock]
        bd = moving_block_bootstrap(X, stat, n_boot=n_boot, block_len=block_len, seed=seed,
                                    n_jobs=n_jobs)                   # (B, R)
        B = bd[:, :len(rows), None]                                  # single 'all' regime column

    if B is None or B.shape[0] < 2:
        return empty

    pt = point.to_numpy()
    se = np.nanstd(B, axis=0, ddof=1)
    tstat = pt / np.where(se > EPS, se, np.nan)
    lo = np.nanpercentile(B, 100 * alpha / 2, axis=0); hi = np.nanpercentile(B, 100 * (1 - alpha / 2), axis=0)

    padj = np.ones_like(pt, dtype=float)
    if rw_by_regime:
        for j in range(pt.shape[1]):
            ptj, Bj = pt[:, j], B[:, :, j]
            fin = np.isfinite(ptj) & np.all(np.isfinite(Bj), axis=0)
            if fin.sum() >= 1:
                padj[fin, j] = romano_wolf_from_boot(ptj[fin], Bj[:, fin])["adj_p"]
    else:
        ptf = pt.ravel(); Bf = B.reshape(B.shape[0], -1)
        fin = np.isfinite(ptf) & np.all(np.isfinite(Bf), axis=0)
        adj = np.ones_like(ptf)
        if fin.sum() >= 1:
            adj[fin] = romano_wolf_from_boot(ptf[fin], Bf[:, fin])["adj_p"]
        padj = adj.reshape(pt.shape)
    reject = padj <= alpha

    mk = lambda M: pd.DataFrame(M, index=rows, columns=cols)
    table = pd.DataFrame(index=rows, columns=cols, dtype=object)
    for i in range(len(rows)):
        for j in range(len(cols)):
            table.iloc[i, j] = _fmt_cell(pt[i, j], se[i, j], padj[i, j])
    table.index.name = "shock"

    jr = []
    for j, c in enumerate(cols):
        fin = np.isfinite(pt[:, j])
        jr.append({"regime": c, "n_shocks": int(fin.sum()),
                   "n_reject": int(np.sum(reject[:, j] & fin)),
                   "min_padj": float(np.nanmin(padj[fin, j])) if fin.any() else np.nan})
    joint = pd.DataFrame(jr).set_index("regime")

    return {"point": point, "se": mk(se), "tstat": mk(tstat), "padj": mk(padj),
            "reject": pd.DataFrame(reject, index=rows, columns=cols),
            "ci_lower": mk(lo), "ci_upper": mk(hi), "table": table, "joint": joint,
            "alpha": alpha, "n_boot": int(B.shape[0]),
            "rw_scope": "by_regime" if rw_by_regime else "joint",
            "n_lags": int(n_lags), "lag_criterion": criterion}


def fevd_correlation(data, spec="informational", n_lags=6, horizon=20, **kw):
    """Forecast-error variance of d-correlation attributed to each shock at `horizon`
    (recursive identification), per regime. Complements the IRF: shares sum to 1."""
    tabs = {}
    groups = ({"all": [data]} if isinstance(data, pd.DataFrame)
              else _group_by_regime(data))
    for regime, frames in groups.items():
        Xs = []; names = None; corr_idx = None
        for df in frames:
            X, nm, ci = build_svar_frame(df, spec, **_bkw(kw))
            if len(X) > n_lags + 5:
                Xs.append(X); names = nm; corr_idx = ci
        if not Xs:
            continue
        X = np.vstack(Xs)
        A, Sigma, _ = var_ols(X, n_lags)
        Theta = orthogonalized_irf(A, Sigma, horizon, "cholesky")
        num = np.zeros(Sigma.shape[0]); tot = 0.0
        for h in range(horizon + 1):
            row = Theta[h][corr_idx, :]
            num += row ** 2; tot += float(np.sum(row ** 2))
        tabs[regime] = pd.Series(num / tot if tot > 0 else num, index=names).drop(names[-1])
    return pd.DataFrame(tabs)


def _group_by_regime(data):
    g = {}
    for item in data:
        g.setdefault(item[1] if len(item) == 3 else "all", []).append(item[-1])
    return g


def _bkw(kw):
    return {k: kw[k] for k in ("n_levels", "target_qty", "corr_method", "corr_window",
                               "wspread_kind") if k in kw}


# ── the better object: spread-conditioned information share IS(S) ─────────────
def spread_conditioned_is(sessions, spread_kind="weighted", n_levels=10, target_qty=None,
                          wspread_kind="cost_to_fill", degree=1, n_lags=0, hac_lags=20):
    """The structural replacement for Table 9: instead of regressing d-correlation on
    spread, make the relative (weighted) spread the conditioning STATE in the
    information-share curve, S = log(spread_ES) - log(spread_SPY), and report IS(S) via
    the state-dependent ECM-SDE. A negative IS_ES gradient in S means ES leads price
    discovery precisely when its book is relatively tighter -- the thesis, read off a
    spread state. Returns the ecm_sde fit (with fit['curve'] the IS(S) table)."""
    def _spr(df, a):
        return (weighted_spread(df, a, target_qty, n_levels, wspread_kind)
                if spread_kind == "weighted" else quoted_spread(df, a))

    def state_fn(df):
        se = _spr(df, "ES"); ss = _spr(df, "SPY")
        return np.log(np.maximum(se, EPS)) - np.log(np.maximum(ss, EPS))

    return es.estimate_sample(sessions, state_fn=state_fn, degree=degree,
                              n_lags=n_lags, hac_lags=hac_lags)


# ── self-test ─────────────────────────────────────────────────────────────────
def _demo_frame(n=8000, seed=0, depth_factor=1.0):
    """Synthetic two-asset book where a common factor drives returns, and ES depth
    (hence its spread) responds to volatility -- enough structure to exercise the IRF."""
    rng = np.random.default_rng(seed)
    s = np.abs(rng.normal(0, 1, n)).cumsum() * 0 + np.abs(rng.standard_normal(n))   # vol proxy
    f = rng.normal(0, 1, n).cumsum()                                            # common log-price factor
    lp1 = f + rng.normal(0, 0.3, n); lp2 = f + rng.normal(0, 0.3, n)
    ms, me = 500 * np.exp(lp1 / 50), 5000 * np.exp(lp2 / 50)
    lev = np.arange(1, 11); dw = np.exp(-0.3 * (lev - 1)); cols = {}
    for a, mid, tick, sgn in (("SPY", ms, 0.01, -0.5), ("ES", me, 0.25, 0.5)):
        depth = 300.0 * np.exp(sgn * s) * depth_factor
        for i, l in enumerate(lev):
            q = np.maximum(depth * dw[i] * (1 + rng.normal(0, 0.05, n)), 1.0)
            cols[f"{a}_bidprice_{l}"] = mid - l * tick; cols[f"{a}_askprice_{l}"] = mid + l * tick
            cols[f"{a}_bidquantity_{l}"] = q; cols[f"{a}_askquantity_{l}"] = q * (1 + rng.normal(0, 0.02, n))
    idx = pd.date_range("2024-08-05 09:30", periods=n, freq="s", tz="America/New_York")
    return pd.DataFrame(cols, index=idx)


def _selftest():
    print("correlation_svar self-test")
    df = _demo_frame()

    # (1) spread observables: positive, weighted >= quoted, finite
    qs = quoted_spread(df, "ES"); ws = weighted_spread(df, "ES"); wd = weighted_spread(df, "ES", kind="depth_weighted")
    md = microprice_dev(df, "ES")
    print("\n(1) spread observables (ES, bps)")
    print("    quoted mean=%.3f  cost-to-fill mean=%.3f  depth-wtd mean=%.3f  microdev std=%.3f"
          % (np.nanmean(qs), np.nanmean(ws), np.nanmean(wd), np.nanstd(md)))
    print("    checks:", np.nanmean(qs) > 0 and np.nanmean(ws) >= np.nanmean(qs) - 1e-9
          and np.nanmean(wd) >= np.nanmean(qs) - 1e-9 and np.isfinite(np.nanstd(md)))

    # (2) VAR / IRF algebra on a known 2-var system: Psi_1 == A_1, impact == chol(Sigma)
    rng = np.random.default_rng(1); T = 6000
    A1 = np.array([[0.4, 0.1], [0.2, 0.3]]); L0 = np.array([[1.0, 0.0], [0.5, 0.8]])
    e = rng.normal(0, 1, (T, 2)) @ L0.T; Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = A1 @ Y[t - 1] + e[t]
    A, Sig, _ = var_ols(Y, 1); Th = orthogonalized_irf(A, Sig, 5, "cholesky")
    print("\n(2) VAR/IRF algebra (recover A_1 and the impact factor)")
    print("    ||A_hat - A_1|| = %.3f   ||Theta_0 - chol(Sigma)|| = %.3f"
          % (np.linalg.norm(A[0] - A1), np.linalg.norm(Th[0] - np.linalg.cholesky(Sig))))
    print("    Psi_1 == A_hat:", np.allclose(_ma_from_var(A, 2)[1], A[0]))
    print("    checks:", np.linalg.norm(A[0] - A1) < 0.05
          and np.allclose(Th[0], np.linalg.cholesky(Sig))
          and np.allclose(_ma_from_var(A, 2)[1], A[0]))

    # (3) Table-9-style IRF across the spec progression + both identifications
    sessions = [("d1", "benchmark", _demo_frame(seed=2, depth_factor=1.5)),
                ("d2", "volatile",  _demo_frame(seed=3, depth_factor=0.7))]
    t_std = correlation_irf(sessions, spec="standard", n_lags=4, horizon=6)
    t_inf = correlation_irf(sessions, spec="informational", n_lags=4, horizon=6)
    t_id = correlation_irf(sessions, spec="informational", n_lags=4, horizon=6, ident="identity")
    print("\n(3) Table-9-style correlation IRF (impact x100)")
    print("    standard spec rows:", list(t_std.index))
    print("    informational spec rows:", list(t_inf.index))
    print(t_inf.round(3).to_string())
    print("    checks:",
          not t_std.empty and not t_inf.empty and not t_id.empty
          and t_std.shape[0] == 2                          # ES & SPY quoted-spread shocks
          and t_inf.shape[0] == 10                         # 5 features x 2 assets
          and set(t_inf.columns) == {"benchmark", "volatile"}
          and np.all(np.isfinite(t_inf.to_numpy())))

    # (4) the better object: spread-conditioned IS(S) curve via ecm_sde
    fit = spread_conditioned_is(sessions, spread_kind="weighted", n_levels=10, degree=1, n_lags=0)
    curve = fit["curve"]
    print("\n(4) spread-conditioned information share IS_ES(S), S = log relative weighted spread")
    print(curve[["S", "IS_ES", "CS_ES"]].round(3).to_string(index=False))
    print("    checks:", isinstance(curve, pd.DataFrame) and "IS_ES" in curve.columns
          and np.all(np.isfinite(curve["IS_ES"].to_numpy())))

    # (5) IRF with bootstrap SE + Romano-Wolf joint significance built in
    sess5 = [(f"d{i}", "benchmark" if i % 2 == 0 else "volatile",
              _demo_frame(n=4000, seed=10 + i, depth_factor=1.4 if i % 2 == 0 else 0.7)) for i in range(6)]
    res = correlation_irf_inference(sess5, spec="informational", n_lags=4, horizon=6,
                                    n_boot=80, alpha=0.10, seed=0)
    pt, se, padj = res["point"].to_numpy(), res["se"].to_numpy(), res["padj"].to_numpy()
    t = np.abs(pt / np.where(se > EPS, se, np.nan))
    fin = np.isfinite(t.ravel())
    imax = int(np.nanargmax(np.where(fin, t.ravel(), -np.inf)))          # largest |t| among finite cells
    print("\n(5) IRF with bootstrap SE + Romano-Wolf joint significance (n_boot=%d, scope=%s)"
          % (res["n_boot"], res["rw_scope"]))
    print(res["table"].to_string())
    print("    per-regime joint:", {k: dict(v) for k, v in res["joint"].T.to_dict().items()})
    shapes_ok = res["se"].shape == res["point"].shape == res["padj"].shape == res["table"].shape
    range_ok = np.all((padj >= -1e-9) & (padj <= 1 + 1e-9)) and np.nanmax(se) > 0
    monotone_ok = np.isclose(padj.ravel()[imax], np.nanmin(padj.ravel()[fin]))   # max |t| -> min adj_p (RW)
    print("    checks:", shapes_ok and range_ok and monotone_ok
          and isinstance(res["table"].iloc[0, 0], str) and res["n_boot"] >= 2)

    # (5b) parallel bootstrap equivalence: n_jobs is a pure speedup. Per-replicate spawned seeds
    #      make the day-cluster draws identical regardless of worker count, so SE/point/padj match.
    rs = correlation_irf_inference(sess5, spec="informational", n_lags=4, horizon=6,
                                   n_boot=80, alpha=0.10, seed=0, n_jobs=1)
    rp = correlation_irf_inference(sess5, spec="informational", n_lags=4, horizon=6,
                                   n_boot=80, alpha=0.10, seed=0, n_jobs=4)

    def _eqnan(a, b):
        a, b = a.to_numpy(), b.to_numpy()
        return bool(np.allclose(np.nan_to_num(a), np.nan_to_num(b))
                    and np.array_equal(np.isnan(a), np.isnan(b)))
    par_ok = _eqnan(rs["se"], rp["se"]) and _eqnan(rs["point"], rp["point"]) and _eqnan(rs["padj"], rp["padj"])
    print("\n(5b) day-cluster bootstrap: serial (n_jobs=1) == threaded (n_jobs=4)")
    print("     SE max|diff|=%.2e  padj max|diff|=%.2e  ->  identical: %s"
          % (float(np.nanmax(np.abs(rs["se"].to_numpy() - rp["se"].to_numpy()))),
             float(np.nanmax(np.abs(rs["padj"].to_numpy() - rp["padj"].to_numpy()))), par_ok))
    print("     checks:", par_ok)


if __name__ == "__main__":
    _selftest()
