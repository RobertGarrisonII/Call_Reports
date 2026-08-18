#!/usr/bin/env python3
"""
lp_table9.py
============
Table 9 estimated by **panel local projections** (Jorda 2005) -- the lag-order-free
counterpart to the SVAR IRF on the window-free RealBar frame.

Why this exists. The SVAR route to Table 9 has one researcher degree of freedom the paper
itself could not resolve: the lag order. Footnote 17 reports AIC pointing at 60 lags and 6
being used because the full model would not run; this stack's own selection machinery shows
the criterion tracking the corr_window MA spike (or its own search bound) on the rolling
frame, and even on the RealBar frame the chosen p* is one argmin among near-ties
(lag_robustness). Every IRF step after impact inherits that choice, because a VAR IRF is the
p-th order model ITERATED: truncate the order and the misspecification compounds at every
horizon.

A local projection removes the choice from the estimand. For each horizon h, the response
theta_h is read off a DIRECT regression of corr_{t+h} on the shock at t -- no iteration, so
nothing compounds. Lagged controls enter only to (a) orthogonalize the shock and (b) soak up
predictable variation for efficiency; the identification is the same recursive ordering the
SVAR uses (regress on the shock plus its contemporaneous PREDECESSORS in the causal ordering
plus p lags of the full system), so at the true model the LP and the Cholesky IRF estimate
the same object (Plagborg-Moller & Wolf 2021), while under lag truncation only the VAR IRF
is biased at h >= 2. That asymmetry is a checkable property, and test_lp_table9.py pins it
on a planted VAR(3) rather than asserting it.

Scaling: theta is per 1-SD STRUCTURAL shock, x100 -- the same units as the SVAR table. The
shock's structural SD is the residual SD of the shock variable on its own controls (the
Cholesky pivot L[j,j]), estimated within the same design.

Panel discipline (same conventions as correlation_svar's panel path):
  * day fixed effects via the within transformation (each session demeaned over its clean
    rows) -- days differ in RV/spread LEVELS by orders of magnitude and pooling them around
    one intercept conflates within-day dynamics with between-day level shifts;
  * leads AND lags built STRICTLY within day -- a response window may not cross the
    overnight seam;
  * halt-aware window invalidation, the idea of irf.local_projection: a bar the halt mask
    NaN'd (or a bar absent from the grid entirely) poisons every observation whose
    [t - p, t + h] window touches it. Dropped, not zero-filled: a response measured across
    a halt is undefined, and zero-filling would manufacture exactly the calm the halt
    interrupted. The per-session frame is therefore built WITHOUT the finite-row mask
    build_svar_frame applies (masking first would splice the two sides of a halt into
    adjacent rows and the window logic could no longer see the gap).

Inference: day-cluster bootstrap (resample whole sessions), because days are the
independent unit everywhere in this stack -- LP residuals are serially correlated by
construction (overlapping response windows), and the naive iid OLS SE understates the
uncertainty whenever shock transmission differs across days (test_lp_table9 pins that too).

Output: tidy rows shock x regime x horizon with theta, bootstrap CI, and -- when the SVAR
is estimable on the same frame at the same p -- the SVAR IRF at the same horizon, so the
gap between the two is visible per cell.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

import correlation_svar as cs
import cross_asset_pd_liquidity as ca

EPS = 1e-12


# ── per-session RealBar frame, NaN rows RETAINED ──────────────────────────────
def build_lp_bar_frame(df, spec="informational", n_levels=10, target_qty=None,
                       wspread_kind="cost_to_fill", bar_seconds=60):
    """The corr_method='bar' assembly of correlation_svar._build_svar_frame_bar, with two
    deliberate differences and nothing else: (1) rows with non-finite entries are KEPT, and
    (2) the output is reindexed onto the COMPLETE bar grid from first to last bar, so a bar
    with no snapshots at all (a full halt window) appears as a NaN row rather than being
    silently absent. Both exist for the same reason: the LP's window invalidation needs bar
    POSITION to equal bar TIME. build_svar_frame drops bad rows before returning, which
    splices the two sides of a halt into adjacent rows -- fine for a VAR fit that treats
    rows exchangeably after lagging, fatal for a projection h bars ahead.

    Same causal ordering contract (ES block, SPY block, dCorr last).
    -> (X [n_bars, k] with NaN rows, names, corr_idx)."""
    feats = cs.PRESETS[spec] if isinstance(spec, str) else list(spec)
    idx = df.index
    if len(idx) < 3:
        return np.empty((0, 0)), [], -1
    r = {a: cs._returns_bps(df, a) for a in ("SPY", "ES")}
    bar = idx.floor("%ds" % int(bar_seconds))
    V = pd.DataFrame(index=idx)
    for a in ("ES", "SPY"):                                                    # futures first
        for f in feats:
            name = f"{cs._LABEL[f]}_{a}"
            if f == "qspr":
                V[name] = cs.quoted_spread(df, a)
            elif f == "wspr":
                V[name] = cs.weighted_spread(df, a, target_qty, n_levels, wspread_kind)
            elif f == "ofi":
                V[name] = ca.order_flow_imbalance(df, a, n_levels)
            elif f == "rv":
                V[name] = pd.Series(r[a], index=idx) ** 2                      # per-bar mean below
            elif f == "mdev":
                V[name] = cs.microprice_dev(df, a, min(n_levels, 5))
    # same coverage floors as the SVAR bar frame (see _build_svar_frame_bar for calibration)
    dt = float(np.median(np.diff(idx.asi8))) / 1e9 if len(idx) > 1 else 1.0
    cap = max(1.0, float(bar_seconds) / max(dt, 1e-9))
    min_sub = max(10, int(0.25 * cap))
    g = V.groupby(bar)
    n_fin = V.notna().groupby(bar).sum()
    bars = pd.date_range(bar.min(), bar.max(), freq="%ds" % int(bar_seconds))
    B = pd.DataFrame(index=pd.Index(bars, name="bar"))
    for c in V.columns:
        base = c.split("_")[0]
        if base in ("OFI",):                          # flow columns SUM, with the half-bar floor
            s = g[c].sum(min_count=1)
            s[n_fin[c].reindex(s.index).fillna(0) < 0.5 * cap] = np.nan
            B[c] = s.reindex(bars)
        else:                                                                  # RV / spreads / mdev
            B[c] = g[c].mean().reindex(bars)
    for c in list(B.columns):                                                  # spread LEVELS -> changes
        if c.split("_")[0] in ("Spread", "WtdSpread"):
            B[c] = B[c].diff()
    rs = pd.Series(r["SPY"], index=idx); re_ = pd.Series(r["ES"], index=idx)
    rho = {}
    for b, ii in rs.groupby(bar).groups.items():
        x = rs.loc[ii].to_numpy(float); y = re_.loc[ii].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_sub and x[m].std() > EPS and y[m].std() > EPS:
            rho[b] = float(np.corrcoef(x[m], y[m])[0, 1])
        else:
            rho[b] = np.nan
    z = pd.Series(rho).reindex(bars)
    B["dCorr"] = (pd.Series(cs._fisher_z(z.to_numpy()), index=B.index).diff()
                  if z.notna().any() else np.nan)
    names = list(B.columns)
    return B.to_numpy(float), names, len(names) - 1


# ── per-day design cross-products ─────────────────────────────────────────────
def _day_lp_grams(X, corr_idx, p, horizons):
    """One session's LP cross-products, all horizons, one pass.

    The full regressor block per row t is C_t = [X_t, X_{t-1}, .., X_{t-p}] (contemporaneous
    system first, then lag blocks) after within-day demeaning over the CLEAN rows; every
    shock's design at every horizon is a COLUMN SUBSET of C (see _shock_cols), so one Gram
    per horizon serves all shocks, and the day-cluster bootstrap becomes weighted Gram sums
    instead of design rebuilds.

    Validity rule: an observation (t, h) exists iff the whole window [t - p, t + h] lies
    inside the day and contains no NaN row -- the gap-count prefix sum of
    irf.local_projection, applied to rows instead of returns. This is what drops (rather
    than zero-fills) response windows that cross a halt, and, because each day is processed
    alone, windows can never cross the overnight seam at all.

    -> {h: (C'C, C'y_h, y_h'y_h, n)} over valid t, or None if the day is unusable."""
    X = np.asarray(X, float)
    T, k = X.shape
    fin = np.all(np.isfinite(X), axis=1)
    if fin.sum() < p + 5:
        return None
    mu = X[fin].mean(axis=0)
    Xd = np.where(np.isfinite(X), X - mu, np.nan)
    ngap = np.concatenate([[0], np.cumsum(~fin)])
    M = np.full((T, k * (p + 1)), np.nan)
    M[:, :k] = Xd
    for L in range(1, p + 1):
        M[L:, L * k:(L + 1) * k] = Xd[:-L]
    y = Xd[:, corr_idx]
    out = {}
    for h in horizons:
        if T - h <= p:
            continue
        t = np.arange(p, T - h)
        t = t[(ngap[t + h + 1] - ngap[t - p]) == 0]          # clean [t-p, t+h] windows only
        if len(t) == 0:
            continue
        C = M[t]; yh = y[t + h]
        out[h] = (C.T @ C, C.T @ yh, float(yh @ yh), len(t))
    return out if out else None


def _shock_cols(j, k, p):
    """Column subset of the full block for shock j: the shock itself first, then its
    contemporaneous PREDECESSORS in the causal ordering (the recursive identification --
    exactly the regressors variable j is orthogonalized against in a Cholesky factor with
    that ordering), then all k*p lag columns."""
    return [j] + list(range(j)) + list(range(k, k * (p + 1)))


def _sigma_shock(G0, n0, j, k, p):
    """Structural shock SD for variable j: residual SD of x_j on its predecessors + lags,
    from the h=0 Gram -- the sample analogue of the Cholesky pivot L[j,j], which converts
    the projection slope (response per UNIT of orthogonalized x_j) into response per 1-SD
    structural shock, the SVAR table's units."""
    cols = _shock_cols(j, k, p)
    W = cols[1:]
    if n0 <= len(W) + 2:
        return np.nan
    _B, rss = cs._gram_solve(G0[np.ix_(W, W)], G0[np.ix_(W, [j])],
                             np.array([[float(G0[j, j])]]))
    return float(np.sqrt(max(rss[0, 0], 0.0) / max(n0 - len(W), 1.0)))


# ── the panel LP estimator ────────────────────────────────────────────────────
def panel_lp_irf(Xs, names, corr_idx, n_lags=6, horizon=10, n_boot=199, alpha=0.05, seed=0):
    """Panel local-projection IRF of the dependent (corr_idx) column to a 1-SD structural
    shock in every other column, horizons 0..horizon, day FE, within-day windows, recursive
    identification at the system's causal ordering, `n_lags` lags of the full system as
    controls. See the module docstring for why the lag count tunes efficiency here rather
    than the estimand.

    SEs/CIs: day-cluster bootstrap -- whole sessions resampled with replacement, theta
    re-estimated per draw (sigma_j included: it is part of the estimator). `se_ols` is the
    NAIVE iid OLS SE, reported so the clustering premium is visible per cell; it is not the
    inference.

    -> tidy DataFrame [shock, horizon, theta_x100, se_x100, ci_lo_x100, ci_hi_x100,
                       se_ols_x100, n_obs, n_days]."""
    p = max(1, int(n_lags)); H = int(horizon)
    horizons = list(range(H + 1))
    days, k = [], None
    for X in Xs:
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] == 0:
            continue
        if k is None:
            k = X.shape[1]
        if X.shape[1] != k:
            continue
        st = _day_lp_grams(X, corr_idx, p, horizons)
        if st is not None:
            days.append(st)
    if not days:
        return pd.DataFrame()
    D = len(days); m_full = k * (p + 1)
    Garr = {h: np.zeros((D, m_full, m_full)) for h in horizons}
    garr = {h: np.zeros((D, m_full)) for h in horizons}
    yyarr = {h: np.zeros(D) for h in horizons}
    narr = {h: np.zeros(D) for h in horizons}
    for d, st in enumerate(days):
        for h, (G, g, yy, n) in st.items():
            Garr[h][d] = G; garr[h][d] = g; yyarr[h][d] = yy; narr[h][d] = n
    shocks = [j for j in range(k) if j != corr_idx]

    def _estimate(counts, want_ols=False):
        sums = {}
        for h in horizons:
            sums[h] = (np.tensordot(counts, Garr[h], axes=1), counts @ garr[h],
                       float(counts @ yyarr[h]), float(counts @ narr[h]))
        G0, _g0, _yy0, n0 = sums[0]
        th = np.full((len(shocks), H + 1), np.nan)
        so = np.full((len(shocks), H + 1), np.nan)
        for a_, j in enumerate(shocks):
            sig = _sigma_shock(G0, n0, j, k, p)
            if not np.isfinite(sig):
                continue
            cols = _shock_cols(j, k, p); m = len(cols)
            for h in horizons:
                G, g, yy, n = sums[h]
                if n <= m + 2:
                    continue
                Gs = G[np.ix_(cols, cols)]
                B, rss = cs._gram_solve(Gs, g[cols][:, None], np.array([[yy]]))
                th[a_, h] = float(B[0, 0]) * sig
                if want_ols:
                    dd = np.sqrt(np.clip(np.diag(Gs), EPS, None))
                    inv00 = float(np.linalg.pinv(Gs / dd[:, None] / dd[None, :])[0, 0]) / dd[0] ** 2
                    s2 = max(float(rss[0, 0]), 0.0) / max(n - m, 1.0)
                    so[a_, h] = sig * np.sqrt(max(s2 * inv00, 0.0))
        return th, so, {h: sums[h][3] for h in horizons}

    th_pt, so_pt, n_by_h = _estimate(np.ones(D), want_ols=True)
    nb = int(n_boot or 0)
    draws = np.full((max(nb, 1), len(shocks), H + 1), np.nan)
    rng = np.random.default_rng(seed)
    for b in range(nb):
        counts = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        draws[b] = _estimate(counts)[0]
    with warnings.catch_warnings():                          # all-NaN slices are legitimate
        warnings.simplefilter("ignore")
        if nb > 1:
            se = np.nanstd(draws, axis=0, ddof=1)
            lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
            hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
        else:
            se = lo = hi = np.full((len(shocks), H + 1), np.nan)
    rows = []
    for a_, j in enumerate(shocks):
        for h in horizons:
            rows.append({"shock": names[j], "horizon": h,
                         "theta_x100": 100.0 * th_pt[a_, h], "se_x100": 100.0 * se[a_, h],
                         "ci_lo_x100": 100.0 * lo[a_, h], "ci_hi_x100": 100.0 * hi[a_, h],
                         "se_ols_x100": 100.0 * so_pt[a_, h],
                         "n_obs": int(n_by_h.get(h, 0)), "n_days": D})
    return pd.DataFrame(rows)


# ── Table 9 by LP: regimes, SVAR comparison column ────────────────────────────
def lp_table9(sessions, spec="informational", n_lags=6, horizon=10, n_boot=199, alpha=0.05,
              seed=0, bar_seconds=60, n_levels=10, target_qty=None,
              wspread_kind="cost_to_fill", min_bars=None, with_svar=True):
    """The full exhibit: panel LP per regime on the RealBar frame, plus -- per cell, when
    the SVAR is estimable on the identical frame at the identical p and ordering -- the
    SVAR IRF at the same horizon. The `svar_x100` column is the point of the table: where
    the two agree, the SVAR's lag choice was innocuous; where they diverge at h >= 2, the
    divergence is the compounding the LP exists to avoid.

    -> tidy DataFrame [regime, shock, horizon, theta_x100, se_x100, ci_lo_x100, ci_hi_x100,
                       se_ols_x100, n_obs, n_days, svar_x100, lp_minus_svar_x100]."""
    p = max(1, int(n_lags)); H = int(horizon)
    if min_bars is None:
        min_bars = p + H + 10
    groups = {}
    for item in ([("all", "all", sessions)] if isinstance(sessions, pd.DataFrame) else sessions):
        regime = item[1] if len(item) == 3 else "all"
        groups.setdefault(regime, []).append(item[-1])
    out = []
    for regime in sorted(groups):
        Xs, names, ci = [], None, None
        for df in groups[regime]:
            X, nm, c = build_lp_bar_frame(df, spec, n_levels, target_qty, wspread_kind,
                                          bar_seconds)
            if c >= 0 and int(np.all(np.isfinite(X), axis=1).sum()) >= min_bars:
                Xs.append(X); names, ci = nm, c
        if not Xs:
            continue
        tab = panel_lp_irf(Xs, names, ci, n_lags=p, horizon=H, n_boot=n_boot,
                           alpha=alpha, seed=seed)
        if tab.empty:
            continue
        tab.insert(0, "regime", regime)
        svar = np.full((len(names), H + 1), np.nan)
        if with_svar:
            # the SVAR fitted on the SAME frame's clean rows, same p, same ordering -- the
            # comparison must isolate estimator (iterated vs direct), nothing else
            try:
                Xc = [x[np.all(np.isfinite(x), axis=1)] for x in Xs]
                if len(Xc) > 1:
                    A, Sigma, _, _ = cs.panel_var_ols(Xc, p, fe=True)
                else:
                    A, Sigma, _ = cs.var_ols(Xc[0], p)
                Theta = cs.orthogonalized_irf(A, Sigma, H, "cholesky")
                for h in range(H + 1):
                    svar[:, h] = 100.0 * Theta[h][ci, :]
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                pass
        name_pos = {nm: j for j, nm in enumerate(names)}
        tab["svar_x100"] = [svar[name_pos[s], h] for s, h in zip(tab["shock"], tab["horizon"])]
        tab["lp_minus_svar_x100"] = tab["theta_x100"] - tab["svar_x100"]
        out.append(tab)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ── rendering (the runner's table style: title / markdown block / italic note) ─
def _wide(sub, value_fmt):
    shocks = list(dict.fromkeys(sub["shock"]))
    hs = sorted(sub["horizon"].unique())
    W = pd.DataFrame(index=pd.Index(shocks, name="shock"),
                     columns=[f"h={h}" for h in hs], dtype=object)
    for _i, r in sub.iterrows():
        W.loc[r["shock"], f"h={int(r['horizon'])}"] = value_fmt(r)
    return W


def render_tables(tab, n_lags, alpha, n_boot):
    """-> list of paper_tables.Table (two per regime: LP with CIs, SVAR comparison)."""
    import paper_tables as pt
    ci_pct = int(round(100 * (1 - alpha)))
    tables = []
    for regime in list(dict.fromkeys(tab["regime"])):
        sub = tab[tab["regime"] == regime]
        n_days = int(sub["n_days"].iloc[0])
        lp = _wide(sub, lambda r: ("--" if not np.isfinite(r["theta_x100"]) else
                                   "%.3f [%.3f, %.3f]" % (r["theta_x100"], r["ci_lo_x100"],
                                                          r["ci_hi_x100"])))
        tables.append(pt.Table(
            f"Table 9 by panel local projections -- {regime}", lp,
            f"Response of RealBar d Fisher-z correlation to a 1-SD structural shock, x100; "
            f"recursive ordering as in the SVAR; {n_lags} within-day control lag(s) (efficiency "
            f"only -- the estimand needs no lag order); day FE; response windows crossing a "
            f"day boundary or a halt-masked bar dropped. [{ci_pct}% day-cluster bootstrap CI, "
            f"n_boot={int(n_boot)}, {n_days} days]."))
        sv = _wide(sub, lambda r: ("--" if not np.isfinite(r["svar_x100"]) else
                                   "%.3f" % r["svar_x100"]))
        tables.append(pt.Table(
            f"SVAR IRF at the same horizons -- {regime}", sv,
            f"Cholesky-orthogonalized VAR({n_lags}) IRF on the identical frame, ordering and "
            f"sample (comparison column): agreement means the lag choice was innocuous; "
            f"divergence at h >= 2 is the truncation compounding the LP avoids."))
    return tables


# ── CLI (same frames pickle / flags as the Table 9 runner) ────────────────────
def build_parser():
    ap = argparse.ArgumentParser(
        description="Table 9 by panel local projections (lag-order-free counterpart)")
    ap.add_argument("--source", choices=["demo", "load"], default="demo")
    ap.add_argument("--pickle", default="", help="glob for a List[(date[,regime],df)] pickle")
    ap.add_argument("--volatile", default="", help="comma-separated YYYY-MM-DD marked volatile")
    ap.add_argument("--spec", default="informational", help="standard | weighted | informational")
    ap.add_argument("--corr-window", type=int, default=100,
                    help="passed through to the (bar-frame) lag selection for parity with the "
                         "runner; the RealBar dependent variable itself is window-free")
    ap.add_argument("--bar-seconds", type=int, default=60)
    ap.add_argument("--n-lags", default="6",
                    help="within-day CONTROL lags: fixed integer, or bic | aic | hq resolved on "
                         "the window-free RealBar frame exactly as the runner does. Controls "
                         "tune efficiency only -- the LP estimand does not depend on this")
    ap.add_argument("--pmax", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=10,
                    help="largest projection horizon (matches the SVAR IRF horizon)")
    ap.add_argument("--n-boot", type=int, default=199,
                    help="day-cluster bootstrap draws; 0 = point estimates only")
    ap.add_argument("--alpha", type=float, default=0.05, help="CI level: (1-alpha) percentile CI")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-bars", type=int, default=None,
                    help="minimum clean bars for a session to enter (default n_lags+horizon+10)")
    ap.add_argument("--no-svar", dest="with_svar", action="store_false", default=True,
                    help="skip the SVAR comparison column")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--halt-mask", dest="halt_mask", action="store_true", default=True)
    ap.add_argument("--no-halt-mask", dest="halt_mask", action="store_false")
    ap.add_argument("--n-demo", type=int, default=8)
    ap.add_argument("--n-demo-bars", type=int, default=6000)
    ap.add_argument("--demo-refresh", type=float, default=0.3)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    warnings.simplefilter("ignore")
    import run_table9_both_ways as rt                        # same loader = same pickle contract
    sessions = rt._load(a)
    print("sessions: %d  regimes: %s" % (len(sessions), ", ".join(sorted({r for _d, r, _f in sessions}))))
    if a.source == "demo":
        print("NOTE: --source demo uses a SYNTHETIC staleness DGP. The numbers below demonstrate")
        print("      the format and that the machinery runs; they say nothing about SPY/ES.")
    # control-lag order: same resolution path as the runner (criterion scored on the
    # window-free RealBar frame). Here a bad p costs efficiency, not the estimand.
    n_lags, crit, ic = cs.resolve_n_lags(sessions, a.n_lags, pmax=a.pmax, spec=a.spec,
                                         corr_window=a.corr_window, bar_seconds=a.bar_seconds,
                                         panel="fe", corr_method="bar")
    if crit is not None:
        if n_lags is None:
            print("could not select a control lag order; pass --n-lags <int>", file=sys.stderr)
            return 1
        print("control lags: p=%d by %s over p<=%d on the RealBar bar frame "
              "(efficiency only -- LP estimates need no lag order)" % (n_lags, crit.upper(), a.pmax))
        if n_lags == 0:
            n_lags = 1
    tab = lp_table9(sessions, spec=a.spec, n_lags=n_lags, horizon=a.horizon, n_boot=a.n_boot,
                    alpha=a.alpha, seed=a.seed, bar_seconds=a.bar_seconds,
                    min_bars=a.min_bars, with_svar=a.with_svar)
    if tab.empty:
        print("no regime had enough usable bars")
        return 1
    pd.set_option("display.width", 240)
    for t in render_tables(tab, n_lags, a.alpha, a.n_boot):
        print()
        print(t.to_string())
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        stem = os.path.join(a.out_dir, "table9_local_projection")
        tab.to_csv(stem + ".csv", index=False)
        with open(stem + ".md", "w") as fh:
            fh.write("\n\n".join(t.to_markdown() for t in render_tables(tab, n_lags, a.alpha,
                                                                        a.n_boot)) + "\n")
        print("\nwrote %s.csv / .md" % stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
