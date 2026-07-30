"""
mwcb_event_study.py
==================
Causal identification of the March-2020 MWCB **reopening asymmetry** as a liquidity-
contagion experiment (item #1 of the redo). At each halt release the futures book was
frozen-then-empty while the ETF primary listing had accumulated orders auction-style --
differential treatment of the two books at a sharp, exogenously-timed boundary.

Outcome (parameter `outcome`, the liquidity object that spills over):
  * 'cost_to_fill' (DEFAULT) : depth-weighted round-trip cost-to-fill spread (bps),
                               HIGHER = MORE ILLIQUID. (correlation_svar.weighted_spread)
  * 'inside_depth'           : summed top-`n_inside` bid+ask size,
                               HIGHER = MORE LIQUID (sign flips vs the spread).
  * 'composite'              : z(cost_to_fill) - z(inside_depth), HIGHER = MORE ILLIQUID.

Estimators, all sharing the outcome parameter:
  * event_study  : the outcome's time profile relative to the release instant (the
                   liquidity analogue of the paper's Figure 3 price plots).
  * did          : 2x2 difference-in-differences (treated MWCB day vs matched control,
                   pre- vs post-release) + a regression DiD with date-clustered SEs.
  * rd_in_time   : sharp regression-discontinuity at the release boundary (local linear
                   each side; the jump at tau=0 is the causal effect).
  * cross_market_spillover : does the source market's illiquidity transmit to the target's,
                   amplified post-release on MWCB days (a triple difference) -- the
                   contagion core.

Treatment is the reopening asymmetry dated at the release boundary; it is NOT a vol
threshold (see stress_index). Controls are matched calm days (stress_index.match_controls
or same time-of-day benchmark). Pure numpy / pandas.
"""
import numpy as np
import pandas as pd

import correlation_svar as csv
import liquidity_curve_metrics as lcm
import price_discovery_shares as pds

EPS = 1e-12
ORIENT = {"cost_to_fill": "illiquidity (higher = worse)",
          "inside_depth": "liquidity (higher = better)",
          "composite": "illiquidity (higher = worse)",
          "auc_decay": "illiquidity: decay-weighted cost-to-demand, bps (higher = worse)",
          "curve_length": "impact convexity: arc length of the supply curve (higher = more nonlinear)"}


# ── outcome (liquidity) observables ───────────────────────────────────────────
def illiquidity(df, asset, outcome="cost_to_fill", n_inside=1, target_qty=None, n_levels=10):
    """Per-bar liquidity outcome for the contagion design (Series indexed like df).
    See module docstring / ORIENT for the sign of each measure."""
    if outcome == "inside_depth":
        d = np.zeros(len(df))
        for i in range(1, n_inside + 1):
            d += df[f"{asset}_bidquantity_{i}"].to_numpy() + df[f"{asset}_askquantity_{i}"].to_numpy()
        return pd.Series(d, index=df.index)
    if outcome == "auc_decay":                              # decay-weighted cost-to-demand (bps); target_qty -> Q0
        return pd.Series(lcm.decay_weighted_cost(df, asset, n_levels=n_levels, decay_scale=target_qty,
                                                 side="both"), index=df.index)
    if outcome == "curve_length":                           # dimensionless impact-convexity index ||L||
        return pd.Series(lcm.impact_curve_length(df, asset, n_levels=n_levels, side="both"), index=df.index)
    cf = csv.weighted_spread(df, asset, target_qty, n_levels, "cost_to_fill")
    if outcome == "cost_to_fill":
        return pd.Series(cf, index=df.index)
    if outcome == "composite":
        dep = np.zeros(len(df))
        for i in range(1, n_inside + 1):
            dep += df[f"{asset}_bidquantity_{i}"].to_numpy() + df[f"{asset}_askquantity_{i}"].to_numpy()
        zc = (cf - np.nanmean(cf)) / (np.nanstd(cf) or 1.0)
        zd = (dep - np.nanmean(dep)) / (np.nanstd(dep) or 1.0)
        return pd.Series(zc - zd, index=df.index)
    raise ValueError(f"unknown outcome {outcome!r}")


# ── cluster-robust OLS (clustered by event date) ─────────────────────────────
def _ols_cluster(Y, X, groups):
    beta, resid = pds._ols(Y, X)
    beta = beta.ravel(); u = resid.ravel()
    XtXinv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    uniq = np.unique(groups)
    for g in uniq:
        idx = np.where(groups == g)[0]
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    G = len(uniq); n, k = X.shape
    adj = (G / (G - 1.0)) * ((n - 1.0) / (n - k)) if (G > 1 and n > k) else 1.0
    V = XtXinv @ meat @ XtXinv * adj
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    return beta, se, beta / np.where(se > EPS, se, np.nan)


# ── bar-level event panel ─────────────────────────────────────────────────────
def build_panel(treated, control, release_by_date, asset="SPY", outcome="cost_to_fill",
                pre_min=5, post_min=5, n_inside=1, target_qty=None, n_levels=10):
    """Stack treated (MWCB) and control sessions into a bar-level panel around each
    release: columns date, t (seconds since release; <0 before), illiq, treated, post.
    `release_by_date` maps each session's date key to its (pseudo-)release Timestamp."""
    rows = []
    for sessions, is_t in ((treated, 1), (control, 0)):
        for date, df in sessions:
            rel = pd.Timestamp(release_by_date[date])
            il = np.asarray(illiquidity(df, asset, outcome, n_inside, target_qty, n_levels), float)
            secs = (df.index - rel).total_seconds().to_numpy() if hasattr(
                (df.index - rel), "total_seconds") else np.asarray((df.index - rel) / np.timedelta64(1, "s"), float)
            keep = (secs >= -pre_min * 60) & (secs <= post_min * 60)
            rows.append(pd.DataFrame({"date": str(date), "t": secs[keep], "illiq": il[keep],
                                      "treated": is_t, "post": (secs[keep] > 0).astype(int)}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["illiq"])


# ── (A) event-study time profile ──────────────────────────────────────────────
def event_study(panel, bin_sec=30, by_treated=True):
    """Mean outcome by event-time bin (seconds; negative = before release), with SEM and
    count. by_treated=True returns treated and control profiles separately."""
    p = panel.copy()
    p["bin"] = (np.floor(p["t"] / bin_sec) * bin_sec).astype(int)
    keys = ["bin", "treated"] if by_treated else ["bin"]
    out = p.groupby(keys)["illiq"].agg(["mean", "sem", "count"]).reset_index()
    return out.sort_values(keys)


# ── (B) difference-in-differences ─────────────────────────────────────────────
def did(panel):
    """2x2 DiD (post-pre)_treated - (post-pre)_control on the outcome, plus a regression
    DiD (illiq ~ treated + post + treated*post) with date-clustered SEs. The treated*post
    coefficient is the causal effect of the reopening asymmetry on the outcome."""
    g = panel.groupby(["treated", "post"])["illiq"].mean()
    cell = lambda tr, po: float(g.get((tr, po), np.nan))
    t_pre, t_post, c_pre, c_post = cell(1, 0), cell(1, 1), cell(0, 0), cell(0, 1)
    X = np.column_stack([np.ones(len(panel)), panel["treated"].to_numpy(),
                         panel["post"].to_numpy(), (panel["treated"] * panel["post"]).to_numpy()])
    beta, se, t = _ols_cluster(panel["illiq"].to_numpy()[:, None], X, panel["date"].to_numpy())
    return {"cells": {"treated_pre": t_pre, "treated_post": t_post,
                      "control_pre": c_pre, "control_post": c_post},
            "treated_diff": t_post - t_pre, "control_diff": c_post - c_pre,
            "did_simple": (t_post - t_pre) - (c_post - c_pre),
            "did_coef": beta[3], "did_se": se[3], "did_t": t[3]}


# ── (C) regression discontinuity in time at the release boundary ──────────────
def rd_in_time(panel, bandwidth_min=5, treated_only=True):
    """Sharp RD at tau=0: local-linear fit illiq ~ 1 + post + tau + post*tau within
    +/- bandwidth_min of the release; the `post` coefficient is the discontinuity (jump).
    Date-clustered SEs."""
    p = panel[panel["treated"] == 1] if treated_only else panel
    h = bandwidth_min * 60.0
    p = p[p["t"].abs() <= h]
    tau = p["t"].to_numpy() / 60.0
    post = (tau > 0).astype(float)
    X = np.column_stack([np.ones(len(tau)), post, tau, post * tau])
    beta, se, t = _ols_cluster(p["illiq"].to_numpy()[:, None], X, p["date"].to_numpy())
    return {"jump": beta[1], "jump_se": se[1], "jump_t": t[1],
            "slope_pre": beta[2], "slope_post": beta[2] + beta[3], "n": len(p)}


# ── (D) cross-market spillover (contagion core) ───────────────────────────────
def _spillover_regression(m):
    """d(target illiq) ~ d(source illiq) interacted with post and treated. The
    spill_post_treated coefficient is the extra source->target transmission post-release
    on MWCB days (the contagion amplification)."""
    ds = m["d_src"].to_numpy(); post = m["post"].to_numpy().astype(float); tr = m["treated"].to_numpy().astype(float)
    X = np.column_stack([np.ones(len(m)), ds, ds * post, ds * post * tr, post, tr, post * tr])
    beta, se, t = _ols_cluster(m["d_tgt"].to_numpy()[:, None], X, m["date"].to_numpy())
    names = ["const", "spill", "spill_post", "spill_post_treated", "post", "treated", "post_treated"]
    return {n: {"coef": float(beta[i]), "se": float(se[i]), "t": float(t[i])} for i, n in enumerate(names)}


def cross_market_spillover(treated, control, release_by_date, source="ES", target="SPY",
                           outcome="cost_to_fill", pre_min=5, post_min=5, target_qty=None, n_levels=10):
    """Build source and target illiquidity panels, merge on (date, event-time), and
    regress the target's illiquidity change on the source's, interacted with post-release
    and treated. Returns the spillover coefficients (see _spillover_regression)."""
    ps = build_panel(treated, control, release_by_date, source, outcome, pre_min, post_min,
                     target_qty=target_qty, n_levels=n_levels).rename(columns={"illiq": "illiq_src"})
    pt = build_panel(treated, control, release_by_date, target, outcome, pre_min, post_min,
                     target_qty=target_qty, n_levels=n_levels).rename(columns={"illiq": "illiq_tgt"})
    m = ps.merge(pt[["date", "t", "illiq_tgt"]], on=["date", "t"]).sort_values(["date", "t"])
    m["d_src"] = m.groupby("date")["illiq_src"].diff()
    m["d_tgt"] = m.groupby("date")["illiq_tgt"].diff()
    m = m.dropna(subset=["d_src", "d_tgt"])
    return _spillover_regression(m)


# ── self-test ─────────────────────────────────────────────────────────────────
def _book(release, treated, seed, pre_min=8, post_min=8, n_levels=10, withdraw=0.30, decay_min=4):
    """Synthetic two-asset book around a release. On treated days both books withdraw
    depth at release (futures more) and recover over decay_min -> cost-to-fill rises and
    inside depth falls post-release; control days are flat."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(release - pd.Timedelta(minutes=pre_min),
                        release + pd.Timedelta(minutes=post_min), freq="s", tz=release.tz)
    secs = (idx - release).total_seconds().to_numpy()
    ramp = np.clip(secs / (decay_min * 60.0), 0.0, 1.0)        # 0 at release -> 1 after decay
    cols = {}
    for asset, base_px, tick, base_depth, wd in (("SPY", 500.0, 0.01, 320.0, withdraw),
                                                 ("ES", 5000.0, 0.25, 220.0, withdraw * 0.6)):
        mid = base_px * np.exp(rng.normal(0, 1, len(idx)).cumsum() / 3000.0)
        factor = np.ones(len(idx))
        if treated:
            wf = wd + (1.0 - wd) * ramp
            factor = np.where(secs > 0, wf, 1.0)
        depth = base_depth * factor * (1 + rng.normal(0, 0.03, len(idx)))
        for l in range(1, n_levels + 1):
            q = np.maximum(depth * np.exp(-0.3 * (l - 1)), 1.0)
            cols[f"{asset}_bidprice_{l}"] = mid - l * tick; cols[f"{asset}_askprice_{l}"] = mid + l * tick
            cols[f"{asset}_bidquantity_{l}"] = q; cols[f"{asset}_askquantity_{l}"] = q
    return pd.DataFrame(cols, index=idx)


def _selftest():
    print("mwcb_event_study self-test")
    tz = "America/New_York"
    rel = pd.Timestamp("2020-03-09 09:49:13", tz=tz)
    treated, control, rby = [], [], {}
    for k in range(4):
        dt = f"T{k}"; treated.append((dt, _book(rel, True, seed=10 + k))); rby[dt] = rel
        dc = f"C{k}"; control.append((dc, _book(rel, False, seed=50 + k))); rby[dc] = rel
    Q = 120.0

    # (1) outcome observables: cost-to-fill rises and inside depth falls post-release on treated
    dft = treated[0][1]
    cf = illiquidity(dft, "SPY", "cost_to_fill", target_qty=Q)
    dep = illiquidity(dft, "SPY", "inside_depth")
    rel_secs = (dft.index - rel).total_seconds().to_numpy()
    pre, post = rel_secs < 0, rel_secs > 0
    print("\n(1) outcome observables (treated SPY)")
    print("    cost-to-fill: pre mean=%.3f -> post mean=%.3f bps  (rises)"
          % (np.nanmean(cf[pre]), np.nanmean(cf[post])))
    print("    inside depth: pre mean=%.1f -> post mean=%.1f       (falls)"
          % (np.nanmean(dep[pre]), np.nanmean(dep[post])))
    print("    checks:", np.nanmean(cf[post]) > np.nanmean(cf[pre]) and np.nanmean(dep[post]) < np.nanmean(dep[pre]))

    # (2) DiD on both outcomes
    pe_cf = build_panel(treated, control, rby, "SPY", "cost_to_fill", target_qty=Q)
    pe_dp = build_panel(treated, control, rby, "SPY", "inside_depth")
    d_cf = did(pe_cf); d_dp = did(pe_dp)
    print("\n(2) difference-in-differences (treated vs control, pre vs post)")
    print("    cost_to_fill : DiD=%.3f bps (t=%.1f)  [+ => reopening makes the ETF book costlier]"
          % (d_cf["did_coef"], d_cf["did_t"]))
    print("    inside_depth : DiD=%.1f    (t=%.1f)  [- => reopening thins the ETF book]"
          % (d_dp["did_coef"], d_dp["did_t"]))
    print("    checks:", d_cf["did_coef"] > 0 and d_cf["did_t"] > 2 and d_dp["did_coef"] < 0 and d_dp["did_t"] < -2)

    # (3) event-study profile + (C) RD-in-time jump
    es = event_study(pe_cf, bin_sec=30)
    es_t = es[es["treated"] == 1]
    rd = rd_in_time(pe_cf, bandwidth_min=4)
    print("\n(3) event study + RD-in-time (cost_to_fill, treated)")
    print("    profile: min bin mean=%.3f (pre) ... max bin mean=%.3f (post)"
          % (es_t["mean"].min(), es_t["mean"].max()))
    print("    RD jump at release = %.3f bps (t=%.1f), slope_pre=%.4f slope_post=%.4f"
          % (rd["jump"], rd["jump_t"], rd["slope_pre"], rd["slope_post"]))
    print("    checks:", rd["jump"] > 0 and rd["jump_t"] > 2 and es_t["mean"].max() > es_t["mean"].min())

    # (4) cross-market spillover estimator (direct synthetic panel with known amplification)
    rng = np.random.default_rng(3)
    parts = []
    for d in range(8):
        n = 300; post = (rng.random(n) > 0.5).astype(int); tr = int(d < 4)
        d_src = rng.normal(0, 1, n)
        d_tgt = (0.2 + 0.8 * (post * tr)) * d_src + rng.normal(0, 0.3, n)   # amp 0.8 only post & treated
        parts.append(pd.DataFrame({"date": f"D{d}", "d_src": d_src, "d_tgt": d_tgt, "post": post, "treated": tr}))
    sp = _spillover_regression(pd.concat(parts, ignore_index=True))
    # also smoke the book-based path
    sp_book = cross_market_spillover(treated, control, rby, "ES", "SPY", "cost_to_fill", target_qty=Q)
    print("\n(4) cross-market spillover (ES -> SPY)")
    print("    baseline spill=%.3f (t=%.1f) ; post*treated amplification=%.3f (t=%.1f)"
          % (sp["spill"]["coef"], sp["spill"]["t"], sp["spill_post_treated"]["coef"], sp["spill_post_treated"]["t"]))
    print("    book-path amplification coef = %.3f (t=%.1f) [+ => futures illiquidity spills to the ETF at reopen]"
          % (sp_book["spill_post_treated"]["coef"], sp_book["spill_post_treated"]["t"]))
    print("    checks:",
          abs(sp["spill"]["coef"] - 0.2) < 0.05 and abs(sp["spill_post_treated"]["coef"] - 0.8) < 0.07
          and sp["spill_post_treated"]["t"] > 3
          and np.isfinite(sp_book["spill_post_treated"]["coef"]))


if __name__ == "__main__":
    _selftest()
