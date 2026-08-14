"""
mean_variance.py
================
Layered mean + variance framework that composes the existing blocks into one pass,
for the SPY (ETF) / ES (future) pair:

  mean (prices)   : per-session VECM on log mid prices -> information shares (Hasbrouck IS,
                    Gonzalo-Granger CS) + the reduced-form return innovations
                    (price_discovery_shares._fit_vecm_fixed / hasbrouck_is / gonzalo_granger)
  variance        : DCC(1,1)-GARCH(1,1)-X on those innovations, with EACH asset's
                    decay-weighted cost-to-demand (AUC_decay) and impact convexity (||L||)
                    as its own GARCH-X variance covariates -> a time-varying correlation
                    rho_t and the liquidity->volatility loadings gamma   (dcc_garch.dcc_garch_x)
  contagion split : mean rho_t in stress vs calm rows, labelled by each session's regime.
                    Because rho_t is the correlation of the *standardized* (devolatilized)
                    residuals, a stress rise in rho_t is the Forbes-Rigobon contagion object,
                    not the mechanical vol-driven rise in raw correlation.

This adds NO new estimators -- only wiring. It is the conditional-mean + conditional-variance
home for both interdependence (mean spillover, handled in the SVAR/VECM layer elsewhere) and
contagion (a stress jump in the standardized correlation).

Design choices baked in (see the discussion):
  * the cointegrated price is the SIMPLE mid (standard Hasbrouck); the microprice-anchored
    AUC_decay / ||L|| enter only as variance covariates, so the inside-imbalance component of
    the microprice is not double-counted on both sides of an equation;
  * liquidity covariates are LAGGED (x_{t-1}) -> predetermined, not contemporaneous;
  * residuals are pooled across sessions for one DCC so rho_t spans calm and stress days
    (per-session VECM means; a handful of overnight-boundary rows are negligible intraday).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import price_discovery_shares as pds
import cross_asset_pd_liquidity as ca
import liquidity_curve_metrics as lcm
import dcc_garch as dg

EPS = 1e-12


def _inside_imbalance(df, asset):
    """Signed inside-size imbalance I = (Qask1 - Qbid1)/(Qask1 + Qbid1). This is the term by
    which the microprice differs from the mid: m_w - m = -(spread/2) * I."""
    qa = df[f"{asset}_askquantity_1"].to_numpy(float); qb = df[f"{asset}_bidquantity_1"].to_numpy(float)
    s = qa + qb
    return np.where(s > EPS, (qa - qb) / s, 0.0)


def _orthogonalize(x, z):
    """Residual of x on [1, z] (OLS), leaving non-finite rows untouched. Used to purge the
    linear inside-imbalance content from a covariate when the microprice is on BOTH sides."""
    x = np.asarray(x, float); z = np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(z)
    if int(ok.sum()) < 5:
        return x
    X = np.column_stack([np.ones(int(ok.sum())), z[ok]])
    b, *_ = np.linalg.lstsq(X, x[ok], rcond=None)
    out = x.copy(); out[ok] = x[ok] - X @ b
    return out


def _session_logprices(df, n_levels, lhs="mid", use_log=True):
    """The (p1, p2) cointegrated price pair for a session under the chosen LHS price, finite-masked
    and log-transformed -- shared by the pooled lag pre-pass and _session_pieces so they agree."""
    if lhs == "wmid":
        ps = np.asarray(lcm.weighted_mid(df, "SPY"), float); pe = np.asarray(lcm.weighted_mid(df, "ES"), float)
    elif lhs == "depth_wmid":
        ps = np.asarray(lcm.depth_weighted_mid(df, "SPY", n_levels), float)
        pe = np.asarray(lcm.depth_weighted_mid(df, "ES", n_levels), float)
    else:
        ps = np.asarray(ca._mid(df, "SPY"), float); pe = np.asarray(ca._mid(df, "ES"), float)
    ok = np.isfinite(ps) & np.isfinite(pe)
    f = np.log if use_log else (lambda x: np.asarray(x, float))
    return f(ps[ok]), f(pe[ok])


def _session_pieces(df, n_lags, n_levels, decay_scale, lhs="mid", anchor="wmid",
                    orth=False, use_log=True, criterion=None, pmax=12):
    """One session -> (shares dict, resid[L,2], X_spy[L,2], X_es[L,2], rho_times[L]) or None.
    lhs    : 'mid' or 'wmid' -> the cointegrated (LHS) price.
    anchor : 'mid' or 'wmid' -> the reference the AUC_decay/||L|| covariates are measured from.
    orth   : if True, residualize each covariate on that asset's inside imbalance (config C).
    criterion: if set, choose the VECM lag by that criterion for this session (else use n_lags)."""
    if lhs == "wmid":
        price_fn = lambda a: np.asarray(lcm.weighted_mid(df, a), float)
    elif lhs == "depth_wmid":
        price_fn = lambda a: np.asarray(lcm.depth_weighted_mid(df, a, n_levels), float)
    else:
        price_fn = lambda a: np.asarray(ca._mid(df, a), float)
    ps, pe = price_fn("SPY"), price_fn("ES")
    ok = np.isfinite(ps) & np.isfinite(pe)
    if int(ok.sum()) < n_lags + 60:
        return None
    dfm = df.iloc[np.flatnonzero(ok)]
    f = np.log if use_log else (lambda x: np.asarray(x, float))
    p1 = f(ps[ok]); p2 = f(pe[ok]); M = p1.shape[0]
    if criterion is not None:
        n_lags, _ = pds.select_lag(p1, p2, pmax=pmax, criterion=criterion)
    alpha, Omega, resid = pds._fit_vecm_fixed(p1, p2, n_lags)        # resid length M-1-n_lags
    lo, hi, mid = pds.hasbrouck_is(alpha, Omega)
    cs = pds.gonzalo_granger(alpha)
    shares = {"alpha_spy": float(alpha[0]), "alpha_es": float(alpha[1]),
              "IS_mid_SPY": float(mid[0]), "IS_mid_ES": float(mid[1]),
              "CS_ES": float(cs[1]), "n_obs": int(M - n_lags)}
    # liquidity covariates on the SAME masked rows, anchored as requested, optionally orthogonalized
    cov = {}
    for a in ("SPY", "ES"):
        auc = np.asarray(lcm.decay_weighted_cost(dfm, a, n_levels=n_levels, decay_scale=decay_scale, anchor=anchor))
        cl = np.asarray(lcm.impact_curve_length(dfm, a, n_levels=n_levels, anchor=anchor))
        if orth:
            imb = _inside_imbalance(dfm, a)
            auc = _orthogonalize(auc, imb); cl = _orthogonalize(cl, imb)
        cov[a] = (auc, cl)
    sl = slice(n_lags, M - 1)                                        # x_{t-1} aligned to resid
    X_spy = np.column_stack([cov["SPY"][0][sl], cov["SPY"][1][sl]])
    X_es = np.column_stack([cov["ES"][0][sl], cov["ES"][1][sl]])
    rho_times = np.asarray(dfm.index[n_lags + 1:M])
    return shares, resid, X_spy, X_es, rho_times


_PRICE_CONFIGS = {
    # name        : (LHS price, curve anchor, orthogonalize covariates)
    "mid":        ("mid",  "wmid", False),   # A: simple-mid cointegration, microprice-anchored curves (cleanest)
    "wmid":       ("wmid", "mid",  False),   # B: microprice cointegration, simple-mid-anchored curves
    "wmid_orth":  ("wmid", "wmid", True),    # C: microprice on both, covariates orthogonalized to inside imbalance
    "depth_wmid": ("depth_wmid", "mid", False),   # D: depth-weighted-mid cointegration, simple-mid-anchored curves
    "mid_depth":  ("mid", "depth_wmid", False),   # E: simple-mid cointegration, depth-weighted-mid-anchored curves
}


def run_mean_variance(sessions, n_lags=5, n_levels=10, decay_scale=None, price="mid",
                      stress_regimes=("volatile", "stress", "crisis"),
                      out_dir=None, write=False, verbose=True, criterion=None, pmax=12,
                      lag_selection="pooled"):
    """Estimate the layered mean+variance object on one session (a DataFrame) or a list of
    (label, regime, df) sessions.

    price selects the cointegration/covariate configuration (all keep the microprice off both
    sides of an equation, so the inside-imbalance term is never double-counted):
      'mid'       (default) ln(simple mid) on the LHS; AUC_decay/||L|| anchored at the microprice.
      'wmid'      ln(microprice) on the LHS (better efficient-price proxy); curves anchored at the
                  simple mid, so the RHS carries no inside imbalance.
      'wmid_orth' ln(microprice) on BOTH the LHS and the curve anchor, with each covariate
                  residualized on that asset's inside-size imbalance to purge the shared term.
      'depth_wmid' ln(depth-weighted mid across n_levels) on the LHS (smoother, reflects deeper
                  liquidity); curves anchored at the simple mid, so the RHS carries no imbalance.
      'mid_depth' ln(simple mid) on the LHS; curves anchored at the depth-weighted mid (cost
                  measured vs a deep fair value -- near-touch marginal cost may be negative).

    Returns information shares (per session + pooled means), the time-varying SPY-ES correlation
    rho_t with a calm-vs-stress split, the liquidity->vol GARCH-X loadings gamma, and the DCC
    persistence."""
    log = (lambda m: print(m)) if verbose else (lambda m: None)
    if price not in _PRICE_CONFIGS:
        raise ValueError("price must be one of %s" % list(_PRICE_CONFIGS))
    lhs, anchor, orth = _PRICE_CONFIGS[price]
    if isinstance(sessions, pd.DataFrame):
        sessions = [("session", "benchmark", sessions)]

    sess_crit = None                                        # per-session criterion passed downstream
    if criterion is not None and lag_selection == "pooled":
        pairs = [_session_logprices(df, n_levels, lhs) for _l, _r, df in sessions]
        n_lags = pds.select_lag_pooled(pairs, pmax=pmax, criterion=criterion)[0]
        log("[mean-variance] pooled %s lag = %d (applied to all sessions)" % (criterion, n_lags))
    elif criterion is not None and lag_selection == "per_day":
        sess_crit = criterion

    rows, R, Xs, Xe, reg = [], [], [], [], []
    for label, regime, df in sessions:
        out = _session_pieces(df, n_lags, n_levels, decay_scale, lhs=lhs, anchor=anchor, orth=orth,
                              criterion=sess_crit, pmax=pmax)
        if out is None:
            log("[mean-variance] %s skipped (insufficient finite mids)" % label); continue
        shares, resid, X_spy, X_es, _t = out
        rows.append({"session": str(label), "regime": regime, **shares})
        R.append(resid); Xs.append(X_spy); Xe.append(X_es)
        reg.append(np.array([regime] * resid.shape[0], dtype=object))
    if not R:
        return {"note": "no usable sessions"}

    resid = np.vstack(R); X_spy = np.vstack(Xs); X_es = np.vstack(Xe); reg = np.concatenate(reg)
    finite = np.all(np.isfinite(np.column_stack([resid, X_spy, X_es])), axis=1)  # so rho aligns to reg
    resid, X_spy, X_es, reg = resid[finite], X_spy[finite], X_es[finite], reg[finite]
    log("[mean-variance] DCC-GARCH-X on %d pooled innovation rows from %d session(s)"
        % (resid.shape[0], len(rows)))
    dccx = dg.dcc_garch_x(resid, names=("SPY", "ES"), X_per_asset=[X_spy, X_es])
    rho = np.asarray(dccx["rho"], float)

    stress = np.isin(reg, np.asarray(stress_regimes, dtype=object))
    rho_stress = float(np.nanmean(rho[stress])) if stress.any() else np.nan
    rho_calm = float(np.nanmean(rho[~stress])) if (~stress).any() else np.nan

    def _g(asset, j):
        g = dccx.get("marginals", {}).get(asset, {}).get("gamma", [])
        g = np.atleast_1d(np.asarray(g, float))
        return float(g[j]) if g.size > j else np.nan
    gamma = {a: {"auc_decay": _g(a, 0), "curve_length": _g(a, 1)} for a in ("SPY", "ES")}

    info = pd.DataFrame(rows)
    res = {"price_config": {"name": price, "lhs_price": lhs, "curve_anchor": anchor, "orthogonalized": orth},
           "info_shares": info,
           "info_shares_pooled": {"CS_ES_mean": float(info["CS_ES"].mean()),
                                  "IS_mid_ES_mean": float(info["IS_mid_ES"].mean())},
           "rho": rho, "regime_rows": reg, "rho_mean": float(np.nanmean(rho)),
           "rho_calm": rho_calm, "rho_stress": rho_stress,
           "rho_diff_stress_minus_calm": (rho_stress - rho_calm)
           if (np.isfinite(rho_stress) and np.isfinite(rho_calm)) else np.nan,
           "gamma_liquidity_to_vol": gamma, "dcc_a": dccx["dcc_a"], "dcc_b": dccx["dcc_b"],
           "n_rows": int(resid.shape[0])}
    if write:
        out_dir = out_dir or os.environ.get("OUT_DIR", "/mnt/user-data/outputs")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "mean_variance_summary.md"), "w") as fh:
            fh.write("# Mean + variance (VECM -> DCC-GARCH-X) summary\n\n")
            fh.write("price config: **%s** (LHS=ln(%s), curve anchor=%s, covariates orthogonalized=%s)\n\n"
                     % (price, lhs, anchor, orth))
            fh.write("## Information shares (per session; mean-layer VECM)\n\n")
            fh.write(info.round(4).to_string(index=False) + "\n\n")
            fh.write("## Time-varying SPY-ES correlation (standardized residuals)\n\n")
            fh.write("mean rho=%.4f | calm=%.4f | stress=%.4f | stress-calm=%.4f | DCC a=%.3f b=%.3f\n\n"
                     % (res["rho_mean"], rho_calm, rho_stress,
                        res["rho_diff_stress_minus_calm"], res["dcc_a"], res["dcc_b"]))
            fh.write("(rho_t is the correlation of the devolatilized residuals, so the stress-calm "
                     "gap is the Forbes-Rigobon contagion object, not the mechanical vol effect.)\n\n")
            fh.write("## Liquidity -> volatility loadings (GARCH-X gamma, per asset)\n\n")
            fh.write(str(gamma) + "\n")
        res["artifact"] = "mean_variance_summary.md"
        log("[mean-variance] wrote mean_variance_summary.md")
    return res


# ── self-test ─────────────────────────────────────────────────────────────────
def _synth(regime, T=1300, n=8, seed=0):
    """Cointegrated SPY/ES log-mids (ES leads via partial SPY adjustment) + a depth book.
    Stress: more common-factor variance (higher return correlation) AND a thinner book
    (higher AUC_decay). Transitory (stationary) microstructure noise so prices cointegrate."""
    rng = np.random.default_rng(seed)
    sig_c = 0.0009 if regime == "volatile" else 0.00035          # common shock sd (stress bigger)
    sig_i = 0.00030
    f = np.cumsum(rng.normal(0, sig_c, T))                       # common efficient (log) price
    spy_w = rng.normal(0, sig_i, T); es_w = rng.normal(0, sig_i, T)   # transitory noise
    f_lag = np.r_[f[0], f[:-1]]
    mid_es = 500.0 * np.exp(f + es_w)
    mid_spy = 500.0 * np.exp(0.7 * f + 0.3 * f_lag + spy_w)       # SPY only partially adjusts -> ES leads
    inside = 80.0 if regime == "volatile" else 320.0             # thinner book in stress
    idx = pd.date_range("2020-03-12 09:30", periods=T, freq="s", tz="America/New_York")
    d = {}
    for asset, mid, tick in (("SPY", mid_spy, 0.01), ("ES", mid_es, 0.25)):
        for sd, s in (("ask", 1), ("bid", -1)):
            for i in range(1, n + 1):
                d[f"{asset}_{sd}price_{i}"] = mid + s * tick * i
                d[f"{asset}_{sd}quantity_{i}"] = inside * np.exp(-0.2 * (i - 1)) * (1 + rng.normal(0, 0.02, T))
    return pd.DataFrame(d, index=idx)


def _selftest():
    print("mean_variance self-test")
    sessions = [("calm1", "benchmark", _synth("benchmark", seed=1)),
                ("calm2", "benchmark", _synth("benchmark", seed=2)),
                ("stress1", "volatile", _synth("volatile", seed=3)),
                ("stress2", "volatile", _synth("volatile", seed=4))]
    all_ok = True
    for price in ("mid", "wmid", "wmid_orth", "depth_wmid", "mid_depth"):
        res = run_mean_variance(sessions, n_lags=5, n_levels=8, price=price, verbose=False)
        info = res["info_shares"]; g = res["gamma_liquidity_to_vol"]
        ok = (len(info) == 4
              and (info["IS_mid_SPY"] + info["IS_mid_ES"] - 1.0).abs().max() < 1e-6      # IS sums to 1
              and info["CS_ES"].between(0, 1).all()
              and np.isfinite(res["rho_mean"]) and -1 <= res["rho_mean"] <= 1
              and res["rho_stress"] > res["rho_calm"]                                    # stress comovement higher
              and all(np.isfinite(v) for asset in g.values() for v in asset.values()))   # gamma finite
        all_ok = all_ok and ok
        print("  [%-9s] LHS=%-4s anchor=%-4s orth=%-5s | CS_ES=%.3f rho calm/stress=%.3f/%.3f | ok=%s"
              % (price, res["price_config"]["lhs_price"], res["price_config"]["curve_anchor"],
                 res["price_config"]["orthogonalized"], res["info_shares_pooled"]["CS_ES_mean"],
                 res["rho_calm"], res["rho_stress"], ok))
    print("  checks:", all_ok)


if __name__ == "__main__":
    _selftest()
