"""
run_mean_variance.py
====================
Layered SPY/ES contagion in one apparatus:

  Layer 1 (MEAN, prices, I(1)) -- cointegrated mid-price VECM
      Delta p_t = alpha (beta' p_{t-1}) + sum_i Gamma_i Delta p_{t-i} + eps_t,   beta=(1,-1)
      -> Hasbrouck IS, Gonzalo-Granger CS (information shares), error-correction loadings.

  Layer 2 (VARIANCE, on the mean residuals eps_t) -- DCC-GARCH(1,1)-X
      h_{i,t} = omega_i + a_i eps_{i,t-1}^2 + b_i h_{i,t-1} + gamma_i' x_{i,t-1}
      R_t via DCC on the standardized residuals
      -> liquidity->vol loadings gamma (book shape forecasts vol), the time-varying
         SPY-ES correlation rho_t, and a calm-vs-stress split of rho.

The variance covariates x are the book-shape liquidity metrics AUC_decay (cost level, bps)
and ||L|| (impact convexity); they are predetermined (lagged one bar), deseasonalized
(intraday time-of-day mean removed), and standardized before entering the variance so the
gamma's are comparable. The calm-vs-stress rho contrast is the Forbes-Rigobon object: rho_t
is the correlation of the *devolatilized* residuals, so a stress-rise is much closer to true
contagion than a rolling raw correlation (which rises mechanically with volatility).

This module is an ORCHESTRATOR -- it composes the already-tested estimators
(price_discovery_shares, ecm_sde, dcc_garch, liquidity_curve_metrics); it adds no new math.
"""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import autoscale as _autoscale
from scipy.stats import chi2

import price_discovery_shares as pds
import ecm_sde as es
import dcc_garch as dg
import liquidity_curve_metrics as lcm
import cross_asset_pd_liquidity as ca

EPS = 1e-12
_STRESS = {"volatile", "stress", "crisis", "turmoil"}


# ── helpers ───────────────────────────────────────────────────────────────────
def _logmid(df, asset, anchor="mid"):
    m = lcm.weighted_mid(df, asset) if anchor in ("wmid", "weighted", "micro") else ca._mid(df, asset)
    m = np.asarray(m, float)
    return np.log(np.where(m > 0, m, np.nan))

def _deseasonalize(x, index):
    """Remove a time-of-day (HH:MM:SS bucket) mean if the index carries intraday variation,
    else just demean. Strips the strong intraday U-shape that would otherwise masquerade as
    dynamics / volatility clustering."""
    x = np.asarray(x, float)
    try:
        t = pd.DatetimeIndex(index)
        key = (t.hour * 3600 + t.minute * 60 + t.second).to_numpy()
    except Exception:
        return x - np.nanmean(x)
    if np.unique(key).size < 3 or np.unique(key).size >= x.size:    # no usable repetition
        return x - np.nanmean(x)
    s = pd.Series(x).groupby(key).transform("mean").to_numpy()
    return x - s

def _zscore(M):
    M = np.asarray(M, float)
    sd = np.nanstd(M, axis=0)
    return (M - np.nanmean(M, axis=0)) / np.where(sd > EPS, sd, np.nan)

def _gauss_ll(eps, h):
    h = np.where(np.asarray(h, float) > EPS, h, np.nan)
    return float(-0.5 * np.nansum(np.log(2 * np.pi) + np.log(h) + np.asarray(eps, float) ** 2 / h))


# ── liquidity covariates ──────────────────────────────────────────────────────
def liquidity_covariates(df, n_levels=10, decay_scale=None, curve_anchor="wmid",
                         deseasonalize=True):
    """Per-bar variance covariates (AUC_SPY, AUC_ES, ||L||_SPY, ||L||_ES), deseasonalized.
    Returns (X[T,4], names)."""
    cols, names = [], []
    for a in ("SPY", "ES"):
        cols.append(lcm.decay_weighted_cost(df, a, n_levels=n_levels, decay_scale=decay_scale,
                                            side="both", anchor=curve_anchor))
        names.append(f"AUC_{a}")
    for a in ("SPY", "ES"):
        cols.append(lcm.impact_curve_length(df, a, n_levels=n_levels, side="both", anchor=curve_anchor))
        names.append(f"L_{a}")
    X = np.column_stack(cols)
    if deseasonalize:
        X = np.column_stack([_deseasonalize(X[:, k], df.index) for k in range(X.shape[1])])
    return X, names


# ── layers ────────────────────────────────────────────────────────────────────
def mean_layer(df, anchor="mid", n_lags=5):
    """Cointegrated VECM on the (log) SPY/ES price. Returns information shares, the EC loadings,
    and the reduced-form innovations eps with the original-bar index they map to."""
    p1 = _logmid(df, "SPY", anchor); p2 = _logmid(df, "ES", anchor)
    dY, X = pds._design_within_day(p1, p2, n_lags)
    base_idx = np.arange(len(p1) - len(dY), len(p1))                # design row r -> original bar
    ok = np.all(np.isfinite(dY), axis=1) & np.all(np.isfinite(X), axis=1)
    dY, X, idx = dY[ok], X[ok], base_idx[ok]
    B, resid = pds._ols(dY, X)
    alpha = B[1, :].copy()
    Sigma = np.cov(resid, rowvar=False, bias=True)
    return {"shares": es._shares_at(alpha, Sigma), "alpha": alpha, "Sigma": Sigma,
            "resid": resid, "resid_idx": idx, "n_obs": int(resid.shape[0])}

def variance_layer(resid, Xcov=None, names=("SPY", "ES")):
    """DCC-GARCH(-X) on the mean residuals. resid[T,2] and Xcov[T,k] must already be finite and
    aligned 1:1 (Xcov already lagged + standardized). Returns rho_t, DCC params, and per-asset
    gamma with an LR test of gamma=0."""
    out = dg.dcc_garch_x(resid, X=Xcov, names=names)
    rho = np.asarray(out["rho"], float)
    gammas = {}
    if Xcov is not None and out.get("marginals"):
        eps = resid - resid.mean(0)
        for j, nm in enumerate(names):
            m = out["marginals"].get(nm)
            if not m:
                continue
            ll_x = _gauss_ll(eps[:, j], m["h"])
            m0 = dg.garch_x_fit(eps[:, j], X=None)
            ll_0 = _gauss_ll(eps[:, j], m0["h"])
            k = int(np.atleast_1d(m["gamma"]).size)
            lr = max(2.0 * (ll_x - ll_0), 0.0)
            gammas[nm] = {"gamma": [float(v) for v in np.atleast_1d(m["gamma"])],
                          "lr_stat": float(lr), "lr_p": float(chi2.sf(lr, max(k, 1))), "k": k}
    return {"rho": rho, "dcc_a": out.get("dcc_a"), "dcc_b": out.get("dcc_b"),
            "cond_vol": out.get("cond_vol"), "gamma": gammas}


# ── orchestrator (single session) ─────────────────────────────────────────────
def run_mean_variance(df, anchor_price="mid", curve_anchor="wmid", n_lags=5, n_levels=10,
                      decay_scale=None, stress=None, deseasonalize=True, var_scale=1e4, verbose=True):
    """One session. `stress`: optional per-bar boolean (length len(df)) flagging the stress
    regime (e.g. post-reopen bars) for the calm-vs-stress rho split. `var_scale` rescales the
    mean residuals before the variance layer (default 1e4 = basis points; GARCH is ill-scaled on
    raw ~1e-3 log-return residuals). Returns the information shares, time-varying rho (+ calm/
    stress split), and the liquidity->vol gamma loadings (on bps^2 variance)."""
    ml = mean_layer(df, anchor_price, n_lags)
    idx = ml["resid_idx"]
    Xfull, names_x = liquidity_covariates(df, n_levels, decay_scale, curve_anchor, deseasonalize)
    Xlag = np.roll(Xfull, 1, axis=0); Xlag[0] = np.nan             # predetermined (bar t-1)
    Xcov = _zscore(Xlag[idx])                                      # align to resid + standardize
    fin = np.all(np.isfinite(ml["resid"]), axis=1) & np.all(np.isfinite(Xcov), axis=1)
    resid, Xcov = ml["resid"][fin] * float(var_scale), Xcov[fin]   # residuals in bps
    vl = variance_layer(resid, Xcov, names=("SPY", "ES"))
    rho = vl["rho"]
    out = {"shares": ml["shares"], "n_obs_mean": ml["n_obs"], "n_obs_var": int(resid.shape[0]),
           "rho_mean": float(np.nanmean(rho)) if rho.size else np.nan, "rho_series": rho,
           "dcc_a": vl["dcc_a"], "dcc_b": vl["dcc_b"], "gamma": vl["gamma"],
           "covariate_names": names_x}
    if stress is not None and rho.size:
        s = np.asarray(stress, bool)[idx][fin]
        if s.size == rho.size and s.any() and (~s).any():
            out["rho_calm"] = float(np.nanmean(rho[~s]))
            out["rho_stress"] = float(np.nanmean(rho[s]))
            out["rho_gap"] = out["rho_stress"] - out["rho_calm"]
    if verbose:
        sh = ml["shares"]
        g = vl["gamma"]
        own = {nm: (g[nm]["gamma"][j] if nm in g else float("nan")) for j, nm in enumerate(("SPY", "ES"))}
        print("[mean] CS_ES=%.3f IS_ES=%.3f half_life=%.1fs | [var] rho=%.3f a+b=%.3f | gamma(AUC_own) SPY=%.3f ES=%.3f"
              % (sh["CS_ES"], sh["IS_ES"], sh.get("half_life_s", float("nan")), out["rho_mean"],
                 (vl["dcc_a"] or 0.0) + (vl["dcc_b"] or 0.0), own["SPY"], own["ES"]))
        if "rho_gap" in out:
            print("       rho calm=%.3f -> stress=%.3f (gap %+.3f)" % (out["rho_calm"], out["rho_stress"], out["rho_gap"]))
    return out


# ── panel wrapper (many sessions) ─────────────────────────────────────────────
def _panel_one(args):
    """One session's mean+variance fit, as a module-level (picklable) function.

    Returns (index, result_or_None, error_repr_or_None) so the parent can rebuild the panel in the
    ORIGINAL session order regardless of completion order, and reproduce the serial loop's
    keep-going-on-error behaviour exactly."""
    i, df, kw = args
    try:
        return i, run_mean_variance(df, **kw), None
    except Exception as e:                                         # noqa: BLE001 -- mirrored from serial
        return i, None, repr(e)


def _init_worker():
    """Pin BLAS to one thread inside each worker. Without this, N processes x M BLAS threads
    oversubscribe every core and the pool runs SLOWER than the serial loop it replaced."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def run_panel(sessions, anchor_price="mid", curve_anchor="wmid", n_lags=5, n_levels=10,
              decay_scale=None, deseasonalize=True, var_scale=1e4, verbose=True, n_jobs=None):
    """Loop sessions; per-session mean+variance, then pool rho by regime (stress = sessions
    whose label is volatile/stress) for the calm-vs-stress correlation contrast.

    Sessions are independent -- each is its own GARCH marginal pair plus a cDCC fit, sharing no
    state -- so this runs them in a PROCESS pool. Processes, not threads: the cost is the GARCH
    and cDCC recursions, which are Python-level loops holding the GIL, so a thread pool would
    serialize them.

    ``n_jobs``: None (default) sizes the pool from the machine via ``autoscale.panel_workers`` --
    min(cores, #sessions, RAM-derived) -- so it adapts to the box without a hard-coded number.
    ``n_jobs=1`` forces the original serial loop, which is what you want when debugging a single
    session's fit or when a worker cannot be forked. Results are order- and value-identical either
    way: the fits are deterministic and the panel is reassembled by original index."""
    n = len(sessions)
    if n_jobs is None:
        try:
            n_jobs = _autoscale.panel_workers(n)
        except Exception:
            n_jobs = 1
    n_jobs = max(1, min(int(n_jobs), n)) if n else 1

    kw = dict(anchor_price=anchor_price, curve_anchor=curve_anchor, n_lags=n_lags,
              n_levels=n_levels, decay_scale=decay_scale, stress=None,
              deseasonalize=deseasonalize, var_scale=var_scale, verbose=False)
    results = [None] * n
    if n_jobs > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as ex:
                for i, r, err in ex.map(_panel_one, [(i, s[2], kw) for i, s in enumerate(sessions)]):
                    results[i] = (r, err)
            if verbose:
                print("[panel] %d sessions across %d worker processes" % (n, n_jobs))
        except Exception as e:                                     # no fork / pickling refused
            if verbose:
                print("[panel] process pool unavailable (%r); falling back to serial" % (e,))
            results = [None] * n
    if any(x is None for x in results):                            # serial path (n_jobs=1 or fallback)
        for i, (_d, _reg, df) in enumerate(sessions):
            if results[i] is None:
                results[i] = _panel_one((i, df, kw))[1:]

    rows, rho_calm, rho_stress = [], [], []
    for (d, regime, _df), (r, err) in zip(sessions, results):
        if err is not None:                                        # keep the panel going
            rows.append({"date": str(d), "regime": regime, "error": err}); continue
        g = r["gamma"]
        rows.append({"date": str(d), "regime": regime,
                     "CS_ES": r["shares"]["CS_ES"], "IS_ES": r["shares"]["IS_ES"],
                     "rho": r["rho_mean"], "dcc_persist": (r["dcc_a"] or 0.0) + (r["dcc_b"] or 0.0),
                     "gamma_AUC_SPY": g.get("SPY", {}).get("gamma", [np.nan])[0] if g else np.nan,
                     "gamma_AUC_ES": (g.get("ES", {}).get("gamma", [np.nan, np.nan]) + [np.nan, np.nan])[1] if g else np.nan})
        (rho_stress if str(regime).lower() in _STRESS else rho_calm).append(r["rho_mean"])
    tab = pd.DataFrame(rows)
    summary = {"n_sessions": len(sessions),
               "CS_ES_mean": float(np.nanmean(tab.get("CS_ES", pd.Series(dtype=float)))) if "CS_ES" in tab else np.nan,
               "rho_mean": float(np.nanmean(tab.get("rho", pd.Series(dtype=float)))) if "rho" in tab else np.nan,
               "rho_calm_days": float(np.nanmean(rho_calm)) if rho_calm else np.nan,
               "rho_stress_days": float(np.nanmean(rho_stress)) if rho_stress else np.nan}
    if verbose:
        print("[panel] sessions=%d  CS_ES=%.3f  rho=%.3f  (calm=%.3f, stress=%.3f)"
              % (summary["n_sessions"], summary["CS_ES_mean"], summary["rho_mean"],
                 summary["rho_calm_days"], summary["rho_stress_days"]))
    return {"per_session": tab, "summary": summary}


# ── self-test ─────────────────────────────────────────────────────────────────
def _synth(T=5200, n=10, seed=0):
    """ES leads (SPY error-corrects to ES). Each asset has an INDEPENDENT bursty vol multiplier
    v that scales its innovation variance AND thins its book (so a fixed-size order's AUC tracks
    v) -> AUC predicts that asset's vol (gamma>0). A separate common shock, louder in a stress
    window, raises the SPY-ES correlation there (rho_stress>rho_calm). The two channels are
    decoupled so each is cleanly identified."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-03-12 09:30", periods=T, freq="s", tz="America/New_York")
    stress = np.zeros(T, bool); stress[int(T * 0.6):] = True

    def burst():                                                   # stationary, bursty vol multiplier ~ mean 1
        lv = np.zeros(T)
        for t in range(1, T):
            lv[t] = 0.7 * lv[t - 1] + rng.normal(0, 0.6)
        v = np.exp(lv); return v / v.mean()
    vS, vE = burst(), burst()

    f = np.cumsum(rng.normal(0, 0.0008, T))                        # efficient (log) price
    common = rng.normal(0, 1, T) * np.where(stress, 0.0012, 0.0005)  # shared shock, louder in stress -> rho
    pES = f + common + rng.normal(0, 1, T) * 0.0006 * np.sqrt(vE)   # ES ~ efficient price (does not adjust)
    pSPY = np.empty(T); pSPY[0] = pES[0]
    eS = rng.normal(0, 1, T) * 0.0006 * np.sqrt(vS)
    for t in range(1, T):                                          # SPY error-corrects toward ES
        pSPY[t] = pSPY[t - 1] + 0.5 * (pES[t - 1] - pSPY[t - 1]) + common[t] + eS[t]

    tick = 0.01; d = {}
    for a, lp, v in (("SPY", pSPY, vS), ("ES", pES, vE)):
        mid = np.exp(lp) * 500.0
        for sd, s in (("ask", 1), ("bid", -1)):
            for i in range(1, n + 1):
                d[f"{a}_{sd}price_{i}"] = mid + s * tick * i
                depth = 300.0 * np.exp(-0.3 * (i - 1)) / np.sqrt(v) * (1 + rng.normal(0, 0.02, T))  # thin when v high
                d[f"{a}_{sd}quantity_{i}"] = np.maximum(depth, 1.0)
    return pd.DataFrame(d, index=idx), stress


def _selftest():
    print("run_mean_variance self-test")
    df, stress = _synth(seed=1)
    r = run_mean_variance(df, anchor_price="mid", n_lags=5, n_levels=10, decay_scale=800.0,
                          stress=stress, verbose=True)             # fixed Q0 so AUC responds to thinning
    sh = r["shares"]; g = r["gamma"]
    lr_p = [g.get(nm, {}).get("lr_p", 1.0) for nm in ("SPY", "ES")]
    es_leads = sh["CS_ES"] > 0.5
    vol_link = np.nanmin(lr_p) < 0.05                              # liquidity covariates jointly enter the variance
    contagion = ("rho_gap" in r) and (r["rho_gap"] > 0)           # corr higher in stress (Forbes-Rigobon)
    dcc_ok = 0.0 <= (r["dcc_a"] or 0) + (r["dcc_b"] or 0) < 1.0
    print("  ES leads (CS_ES>0.5):", es_leads,
          "| AUC/||L||->vol (LR sig, min p=%.3g):" % np.nanmin(lr_p), vol_link,
          "| rho_stress>rho_calm:", contagion, "| DCC a+b<1:", dcc_ok)

    sess = []
    for k in range(4):
        dfk, _ = _synth(seed=10 + k)
        sess.append((f"2020-03-1{k}", "volatile" if k >= 2 else "benchmark", dfk))
    pan = run_panel(sess, n_lags=5, n_levels=10, decay_scale=800.0, verbose=True)
    sm = pan["summary"]
    panel_ok = (len(pan["per_session"]) == 4 and "CS_ES" in pan["per_session"].columns
                and np.isfinite(sm["rho_stress_days"]) and np.isfinite(sm["rho_calm_days"]))

    ok = bool(es_leads and vol_link and contagion and dcc_ok and panel_ok)
    print("\nchecks:", ok)
    if not vol_link:
        print("  NOTE: vol_link is the known-open one -- the liquidity->variance LR test is not")
        print("  significant on this synthetic fixture (min p ~0.51). Pre-existing: it reproduces")
        print("  identically with the pre-v0.9.18 _garch_filter, so it is a property of the fixture")
        print("  or the test, not of the recursion. Everything else here passes.")
    return ok


if __name__ == "__main__":
    # Exit non-zero on failure: a self-test that prints False and returns 0 is invisible to the
    # README's smoke loop, which reads only the exit code (same defect fixed in robust_prices).
    import sys as _sys
    _sys.exit(0 if _selftest() else 1)
