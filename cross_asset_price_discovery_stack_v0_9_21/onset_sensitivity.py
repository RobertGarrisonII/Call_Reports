"""onset_sensitivity.py -- validation diagnostics for the cross-event onset estimator.

The onset transmission rests on two assumptions the synthetic guard cannot test on real data:
  (1) the post-onset window is short enough to PREDATE the basis-feedback cascade -- otherwise the
      structural coefficient is contaminated by the very endogeneity the onset was meant to escape;
  (2) each event supplies DIFFERENTIAL heteroskedasticity across the onset -- otherwise Rigobon is
      unidentified, however clean the eigendecomposition looks (a common-scale variance jump leaves
      M = Om_pre Om_post^{-1} proportional to I, eigenvalues coincident, A not identified).

These tools profile those assumptions on real sessions before any coefficient is trusted:
  * post_window_profile / window_sweep -- where is the onset plateau before the cascade drift, and is
    the pooled slope window-robust or window-driven?
  * id_strength_profile / id_threshold_sweep -- how many events are weakly identified, and does the
    benchmark slope survive dropping them?
  * cholesky_bracket -- does the order-free Rigobon coefficient sit between the two Cholesky orderings?
    Outside the bracket flags label-switching or weak identification.

None of these MAKE the design valid; they reveal whether it is. A window-robust slope built on
strongly-identified events that bracket cleanly is credible; the opposite is a warning to stop.
"""
import numpy as np
import pandas as pd

import onset_response as onr
import rigobon_id as rg

EPS = 1e-12


def post_window_profile(df, onset_ts, post_grid=(60, 120, 300, 600, 1200, 2400), pre_steps=1200,
                        impulse="ES", response="SPY"):
    """For a SINGLE event, the onset transmission as the post-window grows (pre fixed). A flat plateau
    near the onset means the estimate is reading the pre-feedback onset; once the window extends far
    enough to straddle the cascade the structural coefficient is no longer constant and the estimate
    DEPARTS the plateau (in either direction -- under a within-post break Rigobon does not interpolate,
    it destabilises). Read the transmission off the plateau, treat a large departure as the signal that
    the window has reached the feedback region. Returns [post_steps, transmission, id_strength,
    var_ratio, n_post]."""
    rows = []
    for ps in post_grid:
        e = onr.event_onset_estimate(df, onset_ts, pre_steps=pre_steps, post_steps=int(ps),
                                     impulse=impulse, response=response)
        rows.append({"post_steps": int(ps), "transmission": e["transmission"],
                     "id_strength": e["id_strength"], "var_ratio": e["var_ratio"], "n_post": e["n_post"]})
    return pd.DataFrame(rows)


def window_sweep(sessions, events_frame, pre_grid=(600, 1200), post_grid=(120, 300, 600, 1200),
                 degree=1, impulse="ES", response="SPY", release_by_date=None, min_id_strength=0.0):
    """Recompute the cross-event panel and the benchmark f(state) slope for each (pre,post) on the grid.
    A slope stable across a sensible window band is window-robust; one that swings is window-driven (the
    small-post limit is the pre-feedback ideal, instability as post grows is cascade contamination).
    Returns [pre_steps, post_steps, n_events, n_impact_blind, slope, slope_se, r2, median_transmission,
    median_id_strength]."""
    rows = []
    for pre in pre_grid:
        for post in post_grid:
            panel = onr.cross_event_onset(sessions, events_frame, pre_steps=int(pre), post_steps=int(post),
                                          impulse=impulse, response=response, release_by_date=release_by_date)
            fit = onr.fit_stress_response(panel, degree=degree, min_id_strength=min_id_strength)
            rows.append({"pre_steps": int(pre), "post_steps": int(post), "n_events": int(len(panel)),
                         "n_impact_blind": int(panel["impact_blind"].sum()) if len(panel) else 0,
                         "slope": fit["slope"], "slope_se": fit["slope_se"], "r2": fit["r2"],
                         "median_transmission": float(np.nanmedian(panel["transmission"])) if len(panel) else np.nan,
                         "median_id_strength": float(np.nanmedian(panel["id_strength"])) if len(panel) else np.nan})
    return pd.DataFrame(rows)


def id_strength_profile(panel, weak_tol=1e-3):
    """Distribution of Rigobon identification strength across events: the share weakly identified
    (eig_separation below weak_tol), its quantiles, and the share with var_ratio<=1 (no onset variance
    rise -> suspect onset). Weakly-identified events should be dropped or down-weighted before pooling;
    weak ID is the case the eigendecomposition cannot warn you about by looking clean."""
    s = pd.to_numeric(panel.get("id_strength"), errors="coerce").to_numpy()
    v = pd.to_numeric(panel.get("var_ratio"), errors="coerce").to_numpy()
    s = s[np.isfinite(s)]
    return {"n": int(len(s)),
            "share_weak": float(np.mean(s < weak_tol)) if len(s) else np.nan,
            "q10": float(np.quantile(s, 0.10)) if len(s) else np.nan,
            "median": float(np.median(s)) if len(s) else np.nan,
            "q90": float(np.quantile(s, 0.90)) if len(s) else np.nan,
            "share_var_ratio_le1": float(np.nanmean(v <= 1.0)) if np.isfinite(v).any() else np.nan}


def id_threshold_sweep(panel, thresholds=(0.0, 1e-3, 1e-2, 1e-1), degree=1):
    """Benchmark f(state) slope and retained-event count as weakly-identified events are progressively
    dropped (id_strength >= threshold). A slope stable as the weak events fall away is robust; one that
    moves was being driven by poorly-identified onsets. Returns [min_id_strength, n, slope, slope_se, r2]."""
    rows = []
    for thr in thresholds:
        fit = onr.fit_stress_response(panel, degree=degree, impact_blind_only=True, min_id_strength=float(thr))
        rows.append({"min_id_strength": float(thr), "n": fit["n"], "slope": fit["slope"],
                     "slope_se": fit["slope_se"], "r2": fit["r2"]})
    return pd.DataFrame(rows)


def cholesky_bracket(df, onset_ts, pre_steps=1200, post_steps=300, impulse="ES", response="SPY"):
    """Per-event order-free vs ordered check: the Rigobon transmission next to the two Cholesky
    orderings on the post-window covariance. The order-free coefficient should sit BETWEEN the two
    ordering-dependent extremes; outside the bracket flags label-switching or weak identification.
    Returns {rigobon, chol_resp_first, chol_imp_first, within_bracket, id_strength}."""
    idx = df.index
    onset_ts = pd.Timestamp(onset_ts) if onset_ts is not None else idx[len(idx) // 2]
    ridx = idx[1:]
    R = np.column_stack([onr._log_returns(df, response), onr._log_returns(df, impulse)])   # [resp, imp]
    pre_i = np.where(ridx < onset_ts)[0][-pre_steps:]
    post_i = np.where(ridx >= onset_ts)[0][:post_steps]
    out = {"rigobon": np.nan, "chol_resp_first": np.nan, "chol_imp_first": np.nan,
           "within_bracket": False, "id_strength": np.nan}
    if len(pre_i) < 20 or len(post_i) < 20:
        return out
    Om_pre, Om_post = np.cov(R[pre_i].T), np.cov(R[post_i].T)
    res = rg.rigobon_2regime(Om_pre, Om_post, names=[response, impulse])
    rig = float(res["contemp"].loc[response, impulse])
    c_resp_first = float(rg._cholesky_contemp(Om_post, [0, 1])[0, 1])   # response ordered first
    c_imp_first = float(rg._cholesky_contemp(Om_post, [1, 0])[0, 1])    # impulse ordered first
    lo, hi = sorted([c_resp_first, c_imp_first])
    out.update(rigobon=rig, chol_resp_first=c_resp_first, chol_imp_first=c_imp_first,
               within_bracket=bool(lo - 1e-9 <= rig <= hi + 1e-9), id_strength=float(res["eig_separation"]))
    return out
