"""onset_response.py -- the cross-event ONSET driver: the identified estimate of the stress-response
function f(state).

Each event contributes ONE clean onset observation -- the contemporaneous ES->SPY transmission
identified WITHIN-event off the onset variance shift (Rigobon heteroskedasticity ID across the
pre-/post-release boundary), paired with the PREDETERMINED basis dislocation just before the onset.
Pooling these (state, transmission) pairs across events traces f(state). The IMPACT-BLIND events
(scheduled / scheduled-uncertainty / news-classified) form the benchmark; the SALIENCE-CURATED tail
is tested as EXCESS over the benchmark-extrapolated f -- the benchmark carries the exogeneity, so the
excess is identified even when the tail is curated on salience. Inference across the (few, hetero-
geneous) events is Ibragimov-Muller, not the wild bootstrap.

This is the ONSET (pre-feedback) tier -- the cleanly identified headline. Identification sits at the
onset because the within-crisis cascade endogenizes the slope (state and response co-evolve once the
feedback loop engages); that trajectory is DESCRIBED elsewhere (markov_switching_vecm), not fit as
f(state) here. The transmission is the contemporaneous structural coefficient; whether it RISES or
FALLS with dislocation is the empirical question f(state) answers (a reduced-form price-discovery /
arb-link statement, not a claim that the book is the cause).
"""
import numpy as np
import pandas as pd

import cross_asset_pd_liquidity as ca
import liquidity_stress as ls
import rigobon_id as rg
import inference as inf
import noise_robust_cov as ncov

try:                                                          # optional: parallel cross-event onset loop
    from joblib import Parallel, delayed
    _JOBLIB = True
except Exception:                                             # pragma: no cover -- serial fallback
    _JOBLIB = False
_MIN_PARALLEL_EVENTS = 8                                      # below this, pool overhead dominates -> serial

EPS = 1e-12


def _safe_nanmean(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def _robust_window_cov(lp_resp, lp_imp):
    """Noise-corrected 2x2 covariance of [response, impulse] over a window: TSRV (two-scale) variances
    on the diagonal -- removing the bid-ask-bounce inflation that errors-in-variables attenuation would
    otherwise feed into the Rigobon coefficient -- and the Hayashi-Yoshida cross-covariance (unbiased by
    per-leg idiosyncratic noise, which is uncorrelated across the two instruments). Synced grid here, so
    HY reduces toward the sample cross-covariance; the correction lives on the diagonal, which is where
    the bounce confound lives."""
    t = np.arange(len(lp_resp), dtype=float)
    m = ncov.noise_robust_moments(t, np.asarray(lp_resp, float), t, np.asarray(lp_imp, float), var_method="tsrv")
    va, vb, cab = float(m["rv_spy"]), float(m["rv_es"]), float(m["cov"])
    return np.array([[va, cab], [cab, vb]], float)


def _log_returns(df, asset):
    mid = np.asarray(ca._mid(df, asset), float)
    return np.diff(np.log(mid))


def _available_levels(df, asset, side="ask"):
    i = 1
    while f"{asset}_{side}price_{i}" in df.columns:
        i += 1
    return i - 1


def _pre_hollowness(df, asset, n_levels, pre_i):
    """Pre-window mean book hollowness (resiliency/arbitrage-capacity axis) for one leg; level-capped
    and exception-safe (NaN on a too-shallow book)."""
    nl = min(int(n_levels), _available_levels(df, asset))
    try:
        hs = ls.stress_state(df, asset, max(nl, 2))["hollowness"].to_numpy()[1:]
        return _safe_nanmean(hs[pre_i]) if len(pre_i) else np.nan
    except Exception:
        return np.nan


def _pre_cost(df, asset, notional, n_levels, pre_i):
    """Pre-window mean cost-to-fill (bps) for a fixed notional on one leg; level-capped, exception-safe."""
    nl = min(int(n_levels), _available_levels(df, asset))
    try:
        c = ls.cost_to_fill_bps(df, asset, notional, max(nl, 1))[1:]
        return _safe_nanmean(c[pre_i]) if len(pre_i) else np.nan
    except Exception:
        return np.nan


def _crossed_frac(df, asset, sl=None):
    """Share of grid points whose top of book is CROSSED (bid1 > ask1) -- the reconstruction invariant
    violation. A matching engine cannot cross, so a nonzero rate means resting orders that should have
    left are pinning the top, and EVERY book-derived quantity on those bars is contaminated: the spread,
    the hollowness/capacity axes, the cost-to-fill, and the mid that the returns are built from. This is
    therefore the QC weight on the event's whole onset observation, not a cosmetic flag.

    Measured on the SAME frame the estimator consumes rather than inherited from reconstruction: the
    session join drops df.attrs, so lob_stats/consolidated_crossed is unavailable downstream, and a
    measured number cannot silently disagree with the frame that produced the estimate. Pass `sl` to
    restrict to a window (crossing concentrates where message rates spike -- i.e. at the onset -- so the
    window-local rate is the one that bears on the estimate; a session-level rate dilutes it across the
    calm hours). Returns NaN when the columns are absent or no bar has both sides quoted."""
    try:
        bp = np.asarray(df[f"{asset}_bidprice_1"], dtype=float)
        ap = np.asarray(df[f"{asset}_askprice_1"], dtype=float)
    except Exception:
        return np.nan
    if sl is not None:
        bp, ap = bp[sl], ap[sl]
    both = np.isfinite(bp) & np.isfinite(ap)
    return float(np.mean(bp[both] > ap[both])) if bool(both.any()) else np.nan


def event_onset_estimate(df, onset_ts=None, pre_steps=600, post_steps=600, impulse="ES", response="SPY",
                         book_asset=None, n_levels=10, notional=1e6):
    """One event's onset observation: the predetermined basis dislocation (state) and the within-event
    contemporaneous transmission (response<-impulse), identified off the onset variance shift via the
    Rigobon 2-regime (calm pre vs shocked post) heteroskedasticity solution.

    onset_ts splits the session into the calm pre-window and the shocked post-window; if None the
    midpoint is used (for an overnight event pass the first post-gap bar). Returns the transmission
    coefficient, the predetermined |basis| (bps) averaged over the pre-window, the post/pre variance
    ratio, the Rigobon identification strength (eigenvalue separation; ~0 = weak / unidentified), and
    the window sizes. Transmission is NaN when either window is too short to form a covariance."""
    idx = df.index
    if onset_ts is None:
        onset_ts = idx[len(idx) // 2]
    onset_ts = pd.Timestamp(onset_ts)
    ridx = idx[1:]                                             # returns align to idx[1:]
    R = np.column_stack([_log_returns(df, response), _log_returns(df, impulse)])   # [response, impulse]
    pre_all = np.where(ridx < onset_ts)[0]
    post_all = np.where(ridx >= onset_ts)[0]
    pre_i, post_i = pre_all[-pre_steps:], post_all[:post_steps]
    out = dict(transmission=np.nan, transmission_robust=np.nan, state=np.nan, state_holl=np.nan,
               state_holl_imp=np.nan, state_cost=np.nan, state_cost_joint=np.nan, state_cost_max=np.nan,
               state_cost_prod=np.nan, var_ratio=np.nan, id_strength=np.nan, id_strength_robust=np.nan,
               pre_spread_bps=np.nan, pre_noise_bps=np.nan,
               crossed_resp=np.nan, crossed_imp=np.nan, crossed_max=np.nan,
               crossed_resp_sess=np.nan, crossed_imp_sess=np.nan,
               n_pre=int(len(pre_i)), n_post=int(len(post_i)), onset=onset_ts)
    # session-level crossing is available even when the windows are too short to estimate, so record it
    # before the guard -- a dropped event still reports WHY its book was unusable.
    out["crossed_resp_sess"] = _crossed_frac(df, response)
    out["crossed_imp_sess"] = _crossed_frac(df, impulse)
    if len(pre_i) < 20 or len(post_i) < 20:
        return out
    # window-local crossing: the rate over the bars the estimate is actually built from (pre+post), which
    # is the contamination weight for THIS observation. crossed_max is the single sort/filter column.
    sl_win = slice(int(pre_i[0]), int(post_i[-1]) + 2)
    out["crossed_resp"] = _crossed_frac(df, response, sl_win)
    out["crossed_imp"] = _crossed_frac(df, impulse, sl_win)
    _xw = [v for v in (out["crossed_resp"], out["crossed_imp"]) if np.isfinite(v)]
    out["crossed_max"] = float(max(_xw)) if _xw else np.nan
    Om_pre, Om_post = np.cov(R[pre_i].T), np.cov(R[post_i].T)
    res = rg.rigobon_2regime(Om_pre, Om_post, names=[response, impulse])
    out["transmission"] = float(res["contemp"].loc[response, impulse])    # impulse -> response
    out["id_strength"] = float(res["eig_separation"])
    out["var_ratio"] = float(np.trace(Om_post) / (np.trace(Om_pre) + EPS))
    bs = ls.basis_state(df)["abs_basis_bps"].to_numpy()[1:]   # predetermined |basis|, aligned to returns
    out["state"] = float(np.nanmean(bs[pre_i])) if len(pre_i) else np.nan
    # second predetermined axis: pre-window book HOLLOWNESS (arbitrage-capacity / resiliency margin --
    # the dimension a basis dislocation must execute against). Default = the response leg's book (where
    # the transmission lands); pass book_asset=impulse to condition on the OTHER leg's stress instead
    # (the cross-asset propagation channel, robust to own-instrument microstructure contamination).
    # Capped to the book's actual depth and exception-safe: a shallow book yields NaN, never a crash.
    ba = book_asset or response
    nl = min(int(n_levels), _available_levels(df, ba))
    # capacity axes for the leg triple: book_asset leg (default SPY/response), the impulse leg (ES), and
    # the leg-neutral joint round-trip cost below. The book-leg decision is run-all-three-and-compare.
    out["state_holl"] = _pre_hollowness(df, ba, n_levels, pre_i)
    out["state_holl_imp"] = _pre_hollowness(df, impulse, n_levels, pre_i)
    # ── noise-robust transmission + microstructure controls ──────────────────────────────────────
    # Rigobon is robust to TIME-CONSTANT microstructure noise (it cancels in Om_post - Om_pre) but NOT
    # to noise that SHIFTS at the onset -- post-release the touch thins and the BBO flips faster, so
    # mid-return noise variance rises, inflating the post diagonal and ATTENUATING the identified
    # transmission. A hollow pre-window predicts a larger such shift, manufacturing a spurious negative
    # capacity/interaction effect that mimics the spiral. The robust path re-estimates each regime's
    # covariance with noise-corrected (TSRV) variances so the bounce cannot enter the variance shift.
    lp_r = np.log(np.asarray(ca._mid(df, response), float))
    lp_i = np.log(np.asarray(ca._mid(df, impulse), float))
    sl_pre = slice(int(pre_i[0]), int(pre_i[-1]) + 2)         # levels spanning the pre-window returns
    sl_post = slice(int(post_i[0]), int(post_i[-1]) + 2)
    try:
        Om_pre_r = _robust_window_cov(lp_r[sl_pre], lp_i[sl_pre])
        Om_post_r = _robust_window_cov(lp_r[sl_post], lp_i[sl_post])
        res_r = rg.rigobon_2regime(Om_pre_r, Om_post_r, names=[response, impulse])
        out["transmission_robust"] = float(res_r["contemp"].loc[response, impulse])
        out["id_strength_robust"] = float(res_r["eig_separation"])
    except Exception:
        pass
    # pre-window controls: the bounce LEVEL omega (varies with touch fragility even when the quoted
    # spread is pinned at one tick, as it is for SPY/ES) and the quoted spread itself (near-constant
    # here -- carried for completeness, the documented reason a shape/noise state is needed at all).
    def _omega_bps(r):
        r = r[np.isfinite(r)]
        return float(np.sqrt(np.sum(r ** 2) / (2.0 * max(len(r), 1))) * 1e4) if len(r) else np.nan
    out["pre_noise_bps"] = _safe_nanmean([_omega_bps(np.diff(lp_r[sl_pre])), _omega_bps(np.diff(lp_i[sl_pre]))])
    try:
        sps = ls.quoted_spread_bps(df, response)[1:]; spi = ls.quoted_spread_bps(df, impulse)[1:]
        out["pre_spread_bps"] = _safe_nanmean(0.5 * (sps[pre_i] + spi[pre_i]))
    except Exception:
        pass
    # joint round-trip arbitrage cost over the pre-window: the symmetric (leg-neutral) capacity axis --
    # the cost to execute BOTH legs of the basis trade -- an alternative interacting conditioner to
    # single-leg hollowness (pass holl_col="state_cost_joint"). state_cost is the book_asset leg alone.
    out["state_cost"] = _pre_cost(df, ba, notional, n_levels, pre_i)
    c_resp = _pre_cost(df, response, notional, n_levels, pre_i)
    c_imp = _pre_cost(df, impulse, notional, n_levels, pre_i)
    both = np.isfinite(c_resp) and np.isfinite(c_imp)
    # three leg-aggregations of the round-trip arbitrage cost (the book-leg decision is run-all-compare):
    #   joint = SUM  -> the two books' costs stack linearly (legs independent)
    #   max   = BOTTLENECK -> the worse leg gates the round trip (capacity = the binding leg)
    #   prod  = PRODUCT (bps^2) -> the multiplicative axis; a disguised three-way that is expected to be
    #           fat-tailed / unidentified at this event count -- carried as a COMPARATOR, not headline.
    out["state_cost_joint"] = float(c_resp + c_imp) if both else np.nan
    out["state_cost_max"] = float(max(c_resp, c_imp)) if both else np.nan
    out["state_cost_prod"] = float(c_resp * c_imp) if both else np.nan
    return out


def _onset_ts_from_meta(date, meta, tz):
    """Intraday onset Timestamp from a release HH:MM; None for overnight/blank (-> midpoint/open-gap)."""
    et = str(meta.get("event_et", "")).strip().lstrip("~")
    if not et or ":" not in et:
        return None
    try:
        hh, mm = (int(x) for x in et.split(":")[:2])
    except Exception:
        return None
    ts = pd.Timestamp(date).normalize() + pd.Timedelta(hours=hh, minutes=mm)
    if tz is not None and ts.tz is None:
        ts = ts.tz_localize(tz)
    return ts


def _onset_row(item, ev, pre_steps, post_steps, impulse, response, release_by_date, book_asset, n_levels):
    """One event's onset observation (module-level so it is thread-safe / picklable). Returns the
    enriched estimate dict, or None when the session date has no matching event row."""
    date, _regime, df = item
    d0 = pd.Timestamp(date).tz_localize(None).normalize() if pd.Timestamp(date).tz is not None \
        else pd.Timestamp(date).normalize()
    m = ev[ev["_d"] == d0]
    if m.empty:
        return None
    meta = m.iloc[0]
    tz = df.index.tz
    if release_by_date is not None and d0 in release_by_date:
        onset_ts = release_by_date[d0]
    else:
        onset_ts = _onset_ts_from_meta(date, meta, tz)
    est = event_onset_estimate(df, onset_ts, pre_steps, post_steps, impulse, response,
                               book_asset=book_asset, n_levels=n_levels)
    est.update(date=pd.Timestamp(date), category=meta.get("category"),
               selection=meta.get("selection", "salience_curated"),
               impact_blind=bool(meta.get("impact_blind", False)))
    return est


def cross_event_onset(sessions, events_frame, pre_steps=600, post_steps=600,
                      impulse="ES", response="SPY", release_by_date=None, book_asset=None,
                      n_levels=10, n_jobs=1):
    """Loop over events; per-event onset observation; return the cross-event panel. `sessions` is a
    list of (date, regime, df); `events_frame` is a market_shocks.shocks_frame()-style DataFrame with
    date/category/selection/impact_blind (+ optional event_et). `release_by_date` maps a tz-naive
    normalized date -> intraday onset Timestamp; otherwise the per-event event_et is used, falling
    back to the session midpoint (the open-to-open proxy for an overnight event).

    Each event is INDEPENDENT (its estimate depends only on its own session frame), so the loop is
    embarrassingly parallel. n_jobs>1 runs it on a joblib THREAD pool: the per-event cost is dominated
    by GIL-releasing numpy -- the pre/post covariances, the Rigobon solve, and the TSRV / Hayashi-
    Yoshida robust covariance -- so threads overlap that work without process-spawn / pickle / module
    re-import overhead, and the panel is identical to the serial path (input order is preserved and
    there is no shared mutable state). Parallelism engages only above _MIN_PARALLEL_EVENTS; below that
    the pool overhead dominates and it stays serial. (Reconstruction itself -- the event-ordering-
    sensitive book replay -- is NOT parallelized here; this fans out across already-built sessions,
    never within a single feed's event stream.)"""
    ev = events_frame.copy()
    ev["_d"] = pd.to_datetime(ev["date"]).dt.tz_localize(None).dt.normalize()
    items = list(sessions)
    common = (ev, pre_steps, post_steps, impulse, response, release_by_date, book_asset, n_levels)
    if n_jobs and n_jobs > 1 and len(items) >= _MIN_PARALLEL_EVENTS and _JOBLIB:
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_onset_row)(it, *common) for it in items)
    else:
        rows = [_onset_row(it, *common) for it in items]
    rows = [r for r in rows if r is not None]
    cols = ["date", "category", "selection", "impact_blind", "state", "state_holl", "state_holl_imp",
            "state_cost", "state_cost_joint", "state_cost_max", "state_cost_prod", "transmission",
            "transmission_robust", "var_ratio", "id_strength", "id_strength_robust", "pre_spread_bps",
            "pre_noise_bps", "crossed_max", "crossed_resp", "crossed_imp", "crossed_resp_sess",
            "crossed_imp_sess", "n_pre", "n_post"]
    df_out = pd.DataFrame(rows)
    return df_out[[c for c in cols if c in df_out.columns]] if len(df_out) else df_out


def _design(state, degree):
    state = np.asarray(state, float)
    return np.column_stack([state ** p for p in range(degree + 1)])


def fit_stress_response(panel, degree=1, impact_blind_only=True, min_id_strength=0.0):
    """Fit f(state): transmission ~ poly(state, degree), on the IMPACT-BLIND benchmark by default.
    Returns the coefficients with OLS SEs, R^2, the slope, the fitted predictor, and the subset used.
    With degree>=2 the leading coefficient is the convex 'spiral' term. (Cross-event inference proper
    is Ibragimov-Muller -- see onset_inference; these SEs are the within-fit summary.)"""
    sub = panel.copy()
    if impact_blind_only and "impact_blind" in sub:
        sub = sub[sub["impact_blind"]]
    sub = sub[np.isfinite(sub["transmission"]) & np.isfinite(sub["state"])]
    if min_id_strength > 0 and "id_strength" in sub:
        sub = sub[sub["id_strength"] >= min_id_strength]
    n = int(len(sub))
    if n <= degree + 1:
        return {"coef": {}, "se": {}, "n": n, "r2": np.nan, "slope": np.nan, "slope_se": np.nan,
                "degree": degree, "predict": (lambda s: np.full(np.atleast_1d(s).shape, np.nan)), "panel": sub}
    X, y = _design(sub["state"].to_numpy(), degree), sub["transmission"].to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    sigma2 = float(resid @ resid / max(n - X.shape[1], 1))
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    r2 = 1.0 - float(resid @ resid) / (float(np.sum((y - y.mean()) ** 2)) + EPS)
    b_arr = b.copy()
    return {"coef": {f"b{p}": float(b[p]) for p in range(len(b))},
            "se": {f"b{p}": float(se[p]) for p in range(len(se))}, "n": n, "r2": r2,
            "slope": float(b[1]), "slope_se": float(se[1]), "degree": degree,
            "predict": (lambda s: _design(np.atleast_1d(np.asarray(s, float)), degree) @ b_arr),
            "panel": sub}


def excess_over_benchmark(panel, fit):
    """For the SALIENCE-CURATED events, residual = transmission - f_benchmark(state). Test mean excess
    > 0 -- transmission beyond what an impact-blind shock at the same dislocation predicts (the spiral /
    'sincerity' effect) -- with an Ibragimov-Muller t across the curated events (the benchmark carries
    the exogeneity, so this excess is identified even when the curated tail is selected on salience)."""
    cur = panel[(~panel["impact_blind"]) & np.isfinite(panel["transmission"]) & np.isfinite(panel["state"])]
    if cur.empty:
        return {"n": 0, "excess": np.array([]), "mean": np.nan, "im": None, "dates": []}
    pred = np.asarray(fit["predict"](cur["state"].to_numpy())).ravel()
    excess = cur["transmission"].to_numpy() - pred
    return {"n": int(len(excess)), "excess": excess, "mean": float(np.mean(excess)),
            "im": inf.ibragimov_muller(excess), "dates": list(cur["date"])}


def onset_inference(panel, column="transmission", impact_blind_only=True):
    """Ibragimov-Muller across events on a per-event quantity (default the impact-blind benchmark's
    onset transmission): the few-cluster t-test valid where the wild bootstrap is not."""
    sub = panel[panel["impact_blind"]] if (impact_blind_only and "impact_blind" in panel) else panel
    return inf.ibragimov_muller(sub[column].to_numpy())


def run_onset_study(sessions, events_frame, degree=1, pre_steps=600, post_steps=600,
                    impulse="ES", response="SPY", release_by_date=None, min_id_strength=0.0):
    """End-to-end: per-event onset panel -> benchmark f(state) on the impact-blind events ->
    excess-over-benchmark test on the salience-curated tail -> IM inference. Returns all pieces."""
    panel = cross_event_onset(sessions, events_frame, pre_steps, post_steps, impulse, response, release_by_date)
    fit = fit_stress_response(panel, degree=degree, min_id_strength=min_id_strength)
    return {"panel": panel, "fit": fit, "excess": excess_over_benchmark(panel, fit),
            "benchmark_im": onset_inference(panel),
            "n_events": int(len(panel)),
            "n_impact_blind": int(panel["impact_blind"].sum()) if len(panel) else 0}


# ── 2-D stress surface: the liquidity-spiral interaction f(basis, hollowness) ──────────────────────
def _ols_full(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    sigma2 = float(resid @ resid / max(n - k, 1))
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    r2 = 1.0 - float(resid @ resid) / (float(np.sum((y - y.mean()) ** 2)) + EPS)
    return b, se, r2


def fit_stress_surface(panel, basis_col="state", holl_col="state_holl", y_col="transmission",
                       impact_blind_only=True, min_id_strength=0.0, jackknife=True, covar_cols=None):
    """Fit the bilinear stress SURFACE on the impact-blind benchmark:

        transmission = b0 + b1*(basis-b̄) + b2*(holl-h̄) + b3*(basis-b̄)(holl-h̄) + e.

    b3 is the liquidity-spiral interaction -- the cross-sectional complementarity the
    Brunnermeier-Pedersen / limits-of-arbitrage logic predicts (a wide basis bites HARDER when the book
    is hollow). The additive model b1*basis + b2*holl cannot represent this at all, so the interaction
    is the MINIMUM specification in which the mechanism can exist, not a luxury. States are CENTERED
    before forming the product, which removes the mechanical collinearity between the interaction and
    its main effects and leaves only genuine corner-emptiness (read that off interaction_identified).

    Inference note: b3 is a regression coefficient, not a per-event scalar, so Ibragimov-Muller does NOT
    apply to it (IM is the tool for the per-event transmission/excess in fit_stress_response /
    excess_over_benchmark). With few, possibly-influential events the honest companion is the
    leave-one-event-out jackknife band on b3 returned here. Returns coef/se {b0..b3}, r2, n, the
    interaction estimate + t, the centering constants, predict(basis, holl), and the jackknife band."""
    sub = panel.copy()
    if impact_blind_only and "impact_blind" in sub:
        sub = sub[sub["impact_blind"]]
    covar_cols = list(covar_cols) if covar_cols else []
    mask = np.isfinite(sub[basis_col]) & np.isfinite(sub[holl_col]) & np.isfinite(sub[y_col])
    for c in covar_cols:
        mask = mask & np.isfinite(sub[c])
    sub = sub[mask]
    if min_id_strength > 0 and "id_strength" in sub:
        sub = sub[sub["id_strength"] >= min_id_strength]
    n = int(len(sub))
    b_raw = sub[basis_col].to_numpy(); h_raw = sub[holl_col].to_numpy(); y = sub[y_col].to_numpy()
    bbar, hbar = float(np.mean(b_raw)), float(np.mean(h_raw))
    if n <= 4 + len(covar_cols):
        return {"coef": {}, "se": {}, "n": n, "r2": np.nan, "interaction": np.nan, "interaction_se": np.nan,
                "interaction_t": np.nan, "centers": (bbar, hbar), "jackknife": None,
                "predict": (lambda bb, hh: np.full(np.atleast_1d(bb).shape, np.nan)), "panel": sub}
    bc, hc = b_raw - bbar, h_raw - hbar

    def design(bb, hh):
        bb = np.atleast_1d(np.asarray(bb, float)) - bbar
        hh = np.atleast_1d(np.asarray(hh, float)) - hbar
        return np.column_stack([np.ones_like(bb), bb, hh, bb * hh])

    Xcols = [np.ones(n), bc, hc, bc * hc]
    for c in covar_cols:                                       # linear controls, centered (held at mean in predict)
        v = sub[c].to_numpy(float); Xcols.append(v - np.nanmean(v))
    X = np.column_stack(Xcols)
    cond = float(np.linalg.cond(X)) if n > X.shape[1] else np.inf
    if not np.isfinite(cond) or cond > 1e9:
        # design near-singular: the interaction is not SEPARATELY estimable (an unspanned plane / a
        # near-constant capacity axis). Report NaN, not an exploded least-norm coefficient; the verdict
        # belongs to interaction_identified, which flags exactly this case.
        return {"coef": {}, "se": {}, "n": n, "r2": np.nan, "interaction": np.nan, "interaction_se": np.nan,
                "interaction_t": np.nan, "interaction_p": np.nan, "centers": (bbar, hbar), "jackknife": None,
                "ill_conditioned": True, "cond": cond,
                "predict": (lambda bb, hh: np.full(np.atleast_1d(bb).shape, np.nan)), "panel": sub}
    b, se, r2 = _ols_full(X, y)
    from scipy import stats
    t3 = float(b[3] / se[3]) if se[3] > 0 else np.nan
    p3 = float(2 * stats.t.sf(abs(t3), df=max(n - 4, 1))) if np.isfinite(t3) else np.nan
    jack = None
    if jackknife and n > 5:
        loo = []
        for i in range(n):
            m = np.ones(n, bool); m[i] = False
            bi, _, _ = _ols_full(X[m], y[m])
            loo.append(bi[3])
        loo = np.asarray(loo)
        jack = {"b3_loo_min": float(loo.min()), "b3_loo_max": float(loo.max()),
                "b3_loo_sd": float(loo.std(ddof=1)), "b3_loo_range": float(loo.max() - loo.min())}
    return {"coef": {f"b{i}": float(b[i]) for i in range(4)},
            "se": {f"b{i}": float(se[i]) for i in range(4)}, "n": n, "r2": r2,
            "interaction": float(b[3]), "interaction_se": float(se[3]), "interaction_t": t3, "interaction_p": p3,
            "centers": (bbar, hbar), "jackknife": jack,
            "basis_col": basis_col, "holl_col": holl_col, "y_col": y_col,
            "predict": (lambda bb, hh: design(bb, hh) @ b), "panel": sub}


def interaction_identified(panel, basis_col="state", holl_col="state_holl", impact_blind_only=True,
                           vif_tol=5.0, corr_tol=0.9, min_quadrant=3):
    """The b3-analog of id_strength: does the event scatter actually IDENTIFY the interaction, or is it
    extrapolation? If the events cluster on the diagonal (stress in basis AND book together -- the
    natural correlation), basis, holl and basis*holl are collinear, X'X is near-singular and the
    interaction SE explodes -- you would fail to detect a real spiral because the events don't span the
    plane. This reports, on the impact-blind sample: the cross-event corr(basis, holl); the VIF of the
    CENTERED interaction column on the centered main effects (post-centering this is genuine corner-
    emptiness, not the mechanical part); and the median-split quadrant counts, with the sparsest
    OFF-DIAGONAL corner (hi-basis/lo-holl and lo-basis/hi-holl -- the cells that pin b3) called out.
    `identified` requires all three to pass; `reasons` lists any that fail."""
    sub = panel.copy()
    if impact_blind_only and "impact_blind" in sub:
        sub = sub[sub["impact_blind"]]
    sub = sub[np.isfinite(sub[basis_col]) & np.isfinite(sub[holl_col])]
    n = int(len(sub))
    out = {"n": n, "corr": np.nan, "vif_interaction": np.nan, "quadrant_counts": {},
           "min_offdiag_quadrant": 0, "identified": False, "reasons": ["too_few"]}
    if n < 4:
        return out
    b = sub[basis_col].to_numpy(); h = sub[holl_col].to_numpy()
    bc, hc = b - b.mean(), h - h.mean()
    corr = float(np.corrcoef(b, h)[0, 1]) if (bc.std() > 0 and hc.std() > 0) else np.nan
    inter = bc * hc
    Z = np.column_stack([np.ones(n), bc, hc])
    _, _, r2_int = _ols_full(Z, inter)
    vif = float(1.0 / max(1.0 - r2_int, EPS))
    bmed, hmed = np.median(b), np.median(h)
    hi_b, hi_h = b > bmed, h > hmed
    q = {"hi_basis_hi_holl": int(np.sum(hi_b & hi_h)), "hi_basis_lo_holl": int(np.sum(hi_b & ~hi_h)),
         "lo_basis_hi_holl": int(np.sum(~hi_b & hi_h)), "lo_basis_lo_holl": int(np.sum(~hi_b & ~hi_h))}
    min_off = min(q["hi_basis_lo_holl"], q["lo_basis_hi_holl"])
    reasons = []
    if not (np.isfinite(corr) and abs(corr) <= corr_tol):
        reasons.append("states_collinear")
    if not (vif <= vif_tol):
        reasons.append("interaction_vif")
    if not (min_off >= min_quadrant):
        reasons.append("offdiag_corner_sparse")
    out.update(corr=corr, vif_interaction=vif, quadrant_counts=q, min_offdiag_quadrant=int(min_off),
               identified=(len(reasons) == 0), reasons=reasons)
    return out


# ── launcher-grade orchestration: the surface across the leg triple on the robust transmission ─────
def _onset_surface_summary(panel, results, marg):
    nib = int(panel["impact_blind"].sum()) if len(panel) else 0
    L = ["ONSET STRESS-SURFACE  (events=%d, impact-blind=%d)" % (len(panel), nib),
         "1-D marginal f(basis) slope [raw transmission]: %.4f (se %.4f, n %d)"
         % (marg.get("slope", float("nan")), marg.get("slope_se", float("nan")), marg.get("n", 0)),
         "",
         "liquidity-spiral interaction b3 across the capacity axes (transmission_robust; naive=raw):",
         "  %-15s %10s %10s %8s %9s %9s  %s"
         % ("axis", "b3_robust", "b3_naive", "p_robust", "identified", "loo_band", "id_note")]
    for name in ["joint_cost", "cost_max", "cost_prod", "ES_hollowness", "SPY_hollowness"]:
        r = results.get(name)
        if not r:
            continue
        note = "ok" if r["identified"] else ",".join(r["id_reasons"])
        pr = r["p_robust"] if r["p_robust"] is not None else float("nan")
        jk = r.get("jackknife_robust")
        band = jk["b3_loo_range"] if jk else float("nan")
        L.append("  %-15s %10.4f %10.4f %8.3f %9s %9.4f  %s"
                 % (name, r["b3_robust"], r["b3_naive"], pr, str(r["identified"]), band, note))
    L += ["",
          "read: cost aggregations of the round-trip arb cost -- joint_cost (SUM) and cost_max",
          "(BOTTLENECK, the binding leg) are the co-headline; cost_prod (PRODUCT, bps^2) is the",
          "COMPARATOR, expected to come back unidentified / wide loo_band (a disguised three-way the",
          "event count cannot fund). b3 STRONGER under cost_max than joint_cost is itself a bottleneck /",
          "joint-withdrawal signature. ES_hollowness is the cross/contagion leg; SPY_hollowness the",
          "own-leg robustness. A spiral that survives on b3_robust AND is 'identified' is real; else it",
          "is bounce, extrapolation, or an unspanned plane (see id_note / loo_band)."]
    L += _crossing_qc_lines(panel)
    return "\n".join(L)


# Contamination tiers for the onset-window crossed-book rate. A crossed top is impossible in a real
# book, so any nonzero rate is reconstruction error; these thresholds only say how much of the event's
# observation it eats. Tuned to the observed FOMC spread (a few % up to ~75%).
_XED_CLEAN, _XED_SUSPECT = 0.01, 0.10


def _crossing_qc_lines(panel):
    """Reconstruction-QC block: per-event crossed-book rate over the ESTIMATION WINDOW, so contaminated
    events are visible in the log rather than only in the panel CSV. Crossing is not uniform in time --
    it concentrates where message rates spike, i.e. at the onset -- so an event can look acceptable
    session-wide and still have its own pre/post windows badly crossed; both are reported and the gap
    between them is the thing to look at."""
    if not len(panel) or "crossed_max" not in panel.columns:
        return []
    x = panel["crossed_max"]
    ok = x.notna()
    if not bool(ok.any()):
        return ["", "reconstruction QC: crossed-book rate unavailable (no bid1/ask1 columns in frames)."]
    n = int(ok.sum())
    clean = int((x[ok] <= _XED_CLEAN).sum())
    suspect = int(((x[ok] > _XED_CLEAN) & (x[ok] <= _XED_SUSPECT)).sum())
    bad = int((x[ok] > _XED_SUSPECT).sum())
    L = ["",
         "reconstruction QC -- crossed top of book over each event's ESTIMATION WINDOW (impossible in a",
         "real book, so a nonzero rate is reconstruction error contaminating the spread, hollowness,",
         "cost, and mid that this event's observation is built from):",
         "  %d/%d events clean (<=%.0f%%), %d suspect (<=%.0f%%), %d CONTAMINATED (>%.0f%%)"
         % (clean, n, 100 * _XED_CLEAN, suspect, 100 * _XED_SUSPECT, bad, 100 * _XED_SUSPECT)]
    worst = panel.loc[ok].nlargest(min(5, n), "crossed_max")
    if len(worst) and float(worst["crossed_max"].iloc[0]) > _XED_CLEAN:
        L.append("  worst events (window rate | session rate):")
        for _, r in worst.iterrows():
            if not np.isfinite(r["crossed_max"]) or r["crossed_max"] <= _XED_CLEAN:
                continue
            d = pd.Timestamp(r["date"]).date() if pd.notna(r.get("date")) else "?"
            sess = max([v for v in (r.get("crossed_resp_sess"), r.get("crossed_imp_sess"))
                        if v is not None and np.isfinite(v)] or [float("nan")])
            L.append("    %s  window %6.2f%%  session %6.2f%%%s"
                     % (d, 100 * r["crossed_max"], 100 * sess,
                        "   <-- window >> session: onset-local" if np.isfinite(sess)
                        and r["crossed_max"] > 2 * max(sess, 1e-9) else ""))
    if bad:
        L.append("  ACTION: re-fit excluding the contaminated events and compare b3. If b3 moves, the")
        L.append("  surface is reading reconstruction error, not a liquidity spiral.")
    return L


def run_onset_surface(sessions, events_frame, pre_steps=600, post_steps=600, impulse="ES", response="SPY",
                      release_by_date=None, notional=1e6, n_levels=10, min_id_strength=0.0, n_jobs=1):
    """Launcher-grade orchestration of the 2-D stress SURFACE on the cross-event onset panel, run across
    the capacity axes on the NOISE-ROBUST transmission. The round-trip arbitrage cost is aggregated
    three ways -- SUM (joint_cost), BOTTLENECK (cost_max, the binding leg), and PRODUCT (cost_prod, the
    multiplicative comparator expected to be fat-tailed / unidentified) -- and the two single legs (ES
    cross / contagion, SPY own) are carried too. For each axis it reports the interaction b3 on robust
    AND naive transmission side by side (does the spiral survive the bounce correction?), the
    interaction-identification verdict (corr / VIF / off-diagonal corner coverage), and the leave-one-
    event-out jackknife band (where the product's leverage fragility shows). Returns the panel, per-axis
    results, the 1-D marginal slope for context, and a printable summary."""
    panel = cross_event_onset(sessions, events_frame, pre_steps, post_steps, impulse, response,
                              release_by_date=release_by_date, book_asset=response, n_levels=n_levels,
                              n_jobs=n_jobs)
    axes = [("joint_cost", "state_cost_joint"), ("cost_max", "state_cost_max"),
            ("cost_prod", "state_cost_prod"), ("ES_hollowness", "state_holl_imp"),
            ("SPY_hollowness", "state_holl")]
    results = {}
    for name, col in axes:
        if col not in panel.columns:
            continue
        ident = interaction_identified(panel, holl_col=col)
        fr = fit_stress_surface(panel, holl_col=col, y_col="transmission_robust", min_id_strength=min_id_strength)
        fn = fit_stress_surface(panel, holl_col=col, y_col="transmission", min_id_strength=min_id_strength)
        exc = excess_over_surface(panel, fr) if name in ("joint_cost", "cost_max") else None
        results[name] = {"col": col, "identified": ident["identified"], "id_reasons": ident["reasons"],
                         "corr": ident["corr"], "vif": ident["vif_interaction"],
                         "min_offdiag": ident["min_offdiag_quadrant"], "quadrants": ident["quadrant_counts"],
                         "b3_robust": fr["interaction"], "p_robust": fr.get("interaction_p"),
                         "b3_naive": fn["interaction"], "p_naive": fn.get("interaction_p"),
                         "jackknife_robust": fr["jackknife"], "n": fr["n"], "excess": exc}
    marg = fit_stress_response(panel, min_id_strength=min_id_strength)
    return {"panel": panel, "axes": results,
            "marginal": {"slope": marg.get("slope"), "slope_se": marg.get("slope_se"), "n": marg.get("n")},
            "summary": _onset_surface_summary(panel, results, marg),
            "n_events": int(len(panel)), "n_impact_blind": int(panel["impact_blind"].sum()) if len(panel) else 0}


def excess_over_surface(panel, fit):
    """The SURFACE analog of excess_over_benchmark: for the SALIENCE-CURATED events, residual =
    transmission - f_surface(basis, holl), where f_surface is the bilinear surface fit on the IMPACT-
    BLIND benchmark. Tests mean excess > 0 -- transmission beyond what an impact-blind shock at the same
    basis-dislocation AND book-capacity predicts (the spiral's tail, jointly conditioned) -- with an
    Ibragimov-Muller t across the curated events. The benchmark carries the exogeneity, so the excess is
    identified even when the tail is curated on salience.

    Crucially it also reports within_support_frac: the fraction of curated points inside the benchmark's
    (basis, holl) bounding box. A bilinear surface EXTRAPOLATES outside its support, and the curated
    marquee shocks are exactly the points most likely to sit beyond the scheduled backbone's range -- so
    excess for out-of-box points is extrapolation, not interpolation, and must be read with that caveat."""
    bcol = fit.get("basis_col", "state"); hcol = fit.get("holl_col", "state_holl"); ycol = fit.get("y_col", "transmission")
    out = {"n": 0, "excess": np.array([]), "mean": np.nan, "im": None, "within_support_frac": np.nan, "dates": []}
    if fit.get("predict") is None or "impact_blind" not in panel:
        return out
    cur = panel[(~panel["impact_blind"]) & np.isfinite(panel.get(bcol)) & np.isfinite(panel.get(hcol))
                & np.isfinite(panel.get(ycol))]
    if cur.empty:
        return out
    bvals = cur[bcol].to_numpy(); hvals = cur[hcol].to_numpy()
    pred = np.asarray(fit["predict"](bvals, hvals)).ravel()
    if not np.any(np.isfinite(pred)):
        return out                                            # degenerate / ill-conditioned benchmark
    excess = cur[ycol].to_numpy() - pred
    bench = fit.get("panel")
    wf = np.nan
    if bench is not None and len(bench):
        bmin, bmax = float(np.nanmin(bench[bcol])), float(np.nanmax(bench[bcol]))
        hmin, hmax = float(np.nanmin(bench[hcol])), float(np.nanmax(bench[hcol]))
        wf = float(np.mean((bvals >= bmin) & (bvals <= bmax) & (hvals >= hmin) & (hvals <= hmax)))
    fin = np.isfinite(excess)
    return {"n": int(fin.sum()), "excess": excess, "mean": float(np.nanmean(excess)) if fin.any() else np.nan,
            "im": inf.ibragimov_muller(excess[fin]), "within_support_frac": wf,
            "dates": list(cur["date"]) if "date" in cur else []}
