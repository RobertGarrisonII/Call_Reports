"""
kalman_ecm.py
=============
The Kalman arm of the E4 half-life experiment: staleness treated as MISSING DATA
in a latent-price state space, so adjustment speeds are estimated on optimally
filled prices instead of stale repeats.

The problem this closes. The fitted error-correction half-life is 9-12 seconds on
the 1s grid and 177-205 seconds at 10ms -- a twenty-fold disagreement for the same
adjustment process. The staleness hypothesis (memo E4, mechanism ii): at 10ms,
80-89% of rows are stale repeats; a stale row contributes a fake zero return and
yesterday's price to the ECM design, attenuating the measured alpha exactly as
Epps attenuates measured correlation. The state-space treatment makes the
mechanism precise: the EFFICIENT log prices are the state, a quote row is an
OBSERVATION only when the quote actually refreshed, and stale rows are simply
periods with no observation -- which the Kalman filter handles natively (predict
without update). Smoothing then fills the gaps with the model's best
cross-asset-informed interpolation, and the standard ECM estimated on the
SMOOTHED prices carries no staleness attenuation.

Design -- and the pilot lesson that shaped it:

  * kappa comes from JOINT MLE with the error-correction term IN the transition
    (x_t = x_{t-1} + (a1, a2) z_{t-1} + eta), not from a two-step
    smooth-then-regress. The two-step was tried first and FAILS INSTRUCTIVELY at
    85-90% missingness: a random-walk-transition smoother bridges the gaps along
    RW paths, which linearly interpolates the basis and manufactures persistence
    -- on a planted kappa = 0.02 DGP the two-step returned 0.0014 (15x too slow)
    while the raw stale ECM returned 0.061 (3x too fast: with heavy staleness
    the measured "adjustment speed" is really the QUOTE-REFRESH rate). The joint
    MLE, whose likelihood only scores actual observations, recovered 0.014.
    Both failure modes are worth quoting in the paper: staleness does not merely
    attenuate the ECM -- it substitutes the refresh process for the price
    process, in whichever direction the refresh rate sits relative to kappa.
  * Q and R are moment-matched, not estimated: per-leg per-step variance from
    the refresh-return realized variance, cross-correlation from Hayashi-Yoshida
    on the refresh series (both staleness-robust), observation noise
    (obs_noise_ticks x tick / mid)^2. The MLE optimizes ONLY (a1, a2) --
    two parameters, ~40-120 likelihood evaluations, tens of seconds per session
    at 100ms.
  * the RW smoother is retained for what it is good at -- optimal cross-informed
    GAP FILLING (log-price RMSE to the latent path roughly halves on the planted
    DGP) -- and is NOT used for kappa.
  * Filtered, not measured: results are conditional on the linear-Gaussian spec,
    so this arm CORROBORATES the synthetic-staleness bracket (the measured arm
    of E4); agreement between the two is the finding.

Scalar-form implementation (2-vector state, 3-number symmetric covariance,
sequential per-leg updates -- exact for diagonal R) in plain Python floats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cross_asset_pd_liquidity as ca
import noise_robust_cov as nrc
import price_discovery_shares as pds

EPS = 1e-18


# ── the filter/smoother ───────────────────────────────────────────────────────
def rw_kalman_smooth(y, obs_mask, Q, R):
    """2-D random-walk Kalman filter + RTS smoother with PER-LEG missing data.

    y: (T, 2) observations (log prices; entries ignored where obs_mask is False)
    obs_mask: (T, 2) bool -- True where the leg carries a FRESH quote
    Q: (2, 2) per-step transition covariance;  R: length-2 per-leg obs variances
    -> (T, 2) smoothed state means. The first state is initialized diffusely on
    the first observation of each leg."""
    y = np.asarray(y, float)
    m = np.asarray(obs_mask, bool)
    T = len(y)
    q00, q01, q11 = float(Q[0, 0]), float(Q[0, 1]), float(Q[1, 1])
    r0, r1 = float(R[0]), float(R[1])
    # filtered means/covs and one-step-ahead predictions (for RTS)
    fx0 = np.empty(T); fx1 = np.empty(T)
    fp00 = np.empty(T); fp01 = np.empty(T); fp11 = np.empty(T)
    px0 = np.empty(T); px1 = np.empty(T)
    pp00 = np.empty(T); pp01 = np.empty(T); pp11 = np.empty(T)
    big = 1.0                                        # diffuse prior (log-price scale)
    x0 = float(y[m[:, 0], 0][0]) if m[:, 0].any() else 0.0
    x1 = float(y[m[:, 1], 1][0]) if m[:, 1].any() else 0.0
    p00, p01, p11 = big, 0.0, big
    for t in range(T):
        # predict (A = I)
        p00 += q00; p01 += q01; p11 += q11
        px0[t] = x0; px1[t] = x1
        pp00[t] = p00; pp01[t] = p01; pp11[t] = p11
        # sequential scalar updates (exact under diagonal R)
        if m[t, 0]:
            s = p00 + r0
            k0 = p00 / s; k1 = p01 / s
            innov = y[t, 0] - x0
            x0 += k0 * innov; x1 += k1 * innov
            p11 -= k1 * p01; p01 -= k0 * p01; p00 -= k0 * p00
        if m[t, 1]:
            s = p11 + r1
            k1 = p11 / s; k0 = p01 / s
            innov = y[t, 1] - x1
            x0 += k0 * innov; x1 += k1 * innov
            p00 -= k0 * p01; p01 -= k1 * p01; p11 -= k1 * p11
        fx0[t] = x0; fx1[t] = x1
        fp00[t] = p00; fp01[t] = p01; fp11[t] = p11
    # RTS backward pass: J_t = P_f(t) @ inv(P_pred(t+1)); A = I
    sx0 = fx0.copy(); sx1 = fx1.copy()
    for t in range(T - 2, -1, -1):
        a, b, c = pp00[t + 1], pp01[t + 1], pp11[t + 1]
        det = a * c - b * b
        if det <= EPS:
            continue
        i00, i01, i11 = c / det, -b / det, a / det
        j00 = fp00[t] * i00 + fp01[t] * i01
        j01 = fp00[t] * i01 + fp01[t] * i11
        j10 = fp01[t] * i00 + fp11[t] * i01
        j11 = fp01[t] * i01 + fp11[t] * i11
        d0 = sx0[t + 1] - px0[t + 1]
        d1 = sx1[t + 1] - px1[t + 1]
        sx0[t] += j00 * d0 + j01 * d1
        sx1[t] += j10 * d0 + j11 * d1
    return np.column_stack([sx0, sx1])


# ── moment-matched parameters from one session ───────────────────────────────
def _refresh(tsec, mid):
    fin = np.isfinite(mid)
    t, x = tsec[fin], mid[fin]
    chg = np.concatenate([[True], np.diff(x) != 0.0])
    return t[chg], x[chg]


def session_q_r(df, dt, obs_noise_ticks=0.5, ticks=None):
    """(Q, R, obs_mask, y): per-step transition covariance from the refresh-return
    realized variances and the HY cross-covariance (both staleness-robust), and the
    per-leg observation-noise variances from the tick size."""
    ticks = ticks or {"SPY": 0.01, "ES": 0.25}
    idx = df.index
    tsec = idx.view("int64").astype(float) / 1e9
    T = len(df)
    y = np.full((T, 2), np.nan)
    mask = np.zeros((T, 2), bool)
    rv = {}
    refreshed = {}
    for j, a in enumerate(("SPY", "ES")):
        mid = np.asarray(ca._mid(df, a), float)
        fin = np.isfinite(mid)
        lp = np.where(fin & (mid > 0), np.log(np.where(mid > 0, mid, np.nan)), np.nan)
        chg = np.zeros(T, bool)
        fin_ix = np.where(np.isfinite(lp))[0]
        if len(fin_ix):
            chg[fin_ix[0]] = True
            d = np.diff(lp[fin_ix])
            chg[fin_ix[1:][d != 0.0]] = True
        mask[:, j] = chg
        y[:, j] = lp
        tr, xr = tsec[chg], lp[chg]
        rr = np.diff(xr)
        rv[a] = float(np.sum(rr * rr))
        refreshed[a] = (tr, xr)
        med = np.nanmedian(mid)
        y_noise = (obs_noise_ticks * ticks[a] / med) ** 2 if np.isfinite(med) and med > 0 else 1e-12
        if j == 0:
            r0 = y_noise
        else:
            r1 = y_noise
    hy = nrc.hayashi_yoshida(refreshed["SPY"][0], refreshed["SPY"][1],
                             refreshed["ES"][0], refreshed["ES"][1])
    v0 = max(rv["SPY"] / max(T, 1), EPS)
    v1 = max(rv["ES"] / max(T, 1), EPS)
    c01 = hy / max(T, 1)
    cap = 0.98 * np.sqrt(v0 * v1)                   # keep Q positive definite
    c01 = float(np.clip(c01, -cap, cap))
    Q = np.array([[v0, c01], [c01, v1]])
    return Q, np.array([r0, r1]), mask, y


def smooth_session(df, dt, obs_noise_ticks=0.5):
    """-> (mid_spy_smoothed, mid_es_smoothed) as price-level arrays on the frame's
    own grid, plus a diagnostics dict."""
    Q, R, mask, y = session_q_r(df, dt, obs_noise_ticks)
    sm = rw_kalman_smooth(y, mask, Q, R)
    diag = {"stale_frac_SPY": 1.0 - mask[:, 0].mean(), "stale_frac_ES": 1.0 - mask[:, 1].mean(),
            "q_corr": float(Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1] + EPS))}
    return np.exp(sm[:, 0]), np.exp(sm[:, 1]), diag


# ── joint Kalman-VECM MLE (the kappa that staleness cannot reach) ────────────
def vecm_kalman_loglik(a1, a2, y, mask, Q, R, mu):
    """Gaussian log-likelihood of the observed (fresh) quotes under
    x_t = x_{t-1} + (a1, a2) (z_{t-1} - mu) + eta,  z = x_spy - x_es,
    with per-leg missingness. Stale rows contribute prediction steps only --
    the refresh process never enters the likelihood as data."""
    q00, q01, q11 = float(Q[0, 0]), float(Q[0, 1]), float(Q[1, 1])
    r0, r1 = float(R[0]), float(R[1])
    T = len(y)
    x0 = float(y[mask[:, 0], 0][0]) if mask[:, 0].any() else 0.0
    x1 = float(y[mask[:, 1], 1][0]) if mask[:, 1].any() else 0.0
    p00, p01, p11 = 1.0, 0.0, 1.0
    ll = 0.0
    ln2pi = float(np.log(2.0 * np.pi))
    for t in range(T):
        z = x0 - x1 - mu
        x0p = x0 + a1 * z
        x1p = x1 + a2 * z
        f00 = 1.0 + a1; f01 = -a1; f10 = a2; f11 = 1.0 - a2
        t00 = f00 * p00 + f01 * p01; t01 = f00 * p01 + f01 * p11
        t10 = f10 * p00 + f11 * p01; t11 = f10 * p01 + f11 * p11
        p00 = t00 * f00 + t01 * f01 + q00
        p01 = t00 * f10 + t01 * f11 + q01
        p11 = t10 * f10 + t11 * f11 + q11
        x0, x1 = x0p, x1p
        if mask[t, 0]:
            s = p00 + r0
            innov = y[t, 0] - x0
            ll -= 0.5 * (ln2pi + np.log(s) + innov * innov / s)
            k0 = p00 / s; k1 = p01 / s
            x0 += k0 * innov; x1 += k1 * innov
            p11 -= k1 * p01; p01 -= k0 * p01; p00 -= k0 * p00
        if mask[t, 1]:
            s = p11 + r1
            innov = y[t, 1] - x1
            ll -= 0.5 * (ln2pi + np.log(s) + innov * innov / s)
            k1 = p11 / s; k0 = p01 / s
            x0 += k0 * innov; x1 += k1 * innov
            p00 -= k0 * p01; p01 -= k1 * p01; p11 -= k1 * p11
    return float(ll)


def fit_kalman_vecm(df, dt, obs_noise_ticks=0.5, maxiter=120):
    """Joint MLE of the error-correction loadings on one session's frame.
    -> {'alpha_spy', 'alpha_es', 'kappa', 'half_life_s', 'n_evals', diagnostics}.
    kappa = alpha_es - alpha_spy, matching price_discovery_shares' convention."""
    from scipy.optimize import minimize
    Q, R, mask, y = session_q_r(df, dt, obs_noise_ticks)
    lp_s = y[:, 0]; lp_e = y[:, 1]
    mu = float(np.nanmean(lp_s - lp_e))

    def neg(p):
        return -vecm_kalman_loglik(float(p[0]), float(p[1]), y, mask, Q, R, mu)

    res = minimize(neg, x0=np.array([-0.005, 0.005]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": int(maxiter)})
    a1, a2 = float(res.x[0]), float(res.x[1])
    kappa = a2 - a1
    return {"alpha_spy": a1, "alpha_es": a2, "kappa": kappa,
            "half_life_s": (float(np.log(2.0) / kappa * dt) if kappa > 0 else float("nan")),
            "n_evals": int(res.nfev),
            "stale_frac_SPY": float(1.0 - mask[:, 0].mean()),
            "stale_frac_ES": float(1.0 - mask[:, 1].mean())}


# ── the E4 experiment table ──────────────────────────────────────────────────
def half_life_experiment(sessions, dt, n_lags=None, obs_noise_ticks=0.5):
    """Per session: the ECM's kappa/half-life on RAW (stale) mids (pds.estimate_day)
    vs the JOINT Kalman-VECM MLE, which scores only actual quote refreshes.
    -> (per_day DataFrame, summary dict)."""
    if n_lags is None:
        n_lags = ca.frequency_defaults(dt=dt)["n_lags"]
    rows = []
    for date, regime, df in sessions:
        row = {"date": str(date), "regime": regime, "dt_s": float(dt), "n_lags": int(n_lags)}
        try:
            m_spy = np.asarray(ca._mid(df, "SPY"), float)
            m_es = np.asarray(ca._mid(df, "ES"), float)
            raw = pds.estimate_day(m_spy, m_es, n_lags=n_lags)
            k_raw = float(raw.get("kappa", np.nan))
            row["kappa_raw"] = k_raw
            row["half_life_raw_s"] = (float(np.log(2.0) / k_raw * dt)
                                      if np.isfinite(k_raw) and k_raw > 0 else float("nan"))
            kal = fit_kalman_vecm(df, dt, obs_noise_ticks)
            row["kappa_kalman"] = kal["kappa"]
            row["half_life_kalman_s"] = kal["half_life_s"]
            row["mle_evals"] = kal["n_evals"]
            row["stale_frac_SPY"] = kal["stale_frac_SPY"]
            row["stale_frac_ES"] = kal["stale_frac_ES"]
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)
    per_day = pd.DataFrame(rows).set_index("date")
    summary = {}
    if "half_life_raw_s" in per_day.columns:
        hr = pd.to_numeric(per_day["half_life_raw_s"], errors="coerce")
        hk = pd.to_numeric(per_day["half_life_kalman_s"], errors="coerce")
        both = hr.notna() & hk.notna()
        summary = {"n_days": int(both.sum()),
                   "median_half_life_raw_s": float(hr[both].median()) if both.any() else float("nan"),
                   "median_half_life_kalman_s": float(hk[both].median()) if both.any() else float("nan"),
                   "median_ratio_raw_over_kalman": float((hr[both] / hk[both]).median())
                   if both.any() else float("nan")}
    return per_day, summary


# ── planted-staleness DGP (shared by the gate and the demo runner) ────────────
def staleness_demo_sessions(n_days=2, n_steps=30000, kappa=0.02, keep=(0.15, 0.10),
                            seed=0, dt=0.1):
    """Cointegrated pair with a KNOWN per-step kappa, observed through staleness:
    each leg's quote refreshes with probability keep[j] per step and carries
    forward otherwise. -> (sessions list, truth dict with the latent mids)."""
    rng = np.random.default_rng(seed)
    out, truth = [], {}
    for d in range(n_days):
        w = np.cumsum(rng.normal(0, 8e-5, n_steps))            # common factor
        z = np.zeros(n_steps)                                  # basis, AR(1)
        e = rng.normal(0, 4e-5, n_steps)
        for t in range(1, n_steps):
            z[t] = (1.0 - kappa) * z[t - 1] + e[t]
        lp_spy = np.log(550.0) + w + 0.5 * z
        lp_es = np.log(5500.0) + w - 0.5 * z
        idx = pd.date_range("2024-07-24 09:30:00", periods=n_steps,
                            freq="%dms" % int(dt * 1000), tz="America/New_York")
        cols = {}
        for j, (a, lp, tick) in enumerate((("SPY", lp_spy, 0.01), ("ES", lp_es, 0.25))):
            fresh = rng.random(n_steps) < keep[j]
            fresh[0] = True
            obs = np.exp(lp)
            stale = pd.Series(np.where(fresh, obs, np.nan)).ffill().to_numpy()
            cols[f"{a}_bidprice_1"] = stale - tick / 2
            cols[f"{a}_askprice_1"] = stale + tick / 2
            cols[f"{a}_bidquantity_1"] = np.full(n_steps, 100.0)
            cols[f"{a}_askquantity_1"] = np.full(n_steps, 100.0)
        out.append((f"d{d}", "volatile", pd.DataFrame(cols, index=idx)))
        truth[f"d{d}"] = {"lp_spy": lp_spy, "lp_es": lp_es, "kappa": kappa}
    return out, truth
