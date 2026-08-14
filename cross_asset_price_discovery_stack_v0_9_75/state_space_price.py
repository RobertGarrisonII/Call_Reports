"""state_space_price.py -- Gaussian linear state-space decomposition of observed prices into a
PERMANENT (efficient / random-walk) component and a TRANSITORY (microstructure pricing-error)
component, via a from-scratch Kalman filter + RTS smoother fit by maximum likelihood (numpy/scipy).

Why this exists. The size-weighted micro-price is the conditional efficient price only under a
no-drift stationarity assumption that breaks in a fast directional move -- exactly the crash windows
the contagion paper lives on. Rather than ASSUME the micro-price gap equals the pricing error, this
module *estimates* the latent efficient price as a state and DEFINES the transitory component as the
residual delta_t = y_t - p*_t. Three specifications, in increasing structure:

  local_level         y_t = p*_t + u_t ; p*_t = p*_{t-1} + w_t                      (RW + iid noise)
  local_linear_trend  adds a time-varying drift:  p*_t = p*_{t-1} + mu_{t-1} + w_t ;
                      mu_t = mu_{t-1} + xi_t  -- the efficient price can TREND, so a directional
                      (crash) move is NOT misattributed to transitory noise. This is the drift fix.
  common_factor       two cointegrated prices share ONE efficient price plus asset-specific AR(1)
                      pricing errors:  y^A = p* + s^A,  y^B = p* + s^B.  The smoothed p* is the
                      consensus fair value; s^A, s^B are each leg's deviation from it. The leg whose
                      pricing error is smaller / reverts faster is the price-discovery ANCHOR -- a
                      structural measure complementary to (NOT identical to) the VECM information
                      share (see caveats in `common_factor` docstring).

Notation (Durbin & Koopman): y_t = Z a_t + eps, eps~N(0,H);  a_{t+1} = T a_t + R eta, eta~N(0,Q).
Data are internally demeaned/scaled for conditioning and all outputs are returned in ORIGINAL units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_KAPPA = 1e6          # diffuse prior variance (in standardized units; data are O(1) after scaling)


# ════════════════════════════════════════════════════════════════════════════
# Core engine: Kalman filter (with the Gaussian prediction-error decomposition) + RTS smoother
# ════════════════════════════════════════════════════════════════════════════
def kalman_filter(y, Z, T, R, Q, H, a0, P0, loglik_burn: int = 0):
    """Forward filter. ``y`` is (n_obs, m) (NaN rows allowed = missing observation). Returns filtered
    and one-step-predicted states/covariances, forecast errors v_t and their covariances F_t, and the
    Gaussian log-likelihood (summed over t >= ``loglik_burn`` -- skip the diffuse-initialized obs whose
    F_t is inflated by the prior). Returns None on a non-PD F_t (an infeasible parameter)."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    n_obs, m = y.shape
    n = T.shape[0]
    RQRt = R @ Q @ R.T
    a = np.asarray(a0, float).reshape(n)
    P = np.asarray(P0, float).copy()
    a_filt = np.zeros((n_obs, n)); P_filt = np.zeros((n_obs, n, n))
    a_pred = np.zeros((n_obs, n)); P_pred = np.zeros((n_obs, n, n))
    v_all = np.full((n_obs, m), np.nan); F_all = np.zeros((n_obs, m, m))
    ll = 0.0
    for t in range(n_obs):
        if t == 0:
            ap, Pp = a, P                          # the prior IS the t=0 prediction
        else:
            ap = T @ a
            Pp = T @ P @ T.T + RQRt
        Pp = 0.5 * (Pp + Pp.T)
        a_pred[t] = ap; P_pred[t] = Pp
        obs = ~np.isnan(y[t])
        if not obs.any():                          # no data this step: carry the prediction forward
            a, P = ap, Pp
            a_filt[t] = a; P_filt[t] = P
            continue
        Zt = Z[obs]; Ht = H[np.ix_(obs, obs)]
        v = y[t, obs] - Zt @ ap
        F = Zt @ Pp @ Zt.T + Ht
        F = 0.5 * (F + F.T)
        sign, logdetF = np.linalg.slogdet(F)
        if sign <= 0 or not np.isfinite(logdetF):
            return None
        Finv = np.linalg.inv(F)
        K = Pp @ Zt.T @ Finv
        a = ap + K @ v
        P = Pp - K @ Zt @ Pp
        P = 0.5 * (P + P.T)
        a_filt[t] = a; P_filt[t] = P
        v_all[t, obs] = v; F_all[t][np.ix_(obs, obs)] = F
        if t >= loglik_burn:
            ll += -0.5 * (obs.sum() * np.log(2 * np.pi) + logdetF + v @ Finv @ v)
    return {"a_filt": a_filt, "P_filt": P_filt, "a_pred": a_pred, "P_pred": P_pred,
            "v": v_all, "F": F_all, "loglik": float(ll)}


def rts_smoother(filt: dict, T):
    """Rauch-Tung-Striebel fixed-interval smoother: the full-sample E[a_t | y_1..y_n]."""
    a_filt, P_filt = filt["a_filt"], filt["P_filt"]
    a_pred, P_pred = filt["a_pred"], filt["P_pred"]
    n_obs, n = a_filt.shape
    a_sm = a_filt.copy(); P_sm = P_filt.copy()
    for t in range(n_obs - 2, -1, -1):
        Pp = P_pred[t + 1]
        try:
            C = P_filt[t] @ T.T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:
            C = P_filt[t] @ T.T @ np.linalg.pinv(Pp)
        a_sm[t] = a_filt[t] + C @ (a_sm[t + 1] - a_pred[t + 1])
        P_sm[t] = P_filt[t] + C @ (P_sm[t + 1] - Pp) @ C.T
    return a_sm, P_sm


# ════════════════════════════════════════════════════════════════════════════
# Model matrix builders (theta is UNCONSTRAINED; variances via exp, AR coefs via tanh)
# ════════════════════════════════════════════════════════════════════════════
def _ll_build(theta):
    sw2, su2 = np.exp(theta[0]), np.exp(theta[1])
    return (np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]),
            np.array([[sw2]]), np.array([[su2]]))


def _llt_build(theta):
    sw2, sx2, su2 = np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2])
    Z = np.array([[1.0, 0.0]])
    T = np.array([[1.0, 1.0], [0.0, 1.0]])
    return Z, T, np.eye(2), np.diag([sw2, sx2]), np.array([[su2]])


def _cf_build(theta):
    sw2 = np.exp(theta[0])
    phiA, sA2 = np.tanh(theta[1]), np.exp(theta[2])
    phiB, sB2 = np.tanh(theta[3]), np.exp(theta[4])
    Z = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    T = np.diag([1.0, phiA, phiB])
    Q = np.diag([sw2, sA2, sB2])
    H = 1e-10 * np.eye(2)                            # pricing errors absorb the idiosyncratic deviation
    return Z, T, np.eye(3), Q, H


def _cf_init(theta, y):
    phiA, sA2 = np.tanh(theta[1]), np.exp(theta[2])
    phiB, sB2 = np.tanh(theta[3]), np.exp(theta[4])
    vA = min(sA2 / max(1 - phiA ** 2, 1e-6), _KAPPA)
    vB = min(sB2 / max(1 - phiB ** 2, 1e-6), _KAPPA)
    a0 = np.array([np.nanmean(y[0]), 0.0, 0.0])
    return a0, np.diag([_KAPPA, vA, vB])


# ════════════════════════════════════════════════════════════════════════════
# Standardization + MLE driver
# ════════════════════════════════════════════════════════════════════════════
def _standardize(arr):
    """Demean each column and divide by a common scale (preserves the cross-column common factor)."""
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr[:, None]
    mu = np.nanmean(arr, axis=0)
    z = arr - mu
    scale = float(np.nanstd(z))
    scale = scale if scale > 0 else 1.0
    return z / scale, mu, scale


def _fit(y_std, build, theta0, init_fn, burn, n_restart=2, seed=0):
    """Minimize the negative log-likelihood over the unconstrained theta (L-BFGS-B + restarts)."""
    def nll(theta):
        try:
            Z, T, R, Q, H = build(theta)
            a0, P0 = init_fn(theta, y_std)
            f = kalman_filter(y_std, Z, T, R, Q, H, a0, P0, loglik_burn=burn)
        except (np.linalg.LinAlgError, FloatingPointError):
            return 1e12
        if f is None or not np.isfinite(f["loglik"]):
            return 1e12
        return -f["loglik"]

    best = None
    rng = np.random.default_rng(seed)
    starts = [np.asarray(theta0, float)] + [np.asarray(theta0, float) + rng.normal(0, 0.5, len(theta0))
                                            for _ in range(n_restart)]
    for s in starts:
        try:
            r = minimize(nll, s, method="L-BFGS-B")
        except Exception:
            continue
        if r.success or np.isfinite(r.fun):
            if best is None or r.fun < best.fun:
                best = r
    if best is None:
        raise RuntimeError("state-space MLE failed to converge from all starts")
    return best.x, -best.fun


def _default_init_diag(n_diffuse, y):
    def init(theta, yy):
        n = n_diffuse
        a0 = np.zeros(n); a0[0] = np.nanmean(yy[0])
        return a0, np.eye(n) * _KAPPA
    return init


def _as_series(arr, index):
    return pd.Series(arr, index=index) if index is not None else pd.Series(arr)


def _half_life(phi):
    return float(np.log(0.5) / np.log(phi)) if 0.0 < phi < 1.0 else np.inf


# ════════════════════════════════════════════════════════════════════════════
# Public estimators
# ════════════════════════════════════════════════════════════════════════════
def local_level(price) -> dict:
    """Decompose one price series into a random-walk efficient price + iid transitory noise.
    Returns smoothed ``efficient``, ``transitory`` (= price - efficient), the fitted (sigma_w, sigma_u)
    in original units, and ``noise_share`` = Var(transitory)/Var(price-changes-implied), a simple
    price-(in)efficiency read (microstructure-noise fraction)."""
    idx = price.index if isinstance(price, pd.Series) else None
    y = np.asarray(price, float)
    z, mu, scale = _standardize(y)
    dv = np.nanvar(np.diff(z[:, 0]))
    theta0 = [np.log(max(dv / 2, 1e-6)), np.log(max(dv / 2, 1e-6))]
    theta, ll = _fit(z, _ll_build, theta0, _default_init_diag(1, z), burn=1)
    Z, T, R, Q, H = _ll_build(theta)
    a0, P0 = _default_init_diag(1, z)(theta, z)
    a_sm, _ = rts_smoother(kalman_filter(z, Z, T, R, Q, H, a0, P0, loglik_burn=1), T)
    eff = a_sm[:, 0] * scale + mu[0]
    transitory = y - eff
    sw, su = np.sqrt(Q[0, 0]) * scale, np.sqrt(H[0, 0]) * scale
    return {"efficient": _as_series(eff, idx), "transitory": _as_series(transitory, idx),
            "sigma_w": float(sw), "sigma_u": float(su), "loglik": ll, "model": "local_level",
            "noise_share": float(np.var(transitory) / (np.var(transitory) + np.var(np.diff(eff)) + 1e-30))}


def local_linear_trend(price) -> dict:
    """Like ``local_level`` but the efficient price carries a time-varying DRIFT mu_t, so a directional
    trend lives in the PERMANENT component instead of leaking into the transitory residual. Returns the
    smoothed ``efficient``, ``drift`` (mu_t, per-step, original units), ``transitory``, and variances."""
    idx = price.index if isinstance(price, pd.Series) else None
    y = np.asarray(price, float)
    z, mu, scale = _standardize(y)
    dv = np.nanvar(np.diff(z[:, 0]))
    theta0 = [np.log(max(dv / 2, 1e-6)), np.log(max(dv / 50, 1e-8)), np.log(max(dv / 2, 1e-6))]
    theta, ll = _fit(z, _llt_build, theta0, _default_init_diag(2, z), burn=2)
    Z, T, R, Q, H = _llt_build(theta)
    a0, P0 = _default_init_diag(2, z)(theta, z)
    a_sm, _ = rts_smoother(kalman_filter(z, Z, T, R, Q, H, a0, P0, loglik_burn=2), T)
    eff = a_sm[:, 0] * scale + mu[0]
    drift = a_sm[:, 1] * scale
    transitory = y - eff
    return {"efficient": _as_series(eff, idx), "drift": _as_series(drift, idx),
            "transitory": _as_series(transitory, idx), "sigma_w": float(np.sqrt(Q[0, 0]) * scale),
            "sigma_xi": float(np.sqrt(Q[1, 1]) * scale), "sigma_u": float(np.sqrt(H[0, 0]) * scale),
            "loglik": ll, "model": "local_linear_trend"}


def common_factor(price_a, price_b, names=("A", "B")) -> dict:
    """Two cointegrated prices, ONE common efficient price + asset-specific AR(1) pricing errors.
    Returns the smoothed common ``efficient`` (in each leg's own units), each leg's ``pricing_error``,
    the AR(1) persistence/half-life and stationary std of each error, and ``price_discovery_share`` --
    the precision-weighted share precision_i = (1-phi_i^2)/sigma_si^2, i.e. the leg that deviates
    LESS and reverts FASTER is the anchor.

    CAVEAT (read before citing as 'information share'): the permanent innovation w_t here is COMMON to
    both legs, so this measures relative pricing EFFICIENCY (who tracks fair value more tightly), which
    is related to but NOT the same as Hasbrouck's information share (whose innovations FIRST impound
    information). For first-mover attribution use the VECM IS/CS; report this as a complementary
    structural decomposition. Inputs should share a scale -- pass LOG prices for SPY vs ES."""
    idx = price_a.index if isinstance(price_a, pd.Series) else None
    yA = np.asarray(price_a, float); yB = np.asarray(price_b, float)
    muA, muB = np.nanmean(yA), np.nanmean(yB)
    Y = np.column_stack([yA - muA, yB - muB])
    scale = float(np.nanstd(Y));  scale = scale if scale > 0 else 1.0
    Z_ = Y / scale
    dv = np.nanvar(np.diff(Z_[:, 0]))
    theta0 = [np.log(max(dv / 2, 1e-6)), np.arctanh(0.4), np.log(max(dv / 4, 1e-7)),
              np.arctanh(0.4), np.log(max(dv / 4, 1e-7))]
    theta, ll = _fit(Z_, _cf_build, theta0, _cf_init, burn=1, n_restart=3)
    Z, T, R, Q, H = _cf_build(theta)
    a0, P0 = _cf_init(theta, Z_)
    a_sm, _ = rts_smoother(kalman_filter(Z_, Z, T, R, Q, H, a0, P0, loglik_burn=1), T)
    pstar = a_sm[:, 0]
    effA = pstar * scale + muA
    effB = pstar * scale + muB
    errA = yA - effA
    errB = yB - effB
    phiA, phiB = float(T[1, 1]), float(T[2, 2])
    sA2, sB2 = float(Q[1, 1]) * scale ** 2, float(Q[2, 2]) * scale ** 2
    precA = max(1 - phiA ** 2, 1e-6) / max(sA2, 1e-30)
    precB = max(1 - phiB ** 2, 1e-6) / max(sB2, 1e-30)
    shareA = precA / (precA + precB)
    return {"efficient_a": _as_series(effA, idx), "efficient_b": _as_series(effB, idx),
            "pricing_error_a": _as_series(errA, idx), "pricing_error_b": _as_series(errB, idx),
            "phi_a": phiA, "phi_b": phiB, "half_life_a": _half_life(phiA), "half_life_b": _half_life(phiB),
            "pe_std_a": float(np.std(errA)), "pe_std_b": float(np.std(errB)),
            "price_discovery_share_a": float(shareA), "price_discovery_share_b": float(1 - shareA),
            "names": tuple(names), "loglik": ll, "model": "common_factor"}


# ════════════════════════════════════════════════════════════════════════════
# Self-test: recover KNOWN latent states / parameters from simulated data
# ════════════════════════════════════════════════════════════════════════════
def _ar1(phi, sigma, n, rng):
    e = rng.normal(0, sigma, n)
    x = np.empty(n); x[0] = e[0] / np.sqrt(max(1 - phi ** 2, 1e-6))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _selftest() -> bool:
    rng = np.random.default_rng(7)
    print("state_space_price self-test\n")

    # (1) local level: recover the latent random-walk efficient price from RW + iid noise
    n = 4000; sw, su = 0.02, 0.05
    pstar = 100.0 + np.cumsum(rng.normal(0, sw, n))
    y = pstar + rng.normal(0, su, n)
    r = local_level(y)
    corr = float(np.corrcoef(r["efficient"].to_numpy(), pstar)[0, 1])
    rmse = float(np.sqrt(np.mean((r["efficient"].to_numpy() - pstar) ** 2)))
    ll_ok = (corr > 0.99 and rmse < su and 0.5 * sw < r["sigma_w"] < 2 * sw
             and 0.5 * su < r["sigma_u"] < 2 * su)
    print("(1) local level: RW efficient price + iid noise")
    print(f"    sigma_w {sw}->{r['sigma_w']:.4f} | sigma_u {su}->{r['sigma_u']:.4f} | "
          f"corr(eff,true)={corr:.4f} rmse={rmse:.4f} (< sigma_u={su}) -> {ll_ok}")

    # (2) local linear trend: a TRENDING series -> drift sits in the permanent part, recovered
    n = 4000; drift, sw2, su2 = 0.01, 0.01, 0.05
    pstar2 = 100.0 + drift * np.arange(n) + np.cumsum(rng.normal(0, sw2, n))
    y2 = pstar2 + rng.normal(0, su2, n)
    rt = local_linear_trend(y2)
    corr2 = float(np.corrcoef(rt["efficient"].to_numpy(), pstar2)[0, 1])
    drift_hat = float(np.mean(rt["drift"].to_numpy()))
    # the transitory residual must NOT inherit the trend (its mean-reversion ~ the true iid noise)
    tr = rt["transitory"].to_numpy()
    trend_leak = abs(float(np.polyfit(np.arange(n), tr, 1)[0]))      # slope of residual vs time ~ 0
    llt_ok = (corr2 > 0.99 and 0.3 * drift < drift_hat < 3 * drift and trend_leak < drift / 10)
    print("\n(2) local linear trend: trending efficient price (the crash-drift fix)")
    print(f"    drift {drift}->{drift_hat:.4f} | corr(eff,true)={corr2:.4f} | "
          f"residual trend-leak={trend_leak:.2e} (<< drift) -> {llt_ok}")

    # (3) common factor: shared efficient price; leg A leads (small fast error), B lags (big persistent)
    n = 6000; swc = 0.02
    pc = 100.0 + np.cumsum(rng.normal(0, swc, n))
    phiA_t, sA_t = 0.2, 0.010
    phiB_t, sB_t = 0.85, 0.030
    eA = _ar1(phiA_t, sA_t, n, rng); eB = _ar1(phiB_t, sB_t, n, rng)
    yA = pc + eA; yB = pc + eB
    rc = common_factor(yA, yB, names=("LEAD", "LAG"))
    corrc = float(np.corrcoef(rc["efficient_a"].to_numpy(), pc)[0, 1])
    lead_ok = (rc["price_discovery_share_a"] > 0.6                    # A is the anchor
               and rc["phi_a"] < rc["phi_b"]                          # A reverts faster
               and rc["pe_std_a"] < rc["pe_std_b"]                    # A deviates less
               and corrc > 0.99)
    print("\n(3) common factor: who is the price-discovery anchor?")
    print(f"    phi  A {phiA_t}->{rc['phi_a']:.3f}  B {phiB_t}->{rc['phi_b']:.3f}")
    print(f"    pe_std A {rc['pe_std_a']:.4f} < B {rc['pe_std_b']:.4f} | "
          f"discovery share A={rc['price_discovery_share_a']:.3f} | corr(eff,true)={corrc:.4f} -> {lead_ok}")

    # (4) engine sanity: smoothing reduces state MSE vs filtering (uses full sample)
    f = kalman_filter(*( ( (y - y.mean())[:, None],) + _ll_build([np.log(sw**2), np.log(su**2)])
                         + (np.array([0.0]), np.array([[_KAPPA]])) ), loglik_burn=1)
    a_sm, _ = rts_smoother(f, np.array([[1.0]]))
    mse_filt = np.mean((f["a_filt"][:, 0] - (pstar - y.mean())) ** 2)
    mse_smooth = np.mean((a_sm[:, 0] - (pstar - y.mean())) ** 2)
    smooth_ok = mse_smooth < mse_filt
    print(f"\n(4) engine: smoother MSE {mse_smooth:.4f} < filter MSE {mse_filt:.4f} -> {smooth_ok}")

    ok = bool(ll_ok and llt_ok and lead_ok and smooth_ok)
    print("\nchecks:", ok)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
