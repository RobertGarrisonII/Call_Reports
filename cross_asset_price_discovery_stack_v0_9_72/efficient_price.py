"""efficient_price.py -- bivariate state-space (Kalman) decomposition of two prices (SPY, ES) into a
latent random-walk EFFICIENT (permanent) price and a stationary TRANSITORY (microstructure) component,
estimated by Gaussian MLE via the Kalman filter + RTS smoother. numpy / scipy only.

MODEL (bivariate local level; Harvey 1989 / Hasbrouck 1993 reduced form):

    mu_t = mu_{t-1} + eta_t,   eta_t ~ N(0, Q)        latent efficient (log-)price, random walk
    y_t  = mu_t      + eps_t,  eps_t ~ N(0, R)        observed (log-)mid or micro-price + microstructure noise

with y_t, mu_t in R^2 = (SPY, ES). Q is the EFFICIENT-innovation covariance (PERMANENT comovement);
R is the TRANSITORY covariance (bid-ask bounce + queue-imbalance + basis noise -- MICROSTRUCTURE comovement).

WHY THIS MAKES THE DISENTANGLING AIRTIGHT. The observed return (co)variance is
    Gamma_0 = Cov(Delta y_t)              = Q + 2 R
    Gamma_1 = Cov(Delta y_t, Delta y_{t-1}) = -R
so the OBSERVED cross-asset comovement (Q_12 + 2 R_12) splits into a PERMANENT part Q_12 and a
TRANSITORY part 2 R_12, and R is *identified* off the negative first-order return autocovariance (the
microstructure-noise / Epps signature). The simple-mid return correlation that our PCMOF/SVAR analysis
and the comovement literature use is exactly the (Q + 2R) object; this model isolates the Q
(efficient-price) piece -- the Forbes & Rigobon (2002)-robust object. The question "is FUNDAMENTAL
comovement higher in stress, or is the rise in observed correlation an artifact of transitory
(liquidity/microstructure) comovement?" becomes: compare rho(Q) across regimes, not rho(Q + 2R).

It also REPLACES the *assumed* micro-price correction delta = (s/2) * imbalance with the
MODEL-ESTIMATED transitory component delta_i,t = y_i,t - muhat_i,t (the smoother's efficient price),
so "how much of the observed mid is microstructure" is estimated rather than assumed -- dissolving the
critique that the micro-price is optimal only under no-drift conditions that fail in a fast crash.

Estimation: Gaussian MLE by the prediction-error decomposition (Kalman filter). Q, R are parameterized
by log-Cholesky factors (PSD by construction); the optimizer is seeded by a method-of-moments estimate
from (Gamma_0, Gamma_1) above, which makes the 6-parameter optimization fast and reliable. The latent
efficient price and the transitory component come from the RTS smoother at the MLE.

Bivariate by design (SPY/ES); the n-asset generalization (a common-trend factor model with information
shares) is the documented next step and shares this filter's structure.
"""
from math import log
import numpy as np
import pandas as pd
from scipy.optimize import minimize

_LOG2PI = float(np.log(2.0 * np.pi))


# ───────────────────────── covariance parameterization (PSD by construction) ─────────────────────────
def _chol_to_cov(p):
    """3 unconstrained params -> 2x2 PSD covariance via log-Cholesky (positive diagonal)."""
    l00, l10, l11 = p
    a, c = np.exp(l00), np.exp(l11)
    return np.array([[a * a, a * l10], [a * l10, l10 * l10 + c * c]])


def _cov_to_chol(C):
    """2x2 PSD covariance -> 3 log-Cholesky params (inverse of _chol_to_cov; for initialization)."""
    C = np.asarray(C, float)
    a = max(C[0, 0], 1e-300)
    l00 = 0.5 * log(a)
    l10 = C[1, 0] / np.sqrt(a)
    rem = C[1, 1] - l10 * l10
    l11 = 0.5 * log(max(rem, 1e-300))
    return np.array([l00, l10, l11])


def _psd(M, floor=1e-300):
    """Project a symmetric 2x2 onto the PSD cone (eigenvalue clip)."""
    M = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    w, V = np.linalg.eigh(M)
    return (V * np.clip(w, floor, None)) @ V.T


# ───────────────────────── Kalman filter (scalarized hot path: log-likelihood only) ─────────────────────────
def _loglik_2d(y0, y1, Q, R, P0):
    """Gaussian log-likelihood of the bivariate local level via the prediction-error decomposition.
    Fully scalarized (no per-step numpy allocation) so the MLE inner loop is fast over ~10^4-10^5 rows."""
    q00, q01, q11 = Q[0, 0], Q[0, 1], Q[1, 1]
    r00, r01, r11 = R[0, 0], R[0, 1], R[1, 1]
    m0, m1 = y0[0], y1[0]
    p00, p01, p11 = P0, 0.0, P0                       # diffuse-ish prior on the (random-walk) level
    ll = 0.0
    for t in range(len(y0)):
        pp00, pp01, pp11 = p00 + q00, p01 + q01, p11 + q11        # predict (transition = I)
        s00, s01, s11 = pp00 + r00, pp01 + r01, pp11 + r11        # innovation cov S = Ppred + R
        det = s00 * s11 - s01 * s01
        if det <= 0.0:
            return -1e18
        v0, v1 = y0[t] - m0, y1[t] - m1
        quad = (v0 * (s11 * v0 - s01 * v1) + v1 * (-s01 * v0 + s00 * v1)) / det
        ll += -0.5 * (2.0 * _LOG2PI + log(det) + quad)
        k00 = (pp00 * s11 - pp01 * s01) / det                     # gain K = Ppred S^{-1}
        k01 = (-pp00 * s01 + pp01 * s00) / det
        k10 = (pp01 * s11 - pp11 * s01) / det
        k11 = (-pp01 * s01 + pp11 * s00) / det
        m0 = m0 + k00 * v0 + k01 * v1                             # state update
        m1 = m1 + k10 * v0 + k11 * v1
        b00, b01, b10, b11 = 1.0 - k00, -k01, -k10, 1.0 - k11     # P = (I-K) Ppred, symmetrized
        n00 = b00 * pp00 + b01 * pp01
        n01 = b00 * pp01 + b01 * pp11
        n10 = b10 * pp00 + b11 * pp01
        n11 = b10 * pp01 + b11 * pp11
        p00, p01, p11 = n00, 0.5 * (n01 + n10), n11
    return ll


def _filter_smooth_2d(Y, Q, R, P0):
    """Full filter + RTS smoother (run once at the MLE) -> smoothed latent efficient price muhat (T,2)."""
    T = Y.shape[0]
    I2 = np.eye(2)
    xf = np.empty((T, 2)); Pf = np.empty((T, 2, 2))
    xp = np.empty((T, 2)); Pp = np.empty((T, 2, 2))
    x = Y[0].astype(float).copy()
    P = np.array([[P0, 0.0], [0.0, P0]])
    for t in range(T):
        xpred = x                                     # transition = I
        Ppred = P + Q
        S = Ppred + R
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        Sinv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        K = Ppred @ Sinv
        x = xpred + K @ (Y[t] - xpred)
        P = (I2 - K) @ Ppred
        P = 0.5 * (P + P.T)
        xp[t], Pp[t], xf[t], Pf[t] = xpred, Ppred, x, P
    xs = np.empty((T, 2))
    xs[-1] = xf[-1]
    for t in range(T - 2, -1, -1):
        Pn = Pp[t + 1]
        d = Pn[0, 0] * Pn[1, 1] - Pn[0, 1] * Pn[1, 0]
        inv_n = np.array([[Pn[1, 1], -Pn[0, 1]], [-Pn[1, 0], Pn[0, 0]]]) / d
        C = Pf[t] @ inv_n                             # smoother gain (transition = I)
        xs[t] = xf[t] + C @ (xs[t + 1] - xp[t + 1])
    return xs


# ───────────────────────── comovement decomposition ─────────────────────────
def _corr(C):
    d = np.sqrt(C[0, 0] * C[1, 1])
    return float(C[0, 1] / d) if d > 0 else float("nan")


def _decompose_comovement(Q, R):
    """Split the observed return (co)variance Gamma_0 = Q + 2R into permanent (Q) and transitory (2R)."""
    G0 = Q + 2.0 * R
    obs_x = G0[0, 1]
    return {
        "rho_efficient": _corr(Q),                    # PERMANENT (fundamental) comovement  -- the F&R-robust object
        "rho_transitory": _corr(R),                   # TRANSITORY (microstructure) comovement
        "rho_observed": _corr(G0),                    # what the simple-mid return correlation measures
        "perm_share_of_cross": float(Q[0, 1] / obs_x) if obs_x != 0 else float("nan"),   # Q_12 / (Q_12 + 2 R_12)
        "noise_ratio_0": float(R[0, 0] / Q[0, 0]) if Q[0, 0] > 0 else float("nan"),      # R/Q per asset (noise:signal)
        "noise_ratio_1": float(R[1, 1] / Q[1, 1]) if Q[1, 1] > 0 else float("nan"),
        "Q_var": (float(Q[0, 0]), float(Q[1, 1])),
        "R_var": (float(R[0, 0]), float(R[1, 1])),
    }


# ───────────────────────── public API ─────────────────────────
def fit_efficient_price(prices, log=True, maxiter=500, names=("SPY", "ES")):
    """Fit the bivariate local-level model and return the efficient-price decomposition.

    prices : (T,2) array or DataFrame of POSITIVE prices (or log-prices if log=False), columns (SPY, ES).
    Returns a dict with the MLE covariances Q (permanent) and R (transitory), the smoothed efficient
    price `mu`, the transitory component `delta = y - mu`, fit diagnostics, and the comovement
    decomposition (rho_efficient vs rho_transitory vs rho_observed, perm_share_of_cross, ...).

    Internally the (centered) series is rescaled to unit return-variance for numerical conditioning;
    `mu`, `delta`, Q, R are returned in the ORIGINAL units (the correlations/shares are scale-free).
    """
    idx, cols = None, list(names)
    if isinstance(prices, pd.DataFrame):
        idx, cols = prices.index, list(prices.columns)
        Y = prices.to_numpy(float)
    else:
        Y = np.asarray(prices, float)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("prices must be (T, 2)")
    keep = np.isfinite(Y).all(axis=1)
    Y, idx = Y[keep], (idx[keep] if idx is not None else None)
    if log:
        if np.any(Y <= 0):
            raise ValueError("non-positive prices with log=True")
        Y = np.log(Y)
    T = Y.shape[0]
    if T < 50:
        raise ValueError(f"need >= 50 aligned observations, got {T}")

    off = Y.mean(axis=0)                               # center + rescale to unit return variance (conditioning)
    Yc = Y - off
    dvar = float(np.mean(np.var(np.diff(Yc, axis=0), axis=0))) + 1e-300
    scale = 1.0 / np.sqrt(dvar)
    Ys = Yc * scale

    dY = np.diff(Ys, axis=0)                           # method-of-moments init from Gamma_0, Gamma_1
    G0 = np.cov(dY.T)
    a, b = dY[1:] - dY[1:].mean(0), dY[:-1] - dY[:-1].mean(0)
    G1 = (a.T @ b) / (len(dY) - 1)
    R0 = _psd(-0.5 * (G1 + G1.T))                      # R = -Gamma_1
    Q0 = _psd(G0 - 2.0 * R0)                           # Q = Gamma_0 - 2R
    theta0 = np.concatenate([_cov_to_chol(Q0), _cov_to_chol(R0)])

    y0l, y1l = Ys[:, 0].tolist(), Ys[:, 1].tolist()
    P0 = 1e6

    def negll(theta):
        Q = _chol_to_cov(theta[:3]); R = _chol_to_cov(theta[3:])
        ll = _loglik_2d(y0l, y1l, Q, R, P0)
        return -ll if np.isfinite(ll) else 1e18

    res = minimize(negll, theta0, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7})
    Qs, Rs = _chol_to_cov(res.x[:3]), _chol_to_cov(res.x[3:])
    mu_s = _filter_smooth_2d(Ys, Qs, Rs, P0)

    mu = mu_s / scale + off                            # back to original (log-)units
    delta = (Ys - mu_s) / scale
    Q = Qs / scale ** 2
    R = Rs / scale ** 2
    out = {"Q": Q, "R": R, "loglik": float(-res.fun), "n_obs": T, "converged": bool(res.success),
           "n_iter": int(res.nit), "names": cols, "log": log}
    out.update(_decompose_comovement(Q, R))
    if idx is not None:
        out["mu"] = pd.DataFrame(mu, index=idx, columns=cols)
        out["delta"] = pd.DataFrame(delta, index=idx, columns=cols)
    else:
        out["mu"], out["delta"] = mu, delta
    return out


def run_efficient_price_analysis(sessions, spy_col="SPY_mid", es_col="ES_mid", log=True):
    """Fit the decomposition per session (each day its own random walk) and summarize BY REGIME.
    `sessions` is a list of (label, regime, frame). Returns {'per_session': DataFrame, 'by_regime':
    DataFrame} reporting rho_efficient / rho_transitory / rho_observed and the permanent share of
    cross-asset comovement -- i.e., how much of the observed correlation is fundamental vs microstructure,
    and whether the fundamental piece (rho_efficient) shifts across regimes (the F&R-robust contagion test)."""
    rows = []
    for spec in sessions:
        label, regime, df = (spec if len(spec) == 3 else (spec[0], "all", spec[-1]))
        if spy_col not in df.columns or es_col not in df.columns:
            rows.append({"session": label, "regime": regime, "error": "missing mid columns"}); continue
        try:
            fit = fit_efficient_price(df[[spy_col, es_col]], log=log)
            rows.append({"session": label, "regime": regime, "n_obs": fit["n_obs"],
                         "rho_efficient": fit["rho_efficient"], "rho_transitory": fit["rho_transitory"],
                         "rho_observed": fit["rho_observed"], "perm_share_of_cross": fit["perm_share_of_cross"],
                         "noise_ratio_spy": fit["noise_ratio_0"], "noise_ratio_es": fit["noise_ratio_1"],
                         "converged": fit["converged"]})
        except Exception as exc:
            rows.append({"session": label, "regime": regime, "error": repr(exc)})
    per = pd.DataFrame(rows)
    num = ["rho_efficient", "rho_transitory", "rho_observed", "perm_share_of_cross",
           "noise_ratio_spy", "noise_ratio_es"]
    have = [c for c in num if c in per.columns]
    by_regime = (per.dropna(subset=have).groupby("regime")[have].mean()
                 if have and "regime" in per.columns and per.dropna(subset=have).shape[0] else pd.DataFrame())
    return {"per_session": per, "by_regime": by_regime}


# ───────────────────────── self-test: simulate with known truth and RECOVER it ─────────────────────────
def _simulate(T, Q, R, seed=0, level=100.0):
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((T, 2)) @ np.linalg.cholesky(Q).T
    mu = np.cumsum(eta, axis=0) + level
    eps = rng.standard_normal((T, 2)) @ np.linalg.cholesky(R).T
    return mu + eps, mu, eps


def _selftest():
    ok = True

    # (1) RECOVERY + DISENTANGLING. Design the worry case: PERMANENT comovement LOW (rho_Q = 0.30) but
    #     TRANSITORY (microstructure) comovement HIGH (rho_R = 0.85), with noise variance > per-step
    #     signal variance (R diag > Q diag) as at 1s. The OBSERVED correlation is then badly inflated;
    #     the estimator must recover the truth and attribute most of the observed cross-cov to transitory.
    sQ, sR = 1.0, 4.0
    Qt = sQ * np.array([[1.0, 0.30], [0.30, 1.0]])
    Rt = sR * np.array([[1.0, 0.85], [0.85, 1.0]])
    Y, mu_true, eps_true = _simulate(60000, Qt, Rt, seed=1)
    fit = fit_efficient_price(Y, log=False, maxiter=600)
    Qh, Rh = fit["Q"], fit["R"]
    rho_q_err = abs(fit["rho_efficient"] - 0.30)
    rho_r_err = abs(fit["rho_transitory"] - 0.85)
    qd = max(abs(Qh[0, 0] / sQ - 1), abs(Qh[1, 1] / sQ - 1))      # diagonal relative error
    rd = max(abs(Rh[0, 0] / sR - 1), abs(Rh[1, 1] / sR - 1))
    # design values of the decomposition
    obs_true = (0.30 * sQ + 2 * 0.85 * sR) / (sQ + 2 * sR)        # = 0.789
    perm_share_true = (0.30 * sQ) / (0.30 * sQ + 2 * 0.85 * sR)   # = 0.042
    rec_ok = (rho_q_err < 0.08 and rho_r_err < 0.08 and qd < 0.25 and rd < 0.25 and fit["converged"])
    dis_ok = (abs(fit["rho_observed"] - obs_true) < 0.05 and fit["rho_efficient"] < fit["rho_observed"] - 0.2
              and abs(fit["perm_share_of_cross"] - perm_share_true) < 0.05)
    ok &= rec_ok and dis_ok
    print("(1) recover known (Q,R) from simulation + decompose a high observed corr")
    print("    truth:    rho_efficient=0.300  rho_transitory=0.850  rho_observed=%.3f  perm_share=%.3f"
          % (obs_true, perm_share_true))
    print("    estimated rho_efficient=%.3f  rho_transitory=%.3f  rho_observed=%.3f  perm_share=%.3f"
          % (fit["rho_efficient"], fit["rho_transitory"], fit["rho_observed"], fit["perm_share_of_cross"]))
    print("    Q diag rel-err=%.2f  R diag rel-err=%.2f  converged=%s" % (qd, rd, fit["converged"]))
    print("    -> recovery:%s  disentangling:%s" % (rec_ok, dis_ok))

    # (2) the smoother recovers the latent efficient price: its error variance is far below the raw
    #     observation-noise variance, and the estimated transitory tracks the true microstructure noise.
    sm_err = np.var(fit["mu"][:, 0] - mu_true[:, 0])
    raw_err = np.var(eps_true[:, 0])
    corr_delta = np.corrcoef(fit["delta"][:, 0], eps_true[:, 0])[0, 1]
    smooth_ok = (sm_err < 0.5 * raw_err and corr_delta > 0.5)
    ok &= smooth_ok
    print("\n(2) smoothed efficient price recovers the latent permanent component")
    print("    Var(muhat - mu_true)=%.3f  vs  Var(eps)=%.3f  (smoother cuts noise %.0f%%);  corr(delta,eps)=%.2f -> %s"
          % (sm_err, raw_err, 100 * (1 - sm_err / raw_err), corr_delta, smooth_ok))

    # (3) degenerate control: no microstructure noise (R -> 0) => efficient price == observed, delta ~ 0,
    #     and the model attributes ~all comovement to the permanent component.
    Y2, mu2, _ = _simulate(20000, sQ * np.array([[1.0, 0.5], [0.5, 1.0]]),
                           1e-8 * np.eye(2), seed=2)
    f2 = fit_efficient_price(Y2, log=False, maxiter=400)
    nonoise_ok = (abs(f2["perm_share_of_cross"] - 1.0) < 0.05 and np.std(f2["delta"][:, 0]) < 0.1
                  and abs(f2["rho_efficient"] - 0.5) < 0.08)
    ok &= nonoise_ok
    print("\n(3) control: R->0 (no microstructure) => perm_share=%.3f (~1)  std(delta)=%.3f (~0)  "
          "rho_efficient=%.3f (~0.5) -> %s"
          % (f2["perm_share_of_cross"], np.std(f2["delta"][:, 0]), f2["rho_efficient"], nonoise_ok))

    print("\nchecks:", ok)
    return ok


if __name__ == "__main__":
    _selftest()
