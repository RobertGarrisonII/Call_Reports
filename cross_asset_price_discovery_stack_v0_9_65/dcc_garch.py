"""
dcc_garch.py
============
DCC-GARCH-X estimation in pure numpy/scipy (no arch/statsmodels). Models the
time-varying conditional correlation between SPY and ES so "comovement is strong
in volatile periods" becomes an estimated rho_t object rather than a coarse
volatile/benchmark split. The "-X" enters the univariate variance equations:

  GARCH(1,1)-X:  h_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1} + gamma'*X_{t-1}
  DCC(1,1):      Q_t = (1-a-b)*Qbar + a*z_{t-1}z_{t-1}' + b*Q_{t-1},
                 R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

X can be realized vol, |OFI|, the relative-depth state, etc. (predetermined).
Two-step Engle (2002) estimator: fit each marginal, standardize, fit DCC.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize

EPS = 1e-10


# ── univariate GARCH(1,1)-X ──────────────────────────────────────────────────
def _garch_filter(eps, omega, alpha, beta, gamma, X):
    """GARCH(1,1)-X conditional-variance recursion, driven off Python floats.

    Sequential, so it cannot be vectorised -- but the loop body must not touch numpy. Indexing a
    numpy array element-wise boxes a fresh Python float per access, and this is the single hottest
    function in the stack: 65% of a `mean_variance` run, called ~2,200 times per fit by the
    L-BFGS numerical gradients. The boxing is also what made the per-session process pool scale so
    poorly (per-fit time 18.3s alone -> 57.7s with four workers): the cost is allocator and memory
    traffic, a resource the workers share, not arithmetic, which they do not. Squaring is hoisted
    out as one vectorised op and the rest runs on lists, so the arithmetic and its ordering are
    unchanged -- bit-identical output, and it speeds up the serial path as well as the pooled one."""
    eps = np.asarray(eps, float)
    T = len(eps)
    v = np.var(eps)
    h0 = float(v + EPS) if (np.isfinite(v) and v > 0) else float(EPS)
    if T <= 1:
        return np.full(max(T, 0), h0)
    e2 = (eps * eps).tolist()                                # one vectorised square, then plain floats
    xg = [0.0] * T if X is None else (X @ np.atleast_1d(gamma)).tolist()
    om = float(omega); al = float(alpha); be = float(beta); eps_f = float(EPS)
    out = [h0] * T
    prev = h0
    for t in range(1, T):
        cur = om + al * e2[t - 1] + be * prev + xg[t - 1]
        if not (cur > eps_f):                                # NaN-safe floor (nan > EPS is False)
            cur = eps_f
        out[t] = cur
        prev = cur
    return np.asarray(out)

def garch_x_fit(eps, X=None):
    """Fit GARCH(1,1)-X by Gaussian MLE. eps: mean-zero series. X: T x p or None.
    Returns params dict, conditional variance h, standardized resid z.

    The series is STANDARDIZED to unit variance internally and the parameters mapped back
    (omega, h, gamma scale by var; alpha, beta, z are scale-free; loglik gets the Jacobian
    term). Every production caller feeds RAW log-mid returns whose variance is ~4e-9 at 1s
    and ~4e-11 at 10ms, and against that scale the omega lower bound (1e-9) and the
    _garch_filter variance floor (EPS=1e-10) -- both absolute constants sized for unit-ish
    data -- were BINDING: at 1s the fitted omega pinned to its bound and persistence
    collapsed; at 10ms h was 100% floored, alpha=beta=0, and the 'standardized' residuals
    kept all their volatility clustering, contaminating every DCC rho_t downstream. In the
    scaled domain the constants are what they were always meant to be: negligible."""
    eps = np.asarray(eps, float)
    Xm = None if X is None else (np.asarray(X, float).reshape(-1, 1)
                                 if np.asarray(X).ndim == 1 else np.asarray(X, float))
    fin = np.isfinite(eps) & (np.all(np.isfinite(Xm), axis=1) if Xm is not None else True)
    eps = eps[fin]; Xm = None if Xm is None else Xm[fin]
    v_raw = np.var(eps)
    s = float(np.sqrt(v_raw)) if (np.isfinite(v_raw) and v_raw > 0) else 1.0
    eps = eps / s
    v = np.var(eps)                                          # ~1 by construction
    p = 0 if Xm is None else Xm.shape[1]

    def nll(theta):
        omega, alpha, beta = theta[0], theta[1], theta[2]
        gamma = theta[3:] if p else 0.0
        h = _garch_filter(eps, omega, alpha, beta, gamma, Xm)
        if not np.all(np.isfinite(h)) or np.any(h <= 0):
            return 1e10
        val = 0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + eps ** 2 / h)
        return val if np.isfinite(val) else 1e10

    x0 = [0.1 * v, 0.05, 0.90] + [0.0] * p
    bounds = [(1e-9, 10 * v + 1e-6), (0.0, 0.5), (0.0, 0.999)] + [(-10 * v, 10 * v)] * p
    best = None
    for b0 in (0.90, 0.80, 0.50):
        x0[2] = b0
        r = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    gamma = th[3:] if p else np.array([])
    h = _garch_filter(eps, th[0], th[1], th[2], (gamma if p else 0.0), Xm)
    s2 = s * s
    return {"omega": th[0] * s2, "alpha": th[1], "beta": th[2],
            "gamma": gamma * s2 if p else gamma,
            "loglik": -best.fun - len(eps) * np.log(s),      # raw-scale Gaussian loglik
            "h": h * s2, "z": eps / np.sqrt(h)}


# ── DCC(1,1) second stage ────────────────────────────────────────────────────
def _dcc_corr_path(Z, a, b):
    """Bivariate DCC correlation path r_t, as three running scalars instead of 2x2 arrays.

    The Q recursion is inherently sequential, so the loop cannot be vectorised -- but at k=2 the
    state is only (q11, q12, q22), and carrying it as Python floats removes ~8 numpy dispatches
    per bar. That dispatch overhead, not the algebra, is what made a DCC fit take ~25 s at
    T=1300 and left mean_variance.py unable to finish: `_dcc_filter` is called afresh inside
    EVERY likelihood evaluation, so the per-bar constant is multiplied by several hundred
    thousand. Same recursion, same numbers."""
    Z = np.asarray(Z, float)
    T = Z.shape[0]
    z0, z1 = Z[:, 0], Z[:, 1]
    b11 = float(np.dot(z0, z0) / T); b12 = float(np.dot(z0, z1) / T); b22 = float(np.dot(z1, z1) / T)
    om = 1.0 - a - b
    c11, c12, c22 = om * b11, om * b12, om * b22
    q11, q12, q22 = b11, b12, b22
    r = np.empty(T)
    for t in range(T):
        d1 = q11 if q11 > EPS else EPS
        d2 = q22 if q22 > EPS else EPS
        rt = q12 / math.sqrt(d1 * d2)
        r[t] = -1.0 if rt < -1.0 else (1.0 if rt > 1.0 else rt)
        if t < T - 1:
            x, y = z0[t], z1[t]
            q11 = c11 + a * x * x + b * q11
            q12 = c12 + a * x * y + b * q12
            q22 = c22 + a * y * y + b * q22
    return r


def _dcc_filter(Z, a, b):
    T, k = Z.shape
    if k == 2:                                   # fast path: build the (T,2,2) view from r_t
        r = _dcc_corr_path(Z, a, b)
        R = np.empty((T, 2, 2))
        R[:, 0, 0] = R[:, 1, 1] = 1.0
        R[:, 0, 1] = R[:, 1, 0] = r
        return R
    Qbar = (Z.T @ Z) / T
    Q = Qbar.copy(); R = np.empty((T, k, k))
    for t in range(T):
        d = np.sqrt(np.maximum(np.diag(Q), EPS))
        Rt = np.clip(Q / np.outer(d, d), -1.0, 1.0)
        np.fill_diagonal(Rt, 1.0); R[t] = Rt
        if t < T - 1:
            zz = np.outer(Z[t], Z[t])
            Q = (1 - a - b) * Qbar + a * zz + b * Q
    return R

def dcc_fit(Z):
    """Engle DCC(1,1) on standardized residuals Z (T x k). Returns (a,b) and R_t."""
    T, k = Z.shape

    def nll(theta):
        a, b = theta
        if a < 0 or b < 0 or a + b >= 0.9999:
            return 1e10
        if k == 2:
            # Closed form for the bivariate case, which is the whole point of this module
            # (SPY vs ES). For R_t = [[1, r], [r, 1]]:
            #     log|R_t|   = log(1 - r^2)
            #     z' R_t^-1 z = (z1^2 - 2 r z1 z2 + z2^2) / (1 - r^2)
            # The general path called slogdet AND solve on a 2x2 at EVERY t -- two LAPACK
            # dispatches per bar, ~2600 per likelihood evaluation at T=1300, several hundred
            # thousand per fit. That, not the model, is why a DCC fit took ~25 s and why
            # mean_variance.py could not finish. This is algebraically exact, not an
            # approximation: it returns the same number, vectorised over t.
            r = np.clip(_dcc_corr_path(Z, a, b), -0.999999, 0.999999)
            om = 1.0 - r * r
            z1, z2 = Z[:, 0], Z[:, 1]
            quad = (z1 * z1 - 2.0 * r * z1 * z2 + z2 * z2) / om
            ll = float(np.sum(np.log(om) + quad - (z1 * z1 + z2 * z2)))
            return 0.5 * ll if np.isfinite(ll) else 1e10
        R = _dcc_filter(Z, a, b)
        ll = 0.0
        for t in range(T):
            Rt = R[t]
            sign, logdet = np.linalg.slogdet(Rt)
            if not (sign > 0) or not np.isfinite(logdet):     # NaN-safe (nan<=0 is False)
                return 1e10
            try:
                q = Z[t] @ np.linalg.solve(Rt, Z[t])
            except np.linalg.LinAlgError:
                return 1e10
            ll += logdet + q - Z[t] @ Z[t]
        return 0.5 * ll if np.isfinite(ll) else 1e10

    best = None
    for a0, b0 in ((0.02, 0.96), (0.05, 0.90), (0.10, 0.80)):
        r = minimize(nll, [a0, b0], method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 500})
        if best is None or r.fun < best.fun:
            best = r
    a, b = best.x
    if (not np.isfinite(best.fun)) or a < 0 or b < 0 or a + b >= 1.0:
        a, b = 0.0, 0.0                                       # fall back to constant (Qbar) correlation
    return {"a": a, "b": b, "R": _dcc_filter(Z, a, b), "loglik": -best.fun}


# ── Aielli (2013) corrected cDCC ──────────────────────────────────────────────
# Engle's DCC targets Qbar = E[z z'] but the recursion's invariant is NOT Qbar, so correlation
# targeting via Z'Z/T is INCONSISTENT (Aielli 2013, JBES). The cDCC fix rescales the news term by
# the conditional scale implied by Q_t -- z*_t = diag(Q_t)^{1/2} z_t -- so that E[z*_t z*_t']=S
# holds exactly and S is consistently estimable. This is the variant a tier-1 referee expects for
# a DCC dependence model; it also supplies the correlation path the copula layer conditions on.

def _cdcc_path2(Z, a, b, S):
    """Bivariate cDCC recursion carried as three running scalars (q11, q12, q22).

    This is the hot loop of the whole module -- `dcc_method='cdcc'` is the DEFAULT, the filter is
    re-run inside every likelihood evaluation AND inside every targeting iteration, and the
    generic version spends its time in numpy dispatch: profiling one `dcc_garch_x` fit showed
    1.3M `np.outer` calls and 643k each of `np.linalg.solve` and `slogdet`, all on 2x2 matrices.
    Same recursion, same numbers, no per-bar dispatch. Returns (r_t, Zstar)."""
    T = Z.shape[0]
    z0, z1 = Z[:, 0], Z[:, 1]
    s11, s12, s22 = float(S[0, 0]), float(S[0, 1]), float(S[1, 1])
    om = 1.0 - a - b
    c11, c12, c22 = om * s11, om * s12, om * s22
    q11, q12, q22 = s11, s12, s22
    r = np.empty(T); zs0 = np.empty(T); zs1 = np.empty(T)
    for t in range(T):
        d1 = math.sqrt(q11 if q11 > EPS else EPS)
        d2 = math.sqrt(q22 if q22 > EPS else EPS)
        rt = q12 / (d1 * d2)
        r[t] = -1.0 if rt < -1.0 else (1.0 if rt > 1.0 else rt)
        x = d1 * z0[t]; y = d2 * z1[t]                          # z*_t = diag(Q_t)^{1/2} (.) z_t
        zs0[t] = x; zs1[t] = y
        if t < T - 1:
            q11 = c11 + a * x * x + b * q11
            q12 = c12 + a * x * y + b * q12
            q22 = c22 + a * y * y + b * q22
    return r, np.column_stack([zs0, zs1])


def _cdcc_filter(Z, a, b, S):
    """cDCC recursion Q_t=(1-a-b)S + a z*_{t-1}z*_{t-1}' + b Q_{t-1}, z*_t=diag(Q_t)^{1/2} z_t.
    Returns the correlation path R_t and the rescaled residuals z*_t (for consistent targeting)."""
    T, k = Z.shape
    if k == 2:
        r, Zstar = _cdcc_path2(Z, a, b, S)
        R = np.empty((T, 2, 2))
        R[:, 0, 0] = R[:, 1, 1] = 1.0
        R[:, 0, 1] = R[:, 1, 0] = r
        return R, Zstar
    Q = S.copy(); R = np.empty((T, k, k)); Zstar = np.empty_like(Z)
    for t in range(T):
        d = np.sqrt(np.maximum(np.diag(Q), EPS))
        Rt = np.clip(Q / np.outer(d, d), -1.0, 1.0); np.fill_diagonal(Rt, 1.0); R[t] = Rt
        zstar = d * Z[t]                                       # diag(Q_t)^{1/2} ⊙ z_t
        Zstar[t] = zstar
        if t < T - 1:
            Q = (1 - a - b) * S + a * np.outer(zstar, zstar) + b * Q
    return R, Zstar


def _cdcc_target(Z, a, b, iters=8):
    """Aielli's consistent correlation-targeting estimator of S: the fixed point of
    S = (1/T) Σ z*_t z*_t' (z*_t depends on Q_t depends on S), iterated to convergence."""
    T = len(Z); S = (Z.T @ Z) / T
    for _ in range(iters):
        _, Zstar = _cdcc_filter(Z, a, b, S)
        S = (Zstar.T @ Zstar) / T
    return S


def _cdcc_nll(Z, a, b, S):
    if Z.shape[1] == 2:                       # closed form: log|R|=log(1-r^2), quad as below
        r = np.clip(_cdcc_path2(Z, a, b, S)[0], -0.999999, 0.999999)
        om = 1.0 - r * r
        z1, z2 = Z[:, 0], Z[:, 1]
        ll = float(np.sum(np.log(om) + (z1 * z1 - 2.0 * r * z1 * z2 + z2 * z2) / om
                          - (z1 * z1 + z2 * z2)))
        return 0.5 * ll if np.isfinite(ll) else 1e10
    R, _ = _cdcc_filter(Z, a, b, S); ll = 0.0
    for t in range(len(Z)):
        Rt = R[t]; sign, logdet = np.linalg.slogdet(Rt)
        if not (sign > 0) or not np.isfinite(logdet):
            return 1e10
        try:
            q = Z[t] @ np.linalg.solve(Rt, Z[t])
        except np.linalg.LinAlgError:
            return 1e10
        ll += logdet + q - Z[t] @ Z[t]
    return 0.5 * ll if np.isfinite(ll) else 1e10


def cdcc_fit(Z, target_iters=8):
    """Aielli (2013) cDCC(1,1) on standardized residuals Z (T x k). Consistent two-step: target S
    by the fixed-point estimator, MLE (a,b) with S held, then re-target at the fitted (a,b) and
    refit once. Returns (a, b), the consistent targeting matrix S, the correlation path R_t, loglik."""
    S = _cdcc_target(Z, 0.04, 0.92, iters=target_iters)        # pilot targeting

    def fit_ab(S_):
        def nll(theta):
            a, b = theta
            if a < 0 or b < 0 or a + b >= 0.9999:
                return 1e10
            return _cdcc_nll(Z, a, b, S_)
        best = None
        for a0, b0 in ((0.03, 0.95), (0.06, 0.90), (0.10, 0.80)):
            r = minimize(nll, [a0, b0], method="Nelder-Mead",
                         options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 400})
            if best is None or r.fun < best.fun:
                best = r
        return best

    best = fit_ab(S); a, b = best.x
    if np.isfinite(best.fun) and a >= 0 and b >= 0 and a + b < 1.0:
        S = _cdcc_target(Z, a, b, iters=target_iters)          # refine targeting at the fitted (a,b)
        best = fit_ab(S); a, b = best.x
    if (not np.isfinite(best.fun)) or a < 0 or b < 0 or a + b >= 1.0:
        a, b = 0.0, 0.0
    R, _ = _cdcc_filter(Z, a, b, S)
    return {"a": a, "b": b, "S": S, "R": R, "loglik": -best.fun}


# ── orchestrator ─────────────────────────────────────────────────────────────
def dcc_garch_x(returns, X=None, names=("SPY", "ES"), X_per_asset=None, dcc_method="cdcc"):
    """returns: T x 2 (SPY, ES). X: optional exogenous variance regressors shared by both
    marginals (1d/2d) or None. X_per_asset: optional list of one regressor block per column
    (so each asset's variance loads on its OWN liquidity); overrides X when given. ``dcc_method``
    selects the second-stage correlation recursion: ``"cdcc"`` (default, Aielli 2013 corrected,
    consistent targeting) or ``"engle"`` (the original DCC). Returns time-varying correlation
    rho_t, conditional vols, and stage parameters. Non-finite rows are dropped jointly (so the
    standardized residuals stay aligned); with too little clean data it returns NaNs rather than
    raising."""
    R = np.asarray(returns, float)
    k = R.shape[1]

    def _prep(x):
        if x is None:
            return None
        x = np.asarray(x, float)
        return x.reshape(-1, 1) if x.ndim == 1 else x

    if X_per_asset is not None:
        Xs = [_prep(x) for x in X_per_asset]
        if len(Xs) != k:
            raise ValueError("X_per_asset must have one entry per return column")
    else:
        Xs = [_prep(X)] * k
    fin = np.all(np.isfinite(R), axis=1)
    for x in Xs:
        if x is not None:
            fin = fin & np.all(np.isfinite(x), axis=1)
    R = R[fin]
    Xs = [None if x is None else x[fin] for x in Xs]
    if R.shape[0] < 50:                                      # not enough clean data to fit
        T0 = np.asarray(returns).shape[0]
        return {"rho": np.full(T0, np.nan), "cond_vol": np.full((T0, 2), np.nan),
                "marginals": {}, "dcc_a": np.nan, "dcc_b": np.nan, "R": None,
                "note": "insufficient finite data"}
    eps = R - R.mean(0)
    marg = [garch_x_fit(eps[:, j], Xs[j]) for j in range(R.shape[1])]
    Z = np.column_stack([m["z"] for m in marg])
    dcc = (cdcc_fit if dcc_method == "cdcc" else dcc_fit)(Z)
    rho = dcc["R"][:, 0, 1]
    return {"rho": rho, "cond_vol": np.column_stack([np.sqrt(m["h"]) for m in marg]),
            "marginals": dict(zip(names, marg)), "dcc_a": dcc["a"], "dcc_b": dcc["b"],
            "dcc_method": dcc_method, "R": dcc["R"]}


def _selftest():
    rng = np.random.default_rng(11)
    n = 6000
    # 1) DGP: true DCC correlation path + GARCH vols
    ta, tb = 0.04, 0.93; Qbar = np.array([[1.0, 0.5], [0.5, 1.0]])
    Q = Qbar.copy(); z = np.empty((n, 2)); rho_true = np.empty(n)
    for t in range(n):
        d = np.sqrt(np.diag(Q)); Rt = Q / np.outer(d, d); rho_true[t] = Rt[0, 1]
        z[t] = np.linalg.cholesky(Rt) @ rng.normal(size=2)
        Q = (1 - ta - tb) * Qbar + ta * np.outer(z[t], z[t]) + tb * Q
    eps = np.empty((n, 2)); h = np.full(2, 1.0)
    e = np.empty((n, 2))
    for j in range(2):
        hh = 1.0
        for t in range(n):
            if t > 0:
                hh = 0.02 + 0.08 * e[t - 1, j] ** 2 + 0.90 * hh
            e[t, j] = np.sqrt(hh) * z[t, j]
    out = dcc_garch_x(e)
    corr_path = np.corrcoef(out["rho"][50:], rho_true[50:])[0, 1]
    print("DCC-GARCH fit:")
    print("  dcc a=%.3f b=%.3f (true 0.04/0.93), a+b=%.3f" % (out["dcc_a"], out["dcc_b"], out["dcc_a"] + out["dcc_b"]))
    print("  GARCH SPY: alpha=%.3f beta=%.3f (true 0.08/0.90)"
          % (out["marginals"]["SPY"]["alpha"], out["marginals"]["SPY"]["beta"]))
    print("  corr(estimated rho_t, true rho_t) = %.3f" % corr_path)
    print("  checks: a+b<1:", out["dcc_a"] + out["dcc_b"] < 1,
          "| rho tracks truth:", corr_path > 0.6)

    # 2) GARCH-X: exogenous covariate genuinely raising variance -> gamma>0
    x = np.abs(rng.normal(0, 1, n))                      # predetermined vol covariate
    eg = np.empty(n); hh = 1.0
    for t in range(n):
        if t > 0:
            hh = 0.02 + 0.05 * eg[t - 1] ** 2 + 0.85 * hh + 0.30 * x[t - 1]
        eg[t] = np.sqrt(hh) * rng.normal()
    fx = garch_x_fit(eg, X=x)
    print("\nGARCH-X covariate recovery:")
    print("  gamma=%.3f (true 0.30), sign correct & positive:" % fx["gamma"][0], fx["gamma"][0] > 0.1)

    # 3) non-finite robustness: NaN/inf return segments must not poison the DCC fit.
    rr = rng.normal(0, 0.01, (1500, 2)); rr[300:320, 0] = np.nan; rr[900, 1] = np.inf
    rn = dcc_garch_x(rr)
    print("\nnon-finite robustness:")
    print("  dcc a+b<1 under NaN/inf returns:", bool(rn["dcc_a"] + rn["dcc_b"] < 1.0),
          "| mean rho finite:", bool(np.isfinite(np.nanmean(rn["rho"]))))

if __name__ == "__main__":
    _selftest()
