"""
ecm_sde.py
==========
Liquidity-conditional price discovery as a state-dependent error-correction SDE,
and its Euler-Maruyama estimators on the SPY-ES frames.

Model (log prices P_t = (P^SPY, P^ES)', basis z_t = beta'P_t, beta=(1,-1)):

    dP_t = alpha(S_t) (beta'P_t) dt + Sigma(S_t)^{1/2} dW_t  (+ dJ_t)

S_t is the predetermined relative-liquidity state (e.g. log depth_ES - log depth_SPY,
or the leading FPC). The price-discovery "derivative" alpha(.) and the diffusion
Sigma(.) are now FUNCTIONS of the book state -- this is the continuous-time form of
the liquidity-conditional VECM, i.e. a (smooth-transition / varying-coefficient) ECM.

Basis is a state-modulated Ornstein-Uhlenbeck process:
    dz_t = (beta'alpha(S_t)) z_t dt + (beta'Sigma(S_t)^{1/2}) dW_t,
so the price-discovery SPEED is kappa(S) = -beta'alpha(S) = alpha_ES(S) - alpha_SPY(S)
(half-life ln2/kappa), and the LEADERSHIP allocation is, with psi(S)=(alpha_ES(S),-alpha_SPY(S)),
    CS_ES(S) = |alpha_SPY(S)| / (|alpha_SPY(S)|+|alpha_ES(S)|),
    IS_j(S)  = [(psi(S)'Sigma(S)^{1/2})_j]^2 / (psi(S)'Sigma(S)psi(S)).
The thesis is the SHAPE of these: d IS_ES / dS > 0 (ES book relatively deeper -> ES leads),
steepening under stress.

Euler-Maruyama on a grid t_k = k*dt:
    r_{k+1} = alpha(S_k) z_k dt + eta_{k+1},   eta_{k+1} | S_k ~ N(0, Sigma(S_k) dt).
This is exactly a VECM with state-dependent loadings and heteroskedastic innovations.
Because the information share is a scale-invariant ratio, dt cancels and the discrete
estimate of IS(.) targets the continuous-time object directly (dt only rescales kappa).

Estimators (increasing flexibility):
  state_interacted_vecm  alpha(S)=a0+a1*S+...  -> ONE pooled regression; the gradient a1
                         (coef on z*S) is the direct, well-identified test of the thesis.
  local_coeff_vecm       kernel varying-coefficient: the full curves alpha(.),Sigma(.),IS(.).
  fit_em_mle             the Euler-Maruyama Gaussian pseudo-MLE (affine alpha, state-scaled
                         Sigma) -- the SDE estimator with a proper likelihood for inference.
Built on price_discovery_shares (reuses hasbrouck_is / gonzalo_granger / _psi).
numpy / pandas / scipy.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import price_discovery_shares as pds

EPS = 1e-12


# ── basis & small linear algebra ──────────────────────────────────────────────
def _logmid(m):
    m = np.asarray(m, float)
    return np.log(np.where(m > 0, m, np.nan))

log = logging.getLogger(__name__)


def basis(mid_spy, mid_es, demean=True):
    """z_t = log P^SPY - log P^ES (the no-arbitrage basis), demeaned within the session."""
    z = _logmid(mid_spy) - _logmid(mid_es)
    if demean: z = z - np.nanmean(z)
    return z

def _hac(X, y, L=10):
    """OLS with Newey-West HAC standard errors (z*S regressors are persistent)."""
    XtX = X.T @ X
    XtXinv = np.linalg.pinv(XtX)
    b = XtXinv @ (X.T @ y)
    u = y - X @ b
    n, k = X.shape
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, min(L, n - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += w * (G + G.T)
    cov = XtXinv @ S @ XtXinv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return b, se, u

def _cluster_se(X, u, groups, XtXinv=None):
    """One-way cluster-robust (Liang-Zeger) SEs for an OLS fit, clustering by `groups` (the session/day
    id of each row). meat = Sum_g (X_g'u_g)(X_g'u_g)', with the SAME small-cluster correction
    cross_asset_pd_liquidity.panel_regression uses, c = G/(G-1)*(N-1)/(N-K). This permits ARBITRARY
    within-day serial correlation + heteroskedasticity and treats days as independent -- the correct
    structure when non-contiguous sessions are stacked, and a superset of the pooled Newey-West HAC,
    whose Bartlett windows would otherwise straddle the overnight gaps between days. Returns (se, G).
    With ~14 sessions G is small, so reference the t-stat to t_{G-1}, not the normal."""
    X = np.asarray(X, float); u = np.asarray(u, float)
    if XtXinv is None:
        XtXinv = np.linalg.pinv(X.T @ X)
    g = np.asarray(groups); uniq = np.unique(g); G = len(uniq); N, K = X.shape
    if G < 2 or N <= K:
        # ONE cluster (a single session) has no between-cluster variation, so the Liang-Zeger
        # correction G/(G-1) is undefined. Returning NaN here propagates a NaN t-statistic into
        # whatever table the caller is building -- silently, since a NaN prints as a blank cell
        # rather than raising. That is the same silent-failure shape as a swallowed fetch error.
        # A single contiguous session is exactly the case Newey-West HAC is valid for, so fall
        # back to it and say so, instead of emitting an un-inspectable NaN.
        if N > K:
            S = (X * u[:, None]).T @ (X * u[:, None])
            L = min(10, max(1, N // 20))
            for l in range(1, min(L, N - 1) + 1):
                w = 1.0 - l / (L + 1.0)
                Gm = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
                S += w * (Gm + Gm.T)
            V = XtXinv @ S @ XtXinv
            se = np.sqrt(np.clip(np.diag(V), 0.0, None))
            log.warning("cluster-robust SE requested with G=%d cluster(s) and N=%d rows: the "
                        "G/(G-1) correction is undefined, so falling back to Newey-West HAC "
                        "(valid for one contiguous session). Day-clustered inference needs >=2 "
                        "sessions; with few clusters prefer a wild cluster bootstrap.", G, N)
            return se, G
        log.warning("cluster-robust SE undefined (G=%d, N=%d, K=%d) -- returning NaN SEs.", G, N, K)
        return np.full(K, np.nan), G
    meat = np.zeros((K, K))
    for gi in uniq:
        m = g == gi
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    c = (G / (G - 1.0)) * ((N - 1.0) / (N - K))
    V = (XtXinv @ meat @ XtXinv) * c
    return np.sqrt(np.clip(np.diag(V), 0.0, None)), G

def _psd_ridge(M, frac=1e-8):
    M = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    return M + frac * max(np.trace(M), 1.0) * np.eye(M.shape[0])


# ── shares as a function of the loading vector + local covariance ─────────────
def _shares_at(alpha, Sigma, names=("SPY", "ES"), dt=1.0):
    """Hasbrouck IS (mid), Gonzalo-Granger CS, OU speed kappa and half-life, at one state."""
    a = np.asarray(alpha, float)
    lo, hi, mid = pds.hasbrouck_is(a, _psd_ridge(Sigma))
    cs = pds.gonzalo_granger(a)
    kappa = (a[1] - a[0]) / dt                                  # alpha_ES - alpha_SPY, per second
    out = {"alpha_SPY": a[0], "alpha_ES": a[1], "kappa": kappa,
           "half_life_s": (np.log(2.0) / kappa) if kappa > EPS else np.nan}
    for j, nm in enumerate(names):
        out[f"IS_{nm}"] = mid[j]; out[f"CS_{nm}"] = cs[j]
    return out


# ── (1) state-interacted VECM: alpha(S)=a0+a1 S(+...) -- the powerful test ─────
def state_interacted_vecm(mid_spy, mid_es, S, degree=1, n_lags=0, center_state=True,
                          hac_lags=10, names=("SPY", "ES"), _z=None, _ret=None, _Sc=None,
                          _groups=None):
    """Pooled Euler-Maruyama regression r = alpha(S) z + (lags) + eta with alpha(S) a
    degree-`degree` polynomial in S. Returns a0, a1(=d alpha/dS) per equation -- the coefficient
    on z*S is the single-number test of liquidity-conditioning. SEs: Newey-West HAC by default
    (correct for ONE continuous session); pass `_groups` (a per-row session id) on the pooled-panel
    path to get DAY-CLUSTERED SEs instead, which is the correct standard error when non-contiguous
    sessions are stacked (the pooled HAC otherwise leaks across the overnight gaps)."""
    if _z is None:
        z = basis(mid_spy, mid_es); p1 = _logmid(mid_spy); p2 = _logmid(mid_es)
        r = np.column_stack([np.diff(p1), np.diff(p2)]); z = z[:-1]
        S = np.asarray(S, float)[:len(z)]
        ok = np.isfinite(z) & np.isfinite(S) & np.isfinite(r).all(1)
        z, S, r = z[ok], S[ok], r[ok]
        Sc = S - S.mean() if center_state else S
        S_meta = S
    else:
        z, r, Sc = _z, _ret, _Sc                                # pre-stacked (panel path)
        S_meta = _Sc
    cols = [z * Sc ** d for d in range(degree + 1)]             # z, z*S, z*S^2, ...
    labels = [f"z*S^{d}" if d else "z" for d in range(degree + 1)]
    if n_lags:                                                  # optional short-run terms
        dp = r
        for L in range(1, n_lags + 1):
            cols.append(np.r_[np.zeros(L), dp[:-L, 0]]); cols.append(np.r_[np.zeros(L), dp[:-L, 1]])
    X = np.column_stack([np.ones(len(z))] + cols)               # const + EC poly (+ lags)
    fit = {"degree": degree, "labels": labels, "state_mean": float(np.mean(S_meta)),
           "n": len(z), "names": names}
    B = np.zeros((2, X.shape[1])); resid = np.zeros_like(r)
    XtXinv = np.linalg.pinv(X.T @ X)
    G = None
    for j in range(2):
        b, se_hac, u = _hac(X, r[:, j], L=hac_lags); B[j] = b; resid[:, j] = u
        if _groups is not None:
            se, G = _cluster_se(X, u, _groups, XtXinv)         # day-clustered (stacked-panel path)
        else:
            se = se_hac                                        # HAC (single continuous session)
        fit[f"a0_{names[j]}"] = b[1]                            # alpha_j at the mean state
        fit[f"t_a0_{names[j]}"] = b[1] / (se[1] + EPS)
        if degree >= 1:
            fit[f"a1_{names[j]}"] = b[2]                        # d alpha_j / dS  (the thesis)
            fit[f"se_a1_{names[j]}"] = se[2]
            fit[f"t_a1_{names[j]}"] = b[2] / (se[2] + EPS)
    fit["se_kind"] = "cluster" if _groups is not None else "HAC"
    if G is not None:
        fit["n_clusters"] = int(G)
    fit["_B"] = B; fit["_resid"] = resid; fit["_Sigma_const"] = np.cov(resid, rowvar=False, bias=True)
    fit["_S"] = S_meta; fit["_state_centered"] = center_state
    return fit

def _alpha_of_s(fit, s):
    """Evaluate alpha(s) = sum_d a_d * (s - Sbar)^d from a fitted interacted model."""
    sc = s - fit["state_mean"] if fit["_state_centered"] else s
    B = fit["_B"]; deg = fit["degree"]
    return np.array([sum(B[j, 1 + d] * sc ** d for d in range(deg + 1)) for j in range(2)])

def is_curve(fit, grid=None, local_sigma=True, bandwidth=None, dt=1.0):
    """Evaluate the state-varying shares IS(s), CS(s), kappa(s) over a grid of states.
    Sigma(s) is a kernel-local residual covariance (local_sigma) or the constant Sigma."""
    S = fit["_S"]; resid = fit["_resid"]; names = fit["names"]
    if grid is None: grid = np.quantile(S, np.linspace(0.05, 0.95, 19))
    h = bandwidth or (1.5 * np.std(S) * len(S) ** (-0.2))
    rows = []
    for s in grid:
        a = _alpha_of_s(fit, s)
        if local_sigma:
            w = np.exp(-0.5 * ((S - s) / h) ** 2); w = w / (w.sum() + EPS)
            rm = resid - (w[:, None] * resid).sum(0)
            Sig = (w[:, None] * rm).T @ rm
        else:
            Sig = fit["_Sigma_const"]
        rows.append({"S": float(s), **_shares_at(a, Sig, names=names, dt=dt)})
    return pd.DataFrame(rows)


# ── (2) local (kernel) varying-coefficient VECM: the full curves ──────────────
def local_coeff_vecm(mid_spy, mid_es, S, grid=None, bandwidth=None, dt=1.0,
                     n_boot=0, names=("SPY", "ES"), seed=0):
    """Nonparametric alpha(.) and Sigma(.): at each s, kernel-weighted WLS of r on z gives
    alpha_hat(s); the local residual covariance gives Sigma_hat(s); then IS(s), CS(s),
    kappa(s). Optional moving-block bootstrap bands on IS_ES(s)."""
    z = basis(mid_spy, mid_es); p1 = _logmid(mid_spy); p2 = _logmid(mid_es)
    r = np.column_stack([np.diff(p1), np.diff(p2)]); z = z[:-1]; S = np.asarray(S, float)[:len(z)]
    ok = np.isfinite(z) & np.isfinite(S) & np.isfinite(r).all(1); z, S, r = z[ok], S[ok], r[ok]
    if grid is None: grid = np.quantile(S, np.linspace(0.05, 0.95, 19))
    h = bandwidth or (1.06 * np.std(S) * len(S) ** (-0.2))

    def fit_at(s, zz, SS, rr):
        w = np.exp(-0.5 * ((SS - s) / h) ** 2)
        Xw = np.column_stack([np.ones_like(zz), zz]) * np.sqrt(w)[:, None]
        a = np.zeros(2); res = np.zeros_like(rr)
        for j in range(2):
            b, *_ = np.linalg.lstsq(Xw, rr[:, j] * np.sqrt(w), rcond=None)
            a[j] = b[1]; res[:, j] = rr[:, j] - (b[0] + b[1] * zz)
        wn = w / (w.sum() + EPS); rm = res - (wn[:, None] * res).sum(0)
        Sig = (wn[:, None] * rm).T @ rm
        return a, Sig

    rows = []
    for s in grid:
        a, Sig = fit_at(s, z, S, r)
        rows.append({"S": float(s), **_shares_at(a, Sig, names=names, dt=dt)})
    out = pd.DataFrame(rows)

    if n_boot:
        rng = np.random.default_rng(seed); n = len(z); bl = max(10, int(n ** (1 / 3)))
        BS = np.full((n_boot, len(grid)), np.nan)
        for b in range(n_boot):
            idx = []
            while len(idx) < n:
                st = rng.integers(0, n - bl); idx.extend(range(st, st + bl))
            idx = np.array(idx[:n])
            for gi, s in enumerate(grid):
                try:
                    a, Sig = fit_at(s, z[idx], S[idx], r[idx])
                    BS[b, gi] = _shares_at(a, Sig, names=names)[f"IS_{names[1]}"]
                except Exception:
                    pass
        out[f"IS_{names[1]}_lo"] = np.nanpercentile(BS, 2.5, axis=0)
        out[f"IS_{names[1]}_hi"] = np.nanpercentile(BS, 97.5, axis=0)
    return out


# ── (3) Euler-Maruyama Gaussian pseudo-MLE (affine alpha, state-scaled Sigma) ──
def _unpack(theta):
    # alpha_SPY(S)=aS0+aS1 S ; alpha_ES(S)=aE0+aE1 S ; Sigma(S)=D(S) C D(S),
    # log-vols linear in S, fixed correlation rho=tanh(g).
    aS0, aS1, aE0, aE1, lv1, sv1, lv2, sv2, g = theta
    return (np.array([aS0, aE0]), np.array([aS1, aE1]),
            np.array([lv1, lv2]), np.array([sv1, sv2]), np.tanh(g))

def _sigma_of_s(lv, sv, rho, s):
    sd = np.exp(lv + sv * s)
    return np.array([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]])

def em_negloglik(theta, r, z, S, dt=1.0):
    a0, a1, lv, sv, rho = _unpack(theta)
    if abs(rho) >= 0.999: return 1e12
    sd = np.exp(lv[None, :] + np.outer(S, sv)) * np.sqrt(dt)    # n x 2 local sds
    mu = (a0[None, :] + np.outer(S, a1)) * z[:, None] * dt      # drift increments
    e = r - mu
    s1, s2 = sd[:, 0], sd[:, 1]
    det = (s1 * s2) ** 2 * (1 - rho ** 2)
    if np.any(det <= 0) or not np.all(np.isfinite(det)): return 1e12
    q = (e[:, 0] ** 2 / s1 ** 2 - 2 * rho * e[:, 0] * e[:, 1] / (s1 * s2)
         + e[:, 1] ** 2 / s2 ** 2) / (1 - rho ** 2)
    return float(0.5 * np.sum(np.log((2 * np.pi) ** 2 * det) + q))

def fit_em_mle(mid_spy, mid_es, S, dt=1.0, center_state=True, names=("SPY", "ES")):
    """Maximize the Euler-Maruyama Gaussian pseudo-likelihood. Returns affine loadings
    alpha(S)=a0+a1 S, the state-dependent diffusion, and the implied IS at the mean state.
    QMLE: consistent as dt->0, span->infinity; O(dt) discretization bias at fixed dt is
    negligible here (dt << the seconds-to-minutes mean-reversion scale)."""
    z = basis(mid_spy, mid_es); p1 = _logmid(mid_spy); p2 = _logmid(mid_es)
    r = np.column_stack([np.diff(p1), np.diff(p2)]); z = z[:-1]; S = np.asarray(S, float)[:len(z)]
    ok = np.isfinite(z) & np.isfinite(S) & np.isfinite(r).all(1); z, S, r = z[ok], S[ok], r[ok]
    Sc = S - S.mean() if center_state else S
    # warm start from OLS interacted fit + residual vols
    fit0 = state_interacted_vecm(None, None, None, degree=1, center_state=False,
                                 _z=z, _ret=r, _Sc=Sc, names=names)
    sd0 = np.log(np.sqrt(np.clip(np.diag(fit0["_Sigma_const"]), 1e-12, None)) / np.sqrt(dt))
    th0 = np.array([fit0[f"a0_{names[0]}"] / dt, fit0.get(f"a1_{names[0]}", 0.0) / dt,
                    fit0[f"a0_{names[1]}"] / dt, fit0.get(f"a1_{names[1]}", 0.0) / dt,
                    sd0[0], 0.0, sd0[1], 0.0, 0.0])
    res = minimize(em_negloglik, th0, args=(r, z, Sc, dt), method="Nelder-Mead",
                   options={"maxiter": 8000, "xatol": 1e-7, "fatol": 1e-7})
    a0, a1, lv, sv, rho = _unpack(res.x)
    out = {"converged": bool(res.success), "negloglik": float(res.fun),
           "a0_SPY": a0[0], "a1_SPY": a1[0], "a0_ES": a0[1], "a1_ES": a1[1],
           "rho_mean": float(rho), "state_mean": float(S.mean())}
    out["shares_at_mean"] = _shares_at(a0, _sigma_of_s(lv, sv, rho, 0.0), names=names, dt=dt)
    out["_theta"] = res.x
    return out


# ── panel: pool sessions (basis demeaned within day, state centered globally) ─
def _daycluster_boot_a1(z, r, Sc, groups, degree, n_lags, names, n_boot, seed):
    """Day-cluster (pairs) bootstrap of the gradient a1: resample whole sessions with replacement,
    refit, collect a1. a1 (the z*S slope) is invariant to re-centering S, so resampled rows reuse the
    global Sc. A cross-check on the analytic cluster SE; like it, few-cluster-limited at ~14 days."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); idx_by = {g: np.where(groups == g)[0] for g in uniq}
    draws = {nm: [] for nm in names}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by[g] for g in pick])
        f = state_interacted_vecm(None, None, None, degree=degree, n_lags=n_lags, center_state=False,
                                  _z=z[sel], _ret=r[sel], _Sc=Sc[sel], names=names)
        for nm in names:
            draws[nm].append(f.get(f"a1_{nm}", np.nan))
    out = {"n_boot": int(n_boot)}
    for nm in names:
        d = np.asarray(draws[nm], float)
        out[f"a1_{nm}_se"] = float(np.nanstd(d, ddof=1))
        out[f"a1_{nm}_lo"] = float(np.nanpercentile(d, 2.5))
        out[f"a1_{nm}_hi"] = float(np.nanpercentile(d, 97.5))
    return out


def estimate_sample(sessions, state_fn, degree=1, n_lags=0, hac_lags=20, names=("SPY", "ES"),
                    price_fn=None, n_boot=0, seed=0):
    """sessions: iterable of (label, regime, df). state_fn(df) -> the S series (e.g.
    ca.relative_depth_state or fl.relative_fpc_state). price_fn(df, asset) -> the level price
    (default = top-of-book mid; pass robust_prices.price_fn for a noise-robust observable). Pools the
    Euler-Maruyama interacted regression across days (basis demeaned WITHIN each day) and returns the
    loading gradient a1 with DAY-CLUSTERED t-stats: the cluster is the session, so the SE allows
    arbitrary within-day correlation and treats days as independent. (A single-session HAC, applied to
    the stacked series, would run its Bartlett windows across the overnight gaps -- the wrong
    dependence and order-dependent; the cluster SE fixes both.) The t is referenced to t_{G-1}; with
    ~14 sessions inference is few-cluster-limited, so pass n_boot>0 for a day-cluster (pairs) bootstrap
    cross-check, and treat a wild-cluster bootstrap as the belt-and-braces refinement. NOTE: keep
    n_lags=0 here (the driver default) -- lagged-difference columns built on the stacked series would
    themselves cross session boundaries; if lags are needed they must be built within-session."""
    from cross_asset_pd_liquidity import _mid as camid
    if price_fn is None:
        price_fn = camid
    Zs, Rs, Ss, Gs = [], [], [], []
    for gi, (label, regime, df) in enumerate(sessions):
        ps = price_fn(df, "SPY"); pe = price_fn(df, "ES")
        z = basis(ps, pe)
        p1 = _logmid(ps); p2 = _logmid(pe)
        r = np.column_stack([np.diff(p1), np.diff(p2)]); z = z[:-1]
        S = np.asarray(state_fn(df), float)[:len(z)]
        ok = np.isfinite(z) & np.isfinite(S) & np.isfinite(r).all(1)
        Zs.append(z[ok]); Rs.append(r[ok]); Ss.append(S[ok]); Gs.append(np.full(int(ok.sum()), gi))
    z = np.concatenate(Zs); r = np.vstack(Rs); S = np.concatenate(Ss); groups = np.concatenate(Gs)
    Sc = S - S.mean()
    fit = state_interacted_vecm(None, None, None, degree=degree, n_lags=n_lags,
                                center_state=False, hac_lags=hac_lags, _z=z, _ret=r, _Sc=Sc,
                                _groups=groups, names=names)
    fit["state_mean"] = float(S.mean()); fit["_S"] = S
    G = fit.get("n_clusters", len(np.unique(groups)))
    if degree >= 1 and G and G > 1:                            # few-cluster t-test (df = G-1)
        from scipy import stats as _stats
        for nm in names:
            fit[f"p_a1_{nm}"] = float(2.0 * _stats.t.sf(abs(fit[f"t_a1_{nm}"]), df=G - 1))
    if n_boot and degree >= 1:
        fit["bootstrap"] = _daycluster_boot_a1(z, r, Sc, groups, degree, n_lags, names, n_boot, seed)
    fit["curve"] = is_curve(fit, local_sigma=True)
    return fit


# ── self-test: known state-dependent DGP (loadings steepen with S) ────────────
def _selftest():
    rng = np.random.default_rng(0)
    n = 80000
    # S: exogenous AR(1), standardized & demeaned (mean~0, bulk in +/-2.5).
    S = np.zeros(n)
    for k in range(1, n): S[k] = 0.995 * S[k - 1] + rng.normal(0, 0.07)
    S = (S - S.mean()) / (S.std() + 1e-12); S = np.clip(S, -3, 3)
    # loadings: alpha_SPY(S) = -(g0 + g1 S) stays NEGATIVE over the whole range (valid EC
    # everywhere); magnitude rises with S so SPY adjusts more -> ES leads more. alpha_ES const.
    g0, g1, e0 = 0.060, 0.020, 0.004                            # truth: d alpha_SPY/dS = -g1
    Sig = np.array([[6e-4, 2e-4], [2e-4, 6e-4]]); Ltr = np.linalg.cholesky(Sig); sig_f = 4e-4
    p1 = np.zeros(n); p2 = np.zeros(n)
    for k in range(1, n):
        zk = p1[k - 1] - p2[k - 1]; df = rng.normal(0, sig_f)
        aS = -(g0 + g1 * S[k - 1]); aE = e0
        e = Ltr @ rng.normal(0, 1, 2)
        p1[k] = p1[k - 1] + df + aS * zk + e[0]
        p2[k] = p2[k - 1] + df + aE * zk + e[1]
    msp, mes = np.exp(p1), np.exp(p2)

    # (1) interacted estimator recovers the gradient (clean truth: a1_SPY = -g1, a1_ES = 0)
    fit = state_interacted_vecm(msp, mes, S, degree=1)
    print("(1) state-interacted loadings   [truth: d alpha_SPY/dS = -%.3f, d alpha_ES/dS = 0]" % g1)
    print("    alpha_SPY: a0(@mean)=%.4f (t=%.1f)   a1=d/dS=%.4f (t=%.1f)"
          % (fit["a0_SPY"], fit["t_a0_SPY"], fit["a1_SPY"], fit["t_a1_SPY"]))
    print("    alpha_ES : a0(@mean)=%.4f (t=%.1f)   a1=d/dS=%.4f (t=%.1f)"
          % (fit["a0_ES"], fit["t_a0_ES"], fit["a1_ES"], fit["t_a1_ES"]))
    sign_ok = (fit["a1_SPY"] < 0 and abs(fit["t_a1_SPY"]) > 3
               and abs(fit["a1_SPY"] + g1) < 0.01 and abs(fit["a1_ES"]) < abs(fit["a1_SPY"]))

    # (2) implied IS(S)/CS(S) curve is increasing in S; system mean-reverting throughout
    cur = is_curve(fit, grid=np.linspace(-2, 2, 9), local_sigma=True)
    mono = np.corrcoef(cur.S, cur.IS_ES)[0, 1]; cs_mono = np.corrcoef(cur.S, cur.CS_ES)[0, 1]
    print("\n(2) IS_ES(S) curve")
    print(cur[["S", "alpha_SPY", "alpha_ES", "CS_ES", "IS_ES", "kappa", "half_life_s"]].round(4).to_string(index=False))
    print("    corr(S, IS_ES)=%.3f  corr(S, CS_ES)=%.3f  kappa>0 all=%s"
          % (mono, cs_mono, bool((cur.kappa > 0).all())))

    # (3) consistency: IS at the mean state ~ the global pds information share
    ga, gO, _ = pds._fit_vecm_fixed(np.log(msp), np.log(mes), 5)
    glob = pds.hasbrouck_is(ga, gO)[2][1]
    at_mean = _shares_at(_alpha_of_s(fit, S.mean()), fit["_Sigma_const"])["IS_ES"]
    print("\n(3) consistency: IS_ES at mean state=%.3f  vs global pds IS_mid_ES=%.3f" % (at_mean, glob))

    # (4) local (kernel) curve recovers the increasing trend (noisier: data-hungry)
    loc = local_coeff_vecm(msp, mes, S, grid=np.linspace(-2, 2, 9), bandwidth=0.4)
    loc_mono = np.corrcoef(loc.S, loc.IS_ES)[0, 1]; agree = np.corrcoef(cur.IS_ES, loc.IS_ES)[0, 1]
    print("\n(4) local varying-coefficient: corr(S,IS_ES)=%.3f  corr(local,interacted)=%.3f" % (loc_mono, agree))

    # (5) Euler-Maruyama pseudo-MLE recovers the affine loadings
    mle = fit_em_mle(msp, mes, S)
    print("\n(5) Euler-Maruyama pseudo-MLE")
    print("    alpha_SPY: a0=%.4f a1=%.4f | alpha_ES a0=%.4f a1=%.4f | rho=%.2f | conv=%s"
          % (mle["a0_SPY"], mle["a1_SPY"], mle["a0_ES"], mle["a1_ES"], mle["rho_mean"], mle["converged"]))
    mle_ok = mle["a1_SPY"] < 0 and abs(mle["a1_SPY"]) > abs(mle["a1_ES"])

    kappa_mono = bool(np.all(np.diff(cur.kappa) > 0))
    print("\nchecks:", sign_ok, kappa_mono, cs_mono > 0.85, mono > 0.7,
          abs(at_mean - glob) < 0.05, loc_mono > 0.5, mle_ok)

if __name__ == "__main__":
    _selftest()
