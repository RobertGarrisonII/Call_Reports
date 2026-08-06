"""
copula_garch.py — copula dependence for SPY/ES, with tail dependence and a liquidity-conditional hook.

WHY (vs the Gaussian DCC in mean_variance / dcc_garch). A Gaussian DCC summarizes dependence by a
time-varying *linear* correlation rho_t whose tail-dependence coefficient is **zero** for any rho<1:
it asserts that far enough into the tail SPY and ES become independent — backwards for a crash. A copula
keeps the GARCH(-X) margins but replaces the dependence with a function that can carry **tail dependence**
and **asymmetry**, which is exactly the contagion dimension a tariff gap-down lives in.

WHAT THIS PROVIDES (in increasing sophistication):
  1. ``garch_margins``        — GARCH(1,1)-X margins (reuses dcc_garch.garch_x_fit) -> standardized
                                residuals -> rank PIT (pseudo-observations; robust two-step / pseudo-ML).
  2. ``select_copula``        — constant copulas {gaussian, t, clayton, gumbel, bb1, sjc} by BIC, each with its
                                lower/upper tail-dependence coefficients. BB1 (the two-parameter Clayton-Gumbel)
                                carries BOTH tails with an analytic density and nests Clayton at delta=1, giving
                                a clean nested LR test for upper-tail dependence beyond the lower tail; SJC is
                                the symmetrized-Joe-Clayton alternative, kept mainly for the dynamic extension.
                                lower/upper tail-dependence coefficients (lambda_L, lambda_U).
  3. ``t_copula_dcc``         — the *dynamic* baseline: DCC correlation recursion (rho_t) evaluated under a
                                t-copula, so you get time-varying rho_t AND symmetric joint tail dependence
                                lambda_t = lambda(rho_t, nu); reported against the Gaussian-DCC rho_t delta.
  4. ``liquidity_conditional_dependence`` — the headline: tail dependence as a function of the book state.
                                Fits the selected copula in thin- vs deep-book windows (and block-wise over
                                time) and reports whether lower-tail (joint-crash) dependence **rises as the
                                books thin** — a more direct liquidity-*contagion* statement than liquidity->vol.

SIGN / TAIL CONVENTION. lambda_L is *lower*-tail (joint crash) dependence, lambda_U upper-tail. Clayton has
lambda_L>0, lambda_U=0 (crash-only); Gumbel the reverse; t is symmetric; SJC estimates both. A contagion
signature is lambda_L large and rising in stress / as liquidity thins, with lambda_L > lambda_U.

ESTIMATION NOTES. Margins and dependence are fit in two stages (IFM); PIT uses empirical ranks
(pseudo-observations), which is robust to margin tail mis-specification and is standard for copula GARCH.
``t_copula_dcc`` is a two-step approximation (nu from the constant t-copula, then the DCC recursion on the
t-scores) — robust and fast; a full one-step joint MLE or a score-driven (GAS) time-varying *asymmetric*
copula (Creal-Koopman-Lucas 2013; Patton 2006 SJC dynamics) is the more sophisticated extension and is
noted where it would slot in. Everything is numpy/scipy only.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

import dcc_garch as dg

log = logging.getLogger("copula_garch")
_EPS = 1e-10
_CLIP = 1e-5                                              # clamp uniforms away from {0,1}


# ════════════════════════════════════════════════════════════════════════════
# Margins -> pseudo-observations
# ════════════════════════════════════════════════════════════════════════════
def pseudo_obs(Z: np.ndarray) -> np.ndarray:
    """Rank PIT (pseudo-observations) of an array of standardized residuals, column-wise:
    u_it = rank(z_it) / (T+1). Robust to margin tail mis-specification (Genest et al.)."""
    Z = np.asarray(Z, float)
    ranks = np.argsort(np.argsort(Z, axis=0), axis=0) + 1.0
    return ranks / (Z.shape[0] + 1.0)


def garch_margins(returns: np.ndarray, X=None, X_per_asset=None):
    """Fit GARCH(1,1)-X margins to each return series (reusing dcc_garch.garch_x_fit) and return
    ``(U, Z, marg)``: U the rank-PIT uniforms (T x k), Z the standardized residuals, marg the per-asset
    GARCH parameter dicts. Pass ``X``/``X_per_asset`` (e.g. an AUC_decay liquidity covariate) for
    GARCH-X margins exactly as in dcc_garch."""
    R = np.asarray(returns, float)
    R = R[np.isfinite(R).all(axis=1)]
    eps = R - R.mean(axis=0, keepdims=True)
    k = R.shape[1]
    Xs = X_per_asset if X_per_asset is not None else [X] * k
    marg = [dg.garch_x_fit(eps[:, j], Xs[j]) for j in range(k)]
    Z = np.column_stack([m["z"] for m in marg])
    U = pseudo_obs(Z)
    return U, Z, marg


# ════════════════════════════════════════════════════════════════════════════
# Bivariate copula densities, tail dependence, fit, simulate
# ════════════════════════════════════════════════════════════════════════════
def _clip(u):
    return np.clip(u, _CLIP, 1.0 - _CLIP)


# ---- Gaussian ----
def _gauss_logpdf(u, v, rho):
    x = stats.norm.ppf(_clip(u)); y = stats.norm.ppf(_clip(v))
    rho = np.clip(rho, -0.999, 0.999)
    return -0.5 * np.log(1 - rho**2) - (rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho**2))


# ---- Student-t ----
def _t_logpdf(u, v, rho, nu):
    rho = np.clip(rho, -0.999, 0.999); nu = max(nu, 2.05)
    x = stats.t.ppf(_clip(u), nu); y = stats.t.ppf(_clip(v), nu)
    from scipy.special import gammaln
    c0 = gammaln((nu + 2) / 2) + gammaln(nu / 2) - 2 * gammaln((nu + 1) / 2) - 0.5 * np.log(1 - rho**2)
    quad = (x**2 - 2 * rho * x * y + y**2) / (nu * (1 - rho**2))
    return (c0 - (nu + 2) / 2 * np.log1p(quad)
            + (nu + 1) / 2 * (np.log1p(x**2 / nu) + np.log1p(y**2 / nu)))


def _t_tail(rho, nu):
    lam = 2.0 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
    return float(lam), float(lam)


# ---- Clayton (lower tail) ----
def _clayton_logpdf(u, v, th):
    u = _clip(u); v = _clip(v); th = max(th, 1e-3)
    s = u**(-th) + v**(-th) - 1.0
    return np.log1p(th) - (1 + th) * (np.log(u) + np.log(v)) - (2 + 1 / th) * np.log(s)


# ---- Gumbel (upper tail) ----
def _gumbel_logpdf(u, v, th):
    u = _clip(u); v = _clip(v); th = max(th, 1.0 + 1e-6)
    a = -np.log(u); b = -np.log(v)
    sab = a**th + b**th
    A = sab**(1 / th)
    return (-A - np.log(u) - np.log(v) + (th - 1) * (np.log(a) + np.log(b))
            + (1 / th - 2) * np.log(sab) + np.log(A + th - 1))


# ---- BB1 (Joe two-parameter "Clayton-Gumbel"): BOTH tails, analytic density ----
# C(u,v) = [1 + ((u^-th - 1)^d + (v^-th - 1)^d)^(1/d)]^(-1/th), th>0, d>=1.
# Nests Clayton at d=1 (lambda_U->0) and approaches Gumbel as th->0.
# lambda_L = 2^(-1/(th*d)), lambda_U = 2 - 2^(1/d).  Density verified to reduce to Clayton at d=1.
def _bb1_parts(u, v, th, d):
    x = np.maximum(u**(-th) - 1.0, 1e-300)
    y = np.maximum(v**(-th) - 1.0, 1e-300)
    w = x**d + y**d
    z = w**(1.0 / d)
    return x, y, w, z


def _bb1_logpdf(u, v, th, d):
    u = _clip(u); v = _clip(v); th = max(th, 1e-3); d = max(d, 1.0)
    x, y, w, z = _bb1_parts(u, v, th, d)
    bracket = (1.0 + th * d) * z + th * (d - 1.0)
    return (-(th + 1.0) * (np.log(u) + np.log(v))
            + (d - 1.0) * (np.log(x) + np.log(y))
            + (-1.0 / th - 2.0) * np.log1p(z)
            + ((1.0 - 2.0 * d) / d) * np.log(w)
            + np.log(np.maximum(bracket, 1e-300)))


def _bb1_tail(th, d):
    return float(2.0**(-1.0 / (th * d))), float(2.0 - 2.0**(1.0 / d))


def _bb1_hfunc(v, u, th, d):
    """Conditional CDF C_{2|1}(v|u) = dC/du, monotone 0->1 in v (for inverse-CDF sampling)."""
    x, y, w, z = _bb1_parts(u, v, th, d)
    return u**(-th - 1.0) * x**(d - 1.0) * (1.0 + z)**(-1.0 / th - 1.0) * w**((1.0 - d) / d)


# ---- SJC (symmetrized Joe-Clayton): estimable lower AND upper tail ----
def _jc_cdf(u, v, lU, lL):
    lU = min(max(lU, 1e-4), 1 - 1e-4); lL = min(max(lL, 1e-4), 1 - 1e-4)
    kap = 1.0 / np.log2(2 - lU); gam = -1.0 / np.log2(lL)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        a = (1 - (1 - u)**kap)
        b = (1 - (1 - v)**kap)
        inner = (a**(-gam) + b**(-gam) - 1.0)**(-1.0 / gam)
        return 1.0 - (1.0 - inner)**(1.0 / kap)


def _sjc_cdf(u, v, lU, lL):
    return 0.5 * (_jc_cdf(u, v, lU, lL) + _jc_cdf(1 - u, 1 - v, lL, lU) + u + v - 1.0)


def _sjc_logpdf(u, v, lU, lL, h=1e-4):
    """SJC density via the central mixed second difference of the cdf (gradient-free MLE-friendly).
    Clamped away from the boundary; floored positive."""
    u = _clip(u); v = _clip(v)
    up = np.minimum(u + h, 1 - _CLIP); um = np.maximum(u - h, _CLIP)
    vp = np.minimum(v + h, 1 - _CLIP); vm = np.maximum(v - h, _CLIP)
    c = (_sjc_cdf(up, vp, lU, lL) - _sjc_cdf(up, vm, lU, lL)
         - _sjc_cdf(um, vp, lU, lL) + _sjc_cdf(um, vm, lU, lL)) / ((up - um) * (vp - vm))
    return np.log(np.maximum(c, 1e-300))


# ---- family registry: logpdf, init/bounds, tail, simulate ----
def _fit_copula(name, u, v):
    """MLE-fit one constant copula; returns dict(name, params, loglik, k, aic, bic, lambda_L, lambda_U)."""
    u = np.asarray(u, float); v = np.asarray(v, float); T = len(u)

    if name == "gaussian":
        nll = lambda p: -np.sum(_gauss_logpdf(u, v, p[0]))
        r = optimize.minimize_scalar(lambda r: nll([r]), bounds=(-0.999, 0.999), method="bounded")
        ll, k, params, (lL, lU) = -r.fun, 1, {"rho": float(r.x)}, (0.0, 0.0)
    elif name == "t":
        x0 = [float(np.corrcoef(stats.norm.ppf(_clip(u)), stats.norm.ppf(_clip(v)))[0, 1]), 8.0]
        res = optimize.minimize(lambda p: -np.sum(_t_logpdf(u, v, p[0], p[1])), x0,
                                method="L-BFGS-B", bounds=[(-0.999, 0.999), (2.1, 60.0)])
        ll, k, params = -res.fun, 2, {"rho": float(res.x[0]), "nu": float(res.x[1])}
        lL, lU = _t_tail(res.x[0], res.x[1])
    elif name == "clayton":
        res = optimize.minimize_scalar(lambda t: -np.sum(_clayton_logpdf(u, v, t)),
                                       bounds=(1e-2, 30.0), method="bounded")
        th = float(res.x); ll, k, params = -res.fun, 1, {"theta": th}
        lL, lU = 2.0**(-1.0 / th), 0.0
    elif name == "gumbel":
        res = optimize.minimize_scalar(lambda t: -np.sum(_gumbel_logpdf(u, v, t)),
                                       bounds=(1.001, 30.0), method="bounded")
        th = float(res.x); ll, k, params = -res.fun, 1, {"theta": th}
        lL, lU = 0.0, 2.0 - 2.0**(1.0 / th)
    elif name == "bb1":
        th0 = optimize.minimize_scalar(lambda t: -np.sum(_clayton_logpdf(u, v, t)),
                                       bounds=(1e-2, 30.0), method="bounded").x
        best = None                                       # multi-start over delta (boundary at 1)
        for d0 in (1.05, 1.4, 2.0):
            try:
                r = optimize.minimize(lambda p: -np.sum(_bb1_logpdf(u, v, p[0], p[1])),
                                      [max(th0, 0.05), d0], method="L-BFGS-B",
                                      bounds=[(1e-2, 30.0), (1.0, 30.0)])
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                continue
        th, d = float(best.x[0]), float(max(best.x[1], 1.0))
        ll, k, params, (lL, lU) = -best.fun, 2, {"theta": th, "delta": d}, _bb1_tail(th, d)
    elif name == "sjc":
        res = optimize.minimize(lambda p: -np.sum(_sjc_logpdf(u, v, p[0], p[1])), [0.3, 0.3],
                                method="Nelder-Mead",
                                options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 600})
        lU_, lL_ = float(np.clip(res.x[0], 1e-3, 0.999)), float(np.clip(res.x[1], 1e-3, 0.999))
        ll, k, params, (lL, lU) = -res.fun, 2, {"lambda_U": lU_, "lambda_L": lL_}, (lL_, lU_)
    else:
        raise ValueError(f"unknown copula {name!r}")

    return {"name": name, "params": params, "loglik": float(ll), "k": k,
            "aic": float(-2 * ll + 2 * k), "bic": float(-2 * ll + k * np.log(T)),
            "lambda_L": float(lL), "lambda_U": float(lU)}


_FAMILIES = ("gaussian", "t", "clayton", "gumbel", "bb1", "sjc")


def select_copula(U_or_Z, families=_FAMILIES, is_uniform=False) -> dict:
    """Fit each constant copula by MLE and rank by BIC. Input is either rank-PIT uniforms
    (``is_uniform=True``) or standardized residuals (rank PIT applied here). Returns
    ``{"table": DataFrame, "best": name, "fits": {name: dict}}`` with tail-dependence coefficients."""
    U = np.asarray(U_or_Z, float)
    if not is_uniform:
        U = pseudo_obs(U)
    u, v = U[:, 0], U[:, 1]
    fits = {}
    for name in families:
        try:
            fits[name] = _fit_copula(name, u, v)
        except Exception as exc:
            log.warning("copula %s failed: %s", name, exc)
    rows = [{"copula": f["name"], "loglik": f["loglik"], "k": f["k"], "aic": f["aic"],
             "bic": f["bic"], "lambda_L": f["lambda_L"], "lambda_U": f["lambda_U"]}
            for f in fits.values()]
    table = pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
    best = table["copula"].iloc[0] if len(table) else None
    out = {"table": table, "best": best, "fits": fits}
    if "bb1" in fits and "clayton" in fits:                # nested: Clayton = BB1 at delta=1
        lr = max(2.0 * (fits["bb1"]["loglik"] - fits["clayton"]["loglik"]), 0.0)
        # delta=1 is on the boundary -> null is the 1/2 chi2_0 + 1/2 chi2_1 mixture (Self-Liang 1987)
        p = 0.5 * (1.0 - stats.chi2.cdf(lr, 1)) if lr > 0 else 1.0
        out["nested_lr"] = {"stat": float(lr), "df": 1, "p_boundary": float(p),
                            "note": "BB1 nests Clayton at delta=1 (boundary); p uses the "
                                    "1/2 chi2_0 + 1/2 chi2_1 mixture. Small p => upper-tail "
                                    "dependence beyond the lower tail."}
    return out


# ---- simulators (for the self-test / scenario work) ----
def simulate_copula(name, n, params, rng=None):
    rng = rng or np.random.default_rng(0)
    if name == "gaussian":
        rho = params["rho"]
        Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
        return stats.norm.cdf(Z)
    if name == "t":
        rho, nu = params["rho"], params["nu"]
        Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
        W = rng.chisquare(nu, size=n) / nu
        Tt = Z / np.sqrt(W)[:, None]
        return stats.t.cdf(Tt, nu)
    if name == "clayton":                                 # Marshall-Olkin gamma frailty (lower tail)
        th = params["theta"]
        g = rng.gamma(1.0 / th, 1.0, size=n)
        e = rng.exponential(1.0, size=(n, 2))
        return (1.0 + e / g[:, None])**(-1.0 / th)
    if name == "bb1":                                     # inverse conditional CDF (Rosenblatt)
        th, d = params["theta"], params["delta"]
        u = rng.uniform(size=n); t = rng.uniform(size=n); v = np.empty(n)
        for i in range(n):
            f = lambda vv: _bb1_hfunc(vv, u[i], th, d) - t[i]
            try:
                v[i] = optimize.brentq(f, 1e-9, 1 - 1e-9, maxiter=100, xtol=1e-10)
            except Exception:
                v[i] = t[i]
        return np.column_stack([u, v])
    raise ValueError(f"no simulator for {name!r}")


# ════════════════════════════════════════════════════════════════════════════
# Dynamic: t-copula with DCC-driven correlation
# ════════════════════════════════════════════════════════════════════════════
def t_copula_dcc(Z: np.ndarray, nu: float | None = None) -> dict:
    """Time-varying t-copula: estimate nu from the constant t-copula, transform the rank-PIT to
    t-scores, run the DCC(1,1) correlation recursion (dcc_garch.dcc_fit) to get rho_t, and map to a
    time-varying symmetric tail dependence lambda_t = lambda(rho_t, nu). Returns rho_t, lambda_t, nu,
    the DCC (a,b), and the Gaussian-DCC rho_t baseline for the delta. Two-step (nu then DCC); a full
    joint MLE or a score-driven copula is the heavier alternative."""
    Z = np.asarray(Z, float)
    U = pseudo_obs(Z)
    if nu is None:
        nu = select_copula(U, families=("t",), is_uniform=True)["fits"]["t"]["params"]["nu"]
    # t-scores for the dependence recursion, and Gaussian z-scores for the linear-DCC baseline
    Xt = stats.t.ppf(_clip(U), nu)
    Xg = stats.norm.ppf(_clip(U))
    (a, b), R_t = (lambda f: ((f["a"], f["b"]), f["R"]))(dg.dcc_fit(Xt / Xt.std(axis=0, keepdims=True)))
    Rg = dg.dcc_fit(Xg)["R"]
    rho_t = R_t[:, 0, 1]
    rho_g = Rg[:, 0, 1]
    lam_t = np.array([_t_tail(np.clip(r, -0.999, 0.999), nu)[0] for r in rho_t])
    return {"rho_t": rho_t, "lambda_t": lam_t, "nu": float(nu), "dcc_ab": (float(a), float(b)),
            "rho_t_gaussian": rho_g, "rho_mean": float(np.nanmean(rho_t)),
            "lambda_mean": float(np.nanmean(lam_t)), "rho_mean_gaussian": float(np.nanmean(rho_g))}


# ════════════════════════════════════════════════════════════════════════════
# Liquidity-conditional tail dependence (the headline)
# ════════════════════════════════════════════════════════════════════════════
def liquidity_conditional_dependence(Z: np.ndarray, liq_state: np.ndarray, day_id=None, tod_s=None,
                                     copula: str = "t", thin_is_high: bool = True,
                                     state_level: str = "day", tod_trim_min: float = 15.0,
                                     session_seconds: float = 23400.0, n_blocks: int = 8) -> dict:
    """Does joint tail dependence rise with thinness/stress? Controls for the two confounds that wreck a
    naive per-second split: (1) **open/close seasonality** — trims the first/last ``tod_trim_min`` minutes
    (where the cash auction and continuous futures desync); (2) **per-second vs regime** — with
    ``state_level='day'`` it splits by DAY (stress-vs-calm days), aggregating ``liq_state`` to a per-day
    mean and cutting at the across-day median, rather than mixing intraday and cross-day variation.
    Fits the SELECTED ``copula`` (not hard-wired Clayton) in the thin/stress vs deep/calm bucket and
    reports lower-tail dependence by bucket + the thin-minus-deep delta, plus the across-day correlation
    of per-day lambda_L with the day-mean state (n_days points, cleaner than arbitrary blocks). Pass
    ``day_id``/``tod_s`` from the pooled extract; without them it falls back to a per-observation median
    split with no seasonal control. ``thin_is_high``: higher state = thinner/worse liquidity (or higher
    stress). A POSITIVE delta/correlation is the liquidity-contagion direction; negative is tail
    *decoupling* under stress."""
    Z = np.asarray(Z, float); s = np.asarray(liq_state, float); n = len(Z)
    day = np.asarray(day_id)[-n:] if day_id is not None else None
    tod = np.asarray(tod_s, float)[-n:] if tod_s is not None else None
    fam = "t" if copula == "gaussian" else copula        # gaussian lambda_L==0 (uninformative); use t
    m = np.isfinite(Z).all(axis=1) & np.isfinite(s)
    if tod is not None and tod_trim_min > 0:             # drop open/close microstructure
        lo = tod_trim_min * 60.0
        m &= (tod >= lo) & (tod <= session_seconds - lo)
    Z, s = Z[m], s[m]
    day = day[m] if day is not None else None
    U = pseudo_obs(Z)
    out = {"copula": fam, "tod_trim_min": tod_trim_min,
           "fell_back_from_gaussian": bool(copula == "gaussian")}

    if state_level == "day" and day is not None:         # stress-vs-calm DAYS
        days = np.unique(day)
        dmean = {int(d): float(np.nanmean(s[day == d])) for d in days}
        cut = float(np.nanmedian(list(dmean.values())))
        thin_days = {d for d, v in dmean.items() if (v > cut if thin_is_high else v < cut)}
        thin = np.array([int(d) in thin_days for d in day])
        out.update({"state_level": "day", "n_days": int(len(days)), "n_thin_days": int(len(thin_days))})
    else:                                                # per-observation median split
        cut = float(np.nanmedian(s))
        thin = s > cut if thin_is_high else s < cut
        out["state_level"] = "obs"

    def _fit(mask):
        if int(mask.sum()) < 50:
            return np.nan, np.nan
        f = _fit_copula(fam, U[mask, 0], U[mask, 1])
        return f["lambda_L"], f["lambda_U"]
    (lLt, lUt), (lLd, lUd) = _fit(thin), _fit(~thin)
    out.update({"n_thin": int(thin.sum()), "n_deep": int((~thin).sum()),
                "lambda_L_thin": lLt, "lambda_L_deep": lLd, "lambda_L_delta": lLt - lLd,
                "lambda_U_thin": lUt, "lambda_U_deep": lUd})

    # across-day (or block) correlation of per-unit lambda_L with the unit's mean state
    if day is not None:
        groups = [(int(d), np.where(day == d)[0]) for d in np.unique(day)]
    else:
        groups = list(enumerate(np.array_split(np.arange(len(s)), n_blocks)))
    pd_lam, pd_liq, pd_ids = [], [], []
    for d, b in groups:
        if len(b) >= 50:
            f = _fit_copula(fam, U[b, 0], U[b, 1])
            pd_lam.append(f["lambda_L"]); pd_liq.append(float(np.nanmean(s[b]))); pd_ids.append(d)
    corr = (float(np.corrcoef(pd_liq, pd_lam)[0, 1])
            if len(pd_lam) >= 3 and np.ptp(pd_liq) > _EPS else np.nan)
    out["day_corr_lambdaL_liq"] = corr
    out["block_corr_lambdaL_liq"] = corr                 # back-compat alias
    out["per_day_lambda_L"] = pd_lam; out["per_day_liq"] = pd_liq; out["per_day_ids"] = pd_ids
    return out


# ════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════
def _pooled_log_returns(sessions, names=("SPY", "ES")):
    """Within-session log returns of each asset's mid, concatenated across sessions (no overnight gaps)."""
    cols = [f"{a}_mid" for a in names]
    chunks = []
    for spec in sessions:
        df = spec[2] if len(spec) >= 3 else spec[1]
        if all(c in df.columns for c in cols):
            p = df[cols].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.diff(np.log(p), axis=0)
            chunks.append(r[np.isfinite(r).all(axis=1)])
    return np.vstack(chunks) if chunks else np.empty((0, len(names)))


def _pooled_returns_and_liq(sessions, names=("SPY", "ES"), liq_asset="SPY", n_levels=10):
    """Pool log returns, a SPY liquidity state (AUC_decay), and per-row metadata across sessions with
    matched within-session alignment (liquidity at the END of each return interval). Returns
    ``(R, liq, day_id, tod_s)``: day_id an integer session index, tod_s seconds since the 09:30 open —
    so the copula liquidity split can aggregate to the day level and trim open/close seasonality."""
    import liquidity_curve_metrics as lcm
    cols = [f"{a}_mid" for a in names]
    rch, lch, dch, tch = [], [], [], []
    for k, spec in enumerate(sessions):
        df = spec[2] if len(spec) >= 3 else spec[1]
        if not all(c in df.columns for c in cols):
            continue
        p = df[cols].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(p), axis=0)
        good = np.isfinite(r).all(axis=1)
        try:
            auc = np.asarray(lcm.decay_weighted_cost(df, liq_asset, n_levels), float)[1:]  # align to diff
        except Exception:
            auc = np.full(r.shape[0], np.nan)
        try:                                              # seconds since session open, end-of-interval
            tod = (df.index[1:] - df.index[0]).total_seconds().to_numpy(float)
        except Exception:
            tod = np.arange(1, r.shape[0] + 1, dtype=float)
        rch.append(r[good]); lch.append(auc[good])
        dch.append(np.full(int(good.sum()), k)); tch.append(tod[good])
    if not rch:
        return np.empty((0, len(names))), np.empty(0), np.empty(0, int), np.empty(0)
    return (np.vstack(rch), np.concatenate(lch),
            np.concatenate(dch).astype(int), np.concatenate(tch))


def run_copula_analysis(returns=None, sessions=None, liq_state=None, names=("SPY", "ES"),
                        X=None, out_dir: str | None = None, write: bool = False) -> dict:
    """End-to-end: GARCH-X margins -> rank PIT -> constant-copula selection (with tail dependence) ->
    t-copula-DCC dynamic baseline (vs Gaussian-DCC rho) -> optional liquidity-conditional tail
    dependence. Pass ``returns`` (T x k) or ``sessions`` ([(label, regime, df)] with mid columns).
    Returns a dict of results; writes ``copula_garch_summary.md`` and tables if ``write``."""
    day_id = tod_s = None
    if returns is None:
        if sessions is None:
            raise ValueError("pass returns or sessions")
        if isinstance(liq_state, str) and liq_state == "auto":
            returns, liq_state, day_id, tod_s = _pooled_returns_and_liq(sessions, names)
        else:
            returns = _pooled_log_returns(sessions, names)
    elif isinstance(liq_state, str) and liq_state == "auto":
        liq_state = None                                  # 'auto' needs sessions; ignore on raw returns
    if len(returns) < 100:
        raise ValueError(f"need >=100 joint return obs, got {len(returns)}")

    U, Z, marg = garch_margins(returns, X=X)
    sel = select_copula(U, is_uniform=True)
    dyn = t_copula_dcc(Z)
    fam = sel["best"]                                     # split on the SELECTED family, not hard-wired
    liq = vol = None
    if liq_state is not None:                             # (1) split by liquidity (AUC_decay), day-level
        s = np.asarray(liq_state, float)[-len(Z):]
        if np.isfinite(s).sum() >= 100:
            liq = liquidity_conditional_dependence(Z, s, day_id=day_id, tod_s=tod_s, copula=fam)
    if day_id is not None:                                # (2) clean stress proxy: per-day realized vol
        dd = np.asarray(day_id)[-len(Z):]
        r0 = np.asarray(returns, float)[-len(Z):, 0]
        dvol = {int(d): float(np.nanstd(r0[dd == d])) for d in np.unique(dd)}
        vol_state = np.array([dvol[int(d)] for d in dd])  # high vol = stress = "thin" bucket
        vol = liquidity_conditional_dependence(Z, vol_state, day_id=day_id, tod_s=tod_s,
                                               copula=fam, thin_is_high=True)

    res = {"selection": sel, "dynamic": dyn, "liquidity_conditional": liq,
           "volatility_conditional": vol, "n_obs": int(len(Z)), "names": names}
    if write:
        out_dir = out_dir or os.environ.get("OUT_DIR", "/mnt/user-data/outputs")
        os.makedirs(out_dir, exist_ok=True)
        sel["table"].to_csv(os.path.join(out_dir, "copula_selection.csv"), index=False)
        with open(os.path.join(out_dir, "copula_garch_summary.md"), "w") as fh:
            fh.write(_format_md(res))
    return res


def _format_md(res) -> str:
    sel, dyn, liq = res["selection"], res["dynamic"], res["liquidity_conditional"]
    L = ["# Copula-GARCH dependence (SPY/ES)\n",
         "GARCH(1,1)-X margins, rank PIT. Tail dependence: lambda_L lower (joint crash), lambda_U upper.\n",
         "## Constant-copula selection (by BIC)\n",
         "| copula | loglik | k | bic | lambda_L | lambda_U |", "|---|---|---|---|---|---|"]
    for _, r in sel["table"].iterrows():
        L.append("| %s | %.1f | %d | %.1f | %.3f | %.3f |"
                 % (r["copula"], r["loglik"], r["k"], r["bic"], r["lambda_L"], r["lambda_U"]))
    br = sel["table"].set_index("copula").loc[sel["best"]]
    blL, blU = float(br["lambda_L"]), float(br["lambda_U"])
    if sel["best"] == "gaussian":
        tail_msg = "zero tail dependence — no joint-extreme clustering beyond linear correlation"
    elif blL > 0 and blU == 0:
        tail_msg = "lower-tail-only (joint crashes cluster, melt-ups do not) — a joint-crash signature"
    elif blU > 0 and blL == 0:
        tail_msg = "upper-tail-only (joint melt-ups cluster, crashes do not)"
    elif abs(blL - blU) < 0.02:
        tail_msg = "symmetric (joint crashes and melt-ups cluster about equally)"
    elif blL > blU:
        tail_msg = "both tails, lower > upper (joint crashes cluster more than melt-ups)"
    else:
        tail_msg = "both tails, upper >= lower (joint melt-ups cluster at least as much as crashes)"
    L.append("\n**Best by BIC: `%s`** (lambda_L=%.3f, lambda_U=%.3f) — %s. Gaussian forces "
             "lambda_L=lambda_U=0, so the BIC gap to it is the tail content the linear model misses.\n"
             % (sel["best"], blL, blU, tail_msg))
    if "nested_lr" in sel:
        nl = sel["nested_lr"]
        verdict = ("upper-tail dependence beyond the lower tail IS warranted"
                   if nl["p_boundary"] < 0.05 else
                   "no upper-tail dependence beyond the lower tail")
        L.append("**Nested LR (BB1 vs Clayton; delta=1 boundary): Lambda=%.2f, p=%.3f** — %s. "
                 "BB1 carries both tails (lambda_L=2^(-1/(theta*delta)), lambda_U=2-2^(1/delta)) and "
                 "nests Clayton at delta=1, so this isolates whether joint melt-ups co-occur beyond the "
                 "joint crashes.\n" % (nl["stat"], nl["p_boundary"], verdict))
    L.append("## Dynamic: t-copula-DCC vs Gaussian-DCC\n")
    L.append("- t-copula nu: %.2f  (finite nu => non-zero tail dependence; Gaussian DCC is nu->inf)\n"
             % dyn["nu"])
    L.append("- mean rho_t (t-copula-DCC): %.3f ; mean rho_t (Gaussian DCC): %.3f\n"
             % (dyn["rho_mean"], dyn["rho_mean_gaussian"]))
    L.append("- mean lambda_t (time-varying lower=upper tail dependence): %.3f — the joint-crash "
             "probability the linear correlation cannot see.\n" % dyn["lambda_mean"])
    vol = res.get("volatility_conditional")

    def _emit_cond(title, c, state_word):
        if c is None:
            return
        lvl = ("by day (%d days, %d in the thin/stress half)"
               % (c.get("n_days", 0), c.get("n_thin_days", 0))) if c.get("state_level") == "day" \
               else "per-observation median split"
        rises = np.isfinite(c["lambda_L_delta"]) and c["lambda_L_delta"] > 0
        direction = ("RISES with %s — the liquidity-contagion direction" % state_word) if rises else \
                    ("FALLS with %s — tail *decoupling*, not contagion amplification" % state_word)
        L.append("## %s-conditional tail dependence (`%s`; %s; open/close %g-min trimmed)\n"
                 % (title, c["copula"], lvl, c["tod_trim_min"]))
        L.append("- lambda_L thin/stress: %.3f (n=%d) ; deep/calm: %.3f (n=%d) ; **delta: %.3f** — "
                 "lower-tail dependence %s\n" % (c["lambda_L_thin"], c["n_thin"], c["lambda_L_deep"],
                                                 c["n_deep"], c["lambda_L_delta"], direction))
        corr = c.get("day_corr_lambdaL_liq", np.nan)
        if np.isfinite(corr):
            L.append("- across-day corr(per-day lambda_L, day-mean %s state): %.2f\n" % (state_word, corr))

    if liq is not None or vol is not None:
        _emit_cond("Liquidity", liq, "thinner books (higher AUC_decay)")
        _emit_cond("Volatility", vol, "higher realized volatility (stress)")
        if (liq is not None and vol is not None
                and np.isfinite(liq["lambda_L_delta"]) and np.isfinite(vol["lambda_L_delta"])
                and liq["lambda_L_delta"] * vol["lambda_L_delta"] < 0):
            L.append("\n*Note: the AUC_decay (liquidity) and realized-volatility (stress) splits disagree "
                     "in sign. On crash days the reconstructed book is densely quoted at all levels, so "
                     "decay-weighted cost can label them as 'deep' — the volatility split is the cleaner "
                     "stress read; treat the AUC_decay split as confounded by book density / intraday "
                     "seasonality.*\n")
    return "\n".join(L) + "\n"


# ════════════════════════════════════════════════════════════════════════════
# Self-test (synthetic; no market data / binary)
# ════════════════════════════════════════════════════════════════════════════
def _selftest() -> bool:
    print("copula_garch self-test\n")
    rng = np.random.default_rng(7)

    # 1) Clayton recovery: theta=2 -> lambda_L = 2^(-1/2) = 0.7071, lambda_U = 0
    th_true = 2.0
    U = simulate_copula("clayton", 6000, {"theta": th_true}, rng)
    fc = _fit_copula("clayton", U[:, 0], U[:, 1])
    lL_true = 2.0**(-1.0 / th_true)
    clayton_ok = (abs(fc["params"]["theta"] - th_true) < 0.3 and abs(fc["lambda_L"] - lL_true) < 0.05
                  and fc["lambda_U"] == 0.0)
    print("Clayton: theta_hat=%.2f (true %.1f), lambda_L=%.3f (true %.3f), lambda_U=%.1f : %s"
          % (fc["params"]["theta"], th_true, fc["lambda_L"], lL_true, fc["lambda_U"], clayton_ok))

    # 2) selection picks the lower-tail copula on Clayton data; gaussian gives zero tail
    sel = select_copula(U, is_uniform=True)
    sel_ok = (sel["best"] in ("clayton", "bb1", "sjc")
              and float(sel["table"].set_index("copula").loc["gaussian", "lambda_L"]) == 0.0)
    if "sjc" in sel["fits"]:
        sjc = sel["fits"]["sjc"]
        sjc_ok = sjc["lambda_L"] > sjc["lambda_U"]         # asymmetric copula sees lower>upper
    else:
        sjc_ok = True
    print("selection best=%s (gaussian lambda_L=0); SJC lambda_L>lambda_U: %s : %s"
          % (sel["best"], sjc_ok, sel_ok and sjc_ok))

    # 2b) BB1 (Clayton-Gumbel): recover both tails; nested LR vs Clayton.
    #     theta=1.5, delta=1.3 -> lambda_L=2^(-1/1.95)=0.701, lambda_U=2-2^(1/1.3)=0.296  (lower>upper)
    th_b, de_b = 1.5, 1.3
    Ubb = simulate_copula("bb1", 5000, {"theta": th_b, "delta": de_b}, rng)
    fb = _fit_copula("bb1", Ubb[:, 0], Ubb[:, 1])
    lLb, lUb = _bb1_tail(th_b, de_b)
    bb1_ok = (abs(fb["lambda_L"] - lLb) < 0.07 and abs(fb["lambda_U"] - lUb) < 0.10
              and fb["lambda_L"] > fb["lambda_U"] > 0.0)
    print("BB1: theta=%.2f delta=%.2f, lambda_L=%.3f (true %.3f) lambda_U=%.3f (true %.3f), L>U>0 : %s"
          % (fb["params"]["theta"], fb["params"]["delta"], fb["lambda_L"], lLb,
             fb["lambda_U"], lUb, bb1_ok))
    sel_bb1 = select_copula(Ubb, is_uniform=True)          # BB1 data: LR should reject Clayton
    p_bb1 = sel_bb1["nested_lr"]["p_boundary"]
    p_clay = sel["nested_lr"]["p_boundary"]                # Clayton data: LR should NOT reject
    lr_ok = (p_bb1 < 0.05 and p_clay > 0.05 and "bb1" in sel["fits"])
    print("nested LR: BB1 data Lambda=%.1f p=%.3f (reject Clayton); Clayton data Lambda=%.1f p=%.3f "
          "(keep Clayton) : %s" % (sel_bb1["nested_lr"]["stat"], p_bb1,
                                   sel["nested_lr"]["stat"], p_clay, lr_ok))

    # 3) t-copula-DCC tracks a rising correlation, with finite nu => non-zero tail dependence
    n1 = n2 = 2500
    Z1 = rng.multivariate_normal([0, 0], [[1, 0.30], [0.30, 1]], size=n1)
    Z2 = rng.multivariate_normal([0, 0], [[1, 0.85], [0.85, 1]], size=n2)
    Z = np.vstack([Z1, Z2])
    dyn = t_copula_dcc(Z)
    rho = dyn["rho_t"]
    dcc_ok = (np.nanmean(rho[:n1]) < np.nanmean(rho[n1:]) - 0.1     # correlation rose
              and dyn["nu"] < 60 and dyn["lambda_mean"] >= 0
              and np.nanmean(dyn["lambda_t"][n1:]) > np.nanmean(dyn["lambda_t"][:n1]))  # tail dep rose too
    print("t-copula-DCC: rho %.2f->%.2f, nu=%.1f, lambda_t %.3f->%.3f : %s"
          % (np.nanmean(rho[:n1]), np.nanmean(rho[n1:]), dyn["nu"],
             np.nanmean(dyn["lambda_t"][:n1]), np.nanmean(dyn["lambda_t"][n1:]), dcc_ok))

    # 4) liquidity/stress-conditional, DAY-LEVEL with open/close trim: tail dep HIGHER on stress days.
    #    8 calm days (Clayton theta=0.4 -> lambda_L=0.177) + 4 stress days (theta=3 -> lambda_L=0.794);
    #    liq_state high on stress days; tod spans the session; planted garbage in trimmed edges.
    rng4 = np.random.default_rng(11); nd, per = 12, 1800
    Zc, dayc, todc, liqc = [], [], [], []
    for d in range(nd):
        stress = d >= 6                                  # 6 calm / 6 stress -> median cleanly separates
        U = simulate_copula("clayton", per, {"theta": 3.0 if stress else 0.4}, rng4)
        Z = stats.norm.ppf(_clip(U))
        tod = np.linspace(0.0, 23400.0, per)
        edge = (tod < 900) | (tod > 22500)               # first/last 15 min -> scramble to test the trim
        Z[edge] = rng4.normal(0, 1, size=(int(edge.sum()), 2))
        Zc.append(Z); dayc.append(np.full(per, d)); todc.append(tod)
        liqc.append(np.full(per, 1.0 if stress else -1.0) + rng4.normal(0, 0.05, per))
    Zc = np.vstack(Zc); dayc = np.concatenate(dayc); todc = np.concatenate(todc); liqc = np.concatenate(liqc)
    lc = liquidity_conditional_dependence(Zc, liqc, day_id=dayc, tod_s=todc, copula="clayton",
                                          state_level="day", tod_trim_min=15)
    liq_ok = (lc["lambda_L_thin"] > lc["lambda_L_deep"] + 0.2 and lc["n_thin_days"] == 6
              and lc["state_level"] == "day"
              and np.isfinite(lc["day_corr_lambdaL_liq"]) and lc["day_corr_lambdaL_liq"] > 0.5)
    print("liquidity-conditional (day-level): lambda_L stress=%.3f > calm=%.3f (delta=%.3f), "
          "day-corr=%.2f, stress-days=%d : %s"
          % (lc["lambda_L_thin"], lc["lambda_L_deep"], lc["lambda_L_delta"],
             lc["day_corr_lambdaL_liq"], lc["n_thin_days"], liq_ok))

    # 5) end-to-end on synthetic returns + liquidity state, markdown renders
    rr = rng.multivariate_normal([0, 0], [[1e-4, 0.6e-4], [0.6e-4, 1e-4]], size=1500)
    res = run_copula_analysis(returns=rr, liq_state=rng.normal(size=1500), write=False)
    md = _format_md(res)
    e2e_ok = (res["n_obs"] > 1000 and "best" in res["selection"] and "lambda_t" in res["dynamic"]
              and "Copula-GARCH" in md and len(md) > 400)
    print("end-to-end: n=%d best=%s | markdown renders: %s"
          % (res["n_obs"], res["selection"]["best"], e2e_ok))

    ok = all([clayton_ok, sel_ok, sjc_ok, bb1_ok, lr_ok, dcc_ok, liq_ok, e2e_ok])
    print("\nchecks:", ok)
    return ok


if __name__ == "__main__":
    _selftest()
