"""
rigobon_id.py
============
Identification through heteroskedasticity (Rigobon 2003) for the cross-market structural
system -- item #2. The SVAR of the paper (Tables 9/11/13) is identified only by the
imposed "futures lead" recursive ordering. Rigobon's result removes that assumption: if
the contemporaneous structural matrix A is constant across regimes but the structural
shock variances differ (the calm / elevated / stress regimes from `stress_index`), the
regime-varying reduced-form residual covariances over-determine A and pin it down with no
ordering. The contagion payoff: the contemporaneous cross-market coefficient -- and
whether it is asymmetric (futures->ETF vs ETF->futures) -- becomes an estimate, not an
assumption.

Structural form  A y_t = eps_t,  eps_t ~ (0, D_s) diagonal, D_s regime-specific.
Reduced form     u_t = A^{-1} eps_t,  so  Omega_s = A^{-1} D_s A^{-1'}.
Two regimes give  M = Omega_1 Omega_2^{-1} = A^{-1} (D_1 D_2^{-1}) A^{-1 -1}, an eigenvalue
problem: the eigenvectors of M are the columns of A^{-1}, the eigenvalues the structural
shock-variance ratios. Identification needs DISTINCT eigenvalues (the regimes must differ
in their *relative* heteroskedasticity, not just scale). Three+ regimes over-identify
(testable via `overid_statistic`).

Also includes the Forbes-Rigobon (2002) heteroskedasticity-adjusted cross-market
correlation -- the canonical contagion-vs-interdependence test: a crisis rise in
comovement that survives the vol adjustment is contagion; one that does not is just
interdependence amplified by higher volatility.

Pure numpy / pandas.
"""
from itertools import permutations
import numpy as np
import pandas as pd

EPS = 1e-12


# ── regime residual covariances ───────────────────────────────────────────────
def regime_residual_cov(U, regimes):
    """{regime: Omega_r} from reduced-form residuals U (T x k) and regime labels (len T).
    In practice U are the residuals of ONE reduced-form VAR fit on the pooled data
    (constant coefficients across regimes -- the heteroskedasticity is in the shocks)."""
    U = np.asarray(U, float); regimes = np.asarray(regimes)
    out = {}
    for r in pd.unique(regimes[pd.notna(regimes)]):
        rows = U[regimes == r]
        rows = rows[np.all(np.isfinite(rows), axis=1)]
        if len(rows) > rows.shape[1]:
            out[r] = np.cov(rows.T)
    return out


# ── structural normalization (unit diagonal, resolve permutation/sign) ────────
def _normalize_structural(A):
    """Permute rows of A for diagonal dominance, then scale each row to unit diagonal so
    the own-variable coefficient is 1. Resolves the eigenvector permutation and sign."""
    k = A.shape[0]
    best = None
    for perm in permutations(range(k)):
        Ap = A[list(perm)]
        score = np.prod(np.abs(np.diag(Ap)))
        if best is None or score > best[0]:
            best = (score, Ap)
    Ap = best[1]
    d = np.diag(Ap).copy(); d[np.abs(d) < EPS] = EPS
    return Ap / d[:, None]


# ── core: 2-regime identification via simultaneous diagonalization ────────────
def rigobon_2regime(Omega1, Omega2, names=None):
    """Identify the contemporaneous structural matrix A from two regime covariances by
    the generalized-eigenvalue (simultaneous-diagonalization) solution. Returns A (unit
    diagonal), the contemporaneous spillover matrix C = I - A (C[i,j] = effect of variable
    j on i), the structural shock-variance ratios (eigenvalues), and the eigenvalue
    separation (the identification strength -- near-equal ratios mean weak identification)."""
    Om1 = np.asarray(Omega1, float); Om2 = np.asarray(Omega2, float)
    M = Om1 @ np.linalg.inv(Om2)
    lam, V = np.linalg.eig(M)
    lam = np.real(lam); V = np.real(V)
    A = _normalize_structural(np.linalg.inv(V))
    C = np.eye(A.shape[0]) - A
    sep = float(np.min(np.abs(np.subtract.outer(lam, lam) + np.eye(len(lam)) * 1e9)))
    nm = list(names) if names is not None else [f"y{i}" for i in range(A.shape[0])]
    return {"A": A, "contemp": pd.DataFrame(C, index=nm, columns=nm),
            "var_ratios": np.sort(lam)[::-1], "eig_separation": sep, "names": nm}


def overid_statistic(Omega_by_regime, A):
    """With 3+ regimes, A must diagonalize EVERY regime: A Omega_r A' should be diagonal.
    Returns the summed off-diagonal mass (Frobenius, normalized by the diagonal) across
    regimes -- ~0 if the constant-A heteroskedasticity model holds, large if it fails."""
    tot = 0.0; n = 0
    for Om in Omega_by_regime.values():
        D = A @ np.asarray(Om, float) @ A.T
        off = D - np.diag(np.diag(D))
        denom = np.sum(np.diag(D) ** 2) + EPS
        tot += np.sum(off ** 2) / denom; n += 1
    return tot / max(n, 1)


# ── identifying-variation diagnostic (the weak-identification check) ──────────
def _pair_strength(Oa, Ob):
    Oa = np.asarray(Oa, float); Ob = np.asarray(Ob, float)
    ratios = np.diag(Oa) / np.maximum(np.diag(Ob), EPS)            # per-asset variance ratio a:b
    log_r = np.log(np.maximum(ratios, EPS))
    spread = float(np.exp(log_r.max() - log_r.min()) - 1.0)        # 0 iff every asset scales equally
    lam = np.sort(np.real(np.linalg.eigvals(Oa @ np.linalg.inv(Ob))))
    rel_gap = float(abs(lam[-1] - lam[0]) / (np.mean(np.abs(lam)) + EPS))
    return ratios, spread, rel_gap


def identification_diagnostic(Omega_by_regime, names=None, spread_tol=0.15, gap_tol=0.10):
    """Pre-estimation check that the regimes supply IDENTIFYING variation for Rigobon ID. Het-based
    identification FAILS when the regimes differ only in scale -- if every structural shock variance
    scales by a common factor, M = Omega_a Omega_b^{-1} is proportional to I, its eigenvalues coincide
    and A is not identified, however clean the eigen-decomposition looks numerically. The identifying
    signal is *relative* heteroskedasticity: the variances must move by DIFFERENT factors across assets.

    Returns, for the most-separated regime pair: the per-asset variance ratios, their SPREAD
    (max/min - 1; ~0 means common-scale = not identified), the relative eigenvalue gap, and an
    ``identified`` verdict. This is the variance-ratio test a referee expects alongside the structural
    estimate; it does NOT test regime EXOGENEITY (an identifying assumption -- use externally-timed
    regimes such as the dated MWCB halts or time-of-day buckets, not realized vol of the same series)."""
    keys = list(Omega_by_regime.keys())
    if len(keys) < 2:
        raise ValueError("need >=2 regimes")
    best = None
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ratios, spread, gap = _pair_strength(Omega_by_regime[keys[i]], Omega_by_regime[keys[j]])
            if best is None or spread > best["var_ratio_spread"]:
                best = {"pair": (keys[i], keys[j]), "variance_ratios": ratios,
                        "var_ratio_spread": spread, "rel_eig_gap": gap}
    best["identified"] = bool(best["var_ratio_spread"] > spread_tol and best["rel_eig_gap"] > gap_tol)
    best["names"] = list(names) if names is not None else None
    return best


# ── end-to-end against the recursive (Cholesky) alternative ───────────────────
def _cholesky_contemp(Omega, order):
    """Contemporaneous spillover matrix implied by a recursive (Cholesky) ordering
    `order` (list of variable indices, first = most exogenous). Demonstrates ordering
    dependence: each ordering forces one direction of every pair to zero."""
    k = Omega.shape[0]
    P = np.argsort(order)
    Om_p = np.asarray(Omega, float)[np.ix_(order, order)]
    L = np.linalg.cholesky(Om_p + np.eye(k) * EPS)
    A_p = _normalize_structural(np.linalg.inv(L))
    A = A_p[np.ix_(P, P)]
    return np.eye(k) - A


def rigobon_identify(U, regimes, calm="calm", stress="stress", names=None):
    """End-to-end order-free identification of the contemporaneous cross-market matrix
    using the most-heteroskedastic regime pair (calm vs stress by default). Returns the
    Rigobon `contemp` matrix alongside the two Cholesky orderings, so the ordering
    dependence the paper assumes away is visible next to the order-free answer. With 3+
    regimes also returns the over-identification statistic."""
    cov = regime_residual_cov(U, regimes)
    if calm not in cov or stress not in cov:
        keys = list(cov.keys())
        if len(keys) < 2:
            raise ValueError("need >=2 regimes with enough observations")
        calm, stress = keys[0], keys[-1]
    rig = rigobon_2regime(cov[calm], cov[stress], names)
    k = rig["A"].shape[0]; nm = rig["names"]
    chol_fwd = pd.DataFrame(_cholesky_contemp(cov[stress], list(range(k))), index=nm, columns=nm)
    chol_rev = pd.DataFrame(_cholesky_contemp(cov[stress], list(range(k))[::-1]), index=nm, columns=nm)
    out = {"contemp_rigobon": rig["contemp"], "contemp_cholesky_fwd": chol_fwd,
           "contemp_cholesky_rev": chol_rev, "var_ratios": rig["var_ratios"],
           "eig_separation": rig["eig_separation"], "regimes_used": (calm, stress), "A": rig["A"]}
    if len(cov) >= 3:
        out["overid"] = overid_statistic(cov, rig["A"])
    return out


# ── Forbes-Rigobon adjusted correlation (contagion vs interdependence) ────────
def forbes_rigobon_corr(x_crisis, y_crisis, x_calm, y_calm):
    """Heteroskedasticity-adjusted cross-market correlation (Forbes-Rigobon 2002).
    A crisis rise in the raw correlation is partly mechanical when the source market's
    variance rises; rho_adjusted strips that out. contagion = rho_adjusted(crisis) -
    rho(calm): > 0 is genuine contagion, ~0 is interdependence amplified by volatility.
    `x` is the source market (its variance increase drives the adjustment)."""
    xc, yc = np.asarray(x_crisis, float), np.asarray(y_crisis, float)
    xt, yt = np.asarray(x_calm, float), np.asarray(y_calm, float)
    rc = float(np.corrcoef(xc, yc)[0, 1]); rt = float(np.corrcoef(xt, yt)[0, 1])
    delta = float(np.var(xc) / max(np.var(xt), EPS) - 1.0)
    r_adj = rc / np.sqrt(1.0 + delta * (1.0 - rc ** 2))
    return {"rho_crisis": rc, "rho_calm": rt, "rho_adjusted": r_adj,
            "delta_var": delta, "contagion": r_adj - rt}


# ── self-test ─────────────────────────────────────────────────────────────────
def _sim_two_regime(g12, g21, sig_calm, sig_stress, n=12000, seed=0):
    """Reduced-form residuals from a known structural system A y = eps with regime-
    dependent diagonal shock variances. A = [[1,-g12],[-g21,1]] -> true contemp
    C = [[0,g12],[g21,0]]."""
    rng = np.random.default_rng(seed)
    A = np.array([[1.0, -g12], [-g21, 1.0]]); P = np.linalg.inv(A)
    eps_c = rng.normal(0, 1, (n, 2)) * np.sqrt(sig_calm)
    eps_s = rng.normal(0, 1, (n, 2)) * np.sqrt(sig_stress)
    U = np.vstack([eps_c @ P.T, eps_s @ P.T])
    reg = np.array(["calm"] * n + ["stress"] * n)
    return U, reg


def _selftest():
    print("rigobon_id self-test")

    # (1) recover a known contemporaneous matrix, order-free
    g12, g21 = 0.4, 0.2                       # true: ES->ETF 0.4, ETF->ES 0.2 (asymmetric)
    U, reg = _sim_two_regime(g12, g21, sig_calm=[1.0, 1.0], sig_stress=[3.0, 7.0])
    res = rigobon_identify(U, reg, names=["ETF", "ES"])
    C = res["contemp_rigobon"]
    print("\n(1) order-free identification (true C: ETF<-ES=%.2f, ES<-ETF=%.2f)" % (g12, g21))
    print("    Rigobon contemp:\n" + C.round(3).to_string())
    print("    variance ratios=%s  eig separation=%.2f" % (np.round(res["var_ratios"], 2), res["eig_separation"]))
    print("    checks:",
          abs(C.loc["ETF", "ES"] - g12) < 0.05 and abs(C.loc["ES", "ETF"] - g21) < 0.05
          and res["eig_separation"] > 0.02)        # distinct variance ratios => identified

    # (2) Cholesky is ordering-dependent; Rigobon is unique
    cf = res["contemp_cholesky_fwd"]; cr = res["contemp_cholesky_rev"]
    print("\n(2) recursive (Cholesky) ordering dependence vs the order-free answer")
    print("    Cholesky [ETF first] ES<-ETF=%.3f, ETF<-ES=%.3f  (forces one direction to 0)"
          % (cf.loc["ES", "ETF"], cf.loc["ETF", "ES"]))
    print("    Cholesky [ES first]  ES<-ETF=%.3f, ETF<-ES=%.3f"
          % (cr.loc["ES", "ETF"], cr.loc["ETF", "ES"]))
    print("    -> the two orderings disagree; Rigobon recovers both directions at once")
    print("    checks:",
          (abs(cf.loc["ETF", "ES"]) < 1e-6 or abs(cf.loc["ES", "ETF"]) < 1e-6)      # fwd zeroes a direction
          and (abs(cr.loc["ETF", "ES"]) < 1e-6 or abs(cr.loc["ES", "ETF"]) < 1e-6)  # rev zeroes the other
          and abs(C.loc["ETF", "ES"]) > 0.05 and abs(C.loc["ES", "ETF"]) > 0.05)    # rigobon keeps both

    # (3) over-identification: a 3rd regime consistent with the same A does not reject
    rng = np.random.default_rng(5); A = res["A"]; P = np.linalg.inv(A)
    eps3 = rng.normal(0, 1, (12000, 2)) * np.sqrt([5.0, 2.0])           # same A, new variances
    U3 = np.vstack([U, eps3 @ P.T]); reg3 = np.concatenate([reg, ["elevated"] * 12000])
    cov3 = regime_residual_cov(U3, reg3)
    oid_good = overid_statistic(cov3, A)
    oid_bad = overid_statistic(cov3, np.array([[1.0, -0.9], [-0.9, 1.0]]))   # wrong A
    print("\n(3) over-identification with a third regime")
    print("    off-diagonal mass: correct A=%.4f  vs wrong A=%.4f" % (oid_good, oid_bad))
    print("    checks:", oid_good < 0.05 and oid_bad > 5 * oid_good)

    # (4) Forbes-Rigobon: pure heteroskedasticity is NOT contagion; a crisis linkage is
    rng = np.random.default_rng(9); n = 20000
    xc = rng.normal(0, 2, n); xt = rng.normal(0, 1, n)                  # source vol up 4x in crisis
    yt = 0.5 * xt + rng.normal(0, 1, n)
    y_het = 0.5 * xc + rng.normal(0, 1, n)                             # SAME beta -> no contagion
    y_con = 0.9 * xc + rng.normal(0, 1, n)                             # larger crisis beta -> contagion
    fr_het = forbes_rigobon_corr(xc, y_het, xt, yt)
    fr_con = forbes_rigobon_corr(xc, y_con, xt, yt)
    print("\n(4) Forbes-Rigobon adjusted correlation (contagion vs interdependence)")
    print("    pure het : rho_crisis=%.3f rho_calm=%.3f rho_adj=%.3f -> contagion=%.3f (~0)"
          % (fr_het["rho_crisis"], fr_het["rho_calm"], fr_het["rho_adjusted"], fr_het["contagion"]))
    print("    linkage  : rho_crisis=%.3f rho_calm=%.3f rho_adj=%.3f -> contagion=%.3f (>0)"
          % (fr_con["rho_crisis"], fr_con["rho_calm"], fr_con["rho_adjusted"], fr_con["contagion"]))
    print("    checks:", abs(fr_het["contagion"]) < 0.03 and fr_con["contagion"] > 0.1
          and fr_het["rho_crisis"] > fr_het["rho_calm"])             # raw corr rose even without contagion


if __name__ == "__main__":
    _selftest()
