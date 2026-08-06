"""
cross_impact.py
===============
Multi-asset price-impact and CROSS-IMPACT matrices in the spirit of Cont,
Kukanov & Stoikov (2014) and Cont, Cucuringu & Zhang (2023), as the modern
replacement for the message-count order-flow proportions in OFR 19-04.

Model (contemporaneous):  r^a_t = sum_b  Lambda_{ab} * OFI^b_t + e^a_t
  Lambda is K x K. Diagonal = own (Kyle-style) impact; off-diagonal = CROSS-
  impact / spillover (SPY OFI -> ES return and vice versa). Inference is
  Newey-West HAC. A central CCZ result is that apparent best-level cross-impact
  largely SHRINKS once multi-level OFI is used; compare_impact_depth() exposes
  exactly that, which is itself a publishable robustness point.

Requires cross_asset_pd_liquidity.py (for the OFI and mid helpers).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from cross_asset_pd_liquidity import order_flow_imbalance, _mid

EPS = 1e-12


def _newey_west(X, u, L):
    """HAC covariance of OLS coefficients for design X (T x m) and residual u."""
    T, m = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    return XtX_inv @ S @ XtX_inv


def impact_regression(R, O, hac_lags=10):
    """R, O: T x K arrays of aligned returns and OFI. Returns Lambda, SE, t, R2.
    Lambda[a,b] = impact of asset b's OFI on asset a's return."""
    T, K = R.shape
    X = np.column_stack([np.ones(T), O])                 # const + K OFI columns
    Lam = np.zeros((K, K)); SE = np.zeros((K, K)); R2 = np.zeros(K)
    for a in range(K):
        b, u = np.linalg.lstsq(X, R[:, a], rcond=None)[0], None
        u = R[:, a] - X @ b
        V = _newey_west(X, u, hac_lags)
        Lam[a, :] = b[1:]                                # drop intercept
        SE[a, :] = np.sqrt(np.diag(V))[1:]
        ss = np.sum((R[:, a] - R[:, a].mean()) ** 2)
        R2[a] = 1.0 - np.sum(u ** 2) / (ss + EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = Lam / np.where(SE > EPS, SE, np.nan)
    # Single-observation sensitivity. High-frequency impact OLS is exposed to one colossal
    # co-movement observation -- a halt-reopen seam, a flash print -- carrying a coefficient on its
    # own: the 2026-08-05 run's 2020-days cross-impact reversal (ES<-SPY jumping from ~0 to ~0.3)
    # is the motivating case, estimated halt-included at the time. Refit with the single largest
    # |OFI x return| co-movement removed; `drop1_frac` is the relative coefficient move. A material
    # gap says one observation carries the estimate -- the number to check before interpreting any
    # surprising cell, and it ships with the matrix rather than requiring a post-mortem.
    infl = np.max(np.abs(O), axis=1) * np.max(np.abs(R), axis=1)
    keep = np.ones(T, bool)
    if T > 50:
        keep[int(np.argmax(infl))] = False
    Lam1 = np.zeros((K, K))
    X1 = X[keep]
    for a in range(K):
        b1 = np.linalg.lstsq(X1, R[keep, a], rcond=None)[0]
        Lam1[a, :] = b1[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        drop1 = np.abs(Lam - Lam1) / np.where(np.abs(Lam) > EPS, np.abs(Lam), np.nan)
    # flag only when the move is large RELATIVE TO THE COEFFICIENT *and* larger than its SE -- a
    # true-zero cell moves by epsilon/epsilon otherwise and the relative measure alone cries wolf
    move = np.abs(Lam - Lam1)
    flag = bool(np.any((move > SE) & (move > 0.5 * np.abs(Lam)) & np.isfinite(SE)))
    return {"Lambda": Lam, "SE": SE, "t": tstat, "R2": R2,
            "Lambda_drop1": Lam1, "drop1_frac": drop1, "drop1_flag": flag}


# ── Schneider-Lillo no-dynamic-arbitrage symmetry test ────────────────────────
def _hac_sum_stacked(X, E, L):
    """Newey-West HAC of the SUMMED stacked score Σ_t m_t, m_t = [x_t e_{0,t}; x_t e_{1,t}]
    (equation-major), matching bread = I_k ⊗ (X'X)^{-1}. Sum form (same convention as _newey_west)."""
    T, p = X.shape; k = E.shape[1]
    M = np.zeros((T, k * p))
    for a in range(k):
        M[:, a * p:(a + 1) * p] = X * E[:, a][:, None]
    S = M.T @ M
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = M[l:].T @ M[:-l]
        S += w * (G + G.T)
    return S


def cross_impact_symmetry(R, O, hac_lags=10):
    """Schneider & Lillo (2019) no-dynamic-arbitrage symmetry test for the 2-asset cross-impact
    matrix. Under no dynamic arbitrage the cross terms must satisfy Lambda_12 = Lambda_21. Both
    return equations are estimated on the SAME regressors [const, OFI_1, OFI_2]; the joint HAC
    covariance of the stacked coefficients gives a Wald test of Lambda_12 - Lambda_21 = 0 (the
    per-equation SEs from impact_regression cannot, since they omit the cross-equation covariance).
    A significant asymmetry is evidence against the symmetric (no-arbitrage) cross-impact restriction."""
    R = np.asarray(R, float); O = np.asarray(O, float)
    if R.shape[1] != 2:
        raise ValueError("symmetry test is defined for the 2-asset cross-impact pair")
    T = R.shape[0]; X = np.column_stack([np.ones(T), O]); p = X.shape[1]; k = 2
    XtXinv = np.linalg.pinv(X.T @ X)
    B = XtXinv @ (X.T @ R)                                # p x k; B[:,a] = eq a coeffs
    E = R - X @ B
    V = np.kron(np.eye(k), XtXinv) @ _hac_sum_stacked(X, E, hac_lags) @ np.kron(np.eye(k), XtXinv)
    vecB = np.concatenate([B[:, a] for a in range(k)])   # [const0,L11,L12, const1,L21,L22]
    i12, i21 = 0 * p + 2, 1 * p + 1                       # L12 = eq0 on OFI_2; L21 = eq1 on OFI_1
    d = float(vecB[i12] - vecB[i21])
    se = float(np.sqrt(max(V[i12, i12] + V[i21, i21] - 2.0 * V[i12, i21], EPS)))
    z = d / se
    from math import erfc
    pval = float(erfc(abs(z) / np.sqrt(2.0)))             # two-sided normal
    return {"lambda_12": float(B[2, 0]), "lambda_21": float(B[1, 1]), "asymmetry": d,
            "se": se, "z": float(z), "p_value": pval, "symmetric": bool(pval > 0.05)}


def cross_impact_matrix(df, assets=("SPY", "ES"), n_levels=10, hac_lags=10,
                        standardize=True, min_rest_steps=0):
    """Build aligned returns and (level-1 or multi-level) OFI from the book frame
    and estimate the K x K cross-impact matrix. min_rest_steps>0 applies the
    fleeting-quote filter to the OFI inputs (sub-second grids)."""
    R = np.column_stack([np.diff(np.log(_mid(df, a))) for a in assets])
    O = np.column_stack([order_flow_imbalance(df, a, n_levels, min_rest_steps)[1:] for a in assets])
    ok = np.all(np.isfinite(R), axis=1) & np.all(np.isfinite(O), axis=1)   # drop unaligned/NaN rows jointly
    R, O = R[ok], O[ok]                                                     # else one NaN row poisons mean()/lstsq
    if standardize:
        R = (R - R.mean(0)) / (R.std(0) + EPS)
        O = (O - O.mean(0)) / (O.std(0) + EPS)
    out = impact_regression(R, O, hac_lags)
    out["assets"] = list(assets); out["n_levels"] = n_levels
    return out


def compare_impact_depth(df, assets=("SPY", "ES"), n_levels=10, hac_lags=10):
    """Best-level vs multi-level cross-impact. Returns a tidy DataFrame; watch the
    off-diagonal (cross) terms shrink as depth is added (Cont-Cucuringu-Zhang)."""
    best = cross_impact_matrix(df, assets, n_levels=1, hac_lags=hac_lags)
    multi = cross_impact_matrix(df, assets, n_levels=n_levels, hac_lags=hac_lags)
    rows = []
    for a in range(len(assets)):
        for b in range(len(assets)):
            kind = "own" if a == b else "cross"
            rows.append({"return": assets[a], "ofi": assets[b], "type": kind,
                         "lambda_best": best["Lambda"][a, b], "t_best": best["t"][a, b],
                         "lambda_multi": multi["Lambda"][a, b], "t_multi": multi["t"][a, b]})
    return pd.DataFrame(rows)


def cross_impact_panel(sessions, assets=("SPY", "ES"), n_levels=10, hac_lags=10, min_rest_steps=0):
    """Per-session cross-impact matrices, stacked, with regime means. sessions:
    iterable of (date, regime, df). min_rest_steps>0 engages the fleeting-quote filter
    on the OFI inputs (inert at 1s, relevant sub-second)."""
    rows = []
    for date, regime, df in sessions:
        ci = cross_impact_matrix(df, assets, n_levels=n_levels, hac_lags=hac_lags,
                                 min_rest_steps=min_rest_steps)
        for a in range(len(assets)):
            for b in range(len(assets)):
                # the drop-one leverage diagnostic ships IN the panel: a surprising cell (the
                # 2024-08-05 ES<-SPY = 0.40 is the live case) is checkable from the table itself
                # -- lambda_drop1 far from lambda, or drop1_flag True, says one co-movement
                # observation carries the estimate. drop1_flag is per-DAY (any cell moved by more
                # than its SE and half its size), so it repeats across the day's four rows.
                rows.append({"date": str(date), "regime": regime,
                             "return": assets[a], "ofi": assets[b],
                             "type": "own" if a == b else "cross",
                             "lambda": ci["Lambda"][a, b], "t": ci["t"][a, b],
                             "lambda_drop1": ci["Lambda_drop1"][a, b],
                             "drop1_flag": bool(ci["drop1_flag"])})
    panel = pd.DataFrame(rows)
    summary = panel.groupby(["regime", "type"])["lambda"].agg(["mean", "std", "count"])
    if bool(panel["drop1_flag"].any()):
        flagged = sorted(panel.loc[panel["drop1_flag"], "date"].unique())
        summary.attrs["drop1_flagged_dates"] = flagged
    return panel, summary


def _selftest():
    rng = np.random.default_rng(3)
    T = 20000
    # Known OFI -> return map with genuine CROSS-impact ES->SPY.
    O = rng.normal(0, 1, (T, 2))                          # [OFI_SPY, OFI_ES]
    lam_self, lam_cross = 0.6, 0.25
    e = rng.normal(0, 1, (T, 2))
    r_spy = lam_self * O[:, 0] + lam_cross * O[:, 1] + e[:, 0]
    r_es = lam_self * O[:, 1] + 0.05 * O[:, 0] + e[:, 1]   # weak SPY->ES
    R = np.column_stack([r_spy, r_es])
    out = impact_regression(R, O, hac_lags=10)
    print("estimated cross-impact matrix Lambda (rows=return, cols=OFI):")
    print(pd.DataFrame(out["Lambda"], index=["r_SPY", "r_ES"], columns=["OFI_SPY", "OFI_ES"]).round(3))
    print("t-stats:")
    print(pd.DataFrame(out["t"], index=["r_SPY", "r_ES"], columns=["OFI_SPY", "OFI_ES"]).round(1))
    L = out["Lambda"]
    print("\nchecks")
    print("  own impact ~0.6:", round(L[0, 0], 2), round(L[1, 1], 2))
    print("  ES->SPY cross ~0.25 recovered & significant:",
          abs(L[0, 1] - 0.25) < 0.03 and out["t"][0, 1] > 5)
    print("  SPY->ES cross small:", abs(L[1, 0]) < 0.1)
    print("  R2:", out["R2"].round(3))

if __name__ == "__main__":
    _selftest()
