"""
liquidity_stress.py
===================
Liquidity-stress CONDITIONER and its tier-1 defensibility battery, composed on the curve
primitives in liquidity_curve_metrics.py and the FPCA in functional_liquidity.py.

Why this module exists. The revamped paper conditions price discovery on a liquidity state. In a
TICK-CONSTRAINED book -- SPY and ES sit at the one-tick floor almost always (Table 4: median spread
= 1 tick) -- the quoted spread is censored from below and cannot express moderate stress, so the
conditioner must be book-SHAPE based. The curve primitives already exist and are well-disciplined;
what was missing is (a) the assembled two-axis state, with the RESILIENCY axis wired in (not just
the level), and (b) the validation a referee demands -- that the state carries signal the quoted
spread / Amihud cannot. This module supplies both.

Two axes, both BOOK-DERIVED (read off the resting book -> predetermined within the interval, so
usable as a conditioner without the endogeneity of a trade-derived measure), in cross-asset-
comparable units:
  * LEVEL      cost_bps   = decay_weighted_cost  -- bps to demand liquidity (aggregate thinning)
  * RESILIENCY convexity  = impact_curve_length  -- convexity of the impact curve, ORTHOGONAL to the
                            level (near-touch hollowing -- the dimension that makes a sizeable
                            cross-asset arbitrage order expensive even when total depth looks fine).

Validation battery:
  * incremental_content -- nested partial-R^2 + cluster-robust joint Wald: does the state explain the
                           outcome BEYOND quoted spread + Amihud? (The headline defensibility exhibit.)
  * fpca_alignment      -- the hand-chosen axes ARE the dominant data-driven axes of book variation
                           (FPC1 ~ level, FPC2 ~ resiliency) -- "the data agrees with the axes."
  * lambda_validation   -- a high-stress state actually carries high REALIZED cross-impact (trade-
                           derived impact as the VALIDATOR, never the conditioner).

Pure numpy / pandas; depends on liquidity_curve_metrics, functional_liquidity, inference.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import liquidity_curve_metrics as lcm
import functional_liquidity as fl
import inference as inf
import ecm_sde as ecm
import cross_asset_pd_liquidity as ca

EPS = 1e-12


# ── benchmark liquidity proxies (what the state must beat) ────────────────────
def quoted_spread_bps(df, asset):
    """Quoted BBO spread in bps of mid -- the conventional liquidity proxy. In SPY/ES it is pinned at
    the one-tick floor almost always, hence near-constant: the whole reason a shape-based state is needed."""
    b = df[f"{asset}_bidprice_1"].to_numpy(float); a = df[f"{asset}_askprice_1"].to_numpy(float)
    mid = (a + b) / 2.0
    return np.where(mid > EPS, (a - b) / mid * 1e4, np.nan)


def amihud_illiquidity(returns, dollar_volume):
    """Amihud (2002) illiquidity |return| / dollar volume -- the canonical trade-based liquidity proxy,
    the second benchmark the book-state must add information beyond."""
    r = np.abs(np.asarray(returns, float)); v = np.asarray(dollar_volume, float)
    return np.where(v > EPS, r / v, np.nan)


# ── the two-axis stress state (the conditioner) ───────────────────────────────
def _total_dollar_depth(df, asset, n_levels=10, side="both"):
    """Total resting dollar notional in the top n_levels (sum price*size) -- the aggregate-liquidity
    LEVEL primitive, cross-asset comparable and orthogonal to book SHAPE (a uniform thinning lowers it;
    a pure near-touch hollowing at constant total depth does not)."""
    sides = ("ask", "bid") if side == "both" else (side,)
    tot = []
    for sd in sides:
        dist, qty, mid = lcm.extract_book(df, asset, sd, n_levels, unit="dollar")
        px = mid[:, None] + (1.0 if sd == "ask" else -1.0) * dist
        tot.append(np.nansum(np.abs(px) * qty, axis=1))
    return np.nanmean(np.column_stack(tot), axis=1)


def _near_book_shape(df, asset, n_levels=10):
    import warnings
    da, qa, _ = lcm.extract_book(df, asset, "ask", n_levels, unit="dollar")
    db, qb, _ = lcm.extract_book(df, asset, "bid", n_levels, unit="dollar")
    sa = lcm.near_book_auc(da, qa, kind="shape"); sb = lcm.near_book_auc(db, qb, kind="shape")
    with warnings.catch_warnings():                              # all-NaN rows -> NaN, no warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.column_stack([sa, sb]), axis=1)


def cost_to_fill_bps(df, asset, notional, n_levels=10, side="both"):
    """Cost in bps of mid to execute a fixed DOLLAR notional by walking the book -- the single
    economic cost-to-trade-Q stress summary (mixes depth and shape, as the true execution cost does),
    cross-asset comparable via the fixed notional. side='both' averages the buy/sell legs."""
    sides = ("ask", "bid") if side == "both" else (side,)
    out = []
    for sd in sides:
        dist, qty, mid = lcm.extract_book(df, asset, sd, n_levels, unit="dollar")
        tgt = notional / np.where(mid > EPS, mid, np.nan)       # fixed $ notional -> shares
        cum = np.cumsum(qty, axis=1); prev = cum - qty
        t = np.minimum(tgt, cum[:, -1]); take = np.clip(t[:, None] - prev, 0.0, qty)
        cost = (take * dist).sum(axis=1) / np.maximum(t, EPS)
        out.append(cost / np.where(mid > EPS, mid, np.nan) * 1e4)
    return np.nanmean(np.column_stack(out), axis=1)


def stress_state(df, asset, n_levels=10):
    """Two near-ORTHOGONAL axes of liquidity stress, cross-asset-comparable and BOOK-DERIVED
    (predetermined within the interval -> usable as a conditioner without trade-derived endogeneity):

      depth_illiq -- AGGREGATE THINNING. -log(total dollar depth, top n_levels); higher = thinner book,
                     ~invariant to where in the book the depth sits.
      hollowness  -- NEAR-TOUCH HOLLOWING / RESILIENCY. -(near-book shape convexity); higher = the inside
                     is empty relative to deeper levels -- the dimension that makes a sizeable cross-asset
                     arbitrage order expensive even when TOTAL depth looks fine, and the one the quoted
                     spread is blindest to.

    Both rise with stress and are near-orthogonal by construction (level vs shape). For a single
    economic summary that combines them as the true execution cost does, use cost_to_fill_bps."""
    depth = _total_dollar_depth(df, asset, n_levels)
    illiq = -np.log(np.where(depth > EPS, depth, np.nan))
    hollowness = -_near_book_shape(df, asset, n_levels)
    return pd.DataFrame({"depth_illiq": illiq, "hollowness": hollowness}, index=df.index)


def cross_stress_state(df, assets=("SPY", "ES"), n_levels=10):
    """Both assets' two-axis states aligned in one frame -- for conditioning ONE asset's price
    discovery on the OTHER asset's book stress (the cross-asset liquidity-propagation channel that is
    the paper's contribution). Columns: {asset}_depth_illiq, {asset}_hollowness."""
    out = {}
    for a in assets:
        s = stress_state(df, a, n_levels)
        out[f"{a}_depth_illiq"] = s["depth_illiq"].to_numpy()
        out[f"{a}_hollowness"] = s["hollowness"].to_numpy()
    return pd.DataFrame(out, index=df.index)


def basis_state(df, predetermined=True):
    """The SPY-ES no-arbitrage basis assembled as a conditioning STATE for the stress-response
    function -- the OTHER axis (with the book-stress state) the analysis conditions on.

    z_t = log P^SPY - log P^ES is the (1,-1) cointegrating combination demeaned within the session
    (ecm_sde.basis); its magnitude |z_t| is the size of the law-of-one-price violation, i.e. the
    DIRECT health of the cross-asset arbitrage link -- the mechanism itself, not volatility (its
    symptom). Returns the signed basis and the dislocation magnitude in bps. With predetermined=True
    the series is lagged one step, so the state at t is known at t-1 and can condition the response
    at t without conditioning on the contemporaneous outcome. Pass `abs_basis_bps` as `state=` to
    irf.local_projection_irf for the basis-conditioned onset IRF (theta_low/theta_high then trace the
    response as a function of how dislocated the link already is)."""
    z = ecm.basis(ca._mid(df, "SPY"), ca._mid(df, "ES"), demean=True) * 1e4    # demeaned basis, bps
    s = pd.Series(z, index=df.index)
    if predetermined:
        s = s.shift(1)
    return pd.DataFrame({"basis_bps": s, "abs_basis_bps": s.abs()}, index=df.index)


# ── validation 1: incremental content over the spread / Amihud benchmark ──────
def incremental_content(y, base_X, add_X, clusters=None):
    """Does the block ``add_X`` explain ``y`` BEYOND ``base_X``? Nested OLS giving the partial R^2
    (the R^2 gain), an F-test for the added block, and -- when ``clusters`` are supplied -- a
    cluster-robust (Liang-Zeger) joint Wald chi^2 with p-value on the added coefficients. An intercept
    is added automatically and non-finite rows are dropped jointly. Canonical use:
    base_X = [quoted_spread_bps, amihud], add_X = the stress state -> shows the conditioner carries
    signal the conventional proxies cannot. The headline defensibility exhibit."""
    from scipy import stats
    y = np.asarray(y, float).ravel()
    B = np.asarray(base_X, float); B = B.reshape(len(y), -1) if B.size else np.empty((len(y), 0))
    if B.ndim == 2 and B.shape[0] != len(y):
        B = B.T
    A = np.asarray(add_X, float); A = A.reshape(len(y), -1)
    if A.shape[0] != len(y):
        A = A.T
    n0 = len(y); ones = np.ones((n0, 1))
    Xr = np.column_stack([ones, B]); Xf = np.column_stack([ones, B, A])
    ok = np.all(np.isfinite(np.column_stack([y[:, None], Xf])), axis=1)
    y, Xr, Xf, A = y[ok], Xr[ok], Xf[ok], A[ok]; n = len(y)

    def _fit(X):
        b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b; return b, float(u @ u)

    _, rss_r = _fit(Xr); _, rss_f = _fit(Xf)
    tss = float(((y - y.mean()) ** 2).sum()) + EPS
    r2_r, r2_f = 1.0 - rss_r / tss, 1.0 - rss_f / tss
    q = A.shape[1]; kf = Xf.shape[1]
    F = ((rss_r - rss_f) / q) / (rss_f / max(n - kf, 1) + EPS)
    out = {"r2_base": r2_r, "r2_full": r2_f, "partial_r2": r2_f - r2_r,
           "F": float(F), "p_F": float(stats.f.sf(F, q, max(n - kf, 1))), "n_add": q, "n": n}
    if clusters is not None:
        g = np.asarray(clusters)[ok]
        res = inf.cluster_robust_ols(y, Xf, g)
        idx = np.arange(kf - q, kf); bA = res["b"][idx]; VA = res["vcov"][np.ix_(idx, idx)]
        wald = float(bA @ np.linalg.pinv(VA) @ bA)
        out.update({"wald_cluster": wald, "p_wald": float(stats.chi2.sf(wald, q)),
                    "n_clusters": res["n_clusters"]})
    return out


# ── validation 2: the chosen axes are the data-driven axes (FPCA alignment) ───
def fpca_alignment(df, asset, n_levels=10, representation="log"):
    """Do the interpretable curve axes coincide with the dominant DATA-DRIVEN axes of book variation?
    FPCA the depth profile; check that the depth-illiquidity axis aligns strongly with one principal
    component and the hollowness axis with a DIFFERENT one -- i.e. the hand-chosen level and resiliency
    axes ARE principal modes of book variation, converting "I picked these axes" into a robustness
    result. Returns which FPC each axis aligns with, the |corr|, whether they are distinct, the variance
    shares, and the component labels."""
    Y, _ = fl.depth_profile(df, asset, n_levels, representation=representation)
    res = fl.fpca(Y, n_components=3); st = stress_state(df, asset, n_levels)
    sc = res["scores"]; K = min(3, sc.shape[1])

    def _c(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else 0.0

    def _best(axis):
        cs = [abs(_c(sc[:, k], axis)) for k in range(K)]; k = int(np.argmax(cs)); return k, cs[k]

    kd, cd = _best(st["depth_illiq"].to_numpy()); kh, ch = _best(st["hollowness"].to_numpy())
    return {"depth_FPC": kd + 1, "corr_depth": cd, "hollow_FPC": kh + 1, "corr_hollow": ch,
            "distinct": kd != kh, "evr": res["explained_variance_ratio"][:3].tolist(),
            "labels": [d["label"] for d in fl.interpret_components(res)]}


# ── validation 3: the state carries realized impact (lambda check) ────────────
def lambda_validation(realized_impact, stress_level, stress_convexity=None, clusters=None):
    """Validate that the BOOK-derived stress state actually carries high REALIZED (trade-derived)
    cross-impact: regress realized impact on the stress level (and convexity). A positive, significant
    slope confirms the conditioner means what its label claims. The trade-derived impact is the
    VALIDATOR here, never the conditioner -- preserving the book-derived exogeneity of the state.
    Returns coefficients, cluster-robust (or OLS) SEs/t, and R^2."""
    y = np.asarray(realized_impact, float).ravel()
    cols = [np.ones(len(y)), np.asarray(stress_level, float)]; names = ["const", "cost_bps"]
    if stress_convexity is not None:
        cols.append(np.asarray(stress_convexity, float)); names.append("convexity")
    X = np.column_stack(cols)
    ok = np.all(np.isfinite(np.column_stack([y[:, None], X])), axis=1)
    y, X = y[ok], X[ok]
    if clusters is not None:
        res = inf.cluster_robust_ols(y, X, np.asarray(clusters)[ok]); b, se = res["b"], res["se"]
    else:
        b, se = inf.ols_naive_se(y, X)
    u = y - X @ b; tss = float(((y - y.mean()) ** 2).sum()) + EPS
    return {"names": names, "coef": b, "se": se, "t": b / (se + EPS),
            "r2": 1.0 - float(u @ u) / tss, "n": len(y)}


if __name__ == "__main__":
    import test_liquidity_stress  # noqa: F401  (self-test lives in the guard)
