"""
functional_liquidity.py
=======================
Functional-data view of the limit-order book: treat the depth-by-level profile
at each timestamp as a curve and run functional PCA (FPCA). Instead of hand-
picking entropy / slope / center-of-mass, the data hands you the dominant eigen-
modes of book-shape variation; the leading FPC score is a principled, data-driven
liquidity-state variable that drops straight into
cross_asset_pd_liquidity.liquidity_conditional_vecm.

  depth_profile        per-timestamp depth-by-level curve (the functional object)
  fpca                 functional PCA on a curve matrix (general; SVD-based,
                       L2(weights)-orthonormal eigenfunctions, quadrature-aware)
  asset_fpca           FPCA of one asset's book curves over time
  fpc_state_series     one asset's leading FPC score -> scalar 1s state
  relative_fpc_state   FPCA of the cross-asset shape DIFFERENCE curve -> the
                       data-driven analogue of relative_depth_state (drop-in)
  modes_of_variation   classic FDA mean +/- k*sqrt(lambda)*eigenfunction
  project              project new curves onto a fitted basis

Domain is the level index 1..N (common across time and across SPY/ES, and more
comparable than physical distance since tick sizes differ); quadrature defaults
to uniform, so FPCA reduces to PCA on the sampled curves but keeps the functional
interpretation and the machinery to switch to a physical-distance/pct grid.
numpy / pandas only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-12


def _mid(df, asset):
    return ((df[f"{asset}_bidprice_1"] + df[f"{asset}_askprice_1"]) / 2.0).to_numpy()

def _side_sizes(df, asset, side, n):
    return np.column_stack([df[f"{asset}_{side}quantity_{i}"].to_numpy() for i in range(1, n + 1)])


# ── the functional object ─────────────────────────────────────────────────────
def depth_profile(df, asset, n_levels=10, side="both", representation="log"):
    """Per-timestamp depth-by-level curve. side: 'bid'|'ask'|'both'(mean of the two).
    representation: 'log' = log1p(size) (level + shape variation), 'share' = size /
    total (scale-free shape density, comparable across assets), 'raw' = size.
    Returns (Y, x): Y is (T, n_levels), x is the level grid 1..n. Warmup rows with no
    book are treated as empty (zeros)."""
    if side == "both":
        Q = 0.5 * (_side_sizes(df, asset, "bid", n_levels) + _side_sizes(df, asset, "ask", n_levels))
    else:
        Q = _side_sizes(df, asset, side, n_levels)
    Q = np.nan_to_num(np.maximum(Q, 0.0))
    if representation == "log":
        Y = np.log1p(Q)
    elif representation == "share":
        Y = Q / (Q.sum(axis=1, keepdims=True) + EPS)
    elif representation == "raw":
        Y = Q
    else:
        raise ValueError(f"unknown representation {representation!r}")
    return Y, np.arange(1, n_levels + 1, dtype=float)


# ── functional PCA ────────────────────────────────────────────────────────────
def _orient(phi):
    """Deterministic sign: largest-magnitude loading positive (per column)."""
    idx = np.argmax(np.abs(phi), axis=0)
    s = np.sign(phi[idx, np.arange(phi.shape[1])]); s[s == 0] = 1.0
    return s

def fpca(Y, n_components=3, weights=None, center=True):
    """Functional PCA on a curve matrix Y (T x M) with quadrature `weights` (length
    M; default uniform). Eigenfunctions are L2(weights)-orthonormal; scores are the
    functional projections <Yc_t, phi_k>_w (variance = eigenvalue)."""
    Y = np.asarray(Y, float); T, M = Y.shape
    w = np.ones(M) if weights is None else np.asarray(weights, float)
    mu = Y.mean(axis=0) if center else np.zeros(M)
    Yc = Y - mu
    sw = np.sqrt(w)
    U, S, Vt = np.linalg.svd(Yc * sw, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    phi = (Vt[:k].T) / sw[:, None]                      # M x k, L2(w)-orthonormal
    scores = (Yc * w) @ phi                             # T x k functional projections
    sign = _orient(phi)
    phi *= sign; scores *= sign
    eig = (S[:k] ** 2) / max(T - 1, 1)
    total = (S ** 2).sum() / max(T - 1, 1)
    return {"mean": mu, "components": phi, "scores": scores, "eigenvalues": eig,
            "explained_variance_ratio": eig / (total + EPS), "grid": np.arange(1, M + 1.0),
            "weights": w}

def project(Y_new, res):
    return (np.asarray(Y_new, float) - res["mean"]) * res["weights"] @ res["components"]

def modes_of_variation(res, n_sd=2.0):
    out = {}
    for k in range(res["components"].shape[1]):
        amp = n_sd * np.sqrt(max(res["eigenvalues"][k], 0.0))
        out[k + 1] = {"plus": res["mean"] + amp * res["components"][:, k],
                      "minus": res["mean"] - amp * res["components"][:, k]}
    return out


def interpret_components(res, deadzone=0.05):
    """Economic reading of the data-driven liquidity factors: label each FPC by the shape of its
    loading curve across book levels. A loading of constant sign (no zero crossing) is a LEVEL factor
    -- aggregate depth moves up/down at all price levels together; a single sign change is a SLOPE
    factor -- the book tilts/steepens (near-touch depth vs deep depth move oppositely, i.e. resiliency);
    two crossings are CURVATURE. Crossings are counted after zeroing loadings below ``deadzone`` x the
    peak so noise wiggles near zero don't spuriously inflate the count. Returns, per component, its
    label, variance_explained, and the zero-crossing count -- so the FPC liquidity state has a stated
    interpretation rather than being a black-box score."""
    phi = res["components"]; evr = res["explained_variance_ratio"]
    labels = {0: "level", 1: "slope", 2: "curvature"}
    out = []
    for k in range(phi.shape[1]):
        v = phi[:, k]; vv = np.where(np.abs(v) >= deadzone * (np.max(np.abs(v)) + EPS), v, 0.0)
        s = np.sign(vv); s = s[s != 0]
        ncross = int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0
        out.append({"component": k + 1, "label": labels.get(ncross, f"mode-{k + 1}"),
                    "variance_explained": float(evr[k]), "n_sign_changes": ncross})
    return out


# ── book-specific wrappers ────────────────────────────────────────────────────
def asset_fpca(df, asset, n_levels=10, side="both", representation="log", n_components=3):
    Y, x = depth_profile(df, asset, n_levels, side, representation)
    res = fpca(Y, n_components=n_components); res["grid"] = x
    return res

def fpc_state_series(df, asset, n_levels=10, k=1, side="both", representation="log"):
    """One asset's leading (k-th) FPC score: a data-driven scalar liquidity state
    (1s), oriented so higher = more total book depth. Drop-in for the per-asset
    state in liquidity_conditional_vecm.

    FULL-SESSION LOOK-AHEAD, disclosed: the eigenfunctions (and the mean curve) are
    estimated from the WHOLE session before any score is computed, so the state at 10:00
    embeds book-shape modes fitted partly on 15:30 data. That is standard for functional
    DESCRIPTIVE states and harmless for the within-day conditional estimators here (the
    conditioning variable is deterministic given the day), but this state must NOT be used
    as a real-time signal or in any forecasting exercise without re-fitting the FPCA on an
    expanding window. The same applies to relative_fpc_state below."""
    res = asset_fpca(df, asset, n_levels, side, representation, n_components=max(k, 1))
    return res["scores"][:, k - 1]

def relative_fpc_state(df, n_levels=10, k=1, side="both", representation="log",
                       assets=("SPY", "ES")):
    """FPCA of the cross-asset book DIFFERENCE curve (assets[1] - assets[0]) over time;
    returns the leading FPC score (1s) as a data-driven relative liquidity state -- the
    functional analogue of relative_depth_state. representation='log' captures relative
    depth level + shape (reduces to ~relative_depth_state when only the level moves);
    'share' isolates pure relative shape. Oriented so positive ~ assets[1] (ES) book
    relatively deeper/heavier."""
    a0, a1 = assets
    Y0, _ = depth_profile(df, a0, n_levels, side, representation)
    Y1, x = depth_profile(df, a1, n_levels, side, representation)
    res = fpca(Y1 - Y0, n_components=max(k, 1)); res["grid"] = x
    s = res["scores"][:, k - 1]
    ref = (Y1 - Y0).mean(axis=1)                        # overall relative depth
    if np.corrcoef(s, ref)[0, 1] < 0:
        s = -s
    return s


# ── self-test ─────────────────────────────────────────────────────────────────
def _selftest():
    rng = np.random.default_rng(0)

    # (1) FPCA recovers a known latent 1-D shape mode.
    T, N = 4000, 10
    g = np.arange(1, N + 1)
    mu0 = 5.0 * np.exp(-0.25 * (g - 1))
    mode = np.sin(g / N * np.pi); mode /= np.linalg.norm(mode)
    s_true = np.cumsum(rng.normal(0, 0.1, T)); s_true = (s_true - s_true.mean()) / (s_true.std() + EPS)
    Y = mu0[None, :] + np.outer(s_true, mode) + rng.normal(0, 0.05, (T, N))
    r = fpca(Y, n_components=3)
    evr1 = r["explained_variance_ratio"][0]
    cs = abs(np.corrcoef(r["scores"][:, 0], s_true)[0, 1])
    cf = abs(np.corrcoef(r["components"][:, 0], mode)[0, 1])
    print("(1) FPCA mode recovery:")
    print("    FPC1 variance share = %.3f | corr(score,latent)=%.3f | corr(eigfn,mode)=%.3f" % (evr1, cs, cf))
    print("    checks:", evr1 > 0.8 and cs > 0.95 and cf > 0.95)

    # synthetic book-frame generator (leadership and depth level both driven by s_t)
    def gen(n=4000, NL=10, gain=0.5):
        s = np.zeros(n)
        for t in range(1, n):
            s[t] = 0.99 * s[t - 1] + rng.normal(0, 0.12)
        f = np.cumsum(rng.normal(0, 0.0008, n))
        a_spy = np.clip(0.05 * np.exp(+gain * s), 0.002, 0.45)
        a_es = np.clip(0.03 * np.exp(-gain * s), 0.002, 0.45)
        L = np.linalg.cholesky([[1, .3], [.3, 1]]); e = rng.normal(0, 0.01, (n, 2)) @ L.T
        lp1 = np.zeros(n); lp2 = np.zeros(n)
        for t in range(1, n):
            z = lp1[t - 1] - lp2[t - 1]
            lp1[t] = lp1[t - 1] + (f[t] - f[t - 1]) - a_spy[t] * z + e[t, 0]
            lp2[t] = lp2[t - 1] + (f[t] - f[t - 1]) + a_es[t] * z + e[t, 1]
        ms, me = 500 * np.exp(lp1), 5000 * np.exp(lp2)
        lev = np.arange(1, NL + 1); dw = np.exp(-0.3 * (lev - 1)); cols = {}
        for a, mid, tick in (("SPY", ms, 0.01), ("ES", me, 0.25)):
            depth = 300.0 * np.exp((0.5 if a == "ES" else -0.5) * s)
            for i, l in enumerate(lev):
                q = np.maximum(depth * dw[i] * (1 + rng.normal(0, 0.05, n)), 1.0)
                cols[f"{a}_bidprice_{l}"] = mid - l * tick; cols[f"{a}_askprice_{l}"] = mid + l * tick
                cols[f"{a}_bidquantity_{l}"] = q; cols[f"{a}_askquantity_{l}"] = q * (1 + rng.normal(0, 0.02, n))
        idx = pd.date_range("2024-08-05 09:30", periods=n, freq="s", tz="America/New_York")
        return pd.DataFrame(cols, index=idx)

    df = gen()

    # (2) relative FPC state aligns with the existing relative_depth_state.
    import cross_asset_pd_liquidity as ca
    s_fpc = relative_fpc_state(df, n_levels=10, representation="log")
    s_old = ca.relative_depth_state(df, 10)
    c = np.corrcoef(s_fpc, s_old)[0, 1]
    print("\n(2) relative FPC state vs relative_depth_state: corr = %.3f" % c)
    print("    check (data-driven state recovers the depth-level state):", abs(c) > 0.9)

    # (3) end-to-end: FPC state drives the liquidity-conditional VECM.
    lc = ca.liquidity_conditional_vecm(_mid(df, "SPY"), _mid(df, "ES"), s_fpc, n_lags=5)
    sb = lc["shares_by_state"]
    spread = sb[0.9]["CS_ES"] - sb[0.1]["CS_ES"]
    print("\n(3) liquidity_conditional_vecm with the FPC state:")
    print("    CS_ES @ state q10/q50/q90 = %.3f / %.3f / %.3f | t(delta_ES)=%.2f"
          % (sb[0.1]["CS_ES"], sb[0.5]["CS_ES"], sb[0.9]["CS_ES"], lc["t_delta_es"]))
    print("    check (runs; ES leads more as its book deepens):", np.isfinite(lc["t_delta_es"]) and spread > 0)

if __name__ == "__main__":
    _selftest()
