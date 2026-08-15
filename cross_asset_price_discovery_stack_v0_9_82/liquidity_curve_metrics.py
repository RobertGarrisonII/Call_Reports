"""
liquidity_curve_metrics.py
==========================
Shape / magnitude functionals of the limit-order-book depth curve, computed
per (asset, side, time_bucket). Designed to sit downstream of the SPY/ES
aggregated query once it emits the per-bucket snapshot book profile
(bidprice_i / bidquantity_i / askprice_i / askquantity_i, i = 1..N).

Two conventions throughout:
  * distance is measured from mid; pass unit='pct' for cross-asset work
    (default) or unit='ticks' with tick_size for within-asset work.
  * the "curve" is the marginal depth profile q_i over levels i=1..N.
    Cumulative variants (Q_i = cumsum q) are used where the supply curve
    is the natural object (arc length, slope).

All frame-level functions are vectorized over time buckets (axis=1 over levels).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# Vector transforms (T x N matrices; NaN = absent level, treated as 0 depth)
# ─────────────────────────────────────────────────────────────────────────────
def _clean(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return np.where(np.isfinite(M), M, 0.0)

def _usable_levels(P: np.ndarray, Q: np.ndarray):
    """A book level counts only if BOTH its price and size are finite. A shift / torn read can
    leave one leg NaN (price present, size missing, or vice versa); such a level is not a real
    tradeable level, so we set it to (NaN price, 0 size) -- excluded from price aggregates AND
    carrying no depth, i.e. treated as absent, consistently across every reader. This keeps the
    numerator and denominator of any size-weighted price aggregate aligned (otherwise a present
    size with a NaN price inflates the denominator and biases the aggregate)."""
    P = np.asarray(P, dtype=float); Q = np.asarray(Q, dtype=float)
    ok = np.isfinite(P) & np.isfinite(Q)
    return np.where(ok, P, np.nan), np.where(ok, Q, 0.0)

def l2_normalize(q: np.ndarray) -> np.ndarray:
    """Unit-L2 depth vectors: q_i / ||q||_2. Removes magnitude, keeps shape."""
    q = _clean(q)
    n = np.sqrt(np.sum(q * q, axis=1, keepdims=True))
    return np.divide(q, n, out=np.zeros_like(q), where=n > EPS)

def minmax_scale(q: np.ndarray) -> np.ndarray:
    """Per-curve min-max to [0,1] across levels."""
    q = _clean(q)
    lo = q.min(axis=1, keepdims=True)
    rng = q.max(axis=1, keepdims=True) - lo
    return np.divide(q - lo, rng, out=np.zeros_like(q), where=rng > EPS)

def share_of_book(q: np.ndarray) -> np.ndarray:
    """p_i = q_i / sum_j q_j  (the depth *distribution*)."""
    q = _clean(q)
    s = q.sum(axis=1, keepdims=True)
    return np.divide(q, s, out=np.zeros_like(q), where=s > EPS)

# ─────────────────────────────────────────────────────────────────────────────
# Scalar functionals (return shape (T,))
# ─────────────────────────────────────────────────────────────────────────────
def _arc_length(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    dx = np.diff(X, axis=1)
    dy = np.diff(Y, axis=1)
    return np.sqrt(dx * dx + dy * dy).sum(axis=1)

def _trapz_axis1(Y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Trapezoidal integral of Y (T x M) over the shared grid x (M,), along axis 1.
    Portable across numpy versions (np.trapz was removed in numpy 2.x)."""
    dx = np.diff(np.asarray(x, float))
    return np.sum(0.5 * (Y[:, 1:] + Y[:, :-1]) * dx[None, :], axis=1)

def arc_length_normalized(dist: np.ndarray, q: np.ndarray, cumulative=True) -> np.ndarray:
    """
    Arc length of the depth curve after min-max scaling BOTH axes to [0,1].
    Scaling both axes is what makes the length cross-asset comparable: it is a
    pure shape descriptor (convexity / spread of liquidity across price space),
    invariant to tick size, contract size, and total depth. Raw arc length on
    (pct-distance, shares) is dominated by the size axis and not comparable.
    """
    X = minmax_scale(_clean(dist))
    Y = np.cumsum(_clean(q), axis=1) if cumulative else _clean(q)
    Y = minmax_scale(Y)
    return _arc_length(X, Y)

def entropy(q: np.ndarray, normalized=True) -> np.ndarray:
    """Shannon entropy of the depth distribution. High = liquidity spread
    evenly across levels; low = concentrated. Scale-free → cross-asset safe."""
    p = share_of_book(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(p > EPS, p * np.log(p), 0.0), axis=1)
    if normalized:
        n_active = (p > EPS).sum(axis=1)
        denom = np.log(np.maximum(n_active, 1))
        h = np.divide(h, denom, out=np.zeros_like(h), where=denom > EPS)
    return h

def gini(q: np.ndarray) -> np.ndarray:
    """Gini concentration of depth across levels (0 = uniform, →1 = all in one)."""
    q = np.sort(_clean(q), axis=1)
    T, N = q.shape
    idx = np.arange(1, N + 1)[None, :]
    s = q.sum(axis=1)
    g = (2.0 * (idx * q).sum(axis=1)) / (N * s + EPS) - (N + 1.0) / N
    return np.where(s > EPS, g, 0.0)

def center_of_mass(dist: np.ndarray, q: np.ndarray) -> np.ndarray:
    """First moment: depth-weighted mean distance from mid. How FAR liquidity sits."""
    d, p = _clean(dist), share_of_book(q)
    return (d * p).sum(axis=1)

def dispersion(dist: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Sqrt of depth-weighted variance of distance. How SPREAD liquidity is."""
    d, p = _clean(dist), share_of_book(q)
    com = (d * p).sum(axis=1, keepdims=True)
    return np.sqrt((p * (d - com) ** 2).sum(axis=1))

def depth_slope(dist: np.ndarray, q: np.ndarray) -> np.ndarray:
    """OLS slope of cumulative depth Q_i on distance d_i: shares added per unit
    distance. Larger = book fills up fast near the touch (more liquid)."""
    d = _clean(dist)
    Q = np.cumsum(_clean(q), axis=1)
    dm = d - d.mean(axis=1, keepdims=True)
    Qm = Q - Q.mean(axis=1, keepdims=True)
    num = (dm * Qm).sum(axis=1)
    den = (dm * dm).sum(axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > EPS)

def slippage_to_fill(dist: np.ndarray, q: np.ndarray, target: float) -> np.ndarray:
    """Size-weighted avg distance from mid to fill `target` units (marginal
    execution cost / walk-the-book slippage). NaN where book can't fill target."""
    d, qq = _clean(dist), _clean(q)
    cum = np.cumsum(qq, axis=1)
    fillable = cum[:, -1] >= target
    # remaining size we still take at each level, capped so the cumulative = target
    prev = np.concatenate([np.zeros((qq.shape[0], 1)), cum[:, :-1]], axis=1)
    take = np.clip(target - prev, 0.0, qq)
    cost = (take * d).sum(axis=1) / target
    return np.where(fillable, cost, np.nan)

# ─────────────────────────────────────────────────────────────────────────────
# Near-book functionals: the SAME curve metrics, but on the band from the touch
# (level 1) to the WEIGHTED price -- the economically active near book. The
# weighted price is the depth-weighted average distance (== center_of_mass), or a
# fill-VWAP endpoint for a target size. The window is endogenous (it is built from
# the same depths), so the shape metrics are normalized and the band span is
# reported as its own scalar. The near book is the part withdrawn first under
# stress, so these are more stress-responsive than the full N-level versions.
# ─────────────────────────────────────────────────────────────────────────────
def _band_endpoint(dist: np.ndarray, q: np.ndarray, mode="com", target=None) -> np.ndarray:
    """Outer distance of the near-book band. mode='com' -> the weighted (center-of-mass)
    price distance; mode='fill' -> the distance at which cumulative depth reaches
    `target` (the fill-VWAP boundary), NaN if the book can't fill it."""
    if mode == "com":
        return center_of_mass(dist, q)
    if mode != "fill" or target is None:
        raise ValueError("mode must be 'com', or 'fill' with a target size")
    d = _clean(dist); cum = np.cumsum(_clean(q), axis=1)
    T, N = cum.shape; ar = np.arange(T)
    idx = (cum < target).sum(axis=1)                       # first level reaching target
    fillable = idx < N
    hi = np.clip(idx, 1, N - 1); lo = hi - 1
    c0, c1 = cum[ar, lo], cum[ar, hi]; d0, d1 = d[ar, lo], d[ar, hi]
    w = np.clip(np.nan_to_num(np.divide(target - c0, np.where((c1 - c0) > EPS, c1 - c0, np.nan)),
                              nan=0.0), 0.0, 1.0)
    return np.where(fillable, d0 + w * (d1 - d0), np.nan)


def _interp_cum_on_band(dist: np.ndarray, q: np.ndarray, endpoint: np.ndarray, grid: int):
    """Linearly interpolate the cumulative-depth curve onto `grid` equally-spaced points
    spanning [touch, endpoint] per row. Returns (f in [0,1], Q_on_grid[T,grid], span[T],
    n_in_band[T] = number of discrete levels at or inside the endpoint). Interpolating onto
    a fixed grid keeps the shape metrics stable when only a few discrete levels fall inside
    the band; n_in_band flags rows with too little structure to have a shape."""
    d = _clean(dist); Q = np.cumsum(_clean(q), axis=1)
    d1 = d[:, 0]; e = np.asarray(endpoint, float); span = e - d1
    n_in_band = (d <= e[:, None] + EPS).sum(axis=1)
    f = np.linspace(0.0, 1.0, grid); T, N = d.shape; ar = np.arange(T)
    Qq = np.empty((T, grid))
    for m in range(grid):
        xq = d1 + f[m] * span
        j = np.clip((d < xq[:, None]).sum(axis=1) - 1, 0, N - 2)
        x0, x1 = d[ar, j], d[ar, j + 1]; Q0, Q1 = Q[ar, j], Q[ar, j + 1]
        w = np.clip(np.nan_to_num(np.divide(xq - x0, np.where((x1 - x0) > EPS, x1 - x0, np.nan)),
                                  nan=0.0), 0.0, 1.0)
        Qq[:, m] = Q0 + w * (Q1 - Q0)
    return f, Qq, span, n_in_band


def near_book_span(dist: np.ndarray, q: np.ndarray, mode="com", target=None) -> np.ndarray:
    """Distance from the touch to the weighted (or fill) price. Small = mass packed at the
    inside (liquid / concentrated near book); large = thin near book. Its bid-vs-ask
    asymmetry is near-book directional pressure."""
    return _band_endpoint(dist, q, mode, target) - _clean(dist)[:, 0]


def near_book_auc(dist: np.ndarray, q: np.ndarray, mode="com", target=None, grid=21,
                  kind="shape") -> np.ndarray:
    """Area under the cumulative-depth curve over the near band. kind='shape' (default)
    min-max scales the curve to the unit box -> a dimensionless near-book convexity in
    [0,1] (high = depth builds fast just past the touch); kind='depth' -> the band-average
    cumulative depth (shares), a near-book liquidity magnitude. NaN where the span is ~0."""
    f, Qq, span, n_in_band = _interp_cum_on_band(dist, q, _band_endpoint(dist, q, mode, target), grid)
    if kind == "depth":
        out = _trapz_axis1(Qq, f)                           # f-average of cumulative depth
        return np.where(span > EPS, out, np.nan)
    lo = Qq.min(axis=1, keepdims=True); rng = Qq.max(axis=1, keepdims=True) - lo
    Yn = np.divide(Qq - lo, rng, out=np.zeros_like(Qq), where=rng > EPS)
    out = _trapz_axis1(Yn, f)                               # shape: needs >=2 levels for curvature
    return np.where((span > EPS) & (n_in_band >= 2), out, np.nan)


def near_book_arc_length(dist: np.ndarray, q: np.ndarray, mode="com", target=None,
                         grid=21) -> np.ndarray:
    """Arc length (||L||) of the near-book cumulative curve, both axes min-max scaled to
    [0,1] -> a scale-free near-book shape descriptor (>= sqrt(2); larger = more convex /
    kinked). NaN where the span is ~0."""
    f, Qq, span, n_in_band = _interp_cum_on_band(dist, q, _band_endpoint(dist, q, mode, target), grid)
    lo = Qq.min(axis=1, keepdims=True); rng = Qq.max(axis=1, keepdims=True) - lo
    Yn = np.divide(Qq - lo, rng, out=np.zeros_like(Qq), where=rng > EPS)
    X = np.broadcast_to(f, Qq.shape)
    return np.where((span > EPS) & (n_in_band >= 2), _arc_length(X, Yn), np.nan)


def near_book_metrics(df, asset, side, n_levels, unit="pct", tick_size=None,
                      mode="com", target=None, grid=21):
    """Near-book curve features for one (asset, side): span, AUC (shape + depth), and
    arc length over [touch, weighted price]. mode='com' uses the weighted price; 'fill'
    uses the fill-VWAP boundary for `target` size."""
    dist, q, _ = extract_book(df, asset, side, n_levels, unit, tick_size)
    out = {
        "nb_span":      near_book_span(dist, q, mode, target),
        "nb_auc_shape": near_book_auc(dist, q, mode, target, grid, "shape"),
        "nb_auc_depth": near_book_auc(dist, q, mode, target, grid, "depth"),
        "nb_arc_len":   near_book_arc_length(dist, q, mode, target, grid),
    }
    return pd.DataFrame(out, index=df.index).add_prefix(f"{asset}_{side}_")

# ─────────────────────────────────────────────────────────────────────────────
# Frame API
# ─────────────────────────────────────────────────────────────────────────────
def extract_book(df: pd.DataFrame, asset: str, side: str, n_levels: int,
                 unit: str = "pct", tick_size: float | None = None):
    """
    Pull (distance, size) matrices for one side from the wide output frame.
    Expects columns like f'{asset}_{side}price_i', f'{asset}_{side}quantity_i',
    plus level-1 of both sides to form mid. Returns (dist[T,N], size[T,N], mid[T]).
    """
    px = np.column_stack([df[f"{asset}_{side}price_{i}"].to_numpy()    for i in range(1, n_levels + 1)])
    qty = np.column_stack([df[f"{asset}_{side}quantity_{i}"].to_numpy() for i in range(1, n_levels + 1)])
    px, qty = _usable_levels(px, qty)                       # a level counts only if price AND size are finite
    bid1 = df[f"{asset}_bidprice_1"].to_numpy()
    ask1 = df[f"{asset}_askprice_1"].to_numpy()
    mid = (bid1 + ask1) / 2.0
    raw = np.abs(px - mid[:, None])
    if unit == "pct":
        dist = raw / np.where(mid[:, None] > EPS, mid[:, None], np.nan)
    elif unit == "ticks":
        if tick_size is None:
            raise ValueError("unit='ticks' requires tick_size")
        dist = raw / tick_size
    elif unit == "dollar":
        dist = raw
    else:
        raise ValueError(f"unknown unit {unit!r}")
    return dist, qty, mid

def weighted_mid(df, asset):
    """Size-weighted mid (microprice): ask weighted by bid size and vice-versa, so it leans
    toward the side with less depth. Falls back to the simple mid when inside depth is absent."""
    a1 = df[f"{asset}_askprice_1"].to_numpy(float); b1 = df[f"{asset}_bidprice_1"].to_numpy(float)
    qa = df[f"{asset}_askquantity_1"].to_numpy(float); qb = df[f"{asset}_bidquantity_1"].to_numpy(float)
    w = qa + qb
    return np.where(w > EPS, (a1 * qb + b1 * qa) / w, (a1 + b1) / 2.0)

def depth_weighted_mid(df, asset, n_levels=10):
    """Depth-weighted mid across n_levels -- the microprice generalized from the touch to the
    cumulative book. Per side take the size-weighted average price to n_levels, then cross-weight
    by the OPPOSITE side's total size (same lean logic as the microprice). Reduces EXACTLY to
    weighted_mid at n_levels=1. Smoother than the inside microprice and reflects deeper resting
    liquidity, but no longer the one-step-ahead-of-the-mid predictor the inside microprice is."""
    def _side(side):
        P = np.column_stack([df[f"{asset}_{side}price_{i}"].to_numpy(float) for i in range(1, n_levels + 1)])
        Q = np.column_stack([df[f"{asset}_{side}quantity_{i}"].to_numpy(float) for i in range(1, n_levels + 1)])
        P, Q = _usable_levels(P, Q)                        # exclude torn levels from BOTH price and size
        tot = Q.sum(axis=1)
        pbar = np.divide(np.nansum(P * Q, axis=1), tot, out=np.full(tot.shape, np.nan), where=tot > EPS)
        return pbar, tot
    A, Qa = _side("ask"); B, Qb = _side("bid")
    w = Qa + Qb
    out = np.where(w > EPS, (A * Qb + B * Qa) / w, (A + B) / 2.0)
    return np.where(np.isfinite(out), out, weighted_mid(df, asset))     # fall back to inside microprice


def _cost_curve(df, asset, side, n_levels, anchor="wmid"):
    """Marginal per-level cost in bps from `anchor` (weighted mid by default; 'mid' for the
    simple mid) with cumulative quantity Q and notional V. Cost is signed so deeper = larger
    and >=0 on each side. Returns (cost_bps[T,N], q[T,N], Q[T,N], V[T,N]); absent levels carried."""
    P = np.column_stack([df[f"{asset}_{side}price_{i}"].to_numpy(float)    for i in range(1, n_levels + 1)])
    Qraw = np.column_stack([df[f"{asset}_{side}quantity_{i}"].to_numpy(float) for i in range(1, n_levels + 1)])
    P, q = _usable_levels(P, Qraw)                          # level counts only if price AND size finite
    if anchor in (None, "wmid"):
        m = weighted_mid(df, asset)
    elif anchor == "depth_wmid":
        m = depth_weighted_mid(df, asset, n_levels)        # depth-weighted reference (uses n_levels)
    else:                                                  # "mid": simple quote midpoint
        m = (df[f"{asset}_bidprice_1"].to_numpy(float) + df[f"{asset}_askprice_1"].to_numpy(float)) / 2.0
    m = np.where(m > EPS, m, np.nan)[:, None]
    sign = 1.0 if side == "ask" else -1.0
    cost = sign * (P / m - 1.0) * 1e4                       # bps, >=0 (a_i>=m_w>=b_i)
    cost = np.where(np.isfinite(cost), cost, np.nan)
    return cost, q, np.cumsum(q, axis=1), np.cumsum(_clean(P) * q, axis=1)

def decay_weighted_cost(df, asset, n_levels=10, decay_scale=None, decay_levels=5.0,
                        side="both", anchor="wmid"):
    """Decay-weighted average cost-to-demand liquidity (bps). Each level's marginal cost is
    weighted by the exponential mass of an order whose size ~ Exp(Q0): w_i = e^{-Q_{i-1}/Q0} -
    e^{-Q_i/Q0}, so deeper levels are smoothly downweighted and NO hard fill-size cutoff is
    imposed -- the scale-free analogue of cost-to-fill at a fixed quantity. Q0 = decay_scale if
    given, else decay_levels x the inside size. side='both' averages the buy(ask)/sell(bid)
    legs. anchor in {'wmid' (inside microprice, default), 'mid' (simple), 'depth_wmid' (the
    depth-weighted mid across n_levels)}. HIGHER = more illiquid. Returns (T,).
    Note: with anchor='depth_wmid' the reference can sit outside the touch when the book is
    imbalanced, so near-touch marginal cost may be negative (cost relative to deep fair value)."""
    sides = ("ask", "bid") if side == "both" else (side,)
    out = []
    for sd in sides:
        cost, q, Q, _ = _cost_curve(df, asset, sd, n_levels, anchor)
        Qprev = Q - q
        if decay_scale:
            Q0 = float(decay_scale)
        else:
            inside = q[:, :1]
            Q0 = decay_levels * np.where(inside > EPS, inside, np.nan)
        Q0 = np.where(np.asarray(Q0, float) > EPS, Q0, np.nan)
        w = np.exp(-Qprev / Q0) - np.exp(-Q / Q0)           # mass in each level's quantity band
        w = np.where(np.isfinite(w) & np.isfinite(cost), w, 0.0)
        num = np.sum(w * np.where(np.isfinite(cost), cost, 0.0), axis=1)
        den = np.sum(w, axis=1)
        out.append(np.divide(num, den, out=np.full(num.shape[0], np.nan), where=den > EPS))
    return np.nanmean(np.column_stack(out), axis=1)

def impact_curve_length(df, asset, n_levels=10, side="both", anchor="wmid", normalize=True):
    """Arc length ||L|| of the price-impact supply curve -- cumulative quantity (x) vs marginal
    cost in bps (y) -- from the anchor price (origin) outward across n_levels. anchor in
    {'wmid' (default), 'mid', 'depth_wmid'} (see decay_weighted_cost). With normalize=True
    both axes are min-max scaled per bar, so ||L|| in [sqrt(2), 2] is a dimensionless CONVEXITY
    index of price impact: cross-asset comparable and orthogonal to the cost LEVEL (which
    decay_weighted_cost captures). HIGHER = more convex / more nonlinear impact. side='both'
    averages bid and ask. Returns (T,)."""
    sides = ("ask", "bid") if side == "both" else (side,)
    out = []
    for sd in sides:
        cost, q, Q, _ = _cost_curve(df, asset, sd, n_levels, anchor)
        cc = np.where(np.isfinite(cost), cost, 0.0)
        X = np.concatenate([np.zeros((Q.shape[0], 1)), Q], axis=1)        # prepend origin (0,0)
        Y = np.concatenate([np.zeros((cc.shape[0], 1)), cc], axis=1)
        if normalize:
            X = minmax_scale(X); Y = minmax_scale(Y)
        out.append(_arc_length(X, Y))
    return np.nanmean(np.column_stack(out), axis=1)


def side_curve_metrics(df, asset, side, n_levels, unit="pct", tick_size=None,
                       fill_targets=()):
    """Scalar curve features for one (asset, side), indexed like df."""
    dist, q, _ = extract_book(df, asset, side, n_levels, unit, tick_size)
    out = {
        "total_depth":   _clean(q).sum(axis=1),            # L1 norm
        "l2_norm":       np.sqrt((_clean(q) ** 2).sum(1)),  # ||q||_2
        "linf_norm":     _clean(q).max(axis=1),             # top-level size
        "center_of_mass": center_of_mass(dist, q),
        "dispersion":    dispersion(dist, q),
        "entropy_norm":  entropy(q, normalized=True),
        "gini":          gini(q),
        "depth_slope":   depth_slope(dist, q),
        "arc_len_shape": arc_length_normalized(dist, q, cumulative=True),
    }
    for t in fill_targets:
        out[f"slippage_fill_{int(t)}"] = slippage_to_fill(dist, q, t)
    return pd.DataFrame(out, index=df.index).add_prefix(f"{asset}_{side}_")

def normalized_curves(df, asset, side, n_levels):
    """Return the per-level normalized depth vectors as wide frames."""
    _, q, _ = extract_book(df, asset, side, n_levels)
    cols = [f"{asset}_{side}_L{i}" for i in range(1, n_levels + 1)]
    mk = lambda M, suf: pd.DataFrame(M, index=df.index,
                                     columns=[c + suf for c in cols])
    return {
        "l2":      mk(l2_normalize(q), "_l2n"),
        "minmax":  mk(minmax_scale(q), "_mm"),
        "share":   mk(share_of_book(q), "_shr"),
    }

# ── cross-sectional comparisons ──────────────────────────────────────────────
def _row_cosine(A, B):
    A, B = _clean(A), _clean(B)
    num = (A * B).sum(1)
    den = np.sqrt((A * A).sum(1)) * np.sqrt((B * B).sum(1))
    return np.divide(num, den, out=np.zeros_like(num), where=den > EPS)

def book_asymmetry(df, asset, n_levels, unit="pct", tick_size=None):
    """Bid-vs-ask shape comparison within an asset (per bucket)."""
    db, qb, _ = extract_book(df, asset, "bid", n_levels, unit, tick_size)
    da, qa, _ = extract_book(df, asset, "ask", n_levels, unit, tick_size)
    return pd.DataFrame({
        "shape_cosine_bid_ask": _row_cosine(share_of_book(qb), share_of_book(qa)),
        "depth_imbalance":      (_clean(qb).sum(1) - _clean(qa).sum(1)) /
                                (_clean(qb).sum(1) + _clean(qa).sum(1) + EPS),
        "com_skew":             center_of_mass(da, qa) - center_of_mass(db, qb),
    }, index=df.index).add_prefix(f"{asset}_")

def cross_asset_curve_similarity(df, asset_a, asset_b, side, n_levels):
    """SPY-vs-ES liquidity-shape comovement per bucket (your tandem-trade angle)."""
    _, qa, _ = extract_book(df, asset_a, side, n_levels)
    _, qb, _ = extract_book(df, asset_b, side, n_levels)
    sa, sb = share_of_book(qa), share_of_book(qb)
    da = _clean(qa).sum(1); db = _clean(qb).sum(1)
    return pd.DataFrame({
        f"{side}_shape_cosine":  _row_cosine(sa, sb),
        f"{side}_depth_log_ratio": np.log((da + EPS) / (db + EPS)),
    }, index=df.index).add_prefix(f"{asset_a}_{asset_b}_")


def _selftest():
    rng = np.random.default_rng(0)
    T, N = 5, 10
    idx = pd.date_range("2025-04-03 09:30", periods=T, freq="s", tz="America/New_York")
    cols = {}
    mid = 500.0
    for a, tick in [("SPY", 0.01), ("ES", 0.25)]:
        for i in range(1, N + 1):
            cols[f"{a}_bidprice_{i}"] = mid - i * tick
            cols[f"{a}_askprice_{i}"] = mid + i * tick
            cols[f"{a}_bidquantity_{i}"] = rng.gamma(2, 50, T)
            cols[f"{a}_askquantity_{i}"] = rng.gamma(2, 50, T)
    df = pd.DataFrame(cols, index=idx)
    feats = pd.concat([
        side_curve_metrics(df, "SPY", "bid", N, fill_targets=(500, 2000)),
        side_curve_metrics(df, "SPY", "ask", N),
        book_asymmetry(df, "SPY", N),
        cross_asset_curve_similarity(df, "SPY", "ES", "bid", N),
    ], axis=1)
    ncv = normalized_curves(df, "SPY", "bid", N)
    print("scalar feature columns:", list(feats.columns))
    print("\nsample row:\n", feats.iloc[0].round(4).to_string())
    print("\nL2-normalized bid curve row 0 (||.||=%.4f):"
          % np.linalg.norm(ncv["l2"].iloc[0].to_numpy()),
          ncv["l2"].iloc[0].round(3).to_numpy())
    print("OK — no NaNs in scalars:", not feats.drop(columns=[c for c in feats if 'slippage' in c]).isna().any().any())

    # near-book (touch -> weighted price) metrics: front-loaded vs back-loaded books
    Tn, Nn = 200, 10
    didx = pd.date_range("2025-04-03 09:30", periods=Tn, freq="s", tz="America/New_York")
    levels = np.arange(1, Nn + 1)
    front = np.tile(100.0 * np.exp(-0.35 * (levels - 1)), (Tn, 1))    # decaying: mass near touch, COM spans >=2 levels
    back = np.tile(20.0 * levels, (Tn, 1))                           # increasing: mass far out
    conc = np.tile(np.r_[5000.0, np.ones(Nn - 1)], (Tn, 1))          # degenerate: all mass at level 1
    def _frame(qmat):
        c = {}
        for i in range(1, Nn + 1):
            c[f"X_bidprice_{i}"] = 500.0 - i * 0.01; c[f"X_askprice_{i}"] = 500.0 + i * 0.01
            c[f"X_bidquantity_{i}"] = qmat[:, i - 1]; c[f"X_askquantity_{i}"] = qmat[:, i - 1]
        return pd.DataFrame(c, index=didx)
    sFr = near_book_metrics(_frame(front), "X", "bid", Nn).iloc[0]
    sBk = near_book_metrics(_frame(back), "X", "bid", Nn).iloc[0]
    sCc = near_book_metrics(_frame(conc), "X", "bid", Nn).iloc[0]
    fillv = near_book_metrics(_frame(back), "X", "bid", Nn, mode="fill", target=300.0).iloc[0]
    print("\nnear-book metrics (band = touch -> weighted price)")
    print("    front-loaded book: span=%.3e  auc_shape=%.3f  arc_len=%.3f"
          % (sFr["X_bid_nb_span"], sFr["X_bid_nb_auc_shape"], sFr["X_bid_nb_arc_len"]))
    print("    back-loaded  book: span=%.3e  auc_shape=%.3f  arc_len=%.3f"
          % (sBk["X_bid_nb_span"], sBk["X_bid_nb_auc_shape"], sBk["X_bid_nb_arc_len"]))
    print("    concentrated book: span=%.3e  auc_shape=%s (NaN: <2 levels in band, no curvature)"
          % (sCc["X_bid_nb_span"], f"{sCc['X_bid_nb_auc_shape']:.3f}" if np.isfinite(sCc["X_bid_nb_auc_shape"]) else "nan"))
    print("    fill-VWAP endpoint (target=300): span=%.3e  arc_len=%.3f"
          % (fillv["X_bid_nb_span"], fillv["X_bid_nb_arc_len"]))
    print("    checks:",
          sFr["X_bid_nb_span"] < sBk["X_bid_nb_span"]                 # decaying => mass near touch => smaller span
          and sFr["X_bid_nb_auc_shape"] > sBk["X_bid_nb_auc_shape"]   # front-loaded cumulative is concave-above-diagonal
          and sFr["X_bid_nb_auc_shape"] > 0.5 and sBk["X_bid_nb_auc_shape"] < 0.5
          and np.isnan(sCc["X_bid_nb_auc_shape"])                     # degenerate corner -> NaN shape
          and np.isfinite(fillv["X_bid_nb_arc_len"]))

    # impact-curve OUTCOMES: decay-weighted cost-to-demand (level) + arc length ||L|| (convexity)
    def _bk(tick=0.01, decay=0.0, withdraw=1.0, base=500.0, T=20, n=10, inside=300.0):
        idx = pd.date_range("2020-03-12 10:00", periods=T, freq="s", tz="UTC"); d = {}
        for sd, s in (("ask", 1), ("bid", -1)):
            for i in range(1, n + 1):
                d[f"Z_{sd}price_{i}"] = base + s * tick * i
                d[f"Z_{sd}quantity_{i}"] = inside * np.exp(-decay * (i - 1)) * (withdraw if i == 1 else 1.0)
        return pd.DataFrame(d, index=idx)
    M = lambda f: float(np.nanmean(f))
    auc_narrow = M(decay_weighted_cost(_bk(tick=0.01), "Z", 10))
    auc_wide = M(decay_weighted_cost(_bk(tick=0.05), "Z", 10))                 # wider ladder -> costlier
    auc_full = M(decay_weighted_cost(_bk(withdraw=1.0), "Z", 10, decay_scale=1500))
    auc_wd = M(decay_weighted_cost(_bk(withdraw=0.3), "Z", 10, decay_scale=1500))  # inside withdrawn -> costlier
    L_lin = M(impact_curve_length(_bk(decay=0.0), "Z", 10))                   # linear book -> sqrt(2)
    L_conv = M(impact_curve_length(_bk(decay=0.8), "Z", 10))                  # concentrated -> more convex
    print("\nimpact-curve outcomes (cost level + convexity)")
    print("    AUC bps: ladder x5 %.3f -> %.3f ; inside withdrawn %.3f -> %.3f" % (auc_narrow, auc_wide, auc_full, auc_wd))
    print("    ||L||: linear=%.4f (sqrt2=%.4f), convex=%.4f" % (L_lin, 2 ** 0.5, L_conv))
    print("    checks:",
          auc_wide > auc_narrow and auc_wd > auc_full
          and abs(L_lin - 2 ** 0.5) < 1e-6 and L_conv > L_lin and 2 ** 0.5 <= L_conv <= 2.0 + 1e-9)

    # depth-weighted mid: nests the inside microprice at n=1; lies between the per-side DWAPs
    dfb = _bk(tick=0.02, decay=0.5, n=10)
    wm1 = weighted_mid(dfb, "Z")
    dwm1 = depth_weighted_mid(dfb, "Z", n_levels=1)
    dwmN = depth_weighted_mid(dfb, "Z", n_levels=10)
    # per-side depth-weighted average prices to 10 levels
    Pa = np.column_stack([dfb[f"Z_askprice_{i}"].to_numpy(float) for i in range(1, 11)])
    Qa = _clean(np.column_stack([dfb[f"Z_askquantity_{i}"].to_numpy(float) for i in range(1, 11)]))
    Pb = np.column_stack([dfb[f"Z_bidprice_{i}"].to_numpy(float) for i in range(1, 11)])
    Qb = _clean(np.column_stack([dfb[f"Z_bidquantity_{i}"].to_numpy(float) for i in range(1, 11)]))
    dwap_a = np.nansum(Pa * Qa, 1) / Qa.sum(1); dwap_b = np.nansum(Pb * Qb, 1) / Qb.sum(1)
    auc_dw = M(decay_weighted_cost(dfb, "Z", 10, anchor="depth_wmid"))
    len_dw = M(impact_curve_length(dfb, "Z", 10, anchor="depth_wmid"))
    print("\ndepth-weighted mid")
    print("    nests microprice at n=1:", bool(np.allclose(dwm1, wm1)))
    print("    n=10 within [DWAP_bid, DWAP_ask]:", bool(np.all((dwmN >= dwap_b - 1e-9) & (dwmN <= dwap_a + 1e-9))))
    print("    depth_wmid-anchored AUC=%.3f finite, ||L||=%.4f in[sqrt2,2]" % (auc_dw, len_dw))
    print("    checks:",
          bool(np.allclose(dwm1, wm1))
          and bool(np.all((dwmN >= dwap_b - 1e-9) & (dwmN <= dwap_a + 1e-9)))
          and np.isfinite(auc_dw) and 2 ** 0.5 <= len_dw <= 2.0 + 1e-9)

    # torn level (one leg NaN from a shift) must behave exactly like an absent level
    bt = _bk(tick=0.02, decay=0.4, n=10).astype(float)
    def _tear(which):
        d = bt.copy()
        if which in ("p", "both"): d.iloc[5:15, d.columns.get_loc("Z_askprice_5")] = np.nan
        if which in ("q", "both"): d.iloc[5:15, d.columns.get_loc("Z_askquantity_5")] = np.nan
        return (M(depth_weighted_mid(d, "Z", 10)), M(decay_weighted_cost(d, "Z", 10)))
    p_only, q_only, both = _tear("p"), _tear("q"), _tear("both")
    print("\ntorn-level (one-leg NaN) == absent-level")
    print("    depth_wmid  price/qty/both: %.5f / %.5f / %.5f" % (p_only[0], q_only[0], both[0]))
    print("    checks:", np.isclose(p_only[0], both[0]) and np.isclose(q_only[0], both[0])
          and np.isclose(p_only[1], both[1]) and np.isclose(q_only[1], both[1]))

if __name__ == "__main__":
    _selftest()
