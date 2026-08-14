"""
book_geometry.py
================
The ||L|| (arc-length) measure of the depth profile, implemented AS A CHECK, with
the three statistics that were argued to dominate it: total depth, the depth
centroid, and the Herfindahl of depth shares.

Why "as a check" and not as a headline observable
-------------------------------------------------
In raw units the arc length of a 10-level book curve is degenerate: each segment
contributes sqrt(tick^2 + q_i^2) ~= q_i + tick^2/(2 q_i), and at a $0.01 tick
against 100-10,000-share levels the price term is O(1e-8) of the size term -- the
"length" is total displayed depth to ~ten significant figures. The only version
with content maps the curve to the unit square (x = level index fraction,
y = cumulative depth share) where

    L = sum_i sqrt((1/n)^2 + s_i^2),        s_i = level i's share of total depth,

with L = sqrt(2) exactly at uniform shares and a DISCRETE maximum
L_max(n) = sqrt(1 + 1/n^2) + (n-1)/n at full concentration (the continuum bound
of 2 is NOT attained at finite n; normalizing by 2 would misstate the range).
tau = (L - sqrt2)/(L_max(n) - sqrt2) in [0, 1] is then a genuine concentration
index. But L is a sum of a fixed convex function over the SHARES ALONE: it is
permutation-invariant in the levels (all-at-level-1 and all-at-level-10 give the
same tau), hence Schur-convex, hence the same functional family as the
Herfindahl -- it carries no location information by construction, and its
increment over the Herfindahl is a higher-order moment of the share vector.
The economic content it cannot carry (front- vs back-loading) lives in the depth
CENTROID, which also has the exact execution identity

    VWAP(full sweep of the side) - p_touch = c$,
    c$ = sum_i |p_i - p_touch| q_i / sum_i q_i,

i.e. the centroid IS the sweep concession, gaps included. This module computes
tau alongside {total depth, centroid, Herfindahl} so a run can verify, on real
frames, the two claims made for it: (1) tau is rank-equivalent to the Herfindahl
(Spearman rho ~ 1), and (2) tau adds ~zero incremental R^2 over the covering set
when explaining short-horizon volatility. If either check FAILS on real data,
tau is carrying something the covering set misses and deserves promotion;
until then it is a diagnostic column, not a regressor.

All functions are vectorized over rows (T x N level matrices), reuse the
_usable_levels hygiene of liquidity_curve_metrics (a level counts only if price
AND size are finite), and return NaN -- never a silent 0 -- where a side has no
usable depth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import liquidity_curve_metrics as lcm

EPS = 1e-12


# ── core functionals (T x N -> T) ─────────────────────────────────────────────
def depth_shares(q: np.ndarray) -> np.ndarray:
    """s_i = q_i / sum_j q_j, NaN rows where the side has no usable depth (a
    zero-depth side has no shape; 0 would alias 'uniform')."""
    q = np.where(np.isfinite(q), np.asarray(q, float), 0.0)
    s = q.sum(axis=1, keepdims=True)
    out = np.divide(q, s, out=np.full_like(q, np.nan), where=s > EPS)
    return out


def arc_length_L(q: np.ndarray) -> np.ndarray:
    """Unit-square arc length L = sum_i sqrt((1/n)^2 + s_i^2) on the index-based
    cumulative-share curve. Range [sqrt(2), L_max(n)] -- see module docstring."""
    s = depth_shares(q)
    n = s.shape[1]
    return np.sqrt((1.0 / n) ** 2 + s ** 2).sum(axis=1)


def tau_L_max(n: int) -> float:
    """The DISCRETE maximum of L at full concentration: one segment carries the
    whole rise, the other n-1 are flat. The continuum bound 2 is approached only
    as n -> inf (at n=10, L_max = 1.90499); normalizing by 2 would cap tau at
    ~0.86 and misreport full concentration."""
    n = int(n)
    return float(np.sqrt(1.0 + 1.0 / n ** 2) + (n - 1.0) / n)


def tau(q: np.ndarray) -> np.ndarray:
    """Normalized arc length tau = (L - sqrt2)/(L_max(n) - sqrt2) in [0, 1]:
    0 = uniform depth, 1 = all depth at a single level (ANY single level --
    permutation-invariant by construction; see module docstring)."""
    L = arc_length_L(q)
    n = np.asarray(q).shape[1]
    return np.clip((L - np.sqrt(2.0)) / (tau_L_max(n) - np.sqrt(2.0)), 0.0, 1.0)


def herfindahl(q: np.ndarray) -> np.ndarray:
    """Normalized Herfindahl of depth shares: (sum s_i^2 - 1/n)/(1 - 1/n) in
    [0, 1], same endpoints as tau (0 uniform, 1 concentrated) so the two are
    directly comparable. This is the quadratic member of the same symmetric
    Schur-convex family tau belongs to."""
    s = depth_shares(q)
    n = s.shape[1]
    h = (s ** 2).sum(axis=1)
    return (h - 1.0 / n) / (1.0 - 1.0 / n)


def centroid_concession(px: np.ndarray, qty: np.ndarray, touch: np.ndarray,
                        tick_size: float | None = None):
    """Depth centroid as the exact full-sweep concession:
    c$ = sum_i |p_i - p_touch| q_i / sum_i q_i  (price units), and in ticks when
    tick_size is given. Uses ACTUAL level prices, so price gaps count -- this is
    the location statistic the (permutation-invariant) tau cannot carry.
    Identity: VWAP of sweeping every usable level minus the touch equals c$
    exactly, gaps or no gaps. Returns (c_price, c_ticks_or_None)."""
    px = np.asarray(px, float)
    q = np.where(np.isfinite(qty) & np.isfinite(px), np.asarray(qty, float), 0.0)
    d = np.abs(px - np.asarray(touch, float)[:, None])
    d = np.where(np.isfinite(d), d, 0.0)
    tot = q.sum(axis=1)
    c = np.divide((d * q).sum(axis=1), tot,
                  out=np.full(px.shape[0], np.nan), where=tot > EPS)
    return c, (c / float(tick_size) if tick_size else None)


# ── frame-level assembly ──────────────────────────────────────────────────────
def _side_matrices(df: pd.DataFrame, asset: str, side: str, n_levels: int):
    px = np.column_stack([df[f"{asset}_{side}price_{i}"].to_numpy(float)
                          for i in range(1, n_levels + 1)])
    qty = np.column_stack([df[f"{asset}_{side}quantity_{i}"].to_numpy(float)
                           for i in range(1, n_levels + 1)])
    return lcm._usable_levels(px, qty)


def book_geometry_frame(df: pd.DataFrame, asset: str, n_levels: int = 10,
                        tick_size: float | None = None) -> pd.DataFrame:
    """Per-snapshot geometry columns for one asset:
    tau_bid/ask, hhi_bid/ask, centroid_bid/ask (price; *_ticks when tick_size),
    centroid_asym (bid - ask: positive = bid-side liquidity sits deeper, the
    directional tilt), depth_bid/ask (total usable shares/contracts)."""
    out = {}
    cen = {}
    for side in ("bid", "ask"):
        px, qty = _side_matrices(df, asset, side, n_levels)
        touch = df[f"{asset}_{side}price_1"].to_numpy(float)
        out[f"tau_{side}"] = tau(qty)
        out[f"hhi_{side}"] = herfindahl(qty)
        c, ct = centroid_concession(px, qty, touch, tick_size)
        out[f"centroid_{side}"] = c
        if ct is not None:
            out[f"centroid_{side}_ticks"] = ct
        cen[side] = c
        out[f"depth_{side}"] = np.where(np.isfinite(qty), qty, 0.0).sum(axis=1)
    out["centroid_asym"] = cen["bid"] - cen["ask"]
    return pd.DataFrame(out, index=df.index)


# ── the two checks the measure exists for ─────────────────────────────────────
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > EPS else float("nan")


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    """OLS R^2 with intercept; rows with any non-finite entry dropped."""
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[m], X[m]
    if len(y) < X.shape[1] + 10:
        return float("nan")
    Z = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ b
    tss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - (e ** 2).sum() / tss) if tss > EPS else float("nan")


def tau_redundancy_check(df: pd.DataFrame, asset: str, n_levels: int = 10,
                         tick_size: float | None = None, horizon: int = 60) -> dict:
    """The verdict tau exists to deliver, on one session frame.

    (1) rank equivalence: Spearman rho of tau vs the Herfindahl, per side. The
        Schur-convexity argument predicts rho ~ 1; a materially lower rho means
        the higher-order share moments tau weights are moving independently.
    (2) incremental R^2: |forward mid return over `horizon` steps| regressed on
        the covering set {log total depth, centroid, HHI} (both sides), then
        with tau_bid/tau_ask added. The prediction is an increment of ~0.
    -> {side: spearman_tau_hhi}, r2_covering, r2_with_tau, tau_increment"""
    g = book_geometry_frame(df, asset, n_levels, tick_size)
    mid = (df[f"{asset}_bidprice_1"].to_numpy(float)
           + df[f"{asset}_askprice_1"].to_numpy(float)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log(mid)
    fwd = np.full(len(g), np.nan)
    if len(g) > horizon:
        fwd[:-horizon] = np.abs(lm[horizon:] - lm[:-horizon]) * 1e4
    with np.errstate(divide="ignore", invalid="ignore"):
        ldep = np.log(g["depth_bid"].to_numpy() + g["depth_ask"].to_numpy())
    cover = np.column_stack([ldep, g["centroid_bid"], g["centroid_ask"],
                             g["hhi_bid"], g["hhi_ask"]])
    withtau = np.column_stack([cover, g["tau_bid"], g["tau_ask"]])
    r2_c = _r2(fwd, cover)
    r2_t = _r2(fwd, withtau)
    return {"spearman_tau_hhi": {s: _spearman(g[f"tau_{s}"].to_numpy(),
                                              g[f"hhi_{s}"].to_numpy())
                                 for s in ("bid", "ask")},
            "r2_covering": r2_c, "r2_with_tau": r2_t,
            "tau_increment": (r2_t - r2_c) if np.isfinite(r2_c) and np.isfinite(r2_t)
                             else float("nan"),
            "n_obs": int(np.isfinite(fwd).sum()), "horizon_steps": int(horizon)}


def table_book_geometry(sessions, n_levels: int = 10,
                        tick_sizes: dict | None = None, horizon: int = 60):
    """Per-session geometry summary + the redundancy verdict, both assets.
    `sessions` = List[(date, regime, df)]. tick_sizes e.g. {'SPY': 0.01,
    'ES': 0.25}. -> (per_day DataFrame, verdict dict of cross-day means)."""
    tick_sizes = tick_sizes or {}
    rows = []
    for date, regime, df in sessions:
        for asset in ("SPY", "ES"):
            try:
                g = book_geometry_frame(df, asset, n_levels, tick_sizes.get(asset))
                chk = tau_redundancy_check(df, asset, n_levels,
                                           tick_sizes.get(asset), horizon)
            except KeyError:
                continue
            rows.append({
                "date": str(date), "regime": regime, "asset": asset,
                "tau_bid": float(np.nanmean(g["tau_bid"])),
                "tau_ask": float(np.nanmean(g["tau_ask"])),
                "hhi_bid": float(np.nanmean(g["hhi_bid"])),
                "hhi_ask": float(np.nanmean(g["hhi_ask"])),
                "centroid_bid": float(np.nanmean(g["centroid_bid"])),
                "centroid_ask": float(np.nanmean(g["centroid_ask"])),
                "centroid_asym": float(np.nanmean(g["centroid_asym"])),
                "spearman_tau_hhi_bid": chk["spearman_tau_hhi"]["bid"],
                "spearman_tau_hhi_ask": chk["spearman_tau_hhi"]["ask"],
                "r2_covering": chk["r2_covering"],
                "tau_increment": chk["tau_increment"],
            })
    per_day = pd.DataFrame(rows)
    verdict = {}
    if not per_day.empty:
        verdict = {
            "mean_spearman_tau_hhi": float(np.nanmean(
                per_day[["spearman_tau_hhi_bid", "spearman_tau_hhi_ask"]].to_numpy())),
            "mean_tau_increment": float(np.nanmean(per_day["tau_increment"])),
            "n_session_assets": int(len(per_day)),
            "reading": ("tau is rank-equivalent to the Herfindahl and adds no "
                        "explanatory power over {depth, centroid, HHI}: keep it "
                        "as a diagnostic, not a regressor"
                        if np.nanmean(per_day[["spearman_tau_hhi_bid",
                                               "spearman_tau_hhi_ask"]].to_numpy()) > 0.90
                        and abs(np.nanmean(per_day["tau_increment"])) < 0.01
                        else "tau is NOT fully covered on this sample -- "
                             "inspect before dismissing it"),
        }
    return per_day, verdict
