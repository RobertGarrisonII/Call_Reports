"""
cross_flow.py
============
A continuous, signed, size-aware measure of cross-market order flow to replace the
categorical PCMOF / NCMOF of Garrison, Jain & Paddrik (Eqs. 1-2). Built on the
multi-level Cont-Kukanov-Stoikov OFI (`cross_asset_pd_liquidity.order_flow_imbalance`),
which already nets adds / cancels / trades and is size-aware -- so the leakage in the
count-based, add-only, top-of-book imbalance is fixed at the source.

Three layers:

  1. Signed cross-flow  chi_t in [-1,1] : the product of bounded per-leg directional
     pressures p = tanh(z(OFI)). Positive when both books push the same way, negative
     when opposed, magnitude = strength of the joint push. Its rectified halves
         PCMOF_t = max(chi_t, 0),  NCMOF_t = max(-chi_t, 0)   (both in [0,1])
     reproduce the paper's same-direction / opposite-direction split and its
     two-regressor structure -- continuous, signed-via-magnitude, no dead-zone.
     `discretize` shows the categorical as the thresholded special case (what it drops).

  2. Common / divergent rotation (Hasbrouck-Seppi, ref. 33): a fixed 45-degree rotation
     of the standardized pair into a common flow (z_SPY + z_ES)/sqrt2 -- the principled
     PCMOF, market-wide pressure that lifts return correlation -- and a divergent flow
     (z_SPY - z_ES)/sqrt2 -- the principled NCMOF, relative/arbitrage pressure that lowers
     it. Orthogonal, threshold-free; the variance shares are (1+rho)/2 and (1-rho)/2.

  3. Conditional comovement (the H3 stress object measured directly, not inferred from
     shrinking corner cells): rolling and DCC correlation of the two OFI series, plus a
     Hayashi-Yoshida / Epps-corrected correlation of the two RETURN processes for the
     async, high-frequency LHS of Table 9.

Pure numpy / pandas (+ optional dcc_garch, noise_robust_cov).
"""
import numpy as np
import pandas as pd

import cross_asset_pd_liquidity as ca

EPS = 1e-12


# ── per-leg signed OFI and standardization ────────────────────────────────────
def signed_ofi(df, asset, n_levels=10, min_rest_steps=0):
    """Size/cancel-aware multi-level CKS OFI for one market (length len(df), first 0)."""
    return ca.order_flow_imbalance(df, asset, n_levels, min_rest_steps)


def _standardize(x, robust=True):
    x = np.asarray(x, float)
    if robust:
        med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med))
        s = 1.4826 * mad
        if not (s > EPS):
            s = np.nanstd(x); s = s if s > EPS else 1.0
        return (x - med) / s
    mu = np.nanmean(x); sd = np.nanstd(x)
    return (x - mu) / (sd if sd > EPS else 1.0)


# ── (1) continuous signed cross-flow ──────────────────────────────────────────
def cross_flow(g_spy, g_es, kappa=1.0, squash="tanh", robust=True):
    """chi_t in [-1,1]: product of bounded per-leg directional pressures p = f(z/kappa).
    squash='tanh' (default, smooth), 'clip' (piecewise-linear), or 'raw' (unbounded
    product of z-scores, for diagnostics). kappa softens/sharpens the squash."""
    zs = _standardize(g_spy, robust) / kappa
    ze = _standardize(g_es, robust) / kappa
    if squash == "tanh":
        ps, pe = np.tanh(zs), np.tanh(ze)
    elif squash == "clip":
        ps, pe = np.clip(zs, -1.0, 1.0), np.clip(ze, -1.0, 1.0)
    else:
        ps, pe = zs, ze
    return ps * pe


def pcmof_ncmof(chi):
    """Rectified halves of chi: (PCMOF = max(chi,0), NCMOF = max(-chi,0)). Drop-in for
    the two cross-flow regressors of Eqs. (5)-(6); PCMOF - NCMOF == chi."""
    chi = np.asarray(chi, float)
    return np.maximum(chi, 0.0), np.maximum(-chi, 0.0)


def discretize(chi, neutral=0.1):
    """Collapse continuous chi to the paper's 3-state categorical (Same / Opposite /
    Neutral) by thresholding -- the categorical is this special case, and the dead-zone
    |chi| <= neutral is exactly the information the continuous measure keeps."""
    chi = np.asarray(chi, float)
    out = np.full(chi.shape, "Neutral", dtype=object)
    out[chi > neutral] = "Same"
    out[chi < -neutral] = "Opposite"
    return out


# ── (2) common / divergent rotation (Hasbrouck-Seppi) ─────────────────────────
def common_divergent(g_spy, g_es, robust=True):
    """Fixed 45-degree rotation of the standardized OFI pair into orthogonal common and
    divergent flow. Returns the two signed series, their variance shares (1+rho)/2 and
    (1-rho)/2, the OFI correlation rho, and the realized orthogonality (~0). For >2
    assets this generalizes to PC1 (common) / higher PCs (divergent)."""
    zs = _standardize(g_spy, robust); ze = _standardize(g_es, robust)
    common = (zs + ze) / np.sqrt(2.0)
    divergent = (zs - ze) / np.sqrt(2.0)
    m = np.isfinite(common) & np.isfinite(divergent)
    rho = float(np.corrcoef(zs[m], ze[m])[0, 1]) if m.sum() > 2 else np.nan
    orth = float(np.corrcoef(common[m], divergent[m])[0, 1]) if m.sum() > 2 else np.nan
    return {"common": common, "divergent": divergent,
            "common_share": (1 + rho) / 2.0, "divergent_share": (1 - rho) / 2.0,
            "corr": rho, "orthogonality": orth}


# ── (3) conditional comovement ────────────────────────────────────────────────
def comovement_rolling(g_spy, g_es, window=100):
    """Rolling Pearson correlation of the two OFI series (leading NaNs)."""
    return pd.Series(np.asarray(g_spy, float)).rolling(window).corr(
        pd.Series(np.asarray(g_es, float))).to_numpy()


def comovement_dcc(g_spy, g_es):
    """DCC conditional correlation of the two OFI series (length len-1). None if
    dcc_garch is unavailable or the fit fails. The cleaner H3 stress object: its
    breakdown under volatility is measured, not inferred from shrinking corner cells."""
    try:
        import dcc_garch as dg
        fit = dg.dcc_garch_x(np.column_stack([np.asarray(g_spy, float), np.asarray(g_es, float)]))
        return np.asarray(fit["rho"], float)
    except Exception:
        return None


def return_correlation_hy(t_spy, lp_spy, t_es, lp_es):
    """Hayashi-Yoshida / Epps-corrected correlation of the two RETURN processes for
    asynchronous high-frequency data -- the noise-robust LHS for Table 9, replacing the
    fixed-window Pearson the paper differences. t_* are observation times (floats),
    lp_* the log-prices. Falls back to NaN if noise_robust_cov is unavailable."""
    try:
        import noise_robust_cov as nrc
        cov = nrc.hayashi_yoshida(np.asarray(t_spy, float), np.asarray(lp_spy, float),
                                  np.asarray(t_es, float), np.asarray(lp_es, float))
        v1 = float(np.sum(np.diff(np.asarray(lp_spy, float)) ** 2))
        v2 = float(np.sum(np.diff(np.asarray(lp_es, float)) ** 2))
        den = np.sqrt(v1 * v2)
        return cov / den if den > EPS else np.nan
    except Exception:
        return np.nan


# ── (4) VPIN-style volume-clock reformatting of PCMOF / NCMOF ─────────────────
def volume_buckets(notional, n_buckets=50):
    """Assign each bar to an equal-notional bucket on the cumulative-notional clock; the
    bucket size is V = (total notional)/n_buckets (the VPIN convention). Returns an integer
    bucket id per bar in [0, n_buckets-1]. Because cumulative notional is monotone, buckets
    are contiguous runs of bars."""
    c = np.cumsum(np.asarray(notional, float))
    total = float(c[-1]) if len(c) else 0.0
    if not (total > EPS):
        return np.zeros(len(c), dtype=int)
    bid = np.floor((c - EPS) / (total / n_buckets)).astype(int)
    return np.clip(bid, 0, n_buckets - 1)


def _book_notional_proxy(df, g_s, g_e):
    """Book-native per-bar 'traded notional' proxy used when real trade volume is not
    supplied: the dollar magnitude of net order flow |OFI|*mid, summed across markets. This
    is an ACTIVITY clock, not true volume -- pass real per-bar notional for a true VPIN clock."""
    ms = np.asarray(ca._mid(df, "SPY"), float); me = np.asarray(ca._mid(df, "ES"), float)
    return np.abs(np.asarray(g_s, float)) * ms + np.abs(np.asarray(g_e, float)) * me


def bucketed_cross_flow(df, notional=None, notional_fn=None, n_buckets=50, window=50,
                        theta=0.05, kappa=1.0, robust=True, weight_by_min=True,
                        n_levels=10, min_rest_steps=0):
    """VPIN-style volume-clock PCMOF/NCMOF. Bars are bucketed into equal-notional buckets on
    a COMMON (SPY+ES) notional clock; each market's signed OFI is summed within a bucket and
    co-directionality is read off the bucket pair -- the Epps/seasonality correction in
    information time. The shared clock (not two per-instrument clocks) is what makes the two
    legs comparable.

    Per bucket tau: O_m = sum of OFI_m, I_m = O_m / sum|OFI_m| in [-1,1]; the magnitude
    co-flow chi_tau = tanh(z(O_SPY))*tanh(z(O_ES)) (z standardized ACROSS buckets), with
    PCMOF = max(chi,0), NCMOF = max(-chi,0); the categorical PCMOF_cat/NCMOF_cat classify by
    |I_m|>theta; PCMOF_roll/NCMOF_roll are the n-bucket trailing means (the VPIN 1/n sum).
    Buckets are weighted by min(activity_SPY, activity_ES) (co-flow only counts when both
    trade) unless weight_by_min=False (then by bar count).

    notional: per-bar combined notional, or an (N,2) array of (SPY$, ES$); notional_fn(df)
    -> per-bar notional; if neither is given, a book-native |OFI|*mid proxy is used. Returns
    a per-bucket DataFrame indexed by bucket-end timestamp; the per-bar bucket id is attached
    as `.attrs['bucket_id']` (and `.attrs['bar_index']`) for broadcasting back to bars."""
    g_s = signed_ofi(df, "SPY", n_levels, min_rest_steps)
    g_e = signed_ofi(df, "ES", n_levels, min_rest_steps)
    if notional is None and notional_fn is not None:
        notional = notional_fn(df)
    if notional is None:
        notl = _book_notional_proxy(df, g_s, g_e)
    else:
        arr = np.asarray(notional, float)
        notl = arr.sum(axis=1) if arr.ndim == 2 else arr
    bid = volume_buckets(notl, n_buckets)
    idx = df.index
    grp = pd.DataFrame({"b": bid, "g_s": np.asarray(g_s, float), "g_e": np.asarray(g_e, float),
                        "abs_s": np.abs(np.asarray(g_s, float)), "abs_e": np.abs(np.asarray(g_e, float))})
    agg = grp.groupby("b").agg(O_SPY=("g_s", "sum"), O_ES=("g_e", "sum"),
                               W_SPY=("abs_s", "sum"), W_ES=("abs_e", "sum"), n_bars=("b", "size"))
    ub = agg.index.to_numpy()
    end_pos = np.array([np.where(bid == b)[0][-1] for b in ub])
    agg.index = idx[end_pos]; agg.index.name = "bucket_end"

    O_s, O_e = agg["O_SPY"].to_numpy(), agg["O_ES"].to_numpy()
    W_s, W_e = agg["W_SPY"].to_numpy(), agg["W_ES"].to_numpy()
    I_s = O_s / np.maximum(W_s, EPS); I_e = O_e / np.maximum(W_e, EPS)
    zs = _standardize(O_s, robust) / kappa; ze = _standardize(O_e, robust) / kappa
    chi = np.tanh(zs) * np.tanh(ze)
    pc, nc = np.maximum(chi, 0.0), np.maximum(-chi, 0.0)
    ss = np.where(np.abs(I_s) > theta, np.sign(I_s), 0.0)
    se = np.where(np.abs(I_e) > theta, np.sign(I_e), 0.0)
    pc_cat = ((ss == se) & (ss != 0)).astype(float)
    nc_cat = ((ss == -se) & (ss != 0) & (se != 0)).astype(float)
    w = np.minimum(W_s, W_e) if weight_by_min else agg["n_bars"].to_numpy().astype(float)
    pc_roll = pd.Series(pc).rolling(window, min_periods=1).mean().to_numpy()
    nc_roll = pd.Series(nc).rolling(window, min_periods=1).mean().to_numpy()
    out = pd.DataFrame({"O_SPY": O_s, "O_ES": O_e, "I_SPY": I_s, "I_ES": I_e, "chi": chi,
                        "PCMOF": pc, "NCMOF": nc, "PCMOF_cat": pc_cat, "NCMOF_cat": nc_cat,
                        "PCMOF_roll": pc_roll, "NCMOF_roll": nc_roll, "weight": w,
                        "n_bars": agg["n_bars"].to_numpy()}, index=agg.index)
    out.attrs["bucket_id"] = bid; out.attrs["bar_index"] = idx
    return out


def _broadcast_buckets(values_per_bucket, bid, ub):
    """Map a per-bucket series back to bars via the bucket id (piecewise-constant in time)."""
    lut = np.full(int(ub.max()) + 1, np.nan)
    lut[ub.astype(int)] = np.asarray(values_per_bucket, float)
    return lut[bid]


# ── convenience: all cross-flow regressors from a book frame ──────────────────
def cross_flow_features(df, n_levels=10, min_rest_steps=0, kappa=1.0, squash="tanh",
                        robust=True, window=100, clock="time", notional=None, notional_fn=None,
                        n_buckets=50, bucket_window=50):
    """Compute every cross-flow regressor from a book frame, as a DataFrame indexed like
    df: chi, PCMOF, NCMOF (the continuous Eq.5/6 drop-ins), common, divergent (the HS
    rotation), and ofi_corr_roll (rolling OFI comovement). Hand these to
    `correlation_svar.correlation_irf(spec=[...])` to run Table 9 on the continuous flow.

    clock='time' (default) computes chi/PCMOF/NCMOF per bar; clock='volume' computes them on
    the VPIN-style equal-notional clock (see `bucketed_cross_flow`) and broadcasts each
    bucket's value back to its bars, so the frame stays per-bar and usable as an SVAR
    regressor while the co-directionality is synchronized in information time. common /
    divergent / ofi_corr_roll are always computed on the bar clock."""
    g_s = signed_ofi(df, "SPY", n_levels, min_rest_steps)
    g_e = signed_ofi(df, "ES", n_levels, min_rest_steps)
    if clock == "volume":
        b = bucketed_cross_flow(df, notional, notional_fn, n_buckets, bucket_window,
                                kappa=kappa, robust=robust, n_levels=n_levels, min_rest_steps=min_rest_steps)
        bid = b.attrs["bucket_id"]; ub = np.unique(bid)
        chi = _broadcast_buckets(b["chi"].to_numpy(), bid, ub)
    else:
        chi = cross_flow(g_s, g_e, kappa, squash, robust)
    pc, nc = pcmof_ncmof(chi)
    cd = common_divergent(g_s, g_e, robust)
    return pd.DataFrame({
        "chi": chi, "PCMOF": pc, "NCMOF": nc,
        "common": cd["common"], "divergent": cd["divergent"],
        "ofi_corr_roll": comovement_rolling(g_s, g_e, window),
    }, index=df.index)


# ── self-test ─────────────────────────────────────────────────────────────────
def _ofi_pair(T, rho_target, seed):
    """Two synthetic OFI series with a genuine contemporaneous correlation rho_target
    (Cholesky of the 2x2 target, so negative targets are actually negative)."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0, 1, T); z2 = rng.normal(0, 1, T)
    g_s = z1
    g_e = rho_target * z1 + np.sqrt(max(1 - rho_target * rho_target, 0.0)) * z2
    return g_s * 50.0, g_e * 50.0          # arbitrary scale -> standardization must handle it


def _tiny_book(T=4000, seed=5):
    """Minimal 3-level two-asset book for the from-frame smoke test."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, T).cumsum()
    cols = {}
    for a, base, tick in (("SPY", 500.0, 0.01), ("ES", 5000.0, 0.25)):
        mid = base * np.exp((f + rng.normal(0, 0.3, T)) / 50.0)
        for l in range(1, 4):
            q = np.maximum(200.0 * np.exp(-0.3 * (l - 1)) * (1 + rng.normal(0, 0.1, T)), 1.0)
            cols[f"{a}_bidprice_{l}"] = mid - l * tick; cols[f"{a}_askprice_{l}"] = mid + l * tick
            cols[f"{a}_bidquantity_{l}"] = q; cols[f"{a}_askquantity_{l}"] = q
    idx = pd.date_range("2024-08-05 09:30", periods=T, freq="s", tz="America/New_York")
    return pd.DataFrame(cols, index=idx)


def _common_book(T=6000, seed=11):
    """Two-asset book whose prices are driven by a dominant shared factor (as SPY and ES
    track the same index), so the CKS order flow is genuinely positively correlated and
    same-direction flow dominates."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0, 3e-4, T))                      # common log-price factor (dominant)
    cols = {}
    for a, base, tick in (("SPY", 500.0, 0.01), ("ES", 5000.0, 0.25)):
        mid = base * np.exp(f + np.cumsum(rng.normal(0, 0.5e-4, T)))   # shared trend + small idiosyncratic
        for l in range(1, 4):
            q = np.maximum(200.0 * np.exp(-0.3 * (l - 1)) * (1 + rng.normal(0, 0.08, T)), 1.0)
            cols[f"{a}_bidprice_{l}"] = mid - l * tick; cols[f"{a}_askprice_{l}"] = mid + l * tick
            cols[f"{a}_bidquantity_{l}"] = q; cols[f"{a}_askquantity_{l}"] = q * (1 + rng.normal(0, 0.05, T))
    idx = pd.date_range("2024-08-05 09:30", periods=T, freq="s", tz="America/New_York")
    return pd.DataFrame(cols, index=idx)


def _selftest():
    print("cross_flow self-test")

    # (1) chi decomposition + nesting of the categorical
    gp_s, gp_e = _ofi_pair(60000, +0.6, 1)        # positively correlated flow
    gn_s, gn_e = _ofi_pair(60000, -0.6, 2)        # negatively correlated flow
    chip = cross_flow(gp_s, gp_e); chin = cross_flow(gn_s, gn_e)
    pcp, ncp = pcmof_ncmof(chip)
    catp = discretize(chip)
    same = np.mean(catp == "Same"); opp = np.mean(catp == "Opposite"); neu = np.mean(catp == "Neutral")
    print("\n(1) signed cross-flow chi and its rectified halves")
    print("    positive-corr flow : mean chi=%+.3f  PCMOF mean=%.3f > NCMOF mean=%.3f"
          % (chip.mean(), pcp.mean(), ncp.mean()))
    print("    negative-corr flow : mean chi=%+.3f  (NCMOF should now dominate)" % chin.mean())
    print("    chi range [%.2f, %.2f]; categorical (pos corr) Same/Opp/Neutral = %.2f / %.2f / %.2f (dead-zone dropped)"
          % (chip.min(), chip.max(), same, opp, neu))
    print("    checks:",
          np.allclose(pcp - ncp, chip)                                  # rectified halves reconstruct chi
          and chip.min() >= -1 - 1e-9 and chip.max() <= 1 + 1e-9        # bounded
          and pcp.min() >= 0 and ncp.min() >= 0                         # halves non-negative
          and chip.mean() > 0 and chin.mean() < 0                       # sign tracks comovement
          and pcp.mean() > ncp.mean()                                   # same-direction dominates under +corr
          and same > opp)                                               # categorical agrees in sign

    # (2) common / divergent rotation
    cdp = common_divergent(gp_s, gp_e); cdn = common_divergent(gn_s, gn_e)
    print("\n(2) common / divergent rotation (Hasbrouck-Seppi)")
    print("    +corr: common_share=%.3f  divergent_share=%.3f  (= (1+/-rho)/2; rho=%.3f)  orthogonality=%.4f"
          % (cdp["common_share"], cdp["divergent_share"], cdp["corr"], cdp["orthogonality"]))
    print("    -corr: common_share=%.3f  divergent_share=%.3f  (divergent now dominates)"
          % (cdn["common_share"], cdn["divergent_share"]))
    print("    checks:",
          cdp["common_share"] > 0.5 and cdp["divergent_share"] < 0.5    # +corr -> mostly common
          and cdn["common_share"] < 0.5                                 # -corr -> mostly divergent
          and abs(cdp["common_share"] - (1 + cdp["corr"]) / 2) < 1e-6   # closed form
          and abs(cdp["orthogonality"]) < 0.05)                         # components orthogonal

    # (3) conditional comovement: rolling recovers a time-varying correlation; HY ~ Pearson
    rng = np.random.default_rng(7); T = 40000
    rho_t = 0.2 + 0.6 * (np.sin(np.linspace(0, 8 * np.pi, T)) > 0)      # regime-switching corr
    z = rng.normal(0, 1, T)
    g_s = rho_t * z + np.sqrt(np.clip(1 - rho_t ** 2, 0, 1)) * rng.normal(0, 1, T)
    g_e = rho_t * z + np.sqrt(np.clip(1 - rho_t ** 2, 0, 1)) * rng.normal(0, 1, T)
    roll = comovement_rolling(g_s, g_e, 250)
    ok_roll = np.corrcoef(roll[~np.isnan(roll)], rho_t[~np.isnan(roll)])[0, 1]
    # HY on synchronous returns should match the Pearson correlation of increments
    t = np.arange(2000.0); lp1 = rng.normal(0, 1, 2000).cumsum() * 1e-4
    common = rng.normal(0, 1, 2000)
    lp1 = (0.7 * common + np.sqrt(1 - 0.49) * rng.normal(0, 1, 2000)).cumsum() * 1e-4
    lp2 = (0.7 * common + np.sqrt(1 - 0.49) * rng.normal(0, 1, 2000)).cumsum() * 1e-4
    hy = return_correlation_hy(t, lp1, t, lp2)
    pear = np.corrcoef(np.diff(lp1), np.diff(lp2))[0, 1]
    dcc = comovement_dcc(g_s[:8000], g_e[:8000])
    print("\n(3) conditional comovement of the two OFI series")
    print("    rolling-vs-true-corr tracking = %.3f" % ok_roll)
    print("    Hayashi-Yoshida return-corr = %.3f  vs Pearson of increments = %.3f" % (hy, pear))
    print("    DCC conditional corr: %s" % ("ran, mean=%.3f" % np.nanmean(dcc) if dcc is not None else "skipped"))
    print("    checks:", ok_roll > 0.5 and abs(hy - pear) < 0.05)

    # (4) from a book frame: every regressor present, finite, and chi nests PCMOF/NCMOF
    feat = cross_flow_features(_tiny_book(), n_levels=3)
    print("\n(4) cross_flow_features from a book frame")
    print("    columns:", list(feat.columns))
    print("    checks:",
          set(["chi", "PCMOF", "NCMOF", "common", "divergent", "ofi_corr_roll"]).issubset(feat.columns)
          and np.allclose((feat["PCMOF"] - feat["NCMOF"]).to_numpy(), feat["chi"].to_numpy())
          and np.isfinite(feat["chi"].to_numpy()).all())

    # (5) VPIN-style volume-clock reformatting of PCMOF / NCMOF
    book = _common_book(6000, seed=11)                         # correlated inside-depth flow
    b = bucketed_cross_flow(book, n_buckets=40, window=20, n_levels=3)
    notl = _book_notional_proxy(book, signed_ofi(book, "SPY", 3), signed_ofi(book, "ES", 3))
    per_bucket_notional = pd.Series(notl).groupby(volume_buckets(notl, 40)).sum().to_numpy()
    cv = per_bucket_notional.std() / max(per_bucket_notional.mean(), EPS)   # equal-notional buckets
    pcw = float((b["PCMOF"] * b["weight"]).sum() / max(b["weight"].sum(), EPS))
    ncw = float((b["NCMOF"] * b["weight"]).sum() / max(b["weight"].sum(), EPS))
    feat_v = cross_flow_features(book, n_levels=3, clock="volume", n_buckets=40)
    feat_t = cross_flow_features(book, n_levels=3, clock="time")
    print("\n(5) volume-clock (bucketed) PCMOF / NCMOF")
    print("    %d buckets; per-bucket notional CV=%.3f (equal-notional)" % (len(b), cv))
    print("    weighted PCMOF=%.3f  NCMOF=%.3f  (correlated flow -> PCMOF dominates)" % (pcw, ncw))
    print("    columns:", list(b.columns))
    print("    bar-clock OFI corr=%.3f  vs bucket-clock OFI corr=%.3f"
          % (np.corrcoef(signed_ofi(book, "SPY", 3), signed_ofi(book, "ES", 3))[0, 1],
             np.corrcoef(b["O_SPY"], b["O_ES"])[0, 1]))
    print("    checks:",
          1 < len(b) <= 40 and cv < 0.5                                    # contiguous equal-notional buckets
          and b["PCMOF"].between(0, 1).all() and b["NCMOF"].between(0, 1).all()
          and np.allclose((b["PCMOF"] - b["NCMOF"]).to_numpy(), b["chi"].to_numpy())   # chi identity per bucket
          and b["PCMOF_roll"].between(0, 1).all() and b["NCMOF_roll"].between(0, 1).all()
          and pcw > ncw                                                    # correlated flow -> same-direction dominates
          and len(feat_v) == len(book) and len(b.attrs["bucket_id"]) == len(book)
          and np.allclose((feat_v["PCMOF"] - feat_v["NCMOF"]).to_numpy(), feat_v["chi"].to_numpy())
          and np.isfinite(feat_v["chi"].to_numpy()).all())


if __name__ == "__main__":
    _selftest()
