"""
robust_prices.py
================
Noise-robust price observables for cross-asset price discovery, and the glue that
feeds them into the state-dependent ECM-SDE (ecm_sde.py). The SDE's information share
is only as clean as the price it is computed on: the naive top-of-book mid (b1+a1)/2
carries bid-ask bounce and tick discreteness, which act as i.i.d. measurement noise and
pull every information share toward 0.5. This module replaces that observable with
depth-aware prices and adds pre-averaging so the measure itself is noise-robust.

Two noise levers (both requested: weighted midpoints / area-under-curve / curve-length):
  * OBSERVABLE -- integrate the depth curve instead of reading only the touch:
      microprice            P = b1(1-I) + a1*I,  I = (sum bid depth)/(total depth)
      center_of_mass_price  depth-weighted average price over both sides (book centroid)
      auc_mid               mid + (slip_ask - slip_bid)/2  (cost-to-fill / area-under-curve)
  * MEASURE -- estimate the information share on a coarser grid so additive quote noise
      is a smaller share of the (larger) innovation. Sparse sampling (every q-th snapshot)
      is the canonical cure; pre-averaging (a triangular local average of non-overlapping
      blocks) is the variance-efficient version. NB pre-averaging must be paired with
      subsampling -- differenced at the fine grid it injects an MA that collapses IS to 0.5.
  * STATE -- curve-length-normalized book-shape states (scale-free across regimes).

Compose with ecm_sde (which supplies the SDE: state-interacted loadings, kernel-local
varying coefficients, Euler-Maruyama MLE):

    import ecm_sde as es, robust_prices as rp
    pf  = rp.price_fn("microprice", n_levels=10)                     # df,asset -> robust price
    fit = es.estimate_sample(sessions, state_fn=rp.state_fn("rel_logdepth"), price_fn=pf)
    fit["t_a1_SPY"]   # liquidity-conditioning test, now on noise-robust prices

    # for the noise-robust measure, coarsen the frames first (sparse sample / pre-average):
    coarse = [(l, r, rp.coarsen(df, step=20, pre_average=True)) for l, r, df in sessions]

or the one-call wrapper rp.noise_robust_is(sessions, step=20, ...).

numpy / pandas; builds on liquidity_curve_metrics and cross_asset_pd_liquidity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import liquidity_curve_metrics as lcm
import cross_asset_pd_liquidity as ca

EPS = 1e-12


# ── noise-robust price functionals from the book ──────────────────────────────
def _levels(df, asset, side, n_levels):
    px = np.column_stack([df[f"{asset}_{side}price_{i}"].to_numpy(float) for i in range(1, n_levels + 1)])
    qty = np.column_stack([df[f"{asset}_{side}quantity_{i}"].to_numpy(float) for i in range(1, n_levels + 1)])
    return px, lcm._clean(qty)

def microprice(df, asset, n_levels=5):
    """Imbalance-weighted mid over the top n levels: P = b1(1-I) + a1*I, with
    I = sum bid depth / total depth. Heavy bids (I->1) push price toward the ask.
    Robust to bid-ask bounce because it reads the whole top-n book, not just b1,a1."""
    b1 = df[f"{asset}_bidprice_1"].to_numpy(float); a1 = df[f"{asset}_askprice_1"].to_numpy(float)
    _, qb = _levels(df, asset, "bid", n_levels); _, qa = _levels(df, asset, "ask", n_levels)
    Qb = qb.sum(1); Qa = qa.sum(1); tot = Qb + Qa
    I = np.divide(Qb, tot, out=np.full_like(tot, 0.5), where=tot > EPS)
    return pd.Series(b1 * (1.0 - I) + a1 * I, index=df.index)

def center_of_mass_price(df, asset, n_levels=5):
    """Depth-weighted average price over both sides (the book centroid). Maximally
    noise-robust (averages everything) but tilts toward the deeper side."""
    pb, qb = _levels(df, asset, "bid", n_levels); pa, qa = _levels(df, asset, "ask", n_levels)
    num = (pb * qb).sum(1) + (pa * qa).sum(1); den = qb.sum(1) + qa.sum(1)
    mid = (df[f"{asset}_bidprice_1"].to_numpy(float) + df[f"{asset}_askprice_1"].to_numpy(float)) / 2.0
    return pd.Series(np.divide(num, den, out=mid.copy(), where=den > EPS), index=df.index)

def auc_mid(df, asset, n_levels=5, target=None):
    """Cost-to-fill / area-under-curve mid: mid + (slip_ask - slip_bid)/2, where
    slip_side = size-weighted distance to fill `target` units into that side. Robust
    because filling a fixed notional integrates over levels. target defaults to half
    the median visible depth."""
    db, qb, mid = lcm.extract_book(df, asset, "bid", n_levels, unit="dollar")
    da, qa, _ = lcm.extract_book(df, asset, "ask", n_levels, unit="dollar")
    if target is None:
        tot = np.minimum(lcm._clean(qb).sum(1), lcm._clean(qa).sum(1))
        target = float(np.nanmedian(tot)) * 0.5
    target = max(target, EPS)
    slip_b = lcm.slippage_to_fill(db, qb, target); slip_a = lcm.slippage_to_fill(da, qa, target)
    adj = np.nan_to_num((slip_a - slip_b) / 2.0, nan=0.0)
    return pd.Series(mid + adj, index=df.index)

def robust_price(df, asset, scheme="microprice", n_levels=5, **kw):
    """Dispatch to a noise-robust price (level, not log). scheme in
    {'microprice','com','auc','mid'} ('mid' = naive top-of-book, for comparison)."""
    if scheme == "microprice": return microprice(df, asset, n_levels)
    if scheme == "com":        return center_of_mass_price(df, asset, n_levels)
    if scheme == "auc":        return auc_mid(df, asset, n_levels, **kw)
    if scheme == "mid":        return pd.Series(ca._mid(df, asset), index=df.index)
    raise ValueError(f"unknown scheme {scheme!r}")


# ── curve-length-normalized book state (scale-free) ───────────────────────────
def _side_arclen(df, asset, n_levels):
    out = []
    for side in ("bid", "ask"):
        dist, q, _ = lcm.extract_book(df, asset, side, n_levels, unit="pct")
        out.append(lcm.arc_length_normalized(dist, q, cumulative=True))
    return 0.5 * (out[0] + out[1])

def book_state(df, n_levels=10, kind="rel_logdepth"):
    """Relative ES-vs-SPY book state S_t. 'rel_logdepth' = log total ES depth - log
    SPY depth (reuses cross_asset_pd_liquidity); 'rel_arclen' = ES minus SPY curve-
    length (shape steepness, scale-free); 'norm_slope' = depth slope normalized by
    curve length, ES minus SPY."""
    if kind == "rel_logdepth":
        return ca.relative_depth_state(df, n_levels)
    if kind == "rel_arclen":
        s = _side_arclen(df, "ES", n_levels) - _side_arclen(df, "SPY", n_levels)
        return pd.Series(s, index=df.index)
    if kind == "norm_slope":
        def nslope(asset):
            acc = []
            for side in ("bid", "ask"):
                dist, q, _ = lcm.extract_book(df, asset, side, n_levels, unit="pct")
                sl = lcm.depth_slope(dist, q); al = lcm.arc_length_normalized(dist, q)
                acc.append(sl / np.where(al > EPS, al, np.nan))
            return np.nanmean(np.column_stack(acc), axis=1)
        return pd.Series(nslope("ES") - nslope("SPY"), index=df.index)
    raise ValueError(f"unknown kind {kind!r}")


# ── pre-averaging / sparse sampling (noise-robust measure) ────────────────────
def _preaverage(p, K):
    """Centered triangular (Bartlett) moving average over a window K. On its own this
    only smooths; to de-bias the information share it must be subsampled to a coarser,
    non-overlapping grid (see coarsen) -- differenced at the fine grid it injects an MA
    that collapses IS toward 0.5."""
    if K is None or K < 2: return np.asarray(p, float)
    j = np.arange(1, K, dtype=float); g = np.minimum(j, K - j); g /= g.sum()
    return np.convolve(np.asarray(p, float), g, mode="same")

def coarsen(df, step, pre_average=False):
    """Coarsen a session frame to a slower grid so microstructure noise is a smaller
    share of the innovation. step=q keeps every q-th snapshot (sparse sampling);
    pre_average=True first applies a triangular local average over non-overlapping blocks
    of width `step` (the variance-efficient version). Prices and book/state stay aligned
    because the whole frame is coarsened together."""
    if step is None or step < 2: return df
    if pre_average:
        sm = df.rolling(step, center=True, min_periods=1, win_type="triang").mean()
        return sm.iloc[step // 2::step]
    return df.iloc[::step]


# ── glue for ecm_sde ──────────────────────────────────────────────────────────
def price_fn(scheme="microprice", n_levels=10, **kw):
    """Build a price extractor f(df, asset) -> level-price array for ecm_sde's price_fn
    hook (and pds.estimate_day). Noise robustness of the MEASURE is handled separately by
    coarsening the frames (see coarsen / noise_robust_is step=)."""
    def f(df, asset):
        return robust_price(df, asset, scheme, n_levels, **kw).to_numpy(float)
    return f

def state_fn(kind="rel_logdepth", n_levels=10):
    """Build a state extractor f(df) -> S series for ecm_sde's state_fn argument."""
    return lambda df: book_state(df, n_levels=n_levels, kind=kind)

def noise_robust_is(sessions, scheme="microprice", step=1, pre_average=False,
                    state_kind="rel_logdepth", n_levels=10, degree=1, n_lags=0, hac_lags=20):
    """One call: liquidity-conditional price discovery on noise-robust observables and,
    if step>1, a noise-robust (coarsened) grid. Runs ecm_sde's pooled state-interacted
    Euler-Maruyama estimator. Returns the ecm_sde fit (a1/t_a1 + IS(S) curve)."""
    import ecm_sde as es
    if step and step > 1:
        sessions = [(l, r, coarsen(df, step, pre_average)) for l, r, df in sessions]
    return es.estimate_sample(sessions, state_fn=state_fn(state_kind, n_levels),
                              price_fn=price_fn(scheme, n_levels),
                              degree=degree, n_lags=n_lags, hac_lags=hac_lags)


# ── self-test ─────────────────────────────────────────────────────────────────
def _book_cols(prefix, Pf, rng, n_levels=5, tick=0.01, base_q=200.0):
    """Synthetic book whose depth imbalance encodes the sub-tick position of the latent
    fair price Pf, so microprice ~ Pf while the top-of-book mid carries tick noise."""
    b1 = np.floor(Pf / tick) * tick; frac = (Pf - b1) / tick
    cols = {}
    for i in range(1, n_levels + 1):
        cols[f"{prefix}_bidprice_{i}"] = b1 - (i - 1) * tick
        cols[f"{prefix}_askprice_{i}"] = b1 + tick + (i - 1) * tick
        cols[f"{prefix}_bidquantity_{i}"] = base_q * (0.2 + frac) * rng.uniform(0.8, 1.2, len(Pf))
        cols[f"{prefix}_askquantity_{i}"] = base_q * (0.2 + (1 - frac)) * rng.uniform(0.8, 1.2, len(Pf))
    return cols

def _synth_books(n=9000, seed=0, ksp=0.02, kes=0.002, sig_c=1.2e-5, sig_id=8e-6):
    """Two cointegrated latent prices (ES leads: SPY error-corrects fast) wrapped in
    synthetic books. Returns df, latent Pf_spy, Pf_es, and the noise-free truth IS_ES."""
    import price_discovery_shares as pds
    rng = np.random.default_rng(seed)
    dc = rng.normal(0, sig_c, n); lsp = np.zeros(n); les = np.zeros(n); base = np.log(100.0)
    for t in range(1, n):
        z = lsp[t - 1] - les[t - 1]
        lsp[t] = lsp[t - 1] + dc[t] - ksp * z + rng.normal(0, sig_id)
        les[t] = les[t - 1] + dc[t] + kes * z + rng.normal(0, sig_id)
    Pf_spy = np.exp(base + lsp); Pf_es = np.exp(base + les)
    truth = pds.estimate_day(Pf_spy, Pf_es, n_lags=5)["IS_mid_ES"]
    cols = {**_book_cols("SPY", Pf_spy, rng), **_book_cols("ES", Pf_es, rng)}
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq="100ms", tz="UTC")
    return pd.DataFrame(cols, index=idx), Pf_spy, Pf_es, truth

def _selftest():
    import price_discovery_shares as pds
    df, Pf_spy, Pf_es, truth = _synth_books()

    # (A) noise-robust OBSERVABLE: microprice / AUC track the latent fair price; the raw
    #     top-of-book mid carries tick-discreteness noise.
    raw = robust_price(df, "SPY", "mid", 5).to_numpy()
    mic = robust_price(df, "SPY", "microprice", 5).to_numpy()
    auc = robust_price(df, "SPY", "auc", 5).to_numpy()
    com = robust_price(df, "SPY", "com", 5).to_numpy()
    rmse = lambda x: float(np.sqrt(np.mean((x - Pf_spy) ** 2)))
    print("(A) noise-robust observable  (RMSE to latent fair price, $)")
    print("    raw mid=%.5f  microprice=%.5f  auc=%.5f  com=%.5f" % (rmse(raw), rmse(mic), rmse(auc), rmse(com)))
    a_ok = bool(rmse(mic) < rmse(raw) and rmse(auc) < rmse(raw))
    print("    checks:", a_ok)

    # (B) noise-robust MEASURE: tick noise in the raw mid biases IS_ES toward 0.5. Two
    #     independent cures -- the depth-aware observable (microprice ~ latent price), and
    #     a coarser grid (sparse sample / pre-average) where noise is a smaller share.
    pf_raw = price_fn("mid", 5); pf_mic = price_fn("microprice", 5)
    is_raw = pds.estimate_day(pf_raw(df, "SPY"), pf_raw(df, "ES"), n_lags=5)["IS_mid_ES"]
    is_mic = pds.estimate_day(pf_mic(df, "SPY"), pf_mic(df, "ES"), n_lags=5)["IS_mid_ES"]
    cdf = coarsen(df, step=10)                             # sparse sampling
    is_sparse = pds.estimate_day(pf_raw(cdf, "SPY"), pf_raw(cdf, "ES"), n_lags=5)["IS_mid_ES"]
    pdf = coarsen(df, step=10, pre_average=True)           # pre-average + subsample
    is_pa = pds.estimate_day(pf_mic(pdf, "SPY"), pf_mic(pdf, "ES"), n_lags=5)["IS_mid_ES"]
    print("\n(B) noise-robust information share")
    print("    truth IS_ES=%.3f | raw mid=%.3f | microprice=%.3f | sparse(/10)=%.3f | microprice+pre-avg(/10)=%.3f"
          % (truth, is_raw, is_mic, is_sparse, is_pa))
    b_ok = bool(abs(is_mic - truth) < abs(is_raw - truth)
                and abs(is_sparse - truth) < abs(is_raw - truth) and abs(is_raw - 0.5) < abs(truth - 0.5))
    print("    checks:", b_ok)

    # (C) ecm_sde hook: the noise-robust observable feeds the state-dependent SDE and
    #     returns a well-formed information-share curve in [0,1].
    sessions = [("d1", "benchmark", df)]
    fit = noise_robust_is(sessions, scheme="microprice", n_levels=5, degree=1, n_lags=0)
    is_es_col = fit["curve"]["IS_ES"].to_numpy(float)
    fin = np.isfinite(fit["a1_SPY"]) and np.isfinite(fit["t_a1_SPY"]) and bool(np.all(np.isfinite(is_es_col)))
    inrange = bool(np.all((is_es_col >= -1e-6) & (is_es_col <= 1 + 1e-6)))
    leads = float(np.median(is_es_col)) > 0.5             # DGP has ES leading
    print("\n(C) ecm_sde hook on noise-robust prices")
    print("    a1_SPY=%.4f  t(a1_SPY)=%.2f  median IS_ES=%.3f  curve in [%.3f, %.3f]"
          % (fit["a1_SPY"], fit["t_a1_SPY"], float(np.median(is_es_col)),
             float(np.min(is_es_col)), float(np.max(is_es_col))))
    c_ok = fin and inrange and leads
    print("    checks:", c_ok)
    return bool(a_ok and b_ok and c_ok)

if __name__ == "__main__":
    # A self-test that prints False and still exits 0 is invisible to the smoke loop the
    # README recommends (`for m in ...; do python "$m.py"; done`), so a broken module reads
    # as green. Exit non-zero on failure.
    import sys as _sys
    _sys.exit(0 if _selftest() else 1)
