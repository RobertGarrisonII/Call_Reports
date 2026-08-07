"""
irf.py
======
Dynamic layer over the cross-impact and information-share modules: how a shock
PROPAGATES, with credible identification and state dependence. This is the
dynamic face of the liquidity-conditional result -- and the identification
backstop for the paper's existing "spillover decays within a second" claim.

  local_projection / local_projection_irf
      Jorda (2005) local projections. Misspecification-robust, trivially
      state-dependent (Ramey-Zubairy): the cross-asset response becomes a
      FUNCTION of the relative book state. Block-bootstrap bands.

  structural_vecm_irf
      Structural IRFs from the VECM, identified off the estimated CROSS-IMPACT
      matrix (contemporaneous OFI->return map) instead of an arbitrary Cholesky
      ordering. The information shares come from the same VMA representation, so
      these IRFs make the share numbers transparent rather than black-box.

  fevd_from_irf
      Forecast-error variance decomposition: how much of each asset's return-
      forecast-error variance is driven by each (own / cross) OFI shock.

Interpretation guidance lives in README.md (section 'Interpreting the dynamics').
Requires cross_asset_pd_liquidity.py, cross_impact.py, price_discovery_shares.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import price_discovery_shares as pds
import cross_asset_pd_liquidity as ca
import cross_impact as cimp

EPS = 1e-12


# ── Jorda local projections (state-dependent, block-bootstrap bands) ──────────
def local_projection(y, shock, controls=None, state=None, horizons=range(0, 16),
                     n_boot=300, block=20, seed=0, s_lo=-1.0, s_hi=1.0):
    """Cumulative LP IRF of price (cumsum of return y) to a unit `shock`.
    If `state` given, returns the IRF at s_lo / s_hi (standardized state)."""
    y = np.asarray(y, float); shock = np.asarray(shock, float); T = len(y)
    # Halt-masked returns are NaN, and np.cumsum propagates the FIRST NaN to every later element --
    # so one early halt (2020-03-09: 09:34) turned every cumulative response of the day NaN and the
    # whole LP table came back empty (the 2026-08-05 masked run). nancumsum keeps the running sum
    # finite; the gap-count prefix sum then invalidates exactly the response windows that SPAN a
    # masked return -- a window across the halt is undefined (not zero, which is what nancumsum
    # alone would silently say), while windows entirely before or after stay estimable.
    cumy = np.concatenate([[0.0], np.nancumsum(y)])          # response_h(t)=cumy[t+h+1]-cumy[t]
    ngap = np.concatenate([[0], np.cumsum(~np.isfinite(y))])
    s = None if state is None else (np.asarray(state, float) - np.nanmean(state)) / (np.nanstd(state) + EPS)
    rng = np.random.default_rng(seed); rows = []
    def _nan_row(h):
        r = {"horizon": h, "theta": np.nan, "lo": np.nan, "hi": np.nan}
        if state is not None:
            r.update({"theta_low": np.nan, "low_lo": np.nan, "low_hi": np.nan,
                      "theta_high": np.nan, "high_lo": np.nan, "high_hi": np.nan})
        return r
    for h in horizons:
        n = T - h
        t = np.arange(n)
        Yh = cumy[t + h + 1] - cumy[t]
        cols = [np.ones(n), shock[t]]
        if s is not None:
            cols += [s[t], shock[t] * s[t]]
        if controls is not None:
            cols.append(np.asarray(controls)[t])
        X = np.column_stack(cols)
        spans_gap = (ngap[t + h + 1] - ngap[t]) > 0          # response window crosses the halt
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(Yh) & ~spans_gap
        Xh, Yhh = X[ok], Yh[ok]; m = Xh.shape[0]; ncol = Xh.shape[1]
        if m < ncol + 2:                                     # too few finite obs at this horizon
            rows.append(_nan_row(h)); continue
        b = np.linalg.lstsq(Xh, Yhh, rcond=None)[0]
        theta = b[1]; delta = b[3] if s is not None else 0.0
        blk = max(1, min(block, m)); nblk = int(np.ceil(m / blk))
        th, lo_b, hi_b = [], [], []
        for _ in range(n_boot):
            st = rng.integers(0, m - blk + 1, size=nblk)
            ridx = np.concatenate([np.arange(j, j + blk) for j in st])[:m]
            bb = np.linalg.lstsq(Xh[ridx], Yhh[ridx], rcond=None)[0]
            th.append(bb[1])
            if s is not None:
                lo_b.append(bb[1] + bb[3] * s_lo); hi_b.append(bb[1] + bb[3] * s_hi)
        th = np.array(th)
        row = {"horizon": h, "theta": theta,
               "lo": np.quantile(th, .025), "hi": np.quantile(th, .975)}
        if s is not None:
            lo_b, hi_b = np.array(lo_b), np.array(hi_b)
            row.update({"theta_low": theta + delta * s_lo,
                        "low_lo": np.quantile(lo_b, .025), "low_hi": np.quantile(lo_b, .975),
                        "theta_high": theta + delta * s_hi,
                        "high_lo": np.quantile(hi_b, .025), "high_hi": np.quantile(hi_b, .975)})
        rows.append(row)
    return pd.DataFrame(rows)


def local_projection_irf(df, impulse="ES", response="SPY", state=None,
                         horizons=None, n_lags=None, n_levels=10,
                         n_boot=300, block=None, standardize_shock=True,
                         min_rest_steps=0):
    """Cross-asset LP: response-asset price reaction to an impulse-asset OFI shock,
    controlling for lags of both returns and both OFIs. `state` may be a 1s series
    (e.g. ca.relative_depth_state(df)) for the state-dependent IRF. With
    standardize_shock the response reads as cumulative return per 1-SD OFI impulse
    (comparable to the structural IRF and the standardized cross-impact matrix).
    horizons / n_lags / block default to frequency-scaled values from the grid
    (ca.frequency_defaults); min_rest_steps>0 filters fleeting quotes from the OFI."""
    fd = ca.frequency_defaults(df)
    if horizons is None: horizons = fd["horizons"]
    if n_lags is None: n_lags = fd["n_lags"]
    if block is None: block = fd["block"]
    ret = {a: np.diff(np.log(ca._mid(df, a))) for a in (impulse, response)}
    ofi = {a: ca.order_flow_imbalance(df, a, n_levels, min_rest_steps)[1:] for a in (impulse, response)}
    T = len(ret[response])
    y = ret[response]; shock = ofi[impulse]
    if standardize_shock:
        shock = (shock - shock.mean()) / (shock.std() + EPS)
    base = [ret[response], ret[impulse], ofi[response], ofi[impulse]]
    ctrl = []
    for L in range(1, n_lags + 1):
        for series in base:
            lag = np.full(T, np.nan); lag[L:] = series[:-L]; ctrl.append(lag)
    controls = np.column_stack(ctrl)
    st = None if state is None else np.asarray(state)[1:][:T]
    return local_projection(y, shock, controls=controls, state=st,
                            horizons=horizons, n_boot=n_boot, block=block)


# ── structural VECM IRF identified off the cross-impact matrix ────────────────
def _fit_vecm_full(p1, p2, n_lags):
    dY, X = pds._design_within_day(p1, p2, n_lags)
    ok = np.all(np.isfinite(dY), axis=1) & np.all(np.isfinite(X), axis=1)
    dY, X = dY[ok], X[ok]
    B, resid = pds._ols(dY, X)
    alpha = B[1, :].copy()
    Gammas = []
    for i in range(1, n_lags + 1):
        G = np.zeros((2, 2))
        G[:, 0] = B[2 + 2 * (i - 1), :]
        G[:, 1] = B[2 + 2 * (i - 1) + 1, :]
        Gammas.append(G)
    return alpha, Gammas, np.cov(resid, rowvar=False, bias=True)

def _ma_coefficients(alpha, Gammas, horizons_max, beta=(1.0, -1.0)):
    """VMA (level) impulse responses C_h of a VECM with fixed beta."""
    k = 2; p = len(Gammas)
    Pi = np.outer(np.asarray(alpha, float), np.asarray(beta, float))
    A = [None] * (p + 1)                                     # A[0]=A_1 ... A[p]=A_{p+1}
    A[0] = np.eye(k) + Pi + (Gammas[0] if p >= 1 else 0.0)
    for i in range(1, p):
        A[i] = Gammas[i] - Gammas[i - 1]
    if p >= 1:
        A[p] = -Gammas[p - 1]
    C = [np.eye(k)]
    for h in range(1, horizons_max + 1):
        Ch = np.zeros((k, k))
        for j in range(1, min(h, p + 1) + 1):
            Ch = Ch + A[j - 1] @ C[h - j]
        C.append(Ch)
    return C

def structural_vecm_irf(df, horizons=None, n_lags=None, n_levels=10, hac_lags=10,
                        min_rest_steps=0):
    """Structural IRF of (SPY,ES) prices/returns to a 1-SD OFI shock in each
    market, with B identified from the cross-impact matrix (not Cholesky).
    horizons / n_lags default to frequency-scaled values from the grid;
    min_rest_steps>0 filters fleeting quotes from the OFI used to identify B."""
    fd = ca.frequency_defaults(df)
    if horizons is None: horizons = fd["horizons"]
    if n_lags is None: n_lags = fd["n_lags"]
    p1 = np.log(ca._mid(df, "SPY")); p2 = np.log(ca._mid(df, "ES"))
    alpha, Gammas, _ = _fit_vecm_full(p1, p2, n_lags)
    Lam = cimp.cross_impact_matrix(df, n_levels=n_levels, hac_lags=hac_lags,
                                   standardize=False, min_rest_steps=min_rest_steps)["Lambda"]
    sig = np.array([np.std(ca.order_flow_imbalance(df, a, n_levels, min_rest_steps)[1:])
                    for a in ("SPY", "ES")])
    B = Lam @ np.diag(sig)                                   # 1-SD OFI shock -> price innovation
    H = max(horizons)
    C = _ma_coefficients(alpha, Gammas, H)
    level = np.array([C[h] @ B for h in range(H + 1)])       # price IRF
    ret = np.array([(C[h] - (C[h - 1] if h > 0 else np.zeros((2, 2)))) @ B for h in range(H + 1)])
    return {"level_irf": level, "return_irf": ret, "B": B,
            "assets": ["SPY", "ES"], "shocks": ["OFI_SPY", "OFI_ES"], "horizons": list(range(H + 1))}

def fevd_from_irf(return_irf, H=None):
    """Forecast-error variance decomposition from structural return IRFs.
    Returns a (response x shock) matrix whose ROWS sum to 1.

    ASSUMPTION, stated because this stack's own results contradict it: the decomposition sums
    squared responses PER SHOCK, which partitions the forecast-error variance only if the
    structural shocks are mutually UNCORRELATED with unit variance. B here is identified from
    the cross-impact matrix (B = Lambda diag(sd(OFI))), which normalizes each shock's scale but
    does NOT orthogonalize across shocks -- and the paper's tandem-flow finding is precisely
    that SPY and ES order flow are strongly correlated. With corr(OFI_SPY, OFI_ES) = r != 0 the
    cross terms 2 r C_h b_i b_j' are dropped, so the shares are a DIAGONAL approximation that
    understates common-flow variance. Read them next to the reported `ofi_innovation_corr`:
    near 0 the shares are clean; at the tandem-flow magnitudes they attribute variance to
    'SPY flow' vs 'ES flow' that the data cannot separate."""
    D = np.asarray(return_irf); H = (len(D) - 1) if H is None else H
    num = np.sum(D[:H + 1] ** 2, axis=0)                     # k x k: sum over horizons
    tot = num.sum(axis=1, keepdims=True)
    return num / (tot + EPS)


# ── self-test ────────────────────────────────────────────────────────────────
def _selftest():
    rng = np.random.default_rng(9)

    # (1) LP recovers a known cumulative MA response.
    T = 8000; shock = rng.normal(0, 1, T)
    w = np.array([0.5, 0.3, 0.1])                            # impulse weights
    y = np.zeros(T)
    for L, wl in enumerate(w):
        y[L:] += wl * shock[:T - L]
    y += rng.normal(0, 0.1, T)
    lp = local_projection(y, shock, horizons=range(0, 6), n_boot=200, block=15)
    cum_true = np.cumsum(w)                                  # 0.5, 0.8, 0.9, 0.9...
    print("LP cumulative IRF vs truth:")
    for h in range(5):
        tt = cum_true[min(h, len(w) - 1)]
        print("  h=%d  theta=%.3f  (true %.3f)  [%.3f, %.3f]"
              % (h, lp.theta[h], tt, lp.lo[h], lp.hi[h]))

    # (2) state-dependent LP: response strengthens with the state.
    s = rng.normal(0, 1, T)
    y2 = (0.5 + 0.4 * s) * shock + rng.normal(0, 0.1, T)
    lp2 = local_projection(y2, shock, state=s, horizons=range(0, 3), n_boot=200, block=15)
    print("\nstate-dependent LP at h=0: low=%.3f  high=%.3f  (true 0.1 / 0.9)"
          % (lp2.theta_low[0], lp2.theta_high[0]))
    print("  high > low and bands exclude each other:",
          bool(lp2.theta_high[0] > lp2.theta_low[0] and lp2.high_lo[0] > lp2.low_hi[0]))

    # (3) structural IRF + FEVD from known matrices (ES OFI -> SPY return).
    alpha = np.array([-0.05, 0.03]); Gammas = [np.array([[0.10, 0.0], [0.0, 0.10]])]
    C = _ma_coefficients(alpha, Gammas, 30)
    Lam = np.array([[0.6, 0.25], [0.05, 0.6]]); B = Lam @ np.diag([1.0, 1.0])
    ret = np.array([(C[h] - (C[h - 1] if h > 0 else 0)) @ B for h in range(31)])
    fe = fevd_from_irf(ret, H=30)
    print("\nstructural IRF/FEVD:")
    print("  return_irf[0] == B:", np.allclose(ret[0], B))
    print("  return IRF decays (|h=30| small):", float(np.abs(ret[30]).max()) < 0.05)
    print("  FEVD rows sum to 1:", np.allclose(fe.sum(1), 1.0))
    print("  share of SPY return var from ES-OFI shock: %.3f" % fe[0, 1])
    print("  -> cross-asset propagation registered:", fe[0, 1] > 0.05)

    # (4) non-finite robustness: NaN segments / zero (->log -inf) prices must not crash.
    rng2 = np.random.default_rng(123)
    yy = rng2.normal(0, 0.01, 1500); yy[300:1200] = np.nan; sk = rng2.normal(0, 1, 1500)
    lpn = local_projection(yy, sk, horizons=range(0, 5), n_boot=30)
    pf = np.cumsum(rng2.normal(0, 0.01, 800)) + 100.0; pf[::40] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        af, _, _ = _fit_vecm_full(np.log(np.where(pf > 0, pf, np.nan)), np.log(pf + 0.01), 5)
    print("\nnon-finite robustness:")
    print("  local_projection no-crash, all rows present:", len(lpn) == 5)
    print("  _fit_vecm_full alpha finite under NaN prices:", bool(np.all(np.isfinite(af))))

if __name__ == "__main__":
    _selftest()
