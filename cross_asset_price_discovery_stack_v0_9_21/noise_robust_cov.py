"""
noise_robust_cov.py
===================
Microstructure-noise- and asynchronicity-robust realized (co)variance for the
sub-second (100ms / 10ms) analyses. At those frequencies naive realized variance
EXPLODES (noise accumulates) and naive realized correlation is ATTENUATED by the
Epps effect (SPY and ES update asynchronously). This module supplies the
estimators that fix both, with the same per-session interface as
cross_asset_pd_liquidity.realized_moments, and feeds dcc_garch.

Noise-robust variance (univariate):
  * realized_variance(..., method="rv")      naive (baseline / signature plot)
  * realized_variance(..., method="tsrv")    Two-Scale RV (Zhang-Mykland-Ait-Sahalia 2005)
  * realized_variance(..., method="kernel")  Realized Kernel, Parzen (BNHLS 2008), data-driven bandwidth

Asynchronicity-robust covariance:
  * hayashi_yoshida(...)                      async covariance (Hayashi-Yoshida 2005), no synchronization
  * refresh_time(...) / realized_kernel_cov   refresh-time sync (BNHLS 2011) + multivariate realized kernel

Diagnostics:
  * epps_curve(...)        naive corr vs sampling interval, with the HY reference
  * signature_plot(...)    RV vs sampling interval (naive blows up, TSRV flat)

Convenience:
  * noise_robust_moments(...)   robust RV / cov / corr mirroring realized_moments()

Inputs to the covariance estimators are per-asset TIMESTAMPED log-price series
(t, logprice), ideally the raw irregular quote-update times from MayStreet (that
is where async robustness matters); the variance estimators take any fine series.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-15


# ── kernels ──────────────────────────────────────────────────────────────────
def _parzen(x):
    x = np.asarray(x, float)
    out = np.where(x <= 0.5, 1 - 6 * x ** 2 + 6 * x ** 3, 2 * (1 - x) ** 3)
    return np.where(x > 1, 0.0, out)


# ── univariate noise-robust realized variance ────────────────────────────────
def _tsrv(logp, K=None):
    r = np.diff(logp); n = len(r)
    if K is None:
        K = max(2, int(round(n ** (1.0 / 3.0))))
    rv_all = np.sum(r ** 2)
    rv_avg = np.mean([np.sum(np.diff(logp[k::K]) ** 2) for k in range(K)])
    nbar = (n - K + 1) / K
    tsrv = (rv_avg - (nbar / n) * rv_all) / (1.0 - nbar / n)
    return max(tsrv, EPS)

def _noise_var(r):
    """Bandi-Russell / BNHLS noise-variance estimate omega^2 ~ RV/(2n)."""
    return np.sum(r ** 2) / (2.0 * len(r))

def _bnhls_bandwidth(r):
    n = len(r); om2 = _noise_var(r)
    iv = _tsrv_from_returns(r)
    xi2 = om2 / np.sqrt(max(iv, EPS) ** 2)            # xi^2 ~ omega^2 / sqrt(IQ), IQ~IV^2
    H = 3.5134 * (xi2 ** 0.4) * (n ** 0.6)
    return int(np.clip(round(H), 2, n - 1))

def _tsrv_from_returns(r):
    logp = np.concatenate([[0.0], np.cumsum(r)])
    return _tsrv(logp)

def _realized_kernel_uni(logp, H=None):
    r = np.diff(logp); n = len(r)
    if H is None:
        H = _bnhls_bandwidth(r)
    g0 = np.sum(r * r); rk = g0
    for h in range(1, H + 1):
        gh = np.sum(r[h:] * r[:-h])
        rk += _parzen((h - 1) / H) * 2.0 * gh
    return max(rk, EPS)

def realized_variance(logp, method="tsrv", **kw):
    logp = np.asarray(logp, float)
    if method == "rv":
        return float(np.sum(np.diff(logp) ** 2))
    if method == "tsrv":
        return float(_tsrv(logp, kw.get("K")))
    if method == "kernel":
        return float(_realized_kernel_uni(logp, kw.get("H")))
    if method == "bipower":
        return bipower_variation(logp)
    if method == "threshold":
        return threshold_variance(logp, kw.get("c", 3.0), kw.get("n_iter", 4))
    raise ValueError(f"unknown method {method!r}")


# ── jump-robust variation (MWCB days are jump-prone; H3) ──────────────────────
_MU1 = np.sqrt(2.0 / np.pi)                       # E|Z|, Z ~ N(0,1)

def bipower_variation(logp):
    """Barndorff-Nielsen & Shephard (2004) realized bipower variation -- a JUMP-ROBUST estimator of
    integrated variance: BV = mu1^{-2} * sum_{i>=2} |r_i||r_{i-1}|, mu1 = sqrt(2/pi). A finite-activity
    jump lands in one |r_i| only, multiplied by an O(sqrt(dt)) neighbour, so jumps drop out
    asymptotically while naive RV absorbs them. Pairs with realized_variance(...,'rv') to split the
    quadratic variation into continuous (BV) and jump (RV-BV) parts."""
    r = np.diff(np.asarray(logp, float)); a = np.abs(r)
    return float(_MU1 ** (-2) * np.sum(a[1:] * a[:-1]))


def tripower_quarticity(logp):
    """Realized tripower quarticity (BNS 2006): the jump-robust integrated-quarticity estimate that
    standardizes the jump test. TQ = n * mu_{4/3}^{-3} * sum |r_i|^{4/3}|r_{i-1}|^{4/3}|r_{i-2}|^{4/3}."""
    from math import gamma
    r = np.diff(np.asarray(logp, float)); n = len(r)
    if n < 3:
        return float(np.sum(r ** 4))
    mu43 = 2.0 ** (2.0 / 3.0) * gamma(7.0 / 6.0) / gamma(0.5)
    a = np.abs(r) ** (4.0 / 3.0)
    return float(n * mu43 ** (-3) * np.sum(a[2:] * a[1:-1] * a[:-2]))


def threshold_variance(logp, c=3.0, n_iter=4):
    """Mancini (2009) truncated / threshold realized variance: TRV = sum r_i^2 1{|r_i| <= thr}, with
    thr = c * sigma_hat set from a jump-robust per-step scale and iterated (estimate scale -> truncate
    -> re-estimate). Discards the jump returns, leaving the continuous integrated variance. An
    alternative to bipower with a tunable truncation; the two should agree absent jumps."""
    r = np.diff(np.asarray(logp, float)); n = max(len(r), 1)
    sigma = np.sqrt(max(bipower_variation(logp) / n, EPS))         # per-step continuous scale
    keep = np.ones(len(r), bool)
    for _ in range(int(n_iter)):
        keep = np.abs(r) <= c * sigma
        if keep.sum() < 2:
            break
        sigma = np.sqrt(max(np.mean(r[keep] ** 2), EPS))
    return float(np.sum(r[keep] ** 2))


def jump_test(logp):
    """Barndorff-Nielsen-Shephard / Huang-Tauchen ratio jump test. Returns RV, the jump-robust BV,
    the relative jump RJ=(RV-BV)/RV, the standardized statistic z (~N(0,1) under the no-jump null),
    the upper-tail p-value, and the estimated jump variance max(RV-BV,0). A large positive z flags a
    price discontinuity -- the natural application is the MWCB halt days (H3)."""
    logp = np.asarray(logp, float); r = np.diff(logp); n = len(r)
    rv = float(np.sum(r ** 2)); bv = bipower_variation(logp); tq = tripower_quarticity(logp)
    rj = (rv - bv) / rv if rv > EPS else 0.0
    theta = np.pi ** 2 / 4.0 + np.pi - 5.0                         # ~0.6090
    denom = np.sqrt(theta / max(n, 1) * max(1.0, tq / max(bv ** 2, EPS)))
    z = rj / denom if denom > EPS else 0.0
    from math import erfc
    p = 0.5 * erfc(z / np.sqrt(2.0))                               # one-sided upper (jumps -> z>0)
    return {"rv": rv, "bv": bv, "rj": rj, "z": float(z), "p_value": float(p),
            "jump_var": max(rv - bv, 0.0), "tq": tq}


# ── asynchronicity-robust covariance ─────────────────────────────────────────
def hayashi_yoshida(t1, lp1, t2, lp2):
    """Hayashi-Yoshida covariance of two ASYNCHRONOUS log-price series. O(n+m)."""
    t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
    r1 = np.diff(np.asarray(lp1, float)); a1, b1 = t1[:-1], t1[1:]
    r2 = np.diff(np.asarray(lp2, float)); a2, b2 = t2[:-1], t2[1:]
    cov = 0.0; m = len(r2); lo = 0
    for i in range(len(r1)):
        while lo < m and b2[lo] <= a1[i]:            # first j with overlap on the left
            lo += 1
        j = lo
        while j < m and a2[j] < b1[i]:               # all j overlapping interval i
            cov += r1[i] * r2[j]; j += 1
    return float(cov)


def _hy_cov_shifted(a1, b1, r1, a2, b2, r2, theta):
    """HY cross-covariance with the SECOND series' intervals shifted to (a2-theta, b2-theta].
    Same O(n+m) two-pointer as hayashi_yoshida (shifted Y is still time-sorted for fixed theta)."""
    as2 = a2 - theta; bs2 = b2 - theta
    cov = 0.0; m = len(r2); lo = 0
    for i in range(len(r1)):
        while lo < m and bs2[lo] <= a1[i]:
            lo += 1
        j = lo
        while j < m and as2[j] < b1[i]:
            cov += r1[i] * r2[j]; j += 1
    return cov


def lead_lag(t1, lp1, t2, lp2, max_lag, n_grid=81, normalize=True):
    """Hoffmann-Rosenbaum-Yoshida (2013) lead-lag estimator for two ASYNCHRONOUS price series.

    Scans the shifted HY cross-covariance contrast U(theta) over candidate shifts theta in
    [-max_lag, max_lag] (same time units as t1/t2) and returns the maximizer of |U|. This turns
    the paper's ASSUMED Cholesky ordering ("we take the futures market as the lead") into a
    measured, signed quantity that the data either supports or refutes.

    CONVENTION: a POSITIVE ``lead_lag`` means the FIRST series (t1/lp1) LEADS the second by that
    many time units (the second is a delayed reflection of the first); negative means the second
    leads. ``contrast`` is the cross-correlation-scaled curve over ``thetas`` for plotting/CIs.
    O((n+m) * n_grid); under heavy microstructure noise combine with pre-averaging (the raw HY
    contrast is consistent for the lead-lag but noisy at the finest scales)."""
    t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
    r1 = np.diff(np.asarray(lp1, float)); a1, b1 = t1[:-1], t1[1:]
    r2 = np.diff(np.asarray(lp2, float)); a2, b2 = t2[:-1], t2[1:]
    thetas = np.linspace(-float(max_lag), float(max_lag), int(n_grid))
    cov = np.array([_hy_cov_shifted(a1, b1, r1, a2, b2, r2, th) for th in thetas])
    if normalize:
        denom = np.sqrt(np.sum(r1 * r1) * np.sum(r2 * r2) + EPS)
        contrast = cov / denom
    else:
        contrast = cov
    k = int(np.argmax(np.abs(contrast)))
    th_hat = float(thetas[k])
    leader = "first" if th_hat > 0 else ("second" if th_hat < 0 else "none")
    return {"lead_lag": th_hat, "thetas": thetas, "contrast": contrast,
            "peak_contrast": float(contrast[k]), "leader": leader}


def _previous_tick(t, lp, grid):
    idx = np.searchsorted(t, grid, side="right") - 1
    idx = np.clip(idx, 0, len(lp) - 1)
    return np.asarray(lp)[idx]

def refresh_time(times_list):
    """BNHLS refresh times: the instants at which every series has updated."""
    K = len(times_list); pos = [0] * K; tau = []
    L = [len(t) for t in times_list]
    while all(pos[s] < L[s] for s in range(K)):
        t = max(times_list[s][pos[s]] for s in range(K))
        tau.append(t)
        for s in range(K):
            while pos[s] < L[s] and times_list[s][pos[s]] <= t:
                pos[s] += 1
    return np.array(tau)

def refresh_time_prices(times_list, prices_list):
    tau = refresh_time(times_list)
    P = np.column_stack([_previous_tick(times_list[s], prices_list[s], tau)
                         for s in range(len(times_list))])
    return tau, P

def realized_kernel_cov(R, H=None):
    """Multivariate realized kernel (Parzen) on refresh-time-synced returns R (T x k)."""
    R = np.asarray(R, float); n, k = R.shape
    if H is None:
        H = _bnhls_bandwidth(R[:, 0])
    rk = R.T @ R
    for h in range(1, H + 1):
        G = R[h:].T @ R[:-h]
        rk += _parzen((h - 1) / H) * (G + G.T)
    return rk


# ── diagnostics ──────────────────────────────────────────────────────────────
def epps_curve(t1, lp1, t2, lp2, intervals):
    """Naive realized correlation vs sampling interval (Epps), with HY reference."""
    rows = []
    lo = min(t1[0], t2[0]); hi = max(t1[-1], t2[-1])
    for dt in intervals:
        grid = np.arange(lo, hi + dt, dt)
        r1 = np.diff(_previous_tick(t1, lp1, grid)); r2 = np.diff(_previous_tick(t2, lp2, grid))
        cov = np.sum(r1 * r2); v1 = np.sum(r1 ** 2); v2 = np.sum(r2 ** 2)
        rows.append({"interval": dt, "corr_naive": cov / np.sqrt(v1 * v2 + EPS)})
    hy = hayashi_yoshida(t1, lp1, t2, lp2)
    rv1 = np.sum(np.diff(lp1) ** 2); rv2 = np.sum(np.diff(lp2) ** 2)
    df = pd.DataFrame(rows); df.attrs["corr_HY"] = hy / np.sqrt(rv1 * rv2 + EPS)
    return df

def signature_plot(t, lp, intervals):
    """Naive RV vs sampling interval (blows up as dt->0); TSRV reference (flat)."""
    rows = []
    grid_full = lp
    for dt in intervals:
        grid = np.arange(t[0], t[-1] + dt, dt)
        p = _previous_tick(t, lp, grid)
        rows.append({"interval": dt, "rv_naive": float(np.sum(np.diff(p) ** 2))})
    df = pd.DataFrame(rows); df.attrs["tsrv"] = _tsrv(np.asarray(lp))
    return df


# ── convenience: robust moments mirroring realized_moments() ─────────────────
def noise_robust_moments(t_spy, lp_spy, t_es, lp_es, var_method="tsrv"):
    """Async-robust covariance (Hayashi-Yoshida) + noise-robust variances (TSRV or
    kernel) -> robust correlation. Drop-in for cross_asset_pd_liquidity.realized_moments."""
    rv_spy = realized_variance(lp_spy, method=var_method)
    rv_es = realized_variance(lp_es, method=var_method)
    cov = hayashi_yoshida(t_spy, lp_spy, t_es, lp_es)
    corr = cov / np.sqrt(rv_spy * rv_es + EPS)
    return {"rv_spy": rv_spy, "rv_es": rv_es, "cov": cov,
            "corr": float(np.clip(corr, -1.0, 1.0))}


# ── self-test ────────────────────────────────────────────────────────────────
def _selftest():
    rng = np.random.default_rng(13)
    M = 40000
    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])           # integrated (co)variance over [0,1]
    dW = rng.multivariate_normal([0, 0], Sigma / M, size=M)
    X = np.cumsum(dW, axis=0)                            # efficient log-prices
    times = np.linspace(0.0, 1.0, M)
    omega = 1e-3
    Y = X + rng.normal(0, omega, (M, 2))                 # observed (noisy)

    print("=== univariate noise-robust variance (true IV = 1.0) ===")
    rv = realized_variance(Y[:, 0], "rv")
    ts = realized_variance(Y[:, 0], "tsrv")
    rk = realized_variance(Y[:, 0], "kernel")
    print("  RV naive   = %.3f   (inflated by ~2*M*omega^2 = %.3f)" % (rv, 2 * M * omega ** 2))
    print("  TSRV       = %.3f" % ts)
    print("  Kernel     = %.3f" % rk)
    print("  checks: TSRV & kernel near 1.0:", abs(ts - 1) < 0.1 and abs(rk - 1) < 0.15,
          "| naive inflated:", rv > 1.05)

    print("\n=== async covariance (true cov = 0.5, corr = 0.5) ===")
    # heavy, independent sparse sampling -> pronounced Epps attenuation at fine dt
    i1 = np.where(rng.random(M) < 0.25)[0]
    i2 = np.where(rng.random(M) < 0.25)[0]
    t1, p1 = times[i1], X[i1, 0]; t2, p2 = times[i2], X[i2, 1]
    hy = hayashi_yoshida(t1, p1, t2, p2)
    ep = epps_curve(t1, p1, t2, p2, intervals=[0.0002, 0.0005, 0.001, 0.005, 0.02])
    print("  HY covariance = %.3f  (true 0.5)" % hy)
    print("  HY correlation = %.3f  (Epps-immune)" % ep.attrs["corr_HY"])
    print("  Epps: naive corr by sampling interval (attenuated at fine dt, recovers as dt grows):")
    print("   ", dict(zip(ep.interval, ep.corr_naive.round(3))))
    print("  checks: HY cov ~0.5:", abs(hy - 0.5) < 0.08,
          "| naive(finest) attenuated below true/HY:",
          ep.corr_naive.iloc[0] < ep.attrs["corr_HY"] - 0.10)

    print("\n=== refresh-time + multivariate realized kernel ===")
    tau, P = refresh_time_prices([t1, t2], [p1, p2])
    rkc = realized_kernel_cov(np.diff(P, axis=0))
    print("  refresh-time points: %d ; RK cov_12 = %.3f (true 0.5)" % (len(tau), rkc[0, 1]))
    print("  check: RK cov ~0.5:", abs(rkc[0, 1] - 0.5) < 0.12)

    print("\n=== noise_robust_moments (noisy + async) ===")
    yp1 = X[i1, 0] + rng.normal(0, omega, len(i1))
    yp2 = X[i2, 1] + rng.normal(0, omega, len(i2))
    m = noise_robust_moments(t1, yp1, t2, yp2)
    print("  rv_spy=%.3f rv_es=%.3f cov=%.3f corr=%.3f (true 1,1,0.5,0.5)"
          % (m["rv_spy"], m["rv_es"], m["cov"], m["corr"]))

if __name__ == "__main__":
    _selftest()
