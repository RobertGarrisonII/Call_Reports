"""
pricing_error.py
===============
Hasbrouck (1993) pricing-error decomposition and the dollar welfare cost of the
dislocation -- item #5. The information-share machinery says *who* leads; this says *how
much transitory mispricing* there is and what it costs. The observed log price decomposes
(Beveridge-Nelson) into a random-walk efficient price and a stationary pricing error:

    p_t = m_t + s_t,   m_t = m_{t-1} + w_t,   s_t stationary.

sigma_s = sd(s_t) is Hasbrouck's market-quality measure: the magnitude of transitory
deviations from the efficient price -- exactly the dislocation the MWCB reopening asymmetry
injects. It is **reduced-form identified** (no ordering assumption): from the VMA of
(Delta p, order flow), Delta p_t = theta(L) v_t with innovation covariance Omega,

    sigma_w^2 = theta(1) Omega theta(1)'                 (permanent / efficient variance)
    sigma_s^2 = sum_k theta*_k Omega theta*_k',  theta*_k = sum_{j>k} theta_j  (transitory).

The welfare angle: a stress-period rise in sigma_s is the price-efficiency cost of the
dislocation, and notional x sigma_s is its dollar cost. The cross-market version decomposes
the SPY-ES **basis** -- its transitory variance is the cross-market dislocation, the
contagion object. Pure numpy/pandas; reuses the VAR/VMA in correlation_svar.
"""
import numpy as np
import pandas as pd

import cross_asset_pd_liquidity as ca
import correlation_svar as csv

EPS = 1e-12


# ── core: Beveridge-Nelson permanent / transitory variance ────────────────────
def pricing_error_decomposition(z, p=5, horizon=300, price_row=0):
    """Hasbrouck/Beveridge-Nelson decomposition from a VAR in z (T x k; column `price_row`
    is the price change Delta p, the rest are order-flow / informational proxies). Returns
    sigma_s (transitory pricing-error sd), sigma_w (efficient-price innovation sd), their
    squares, and the noise ratio sigma_s/sigma_w. Units follow z (use bps returns -> bps).
    Reduced-form identified: no Cholesky / ordering."""
    z = np.asarray(z, float)
    A, Sigma, _ = csv.var_ols(z, p)
    Psi = csv._ma_from_var(A, horizon)                       # Psi_0..Psi_H (k x k)
    theta = np.array([Psi[k][price_row, :] for k in range(horizon + 1)])   # (H+1, k)
    theta1 = theta.sum(axis=0)                               # cumulative (long-run) impact
    sigma_w2 = float(theta1 @ Sigma @ theta1)
    tail = np.cumsum(theta[::-1], axis=0)[::-1]              # tail[m] = sum_{j>=m} theta_j
    sigma_s2 = 0.0
    for m in range(1, horizon + 1):                          # theta*_k = tail[k+1], k=0..H-1
        tk = tail[m]
        sigma_s2 += float(tk @ Sigma @ tk)
    sigma_s = np.sqrt(max(sigma_s2, 0.0)); sigma_w = np.sqrt(max(sigma_w2, 0.0))
    return {"sigma_s": sigma_s, "sigma_w": sigma_w, "sigma_s2": sigma_s2, "sigma_w2": sigma_w2,
            "noise_ratio": sigma_s / sigma_w if sigma_w > EPS else np.nan,
            "ma_tail_residual": float(np.abs(Psi[horizon][price_row, :]).sum())}


# ── build the (Delta p, order flow) frame for one market ──────────────────────
def _ret_ofi(df, asset, n_levels):
    mid = np.asarray(ca._mid(df, asset), float)
    r = np.diff(np.log(np.where(mid > EPS, mid, np.nan))) * 1e4          # bps
    ofi = ca.order_flow_imbalance(df, asset, n_levels)[1:]
    z = np.column_stack([r, ofi])
    return z[np.all(np.isfinite(z), axis=1)]


def market_quality(df, asset, n_levels=10, p=5, horizon=300):
    """sigma_s / sigma_w (bps) for one market, from a VAR in (mid return, signed OFI)."""
    return pricing_error_decomposition(_ret_ofi(df, asset, n_levels), p, horizon)


# ── dollar welfare cost ───────────────────────────────────────────────────────
def welfare_cost(sigma_s_bps, notional, expected_abs=True):
    """Dollar cost of transitory mispricing: notional traded x the average fractional
    deviation from the efficient price. sigma_s is in bps; expected_abs=True uses the
    expected absolute deviation E|s| = sqrt(2/pi) sigma_s of a normal pricing error."""
    frac = sigma_s_bps / 1e4
    factor = np.sqrt(2.0 / np.pi) if expected_abs else 1.0
    return float(notional) * frac * factor


def dislocation_cost(dec_calm, dec_stress, notional_stress, expected_abs=True):
    """The dislocation's price-efficiency cost: the rise in sigma_s from calm to stress and
    its dollar cost on the stress-period notional. dec_* are pricing_error_decomposition
    outputs (or market_quality)."""
    d = dec_stress["sigma_s"] - dec_calm["sigma_s"]
    return {"sigma_s_calm": dec_calm["sigma_s"], "sigma_s_stress": dec_stress["sigma_s"],
            "delta_sigma_s_bps": d,
            "stress_total_cost": welfare_cost(dec_stress["sigma_s"], notional_stress, expected_abs),
            "incremental_cost": welfare_cost(max(d, 0.0), notional_stress, expected_abs)}


# ── cross-market relative pricing error (the basis dislocation) ───────────────
def relative_pricing_error(df, asset_a="SPY", asset_b="ES", ewma_halflife=300):
    """Cross-market dislocation: the transitory deviation of the SPY-ES **basis**
    b_t = log mid_a - log mid_b. Because the pair is cointegrated, the basis is stationary
    and its (high-pass) transitory standard deviation -- the basis minus a slow EWMA fair
    value -- is the cross-market mispricing magnitude (bps), the contagion object; its rise
    in stress is liquidity contagion. `sd_basis` is the total demeaned-basis sd. (A full
    bivariate-VECM Hasbrouck pricing-error decomposition is the rigorous extension; the
    high-pass deviation avoids the over-differencing that plagues a VAR on the stationary
    basis's first difference.)"""
    ma = np.asarray(ca._mid(df, asset_a), float); mb = np.asarray(ca._mid(df, asset_b), float)
    basis_bps = (np.log(np.where(ma > EPS, ma, np.nan)) - np.log(np.where(mb > EPS, mb, np.nan))) * 1e4
    s = pd.Series(basis_bps)
    sd_basis = float((s - s.mean()).std())
    transitory = s - s.ewm(halflife=ewma_halflife, min_periods=max(5, ewma_halflife // 5)).mean()
    return {"sigma_s_basis": float(transitory.std()), "sd_basis": sd_basis,
            "ewma_halflife": ewma_halflife}


# ── self-test ─────────────────────────────────────────────────────────────────
def _sim_roll(T, sigma_e, b, x_noise=0.2, seed=0):
    """Roll/MA(1) model: a genuine I(1) price with Delta p_t = e_t + b e_{t-1} (b<0 =
    bid-ask bounce). The efficient-price innovation is (1+b)e_t; the pricing error is the
    transitory bounce. True sigma_s = |b| sigma_e, sigma_w = |1+b| sigma_e -- an unambiguous
    Beveridge-Nelson benchmark. Order flow x is a noisy signal of e."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma_e, T)
    dp = e[1:] + b * e[:-1]
    x = e[1:] + rng.normal(0, x_noise, T - 1)
    return np.column_stack([dp, x]), abs(b) * sigma_e, abs(1 + b) * sigma_e


def _selftest():
    print("pricing_error self-test")

    # (1) recover sigma_s and sigma_w on the Roll/MA(1) model (exact BN benchmark)
    z, true_ss, true_sw = _sim_roll(40000, sigma_e=0.5, b=-0.4, seed=1)
    dec = pricing_error_decomposition(z, p=15, horizon=200)
    print("\n(1) Beveridge-Nelson recovery (Roll model: true sigma_s=%.3f, sigma_w=%.3f)" % (true_ss, true_sw))
    print("    estimated sigma_s=%.3f  sigma_w=%.3f  noise ratio=%.3f  (MA tail residual=%.1e)"
          % (dec["sigma_s"], dec["sigma_w"], dec["noise_ratio"], dec["ma_tail_residual"]))
    print("    checks:", abs(dec["sigma_s"] - true_ss) / true_ss < 0.15
          and abs(dec["sigma_w"] - true_sw) / true_sw < 0.15 and dec["ma_tail_residual"] < 1e-3)

    # (2) a (near-)efficient price (b~0) has ~zero pricing error
    z_eff, _, _ = _sim_roll(40000, sigma_e=0.5, b=-0.02, seed=2)
    dec_eff = pricing_error_decomposition(z_eff, p=15, horizon=200)
    print("\n(2) efficient price -> sigma_s ~ 0")
    print("    sigma_s=%.4f (<< sigma_w=%.3f)" % (dec_eff["sigma_s"], dec_eff["sigma_w"]))
    print("    checks:", dec_eff["sigma_s"] < 0.1 * dec_eff["sigma_w"])

    # (3) stress (bigger bounce) raises sigma_s, and the dollar welfare cost rises with it
    z_calm, _, _ = _sim_roll(40000, sigma_e=0.5, b=-0.30, seed=3)
    z_str, _, _ = _sim_roll(40000, sigma_e=0.7, b=-0.55, seed=4)
    d_calm = pricing_error_decomposition(z_calm, p=15, horizon=200)
    d_str = pricing_error_decomposition(z_str, p=15, horizon=200)
    cost = dislocation_cost(d_calm, d_str, notional_stress=5e9)
    print("\n(3) stress dislocation + dollar welfare cost (notional $5B)")
    print("    sigma_s: calm=%.3f -> stress=%.3f bps  (delta=%.3f)"
          % (cost["sigma_s_calm"], cost["sigma_s_stress"], cost["delta_sigma_s_bps"]))
    print("    incremental dislocation cost = $%.2fM   (stress total = $%.2fM)"
          % (cost["incremental_cost"] / 1e6, cost["stress_total_cost"] / 1e6))
    print("    checks:", cost["sigma_s_stress"] > cost["sigma_s_calm"]
          and cost["incremental_cost"] > 0
          and welfare_cost(2.0, 1e9) > welfare_cost(1.0, 1e9))

    # (4) cross-market basis dislocation: transitory sd ~ total demeaned-basis sd
    rng = np.random.default_rng(7); n = 8000
    f = rng.normal(0, 1, n).cumsum()                            # common efficient log-trend
    bspread = np.zeros(n)
    for t in range(1, n):
        bspread[t] = 0.8 * bspread[t - 1] + rng.normal(0, 0.0005)   # stationary basis dislocation
    cols = {}
    for a, base, tick, sgn in (("SPY", 500.0, 0.01, +0.5), ("ES", 5000.0, 0.25, -0.5)):
        mid = base * np.exp(f / 50.0 + sgn * bspread)
        for l in range(1, 11):
            cols[f"{a}_bidprice_{l}"] = mid - l * tick; cols[f"{a}_askprice_{l}"] = mid + l * tick
            q = np.maximum(300 * np.exp(-0.3 * (l - 1)) * (1 + rng.normal(0, 0.05, n)), 1.0)
            cols[f"{a}_bidquantity_{l}"] = q; cols[f"{a}_askquantity_{l}"] = q
    book = pd.DataFrame(cols, index=pd.date_range("2020-03-09 09:30", periods=n, freq="s", tz="America/New_York"))
    rel = relative_pricing_error(book, "SPY", "ES", ewma_halflife=400)
    print("\n(4) cross-market basis dislocation (true basis sd ~ %.2f bps)"
          % (0.0005 / np.sqrt(1 - 0.8 ** 2) * 1e4))
    print("    sigma_s_basis (transitory) = %.3f bps   sd_basis = %.3f bps" % (rel["sigma_s_basis"], rel["sd_basis"]))
    print("    checks:", rel["sigma_s_basis"] > 0 and rel["sd_basis"] > 0
          and 0.5 * rel["sd_basis"] <= rel["sigma_s_basis"] <= 1.5 * rel["sd_basis"])


if __name__ == "__main__":
    _selftest()
