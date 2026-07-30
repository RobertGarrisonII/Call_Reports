"""
tick_correction.py
==================
Tick-size-aware price-discovery correction -- item #6, referee-proofing the paper's
futures-dominance result. The paper concedes the information-share asymmetry is partly a
discreteness artifact: SPY and ES sit on different price grids (1 ES tick = 0.25 index pts
~ 2.5 SPY ticks in index-point terms; in fractional terms SPY's $0.01/$550 ~ 1.8 bps is
the *coarser* grid, ES's 0.25/5500 ~ 0.45 bps the finer). A coarser grid injects rounding
noise (a price rounded to delta has error ~ U(-delta/2, delta/2), variance delta^2/12; the
return rounding noise has variance ~ delta^2/6), which contaminates the Hasbrouck
information share. The question a referee asks: does the lead survive once both markets are
put on equal footing?

Two corrections, both reusing the VECM / Hasbrouck machinery:
  * common-grid IS  -- round BOTH log prices to the same (coarser) fractional tick so
    neither instrument has a discreteness advantage, then re-estimate. Assumption-light.
  * rounding-corrected IS -- subtract each market's rounding-noise variance delta_i^2/6
    from the innovation covariance diagonal (a first-order de-rounding), then re-estimate.
    A tractable stand-in for the Hasbrouck (1999) rounded-observation state space.

Reports the raw, common-grid, and rounding-corrected information shares side by side, plus
each market's tick-noise share of return variance. Pure numpy/pandas; reuses
markov_switching_vecm (VECM fit, Hasbrouck-IS-from-alpha, Gonzalo-Granger).
"""
import numpy as np
import pandas as pd

import markov_switching_vecm as ms

EPS = 1e-12


# ── tick / rounding primitives ────────────────────────────────────────────────
def fractional_tick(tick, price):
    """Tick as a fraction of price (~ its size in return/log units). bps = 1e4 x this."""
    return float(tick) / float(price)


def rounding_return_var(tick, price):
    """Variance the tick rounding adds to one-period returns: delta^2/6 where
    delta = tick/price (price error ~ U(-delta/2, delta/2) has var delta^2/12; the return
    error is its first difference, var 2*delta^2/12). Upper bound on the contemporaneous
    contribution since the true rounding noise is MA(1) and partly absorbed by VAR lags."""
    d = fractional_tick(tick, price)
    return d * d / 6.0


def round_to_fractional_grid(logp, delta_frac):
    """Round a log-price series to a common fractional (relative) grid of size delta_frac,
    i.e. round the price to multiples of delta_frac * price. Implemented in log space:
    round(logp / delta_frac) * delta_frac."""
    if delta_frac <= EPS:
        return np.asarray(logp, float)
    return np.round(np.asarray(logp, float) / delta_frac) * delta_frac


def _nearest_psd(S):
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    return V @ np.diag(np.clip(w, 1e-12, None)) @ V.T


# ── VECM information share ────────────────────────────────────────────────────
def _fit_vecm(Y, p=1, beta=None):
    b, c, ect = ms._cointegrate(np.asarray(Y, float), beta)
    Yr, X = ms._build_design(np.asarray(Y, float), ect, p)
    B, Sig = ms._wls(Yr, X, np.ones(len(Yr)))
    return B[0, :].copy(), Sig                      # alpha (ECT loadings), innovation cov


def information_share(Y, p=1, beta=None):
    """Hasbrouck information share (lower/upper/mid per asset) and Gonzalo-Granger shares
    from a single-regime VECM on the two log prices."""
    alpha, Sig = _fit_vecm(Y, p, beta)
    lo, hi, mid = ms.hasbrouck_is_from_alpha(alpha, Sig)
    return {"IS_lower": lo, "IS_upper": hi, "IS_mid": mid,
            "GG": ms.gonzalo_granger_shares(alpha), "alpha": alpha, "Sigma": Sig}


# ── the tick-size correction ──────────────────────────────────────────────────
def tick_corrected_information_share(Y, ticks, prices, p=1, beta=None, names=("SPY", "ES")):
    """Compare the information share raw vs two tick corrections. `ticks` and `prices` are
    the (SPY, ES) tick sizes and representative price levels. Returns a DataFrame of the
    mid information share per market under raw / common-grid / rounding-corrected, plus each
    market's tick-noise share of return variance, and a verdict on whether the lead survives."""
    Y = np.asarray(Y, float)
    alpha, Sig = _fit_vecm(Y, p, beta)
    lo, hi, mid_raw = ms.hasbrouck_is_from_alpha(alpha, Sig)

    # tick-noise share of each market's return-innovation variance
    dvar = np.array([rounding_return_var(ticks[i], prices[i]) for i in range(2)])
    tick_share = dvar / np.maximum(np.diag(Sig), EPS)

    # (a) rounding-corrected: subtract rounding-noise variance from the diagonal, keep PSD
    Sig_c = Sig.copy()
    for i in range(2):
        Sig_c[i, i] = max(Sig[i, i] - dvar[i], 1e-12)
    Sig_c = _nearest_psd(Sig_c)
    _, _, mid_corr = ms.hasbrouck_is_from_alpha(alpha, Sig_c)

    # (b) common grid: round both log prices to the coarser fractional tick, refit
    delta = np.array([fractional_tick(ticks[i], prices[i]) for i in range(2)])
    delta_common = float(delta.max())
    Yc = np.column_stack([round_to_fractional_grid(Y[:, i], delta_common) for i in range(2)])
    alpha_g, Sig_g = _fit_vecm(Yc, p, beta)
    _, _, mid_grid = ms.hasbrouck_is_from_alpha(alpha_g, Sig_g)

    tab = pd.DataFrame({"raw": mid_raw, "common_grid": mid_grid, "rounding_corrected": mid_corr,
                        "tick_noise_share": tick_share,
                        "fractional_tick_bps": delta * 1e4}, index=list(names))
    leader_raw = names[int(np.argmax(mid_raw))]
    leader_grid = names[int(np.argmax(mid_grid))]
    leader_corr = names[int(np.argmax(mid_corr))]
    return {"table": tab, "leader_raw": leader_raw, "leader_common_grid": leader_grid,
            "leader_rounding_corrected": leader_corr,
            "lead_survives": (leader_raw == leader_grid == leader_corr),
            "delta_common_bps": delta_common * 1e4}


# ── self-test ─────────────────────────────────────────────────────────────────
def _sim_symmetric(T=30000, seed=0, price_spy=550.0, price_es=5500.0, sigma_w_bps=2.0, e_bps=1.0):
    """Two log prices sharing a common efficient trend f and symmetric transitory AR(1)
    errors -> the TRUE information share is 50/50 by construction. Returns the latent
    (un-rounded) log prices around the given price levels."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, sigma_w_bps / 1e4, T); f = np.cumsum(w)
    e = [np.zeros(T), np.zeros(T)]
    for j in range(2):
        eta = rng.normal(0, e_bps / 1e4, T)
        for t in range(1, T):
            e[j][t] = 0.5 * e[j][t - 1] + eta[t]
    lp_spy = np.log(price_spy) + f + e[0]
    lp_es = np.log(price_es) + f + e[1]
    return np.column_stack([lp_spy, lp_es])


def _round_price(logp, tick):
    """Round a log price to an absolute tick grid: log(round(price/tick)*tick)."""
    p = np.exp(np.asarray(logp, float))
    return np.log(np.maximum(np.round(p / tick) * tick, EPS))


def _selftest():
    print("tick_correction self-test")
    ticks = (0.01, 0.25); prices = (550.0, 5500.0)        # ES coarser in fractional terms (0.45 vs 0.18 bps)

    # (1) symmetric, un-rounded -> IS ~ 50/50 (validates the IS engine)
    Y = _sim_symmetric(seed=1, sigma_w_bps=0.5, e_bps=0.3)
    base = information_share(Y, p=1, beta=1.0)
    print("\n(1) symmetric un-rounded prices -> IS ~ 0.5")
    print("    IS_mid = %s" % np.round(base["IS_mid"], 3))
    print("    fractional ticks (bps): SPY=%.2f  ES=%.2f  (ES is the coarser grid)"
          % (fractional_tick(ticks[0], prices[0]) * 1e4, fractional_tick(ticks[1], prices[1]) * 1e4))
    print("    checks:", abs(base["IS_mid"][0] - 0.5) < 0.12)

    # (2) coarse ES grid spuriously makes the OTHER market (SPY) appear to lead; corrections remove it.
    #     ES tick amplified to 1.0 to expose the artifact (the real 0.25 tick gives a small effect at 1s).
    Y2 = _sim_symmetric(seed=1, sigma_w_bps=0.3, e_bps=0.2)
    ticks_amp = (0.01, 1.0)
    Yr = np.column_stack([_round_price(Y2[:, 0], ticks_amp[0]), _round_price(Y2[:, 1], ticks_amp[1])])
    res = tick_corrected_information_share(Yr, ticks_amp, prices, p=1, beta=1.0)
    print("\n(2) coarse ES grid -> spurious SPY lead; common-grid & rounding corrections remove it")
    print(res["table"].round(4).to_string())
    raw = res["table"].loc["SPY", "raw"]; grid = res["table"].loc["SPY", "common_grid"]
    corr = res["table"].loc["SPY", "rounding_corrected"]
    print("    SPY IS: raw-rounded=%.3f (spurious lead) | common-grid=%.3f | rounding-corrected=%.3f"
          % (raw, grid, corr))
    print("    tick-noise share: SPY=%.3f  ES=%.3f (ES coarser -> larger share)"
          % (res["table"].loc["SPY", "tick_noise_share"], res["table"].loc["ES", "tick_noise_share"]))
    print("    checks:",
          res["table"].loc["ES", "tick_noise_share"] > res["table"].loc["SPY", "tick_noise_share"]
          and abs(raw - 0.5) > 0.1                                       # coarse ES grid creates a clear spurious SPY lead
          and abs(grid - 0.5) < abs(raw - 0.5)                           # common grid removes it
          and abs(corr - 0.5) < abs(raw - 0.5))                          # rounding correction removes it

    # (3) a genuinely leading market survives the correction (real lead, not a tick artifact)
    rng = np.random.default_rng(5); T = 30000
    w = rng.normal(0, 2e-4, T); f = np.cumsum(w)
    # ES leads: ES ~ efficient price; SPY lags via error correction to ES
    lp_es = np.log(5500.0) + f + rng.normal(0, 0.5e-4, T)
    lp_spy = np.zeros(T); lp_spy[0] = np.log(550.0)
    for t in range(1, T):
        lp_spy[t] = lp_spy[t - 1] + 0.5 * ((lp_es[t] - np.log(10.0)) - lp_spy[t - 1]) + rng.normal(0, 0.5e-4)
    Yl = np.column_stack([lp_spy, lp_es])
    resl = tick_corrected_information_share(Yl, ticks, prices, p=1, beta=1.0)
    print("\n(3) ES genuinely leads -> lead survives the corrections")
    print("    leader: raw=%s, common-grid=%s, rounding-corrected=%s"
          % (resl["leader_raw"], resl["leader_common_grid"], resl["leader_rounding_corrected"]))
    print("    ES IS_mid: raw=%.3f common-grid=%.3f corrected=%.3f"
          % (resl["table"].loc["ES", "raw"], resl["table"].loc["ES", "common_grid"], resl["table"].loc["ES", "rounding_corrected"]))
    print("    checks:", resl["leader_raw"] == "ES" and resl["lead_survives"])


if __name__ == "__main__":
    _selftest()
