"""
tandem_order_flow.py
====================
Reproduce **Table 5** of Garrison, Jain & Paddrik ("Cross-Asset Tandem Trading and
Extraordinary Volatility") -- the cross-market order-imbalance *contingency* analysis.
This is the order-FLOW counterpart to the price-discovery VECM; it needs the per-bar
NEW-ORDER buy/sell counts of each market (Eq. 1), not the depth book frames the VECM
consumes. Returns / correlation for 5.III come from the mids.

Three layers, matching the paper:

  5.I   theoretical nulls
          - Panel A: independent bivariate binomial  (null 1A: markets independent)
          - Panel B: comonotone, rho = 1             (null 1B: perfectly integrated)
          - expected return-sign matrices (buy -> +, sell -> -, neutral -> 0)
  5.II  empirical state frequencies (% of bars) per sample, with the PCMOF / NCMOF
        corner-cell shares (PCMOF = Sell-Sell + Buy-Buy, NCMOF = Sell-Buy + Buy-Sell)
  5.III conditional mean d(return-correlation) x1000 and mean (ETF, Future) returns (bps)

Definitions (paper, Eqs. 1-2; theta = 0.05 -> Sell < 0.45, Neutral in [0.45, 0.55],
Buy > 0.55; NA when a market has no new orders in the bar).

Pure numpy / pandas (no statsmodels / scipy needed). Rows index the FUTURE state,
columns the ETF state, exactly as the paper prints the matrices.
"""
from math import lgamma, log, erf, sqrt
import numpy as np
import pandas as pd

STATES = ["Sell", "Neutral", "Buy", "NA"]           # column / row order (NA last)
DIR3 = ["Sell", "Neutral", "Buy"]                   # the directional 3 (no NA)


# ── imbalance and state classification (Eqs. 1-2) ─────────────────────────────
def order_imbalance(buy, sell):
    """Eq. (1): new buys / (new buys + new sells) per bar. NaN when no new orders."""
    buy = np.asarray(buy, float); sell = np.asarray(sell, float)
    tot = buy + sell
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, buy / tot, np.nan)


def classify(oi, lo=0.45, hi=0.55):
    """Bin an imbalance series into Sell / Neutral / Buy / NA (NA = undefined)."""
    oi = np.asarray(oi, float)
    out = np.full(oi.shape, "NA", dtype=object)
    fin = np.isfinite(oi)
    out[fin & (oi < lo)] = "Sell"
    out[fin & (oi >= lo) & (oi <= hi)] = "Neutral"
    out[fin & (oi > hi)] = "Buy"
    return out


def _with_margins(M, kind, grand=None):
    """Append an 'All' row and column. kind='sum' for frequencies (marginal totals),
    kind='mean' for conditional means (marginal = mean over the other axis)."""
    M = M.reindex(index=STATES, columns=STATES)
    out = M.copy()
    if kind == "sum":
        out.loc["All"] = M.sum(axis=0)
        out["All"] = out.sum(axis=1)            # includes the new All row -> total at corner
    else:
        out.loc["All"] = M.mean(axis=0)
        out["All"] = M.mean(axis=1).reindex(out.index)
        if grand is not None:
            out.loc["All", "All"] = grand
    return out


# ── 5.II: empirical state-frequency matrix ────────────────────────────────────
def frequency_matrix(fut_state, etf_state):
    """Table 5.II: percent of bars in each (Future-state, ETF-state) cell, with margins."""
    df = pd.DataFrame({"fut": pd.Categorical(fut_state, categories=STATES),
                       "etf": pd.Categorical(etf_state, categories=STATES)})
    m = pd.crosstab(df["fut"], df["etf"], dropna=False).reindex(index=STATES, columns=STATES, fill_value=0)
    tot = float(m.to_numpy().sum())
    m = m / tot * 100.0 if tot > 0 else m.astype(float)
    return _with_margins(m, "sum")


def pcmof_ncmof(freq):
    """Corner-cell shares from a 5.II frequency matrix (PCMOF = matched, NCMOF = opposed)."""
    return {"PCMOF": float(freq.loc["Sell", "Sell"] + freq.loc["Buy", "Buy"]),
            "NCMOF": float(freq.loc["Sell", "Buy"] + freq.loc["Buy", "Sell"])}


# ── 5.III: conditional mean d-correlation and returns ─────────────────────────
def rolling_dcorr(ret_etf, ret_fut, window=100):
    """First difference of the rolling return correlation (the paper's 100-bar window).
    Leading NaNs where the window is not yet full."""
    s_e = pd.Series(np.asarray(ret_etf, float)); s_f = pd.Series(np.asarray(ret_fut, float))
    rho = s_e.rolling(window).corr(s_f)
    return rho.diff().to_numpy()


def conditional_means(fut_state, etf_state, value):
    """Mean of `value` within each (Future-state, ETF-state) cell, with margins (the
    'All' row/col are conditional means over the other axis; corner is the grand mean)."""
    v = np.asarray(value, float)
    df = pd.DataFrame({"fut": pd.Categorical(fut_state, categories=STATES),
                       "etf": pd.Categorical(etf_state, categories=STATES), "v": v})
    df = df[np.isfinite(df["v"])]
    cell = df.pivot_table(index="fut", columns="etf", values="v", aggfunc="mean", observed=False)
    grand = float(df["v"].mean()) if len(df) else np.nan
    # marginals computed from the data (not from the cell matrix) so they are true conditionals
    row_marg = df.groupby("fut", observed=False)["v"].mean()
    col_marg = df.groupby("etf", observed=False)["v"].mean()
    out = _with_margins(cell, "mean", grand=grand)
    out.loc[DIR3 + ["NA"], "All"] = row_marg.reindex(STATES).values
    out.loc["All", DIR3 + ["NA"]] = col_marg.reindex(STATES).values
    return out


# ── 5.I: theoretical nulls ────────────────────────────────────────────────────
def _binom_pmf(N):
    """Exact Binomial(N, 1/2) pmf via log-gamma (dependency-light)."""
    k = np.arange(N + 1)
    logc = np.array([lgamma(N + 1) - lgamma(i + 1) - lgamma(N - i + 1) for i in k])
    return np.exp(logc + N * log(0.5))


def _ncdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def binom_state_probs(N, lo=0.45, hi=0.55, method="normal"):
    """P(imbalance in Sell / Neutral / Buy) for buys ~ Binomial(N, 1/2). Percent.

    The paper uses the **normal approximation** (method='normal', the default):
    P(Sell) = Phi((lo - 1/2) / sqrt(1/4N)), which reproduces the printed Table 5.I
    marginals -- N = 505 (ETF) -> ~1.23 / 97.54 / 1.23; N = 112 (Future) -> ~14.50 /
    71.00 / 14.50. method='exact' sums the binomial pmf instead (a hair different at
    these N because of the discreteness/continuity correction)."""
    if method == "exact":
        k = np.arange(N + 1); oi = k / N; pmf = _binom_pmf(N)
        p_sell = float(pmf[oi < lo].sum()); p_buy = float(pmf[oi > hi].sum())
    else:
        sd = sqrt(0.25 / N)
        p_sell = _ncdf((lo - 0.5) / sd); p_buy = 1.0 - _ncdf((hi - 0.5) / sd)
    return {"Sell": 100 * p_sell, "Neutral": 100 * (1 - p_sell - p_buy), "Buy": 100 * p_buy, "NA": 0.0}


def theoretical_null(n_etf=505, n_fut=112, rho=0.0, lo=0.45, hi=0.55):
    """Table 5.I frequency panel. rho=0 -> Panel A (independent: outer product of the two
    binomial marginals). rho>=1 -> Panel B (comonotone: the Future marginal on the diagonal,
    zero off-diagonal). Returned with margins; values in percent."""
    pe = binom_state_probs(n_etf, lo, hi); pf = binom_state_probs(n_fut, lo, hi)
    M = pd.DataFrame(0.0, index=STATES, columns=STATES)
    if rho >= 1.0:
        for s in DIR3:
            M.loc[s, s] = pf[s]                      # all mass on matched states
    else:
        for i in STATES:
            for j in STATES:
                M.loc[i, j] = pf[i] * pe[j] / 100.0   # independence (both in %)
    return _with_margins(M, "sum")


def expected_return_signs():
    """Table 5.I right panel: (ETF, Future) expected return sign per cell (buy -> +,
    sell -> -, neutral -> 0). Returned as a string matrix."""
    sign = {"Sell": "-", "Neutral": "0", "Buy": "+", "NA": "NA"}
    M = pd.DataFrame("", index=STATES, columns=STATES)
    for i in STATES:           # Future (row)
        for j in STATES:       # ETF (col)
            M.loc[i, j] = f"({sign[j]}, {sign[i]})"
    return M


# ── the null that actually isolates CROSS-market dependence ───────────────────
# The Binomial(n, 1/2) null above (Table 5.I Panel A) is a joint hypothesis. It says
#   (i)  each market's own order flow is a fair coin across n orders, AND
#   (ii) the two markets are independent of each other.
# Rejecting it tells you nothing about which half failed, and on this data it is (i) that
# fails, overwhelmingly. From the paper's own Table 5.II Panel A the ETF is directional
# (Sell or Buy) in 68.7% of one-second bars against a 2.4% binomial prediction, and the
# future in 83.2% against 29.0%. Order flow inside ONE market is autocorrelated and clustered
# -- order splitting, queue jockeying, momentum -- so the marginals are nothing like coin
# flips. That inflates ALL FOUR corner cells mechanically, with zero cross-market linkage.
#
# To measure cross-market tandem trading you must hold each market's OWN directionality fixed
# and ask only whether the two line up more often than chance. That is independence given the
# observed marginals, and it is the benchmark these functions provide. On Table 5.II Panel A:
#
#            observed   binomial null   independence-given-marginals
#   PCMOF     43.1%        0.4%              28.6%   -> 1.51x   (real, not 100x)
#   NCMOF     15.7%        0.4%              28.6%   -> 0.55x   (a DEFICIT, not an excess)
#
# so the paper's "NCMOF levels are much greater than the predicted values under either form of
# the null" reverses under a correctly specified null, while the PCMOF result survives.
def independence_given_marginals(freq):
    """Expected cell frequencies under CROSS-market independence, holding each market's own
    observed marginal state distribution fixed. Takes and returns a 5.II-style percent matrix
    (rows = Future state, cols = ETF state); the NA row/col is dropped and the 3x3 block is
    renormalised, since an NA bar carries no directional information in either market."""
    M = freq.reindex(index=DIR3, columns=DIR3).astype(float).to_numpy()
    tot = M.sum()
    if tot <= 0:
        return pd.DataFrame(np.nan, index=DIR3, columns=DIR3)
    M = M / tot * 100.0
    E = np.outer(M.sum(1), M.sum(0)) / 100.0
    return pd.DataFrame(E, index=DIR3, columns=DIR3)


def dependence_summary(freq, n_bars=None):
    """Cross-market dependence in the 5.II matrix, separated from each market's own
    directionality.

    Returns observed and independence-implied PCMOF / NCMOF (percent of directional bars),
    their ratios, and the **corner log odds ratio**

        log OR = log( (SellSell * BuyBuy) / (SellBuy * BuySell) )

    which is the marginal-free statement of the paper's claim: it is invariant to the row and
    column totals of the table, so unlike the raw corner percentages it cannot be manufactured
    by within-market clustering, and it is comparable ACROSS the baseline / volatile / MWCB
    panels even though their marginals differ a lot. logOR > 0 = genuine positive tandem
    trading. (Invariance is to the table's margins; re-discretising an underlying continuous
    association at different cut points does move the odds ratio.) With ``n_bars`` (the number of directional bars behind the matrix) the standard
    error sqrt(sum 1/n_ij) and a z-statistic are added."""
    M = freq.reindex(index=DIR3, columns=DIR3).astype(float).to_numpy()
    tot = M.sum()
    if tot <= 0:
        return {}
    P = M / tot * 100.0
    E = independence_given_marginals(freq).to_numpy()
    pc, nc = P[0, 0] + P[2, 2], P[0, 2] + P[2, 0]
    epc, enc = E[0, 0] + E[2, 2], E[0, 2] + E[2, 0]
    a, b, c, d = P[0, 0], P[0, 2], P[2, 0], P[2, 2]      # SellSell, SellBuy, BuySell, BuyBuy
    out = {"PCMOF": pc, "NCMOF": nc, "PCMOF_indep": epc, "NCMOF_indep": enc,
           "PCMOF_ratio": pc / epc if epc > 0 else np.nan,
           "NCMOF_ratio": nc / enc if enc > 0 else np.nan,
           "log_OR": float(np.log((a * d) / (b * c))) if min(a, b, c, d) > 0 else np.nan}
    if n_bars:
        n = np.array([a, b, c, d]) / 100.0 * float(n_bars)
        if n.min() > 0:
            se = float(np.sqrt((1.0 / n).sum()))
            out["log_OR_se"] = se
            out["log_OR_z"] = out["log_OR"] / se if se > 0 else np.nan
    return out


def permutation_null(fut_state, etf_state, n_perm=999, seed=0, lo=0.45, hi=0.55):
    """Distribution-free null for cross-market tandem trading: hold each market's own bar
    sequence intact and shuffle the PAIRING between them. This destroys only the cross-market
    association, preserving each market's exact marginal state distribution -- including the
    over-dispersion, clustering and intraday seasonality that a Binomial(n, 1/2) null assumes
    away, and that are the real reason the paper's Panel A null is rejected.

    Unlike ``theoretical_null`` this is automatically correct at ANY aggregation. The binomial
    null depends entirely on the number of new orders per bar, so it cannot be reused across
    frequencies: at the paper's per-second rates (505 ETF / 112 future) it puts 0.4% in each
    corner, but at a ten-millisecond bar (~5.1 / ~1.1 orders) it puts ~23%, and in action time
    (one order per event) ~50%. Comparing a fine-frequency frequency against the per-second
    0.4% overstates the evidence by one to two orders of magnitude.

    Returns {'PCMOF', 'NCMOF'} -> dict(observed, null_mean, null_lo, null_hi, p_value),
    where p_value is the two-sided permutation p for observed vs the null distribution."""
    fs = np.asarray(fut_state, dtype=object); es = np.asarray(etf_state, dtype=object)
    keep = (fs != "NA") & (es != "NA")
    fs, es = fs[keep], es[keep]
    n = len(fs)
    if n < 10:
        return {}
    def _corners(f, e):
        pc = np.mean(((f == "Sell") & (e == "Sell")) | ((f == "Buy") & (e == "Buy"))) * 100
        nc = np.mean(((f == "Sell") & (e == "Buy")) | ((f == "Buy") & (e == "Sell"))) * 100
        return pc, nc
    obs_pc, obs_nc = _corners(fs, es)
    rng = np.random.default_rng(seed)
    draws = np.empty((int(n_perm), 2))
    for i in range(int(n_perm)):
        draws[i] = _corners(fs, es[rng.permutation(n)])
    out = {}
    for j, (key, obs) in enumerate((("PCMOF", obs_pc), ("NCMOF", obs_nc))):
        d = draws[:, j]
        p = (1.0 + np.sum(np.abs(d - d.mean()) >= abs(obs - d.mean()))) / (len(d) + 1.0)
        out[key] = {"observed": float(obs), "null_mean": float(d.mean()),
                    "null_lo": float(np.percentile(d, 2.5)), "null_hi": float(np.percentile(d, 97.5)),
                    "p_value": float(p)}
    return out


def independence_null_from_counts(n_etf, n_fut, n_draw=None, seed=0, lo=0.45, hi=0.55):
    """Binomial independence null evaluated at the ACTUAL per-bar order counts, rather than at
    a single representative n. ``n_etf`` / ``n_fut`` are the per-bar total new-order counts of
    the two markets (arrays); buys are redrawn as Binomial(n_bar, 1/2) independently in each
    market. This is the frequency-matched version of ``theoretical_null`` -- use it whenever
    the aggregation is not the one the fixed n=505/112 was calibrated to.

    Returns the same {'PCMOF','NCMOF'} percentages, including bars that come out NA."""
    ne = np.asarray(n_etf, float); nf = np.asarray(n_fut, float)
    m = np.isfinite(ne) & np.isfinite(nf)
    ne, nf = ne[m].astype(int), nf[m].astype(int)
    if n_draw:                                            # optional resample to a target size
        r0 = np.random.default_rng(seed)
        idx = r0.integers(0, len(ne), int(n_draw))
        ne, nf = ne[idx], nf[idx]
    rng = np.random.default_rng(seed)
    be = rng.binomial(np.maximum(ne, 0), 0.5); bf = rng.binomial(np.maximum(nf, 0), 0.5)
    se = classify(order_imbalance(be, ne - be), lo, hi)
    sf = classify(order_imbalance(bf, nf - bf), lo, hi)
    return {"PCMOF": float(np.mean(((sf == "Sell") & (se == "Sell")) | ((sf == "Buy") & (se == "Buy"))) * 100),
            "NCMOF": float(np.mean(((sf == "Sell") & (se == "Buy")) | ((sf == "Buy") & (se == "Sell"))) * 100)}


# ── end-to-end: build Table 5.II + 5.III from per-bar series ──────────────────
def table5_from_series(buy_etf, sell_etf, buy_fut, sell_fut,
                       ret_etf=None, ret_fut=None, corr_window=100, ret_in_bps=True,
                       lo=0.45, hi=0.55):
    """Reproduce Table 5.II (+5.III if returns are supplied) for ONE sample.

    Inputs are aligned per-bar arrays (1-second bars in the paper):
      buy/sell_{etf,fut} : new-order counts (Eq. 1 inputs)
      ret_{etf,fut}      : per-bar returns (fractions, or bps if ret_in_bps and you pass bps)
    Returns a dict with 'frequency' (5.II), 'pcmof_ncmof', and -- if returns given --
    'dcorr_x1000', 'ret_etf_bps', 'ret_fut_bps' (5.III)."""
    e = order_imbalance(buy_etf, sell_etf); f = order_imbalance(buy_fut, sell_fut)
    se = classify(e, lo, hi); sf = classify(f, lo, hi)
    out = {"frequency": frequency_matrix(sf, se)}
    out["pcmof_ncmof"] = pcmof_ncmof(out["frequency"])
    # Cross-market dependence separated from each market's own directionality. Report these
    # next to the raw corner shares: the raw shares are inflated by within-market clustering
    # and are NOT a measure of tandem trading on their own (see independence_given_marginals).
    n_dir = int(np.sum((sf != "NA") & (se != "NA")))
    out["dependence"] = dependence_summary(out["frequency"], n_bars=n_dir)
    out["independence_expected"] = independence_given_marginals(out["frequency"])
    # Frequency-matched binomial null: the fixed n=505/112 of theoretical_null is a
    # ONE-SECOND calibration and must not be reused at 10ms or in action time.
    out["binomial_null_matched"] = independence_null_from_counts(
        np.asarray(buy_etf, float) + np.asarray(sell_etf, float),
        np.asarray(buy_fut, float) + np.asarray(sell_fut, float))
    if ret_etf is not None and ret_fut is not None:
        re = np.asarray(ret_etf, float); rf = np.asarray(ret_fut, float)
        scale = 1.0 if ret_in_bps else 1e4
        dcorr = rolling_dcorr(re, rf, corr_window)
        out["dcorr_x1000"] = conditional_means(sf, se, dcorr * 1000.0)
        out["ret_etf_bps"] = conditional_means(sf, se, re * scale)
        out["ret_fut_bps"] = conditional_means(sf, se, rf * scale)
    return out


def table5_from_sessions(sessions, counts_fn, corr_window=100):
    """Pool a list of (label, regime, df)-style sessions into one Table 5, splitting by
    regime. `counts_fn(df) -> (buy_etf, sell_etf, buy_fut, sell_fut, ret_etf, ret_fut)`
    extracts the per-bar arrays from whatever object you carry the message counts in
    (the depth frame alone does NOT contain new-order side counts -- see module docstring).
    Returns {regime: table5_from_series(...)}."""
    by = {}
    bins = {}
    for label, regime, df in sessions:
        arrs = counts_fn(df)
        bins.setdefault(regime, []).append(arrs)
    for regime, lst in bins.items():
        cols = [np.concatenate([np.asarray(a[k], float) for a in lst]) for k in range(6)]
        by[regime] = table5_from_series(*cols, corr_window=corr_window)
    return by


# ── self-test ─────────────────────────────────────────────────────────────────
def _selftest():
    print("tandem_order_flow self-test")

    # (1) theoretical nulls reproduce the printed Table 5.I marginals exactly
    pe = binom_state_probs(505); pf = binom_state_probs(112)
    A = theoretical_null(rho=0.0); B = theoretical_null(rho=1.0)
    print("\n(1) Table 5.I nulls")
    print("    ETF marginal    Sell/Neutral/Buy = %.2f / %.2f / %.2f  (paper 1.22 / 97.57 / 1.22)"
          % (pe["Sell"], pe["Neutral"], pe["Buy"]))
    print("    Future marginal Sell/Neutral/Buy = %.2f / %.2f / %.2f  (paper 14.49 / 71.02 / 14.49)"
          % (pf["Sell"], pf["Neutral"], pf["Buy"]))
    indep_corners = A.loc["Sell", "Sell"] + A.loc["Buy", "Buy"] + A.loc["Sell", "Buy"] + A.loc["Buy", "Sell"]
    print("    Panel A independent corner mass (PCMOF+NCMOF) = %.2f%%  (paper: < 1%%)" % indep_corners)
    print("    checks:",
          abs(pe["Sell"] - 1.22) < 0.05 and abs(pe["Neutral"] - 97.57) < 0.05
          and abs(pf["Sell"] - 14.49) < 0.05 and abs(pf["Neutral"] - 71.02) < 0.05
          and indep_corners < 1.0
          and abs(float(A.loc["All", "All"]) - 100.0) < 1e-6
          and abs(B.loc["Neutral", "Neutral"] - pf["Neutral"]) < 1e-9    # comonotone: marginal on diagonal
          and abs(float(B.loc["Sell", "Buy"])) < 1e-12)                  # zero off-diagonal

    # (2) empirical 5.II/5.III on a synthetic sample with POSITIVELY correlated imbalance
    rng = np.random.default_rng(11)
    T = 60000
    # common directional factor -> correlated buy probabilities -> PCMOF should dominate
    z = rng.normal(0, 1, T)
    p_etf = 1 / (1 + np.exp(-(0.0 + 1.3 * z)))           # P(buy) for ETF orders this bar
    p_fut = 1 / (1 + np.exp(-(0.0 + 1.3 * z)))           # same factor -> positive dependence
    n_etf = rng.poisson(505, T); n_fut = rng.poisson(112, T)
    buy_etf = rng.binomial(n_etf, p_etf); sell_etf = n_etf - buy_etf
    buy_fut = rng.binomial(n_fut, p_fut); sell_fut = n_fut - buy_fut
    # returns driven by the same factor (so buy-buy bars carry positive returns, etc.)
    ret_etf = 0.8 * z + rng.normal(0, 1, T)              # in "bps" units for the test
    ret_fut = 0.8 * z + rng.normal(0, 1, T)
    res = table5_from_series(buy_etf, sell_etf, buy_fut, sell_fut, ret_etf, ret_fut, ret_in_bps=True)
    freq = res["frequency"]; cn = res["pcmof_ncmof"]
    print("\n(2) empirical 5.II / 5.III (synthetic, positively-correlated imbalance)")
    print("    PCMOF = %.1f%%   NCMOF = %.1f%%   (matched-direction clustering dominates)"
          % (cn["PCMOF"], cn["NCMOF"]))
    print("    frequency matrix sums to %.2f%%" % float(freq.loc["All", "All"]))
    rE = res["ret_etf_bps"]
    print("    mean ETF return: Buy-Buy cell = %+.2f vs Sell-Sell cell = %+.2f (bps)"
          % (rE.loc["Buy", "Buy"], rE.loc["Sell", "Sell"]))
    print("    checks:",
          abs(float(freq.loc["All", "All"]) - 100.0) < 1e-6                 # frequencies normalize
          and cn["PCMOF"] > cn["NCMOF"]                                     # positive dependence -> PCMOF
          and cn["PCMOF"] > 2 * cn["NCMOF"]
          and rE.loc["Buy", "Buy"] > rE.loc["Sell", "Sell"]                 # buy-buy carries higher return
          and np.isfinite(res["dcorr_x1000"].loc["Buy", "Buy"]))

    # (3) NA handling: bars with no new orders are classified NA, not forced into Neutral
    b = np.array([10, 0, 7, 0]); s = np.array([2, 0, 9, 0])
    st = classify(order_imbalance(b, s))
    print("\n(3) NA handling")
    print("    counts (10,2)->%s  (0,0)->%s  (7,9)->%s" % (st[0], st[1], st[2]))
    print("    checks:", st[0] == "Buy" and st[1] == "NA" and st[2] == "Sell" and st[3] == "NA")


if __name__ == "__main__":
    _selftest()
