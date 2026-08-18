"""
table5_inference.py
===================
Day-clustered inference for the Table 5 corner log odds ratio.

The problem this closes. STAGE 4 scores the pooled corner log OR with the Woolf
standard error sqrt(sum 1/n_ij), which is valid only if the pooled bars are IID
multinomial draws of the 3x3 state pair. They are nothing of the sort: order flow
is serially dependent within a day (order splitting, queue jockeying, intraday
seasonality), and the coupling itself moves day to day, so the effective sample
is far smaller than the bar count. On the 2026-08-17 run the pooled panels
printed Woolf z = 159-214 -- numbers that are a statement about the number of
bars, not about the evidence. tandem_order_flow already documents the caveat in
dependence_summary; this module replaces the caveat with the correct unit.

Days are the independent unit for inference everywhere in this stack, and the
log OR pools cleanly over them: the pooled panel estimate is a function of the
SUMMED per-day 3x3 count matrices (dependence_summary is scale-invariant, so
summed counts reproduce the percent-matrix point estimate exactly). That makes
day-level resampling exact for the pooled statistic, not an approximation to it:

  * day bootstrap -- resample days with replacement within a panel, recompute
    the pooled log OR from the resampled count matrices. Percentile CI, SE, z.
    Within-day dependence of any form is preserved untouched inside each day's
    matrix; only the day-level variation is resampled, which is exactly the
    variation the Woolf SE ignores.
  * jackknife fallback -- the MWCB panel has ~4 days; a bootstrap over so few
    days undercovers and its percentiles take at most 2^n - 1 distinct values.
    Leave-one-day-out jackknife SE with a normal-approximation CI is reported
    instead, flagged note='exact/caveat' so no reader mistakes it for a
    well-populated interval.
  * within-pair sign-flip -- the volatile-vs-benchmark contrast on per-day log
    ORs, flipping signs only within each (volatile day, matched ~1y-prior
    control) pair. The free day-label permutation confounds regime with a
    decade of market-structure drift; the sign-flip is era-robust by
    construction (same design as price_discovery_shares.compare_regimes pairs
    mode, applied to the tandem statistic). Both p-values are returned: free-
    significant but within-pair-not is the era-confounding signature.

Halt masking is the CALLER's job (the runner mirrors STAGE 4's
market_halts.mask_frame on load): this module sees frames that are already
masked, and NaN trade columns simply produce NA bars that the 3x3 drops.

Pure numpy / pandas on top of tandem_order_flow; no new dependencies.
"""
import numpy as np
import pandas as pd

import tandem_order_flow as tof

# 9 directional cells, row = FUTURE state, col = ETF state (the paper's layout).
CELLS = [f"n_{fr}{er}" for fr in tof.DIR3 for er in tof.DIR3]
_CORNERS = ("n_SellSell", "n_SellBuy", "n_BuySell", "n_BuyBuy")   # a, b, c, d
SMALL_PANEL_DAYS = 6                 # below this the bootstrap undercovers -> jackknife


def _dates(spec):
    """Normalise a date spec (comma string or iterable) to a set of 'YYYY-MM-DD'."""
    if spec is None:
        return set()
    if isinstance(spec, str):
        spec = spec.split(",")
    return {str(d).strip()[:10] for d in spec if str(d).strip()}


def _panel_label(date, regime, mwcb_set):
    """STAGE 4's tagging, verbatim: MWCB days are their own panel, not 'volatile'."""
    if str(date)[:10] in mwcb_set:
        return "C MWCB"
    return "B volatile" if regime == "volatile" else "A baseline"


def _counts_frame(counts9):
    """9-vector of counts (CELLS order) -> DIR3 x DIR3 DataFrame for dependence_summary."""
    return pd.DataFrame(np.asarray(counts9, float).reshape(3, 3),
                        index=tof.DIR3, columns=tof.DIR3)


def _log_or(counts9):
    """Corner log OR straight from counts -- scale-invariant, so identical to the
    percent-matrix log_OR of dependence_summary on the same cells."""
    a, b, c, d = (float(counts9[CELLS.index(k)]) for k in _CORNERS)
    if min(a, b, c, d) <= 0:
        return float("nan")
    return float(np.log((a * d) / (b * c)))


def per_day_matrices(sessions, counts_fn, mwcb, ssr_fn=None):
    """Per-day 3x3 count matrices and per-day dependence statistics.

    sessions  : list of (date, regime, df) as STAGE 4 loads them (already halt-masked
                by the caller).
    counts_fn : df -> (buy_etf, sell_etf, buy_fut, sell_fut, ret_etf, ret_fut) or None
                (mstbook_loader.counts_from_frame); a None frame is skipped, never fatal.
    mwcb      : MWCB dates (comma string or iterable) -> those days get panel 'C MWCB'.
    ssr_fn    : optional df -> (restricted, known) (mstbook_loader.session_is_ssr);
                when given, 'ssr' / 'ssr_known' columns are added so the caller can
                rebuild the MWCB panel excluding restricted sessions (STAGE 4's exSSR).

    Returns a DataFrame indexed by date: panel, the 9 CELLS (raw bar COUNTS -- counts,
    not percents, because counts are what sum across days), n_bars (directional bars),
    log_OR and corner_asym (per-day dependence_summary of that day's matrix). Skipped
    session labels travel in .attrs['skipped'].
    """
    mwcb_set = _dates(mwcb)
    rows, skipped = [], []
    for date, regime, df in sessions:
        arrs = counts_fn(df)
        if arrs is None:
            skipped.append(str(date))
            continue
        be, se, bf, sf = (np.asarray(a, float) for a in arrs[:4])
        st_e = tof.classify(tof.order_imbalance(be, se))
        st_f = tof.classify(tof.order_imbalance(bf, sf))
        counts = [int(np.sum((st_f == fr) & (st_e == er)))
                  for fr in tof.DIR3 for er in tof.DIR3]
        d = tof.dependence_summary(_counts_frame(counts))
        row = {"date": str(date)[:10], "panel": _panel_label(date, regime, mwcb_set)}
        row.update(dict(zip(CELLS, counts)))
        row["n_bars"] = int(sum(counts))
        row["log_OR"] = d.get("log_OR", float("nan"))
        row["corner_asym"] = d.get("corner_asym", float("nan"))
        if ssr_fn is not None:
            restricted, known = ssr_fn(df)
            row["ssr"] = bool(restricted)
            row["ssr_known"] = bool(known)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["panel"] + CELLS + ["n_bars", "log_OR", "corner_asym"])
    out.attrs["skipped"] = skipped
    return out


def _panel_counts(per_day, panel):
    sub = per_day.loc[per_day["panel"] == panel, CELLS]
    return sub.to_numpy(float)


def panel_log_or(per_day, panel):
    """Pooled panel log OR from the SUMMED per-day count matrices.

    Summing counts over days and pooling the bars are the same operation, and the
    log OR is invariant to the count-vs-percent scaling, so this reproduces
    table5_from_sessions' pooled point estimate exactly (gate check 4)."""
    C = _panel_counts(per_day, panel)
    if C.shape[0] == 0:
        return float("nan")
    d = tof.dependence_summary(_counts_frame(C.sum(axis=0)))
    return d.get("log_OR", float("nan"))


def panel_dependence(per_day, panel):
    """Full dependence_summary of the pooled panel, Woolf z included (n_bars = the
    panel's directional bar total). The Woolf z is reported as the scale reference
    STAGE 4 already prints -- inference belongs to day_bootstrap."""
    C = _panel_counts(per_day, panel)
    if C.shape[0] == 0:
        return {}
    tot = C.sum(axis=0)
    return tof.dependence_summary(_counts_frame(tot), n_bars=int(tot.sum()))


def day_bootstrap(per_day, panel, n_boot=2000, seed=0):
    """Day-resampled inference for one panel's pooled log OR.

    Resamples DAYS with replacement, recomputes the pooled log OR from the summed
    resampled count matrices per draw: percentile CI + bootstrap SE + z. Panels with
    fewer than SMALL_PANEL_DAYS days (the MWCB panel) get the leave-one-day-out
    jackknife SE and a normal-approximation CI instead, with note='exact/caveat' --
    a bootstrap over n < 6 days has at most 2^n - 1 distinct resamples and its
    percentiles are not an interval anyone should trust."""
    C = _panel_counts(per_day, panel)
    n = C.shape[0]
    point = panel_log_or(per_day, panel)
    out = {"panel": panel, "n_days": int(n), "log_OR": point,
           "ci_lo": float("nan"), "ci_hi": float("nan"),
           "se_day": float("nan"), "z_day": float("nan")}
    if n == 0 or not np.isfinite(point):
        out["note"] = "empty" if n == 0 else "zero cell in pooled matrix"
        return out
    corner_idx = [CELLS.index(k) for k in _CORNERS]
    K = C[:, corner_idx]                                   # (n_days, 4): a b c d
    if n < SMALL_PANEL_DAYS:
        if n >= 2:
            loo = K.sum(axis=0)[None, :] - K               # leave-one-day-out sums
            with np.errstate(invalid="ignore", divide="ignore"):
                th = np.log((loo[:, 0] * loo[:, 3]) / (loo[:, 1] * loo[:, 2]))
            th = th[np.isfinite(th)]
            if len(th) >= 2:
                m = len(th)
                se = float(np.sqrt((m - 1) / m * np.sum((th - th.mean()) ** 2)))
                out.update(se_day=se, z_day=point / se if se > 0 else float("nan"),
                           ci_lo=point - 1.96 * se, ci_hi=point + 1.96 * se)
        out["note"] = "exact/caveat"                       # jackknife, n_days < 6
        out["method"] = "jackknife"
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    S = K[idx].sum(axis=1)                                 # (n_boot, 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.log((S[:, 0] * S[:, 3]) / (S[:, 1] * S[:, 2]))
    draws = draws[np.isfinite(draws)]
    if len(draws) < max(50, n_boot // 10):
        out["note"] = "degenerate resamples (zero corner cells)"
        return out
    se = float(np.std(draws, ddof=1))
    out.update(ci_lo=float(np.percentile(draws, 2.5)),
               ci_hi=float(np.percentile(draws, 97.5)),
               se_day=se, z_day=point / se if se > 0 else float("nan"),
               method="bootstrap", n_boot_finite=int(len(draws)))
    return out


def pair_contrast(per_day, volatile_dates, benchmark_dates, n_perm=20000, seed=0):
    """Within-pair sign-flip test of the regime difference in per-day log OR.

    Pairs are POSITIONAL (zip of the two date lists), the stack's matched-pair
    convention: pair i is (volatile_dates[i], benchmark_dates[i]), the control
    chosen ~1 year before its volatile partner. Dates absent from per_day (not
    extracted, skipped, or non-finite log OR) degrade to fewer pairs -- mirroring
    price_discovery_shares.compare_regimes pairs mode, which drops incomplete
    pairs rather than failing the test. d_i = logOR(vol_i) - logOR(ben_i); the
    null flips the sign of each d_i independently, so every comparison stays
    within its era. The free day-label permutation p is returned for reference:
    free-significant but within-pair-not flags era confounding, not regime."""
    vol = [str(d)[:10] for d in (volatile_dates if not isinstance(volatile_dates, str)
                                 else volatile_dates.split(",")) if str(d).strip()]
    ben = [str(d)[:10] for d in (benchmark_dates if not isinstance(benchmark_dates, str)
                                 else benchmark_dates.split(",")) if str(d).strip()]
    lor = {str(k)[:10]: float(v) for k, v in per_day["log_OR"].items()}
    d, used_v, used_b = [], [], []
    for v, b in zip(vol, ben):
        lv, lb = lor.get(v, float("nan")), lor.get(b, float("nan"))
        if np.isfinite(lv) and np.isfinite(lb):
            d.append(lv - lb)
            used_v.append(lv); used_b.append(lb)
    d = np.asarray(d, float)
    out = {"n_pairs": int(len(d)), "mean_pair_diff": float("nan"),
           "p_within_pair": float("nan"), "p_free": float("nan")}
    if len(d) == 0:
        return out
    rng = np.random.default_rng(seed)
    obs = float(np.mean(d))
    flips = rng.choice([-1.0, 1.0], size=(int(n_perm), len(d)))
    null = flips @ d / len(d)
    p_wp = (1.0 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (n_perm + 1.0)
    # free permutation over the SAME used days, so the two p-values disagree only
    # through the pairing, never through sample composition
    pool = np.asarray(used_v + used_b, float)
    n_v = len(used_v)
    obs_free = float(np.mean(pool[:n_v]) - np.mean(pool[n_v:]))
    cnt = 0
    for _ in range(int(n_perm)):
        rng.shuffle(pool)
        if abs(float(np.mean(pool[:n_v]) - np.mean(pool[n_v:]))) >= abs(obs_free) - 1e-12:
            cnt += 1
    out.update(mean_pair_diff=obs, p_within_pair=float(p_wp),
               p_free=float((cnt + 1.0) / (n_perm + 1.0)))
    return out
