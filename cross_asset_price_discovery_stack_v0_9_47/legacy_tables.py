"""
legacy_tables.py
================
Side-by-side LEGACY (as in Garrison-Jain-Paddrik) vs UPGRADED estimators — the before/after
comparison that shows exactly which of the paper's conclusions survive the re-flooring. Each
contrast pairs the paper's original method with the stack's replacement on the SAME data:

  comovement      Pearson realized correlation              vs  Hayashi-Yoshida (Epps-robust)
  inference       pooled OLS with iid SEs (the ***)          vs  day-clustered SE + wild-cluster bootstrap
  liquidity       quoted-spread proxy (1 tick, ~constant)    vs  two-axis book-stress state
  identification  recursive Cholesky (futures ordered 1st)   vs  Rigobon heteroskedasticity ID

Each returns a plain DataFrame with a `legacy` column and an `upgraded` column for the report.
Best-effort: a failure in one contrast records an error and leaves the others intact.

NOTE on what a before/after can and cannot show. The point of these tables is METHODOLOGY: same
data, old estimator vs new. Two caveats travel with them. (1) The Epps gap (Pearson vs HY) only
appears under ASYNCHRONOUS sampling; on a synchronous grid the two correlations coincide, which is
correct, not a bug. (2) The wild-cluster bootstrap is unreliable at very few clusters (<~10); with
the paper's MWCB framing (G=4) read the Ibragimov-Muller / randomization route instead. The
t-statistic collapse from iid to day-clustered SEs is the robust headline of the comparison.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import cross_asset_pd_liquidity as cap
import noise_robust_cov as nrc
import inference as inf
import liquidity_stress as ls
import rigobon_id as rig

EPS = 1e-12


def _mid(df, a):
    return cap._mid(df, a)


def _log_mid(df, a):
    return np.log(np.maximum(_mid(df, a), EPS))


def _stack_returns_ofi(sessions, n_levels):
    """Pool aligned SPY return, ES return, ES order-flow imbalance, and a day id across sessions."""
    rs, re, of, day = [], [], [], []
    for i, (_, _, df) in enumerate(sessions):
        r_s = np.diff(_log_mid(df, "SPY")); r_e = np.diff(_log_mid(df, "ES"))
        ofi = np.asarray(cap.order_flow_imbalance(df, "ES", n_levels), float)[1:]
        n = min(len(r_s), len(r_e), len(ofi))
        if n < 5:
            continue
        rs.append(r_s[:n]); re.append(r_e[:n]); of.append(ofi[:n]); day.append(np.full(n, i))
    return (np.concatenate(rs), np.concatenate(re), np.concatenate(of), np.concatenate(day))


# ── contrast 1: comovement — Pearson vs Hayashi-Yoshida ───────────────────────
def comovement_table(sessions):
    pear, hy = [], []
    for _, _, df in sessions:
        ms = _log_mid(df, "SPY"); me = _log_mid(df, "ES")
        rs = np.diff(ms); re = np.diff(me); n = min(len(rs), len(re))
        rs, re = rs[:n], re[:n]
        if n > 5 and rs.std() > EPS and re.std() > EPS:
            pear.append(float(np.corrcoef(rs, re)[0, 1]))
        t = np.arange(len(ms), dtype=float)
        cov = nrc.hayashi_yoshida(t, ms, t, me)
        rv1 = float(np.sum(np.diff(ms) ** 2)); rv2 = float(np.sum(np.diff(me) ** 2))
        if rv1 > EPS and rv2 > EPS:
            hy.append(float(cov) / np.sqrt(rv1 * rv2))
    return pd.DataFrame({"legacy (Pearson)": [float(np.nanmean(pear))],
                         "upgraded (Hayashi-Yoshida)": [float(np.nanmean(hy))]},
                        index=["mean SPY/ES return corr"])


# ── contrast 2: inference — pooled iid SE vs day-clustered + wild bootstrap ────
def inference_table(sessions, n_levels, quick=False):
    rs, _, ofi, day = _stack_returns_ofi(sessions, n_levels)
    X = np.column_stack([np.ones(len(rs)), ofi]); y = rs
    b, se_iid = inf.ols_naive_se(y, X)
    cl = inf.cluster_robust_ols(y, X, day)
    wb = inf.wild_cluster_bootstrap(y, X, day, test_idx=1, n_boot=199 if quick else 999, seed=0)
    return pd.DataFrame(
        {"coef": [b[1]],
         "legacy t (iid, pooled)": [b[1] / (se_iid[1] + EPS)],
         "upgraded t (day-cluster)": [cl["t"][1]],
         "wild-boot p": [wb["p"]],
         "n_obs": [int(len(y))], "n_days": [int(cl["n_clusters"])]},
        index=["SPY_ret ~ ES_OFI"])


# ── contrast 3: liquidity — quoted spread vs two-axis book-stress state ────────
def liquidity_table(sessions, n_levels):
    Y, SP, DI, HO, DAY = [], [], [], [], []
    for i, (_, _, df) in enumerate(sessions):
        y = np.abs(np.diff(_log_mid(df, "SPY")))
        sp = ls.quoted_spread_bps(df, "SPY")[1:]
        st = ls.stress_state(df, "SPY", n_levels)
        di = st["depth_illiq"].to_numpy()[1:]; ho = st["hollowness"].to_numpy()[1:]
        n = min(len(y), len(sp), len(di), len(ho))
        if n < 5:
            continue
        Y.append(y[:n]); SP.append(sp[:n]); DI.append(di[:n]); HO.append(ho[:n]); DAY.append(np.full(n, i))
    Y = np.concatenate(Y); SP = np.concatenate(SP)
    DI = np.concatenate(DI); HO = np.concatenate(HO); DAY = np.concatenate(DAY)
    inc = ls.incremental_content(Y, SP, np.column_stack([DI, HO]), clusters=DAY)
    tab = pd.DataFrame(
        {"legacy (quoted spread)": [inc["r2_base"], np.nan],
         "upgraded (book-stress state)": [inc["r2_full"], inc["partial_r2"]]},
        index=["R2 explaining |SPY ret|", "partial R2 (state | spread)"])
    return tab, inc


# ── contrast 4: identification — Cholesky (futures first) vs Rigobon het-ID ────
def identification_table(sessions):
    uniq = sorted({reg for _, reg, _ in sessions})
    if len(uniq) < 2:
        raise ValueError("Rigobon needs >=2 regimes; demo/--volatile must label at least two")
    U, lab = [], []
    for _, reg, df in sessions:                                  # paper order: FUTURES (ES) first
        rs = np.diff(_log_mid(df, "SPY")); re = np.diff(_log_mid(df, "ES"))
        n = min(len(rs), len(re)); rs, re = rs[:n] - rs[:n].mean(), re[:n] - re[:n].mean()
        U.append(np.column_stack([re, rs])); lab += [reg] * n     # reduced-form innovations (demeaned)
    U = np.vstack(U); lab = np.array(lab)
    res = rig.rigobon_identify(U, lab, calm=uniq[0], stress=uniq[-1], names=["ES", "SPY"])
    cf, rg = res["contemp_cholesky_fwd"], res["contemp_rigobon"]
    return pd.DataFrame(
        {"legacy (Cholesky, futures 1st)": [cf.loc["SPY", "ES"], cf.loc["ES", "SPY"]],
         "upgraded (Rigobon het-ID)": [rg.loc["SPY", "ES"], rg.loc["ES", "SPY"]]},
        index=["SPY <- ES (contemp)", "ES <- SPY (contemp)"])


# ── assembler ─────────────────────────────────────────────────────────────────
def build_legacy_comparison(sessions, args):
    """Run every before/after contrast best-effort; return a dict of DataFrames for the report."""
    n = args.n_levels; quick = bool(getattr(args, "quick", False))
    out = {}
    jobs = [("comovement_Pearson_vs_HY", lambda: comovement_table(sessions)),
            ("inference_iid_vs_clustered", lambda: inference_table(sessions, n, quick)),
            ("liquidity_spread_vs_bookstate", lambda: liquidity_table(sessions, n)),
            ("identification_Cholesky_vs_Rigobon", lambda: identification_table(sessions))]
    for key, fn in jobs:
        try:
            r = fn()
            out[key] = r[0] if isinstance(r, tuple) else r
        except Exception as e:
            out[f"{key}_error"] = str(e)
    return out
