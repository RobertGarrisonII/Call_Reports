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
        # finite rows only: halt-masked returns are NaN, and one NaN row through ols_naive_se /
        # cluster_robust_ols turned the whole inference table's coef and both t's NaN on the
        # 2026-08-05 masked run (while the wild-bootstrap p survived, which is worse than failing
        # loudly). The regression is CONTEMPORANEOUS -- ret_t on OFI_t, no lag windows -- so
        # dropping rows cannot splice anything.
        okr = (np.isfinite(r_s[:n]) & np.isfinite(r_e[:n]) & np.isfinite(ofi[:n]))
        if int(okr.sum()) < 5:
            continue
        rs.append(r_s[:n][okr]); re.append(r_e[:n][okr]); of.append(ofi[:n][okr])
        day.append(np.full(int(okr.sum()), i))
    return (np.concatenate(rs), np.concatenate(re), np.concatenate(of), np.concatenate(day))


# ── contrast 1: comovement — Pearson vs Hayashi-Yoshida ───────────────────────
def comovement_table(sessions):
    pear, hy = [], []
    for _, _, df in sessions:
        ms = _log_mid(df, "SPY"); me = _log_mid(df, "ES")
        rs = np.diff(ms); re = np.diff(me); n = min(len(rs), len(re))
        rs, re = rs[:n], re[:n]
        # pairwise-finite returns: with halt-masked frames rs.std() is NaN, `NaN > EPS` is False,
        # and the four MWCB days silently VANISHED from this table on the 2026-08-05 masked run --
        # a shrunken sample indistinguishable from a full one. Contemporaneous correlation, so the
        # row drop cannot splice; the halt seconds simply contribute nothing.
        okp = np.isfinite(rs) & np.isfinite(re)
        rs, re = rs[okp], re[okp]
        if len(rs) > 5 and rs.std() > EPS and re.std() > EPS:
            pear.append(float(np.corrcoef(rs, re)[0, 1]))
        # HY per contiguous finite SEGMENT, summed. Compressing the NaN out and handing HY the
        # kept timestamps would leave one interval SPANNING the halt on each leg -- and that
        # single gap-return co-movement (a few hundred bp squared against ~1e-4 of daily RV)
        # would dominate both the covariance and the variances, quietly pushing every MWCB day's
        # HY correlation to ~1. Segment-wise accumulation excludes the cross-halt interval
        # entirely, matching what the Pearson column's NaN diffs already do.
        t = np.arange(len(ms), dtype=float)
        msa = np.asarray(ms, float); mea = np.asarray(me, float)
        fin = np.isfinite(msa) & np.isfinite(mea)
        cov = rv1 = rv2 = 0.0
        j = 0
        while j < len(fin):
            if not fin[j]:
                j += 1; continue
            k = j
            while k < len(fin) and fin[k]:
                k += 1
            if k - j > 5:
                cov += float(nrc.hayashi_yoshida(t[j:k], msa[j:k], t[j:k], mea[j:k]))
                rv1 += float(np.sum(np.diff(msa[j:k]) ** 2))
                rv2 += float(np.sum(np.diff(mea[j:k]) ** 2))
            j = k
        if rv1 > EPS and rv2 > EPS:
            hy.append(cov / np.sqrt(rv1 * rv2))
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
        n = min(len(rs), len(re))
        # nanmean, NOT .mean(): on a halt-masked day a plain mean is NaN and demeaning poisons the
        # ENTIRE day's rows -- regime_residual_cov then silently drops the whole day, so the four
        # MWCB days (the stress observations het-ID depends on) vanished wholesale from the regime
        # covariances, deflating the very strength diagnostic added below. The per-row halt NaN is
        # handled downstream by regime_residual_cov's finite-row filter; contemporaneous
        # covariance, so the row drop cannot splice.
        rs, re = rs[:n] - np.nanmean(rs[:n]), re[:n] - np.nanmean(re[:n])
        U.append(np.column_stack([re, rs])); lab += [reg] * n     # reduced-form innovations (demeaned)
    U = np.vstack(U); lab = np.array(lab)
    res = rig.rigobon_identify(U, lab, calm=uniq[0], stress=uniq[-1], names=["ES", "SPY"])
    cf, rg = res["contemp_cholesky_fwd"], res["contemp_rigobon"]
    # IDENTIFICATION STRENGTH IN THE TABLE, not in a docstring. Het-ID pins A down only when the
    # regimes differ in RELATIVE heteroskedasticity; when both legs' variances scale up together
    # under stress (the usual case for a ~0.93-correlated pair), M = Omega_v Omega_b^{-1} is near
    # cI, its eigenvalues nearly coincide, and the eigenvectors -- hence the signs and magnitudes
    # of the "identified" matrix -- are numerical noise. The 2026-08-05 run printed SPY<-ES = -0.33
    # / ES<-SPY = +1.01 (against every other estimator AND its own previous run) with nothing in
    # the output saying whether the rotation was identified at all; the previous run's "Rigobon
    # validates the recursive ordering" carried the same silence. The verdict row makes both
    # readable: when it says 'no', neither run's Rigobon numbers mean anything -- do not quote.
    cov = rig.regime_residual_cov(U, lab)
    diag = rig.identification_diagnostic(cov, names=["ES", "SPY"])
    verdict = "yes" if diag["identified"] else "NO -- do not quote the het-ID column"
    # display the RELATIVE eigenvalue gap -- the statistic the verdict actually thresholds
    # (> 0.10). The absolute gap res["eig_separation"] scales with the overall calm-to-stress
    # variance ratio, so it can read large (or tiny) purely from common scale -- exactly the
    # regime the verdict exists to flag -- and a reader reconciling a big "separation" with a NO
    # verdict would conclude the verdict is a bug and quote the column anyway.
    return pd.DataFrame(
        {"legacy (Cholesky, futures 1st)": [cf.loc["SPY", "ES"], cf.loc["ES", "SPY"],
                                            np.nan, np.nan, ""],
         "upgraded (Rigobon het-ID)": [rg.loc["SPY", "ES"], rg.loc["ES", "SPY"],
                                       diag["var_ratio_spread"], diag["rel_eig_gap"], verdict]},
        index=["SPY <- ES (contemp)", "ES <- SPY (contemp)",
               "var-ratio spread (max/min - 1; ~0 = common-scale, unidentified)",
               "relative eigenvalue gap (verdict thresholds this > 0.10)",
               "identified (relative het present)"])


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
