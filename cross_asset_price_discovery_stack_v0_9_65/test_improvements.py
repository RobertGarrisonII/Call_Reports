#!/usr/bin/env python3
"""test_improvements.py -- the v0.9.65 improvement batch behaves as specified.

  (1) GFEVD reduces EXACTLY to the orthogonal FEVD at R = I, and matches the hand
      analytic answer on a static (H=0, D_0=I) system with correlated shocks
  (2) the mean-group returns the effective-sample-weighted column, and the weighting is
      exactly sum(n_d * resp_d)/sum(n_d) recomputed from its own per_day table
  (3) table_rigobon adds the over-identification row with >=3 exogenous labels and omits
      it with 2
  (4) the Table 9 CLI prints the RealBar own-lag MA(1) diagnostic when a criterion is used
  (5) _gram_solve under Jacobi equilibration recovers lstsq-accuracy coefficients on a
      design with 1e6-spread column scales (where raw normal equations lose digits)
  (6) rank_sample: planted extreme-range days rank on top; pairing obeys the
      same-weekday 350-371d rule (via its --selftest)

Run: python test_improvements.py
"""
from __future__ import annotations

import os
import subprocess as sp
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def check_gfevd():
    import irf as irfm
    rng = np.random.default_rng(3)
    D = rng.normal(size=(7, 2, 2))
    f = irfm.fevd_from_irf(D)
    g = irfm.gfevd_from_irf(D, shock_corr=np.eye(2))
    reduces = np.allclose(f, g, atol=1e-12)
    # static analytic case: D_0 = I only, R = [[1, r], [r, 1]] -> theta row i proportional
    # to [1, r^2] (own) / [r^2, 1] (other), normalized
    r = 0.6
    D0 = np.zeros((1, 2, 2)); D0[0] = np.eye(2)
    R = np.array([[1.0, r], [r, 1.0]])
    g0 = irfm.gfevd_from_irf(D0, shock_corr=R)
    want = np.array([[1.0, r ** 2], [r ** 2, 1.0]])
    want = want / want.sum(axis=1, keepdims=True)
    analytic = np.allclose(g0, want, atol=1e-12)
    ok = reduces and analytic
    print("(1) GFEVD == FEVD at R=I (%s); static correlated-shock case matches the "
          "hand answer [[1,r^2],[r^2,1]] row-normalized (%s) : %s"
          % (reduces, analytic, ok))
    return ok


def check_weighted_mean_group():
    import correlation_svar as cs
    from test_panel_svar import _book_frame
    sessions = [("d1", "volatile", _book_frame(seed=5, n=3000)),
                ("d2", "volatile", _book_frame(seed=6, n=6000, flip_ofi=True)),
                ("d3", "benchmark", _book_frame(seed=7, n=4000))]
    mg = cs.correlation_irf_mean_group(sessions, spec="informational", n_levels=3, n_lags=2,
                                       corr_method="bar", bar_seconds=30)
    wm = mg["point_weighted"]
    shaped = (not wm.empty and list(wm.columns) == list(mg["point"].columns)
              and list(wm.index) == list(mg["point"].index))
    # the weighted column must equal sum(n*resp)/sum(n) recomputed from per_day
    pd_ = mg["per_day"]
    ok_math = True
    for (shock, regime), grp in pd_.groupby(["shock", "regime"]):
        want = float(np.sum(grp["n_obs"] * grp["resp_x100"]) / np.sum(grp["n_obs"]))
        got = float(wm.loc[shock, regime])
        ok_math &= abs(got - want) < 1e-9
    # with unequal day lengths in 'volatile', weighted must sit CLOSER to the longer
    # day's response than the unweighted mean does (|w-b| = (na/(na+nb))|a-b| < |u-b|)
    vol = pd_[pd_.regime == "volatile"]
    tilt = True
    for shock, grp in vol.groupby("shock"):
        by_n = grp.sort_values("n_obs")
        if len(by_n) == 2 and abs(by_n.iloc[0]["resp_x100"] - by_n.iloc[1]["resp_x100"]) > 1e-6:
            b = by_n.iloc[1]["resp_x100"]
            u = float(mg["point"].loc[shock, "volatile"])
            w = float(wm.loc[shock, "volatile"])
            tilt &= abs(w - b) <= abs(u - b) + 1e-12
    ok = shaped and ok_math and tilt
    print("(2) weighted MG: shape matches (%s); equals sum(n*resp)/sum(n) from per_day "
          "(%s); tilts toward the longer day (%s) : %s" % (shaped, ok_math, tilt, ok))
    return ok


def check_rigobon_overid_row():
    import paper_tables as pt
    from test_panel_svar import _book_frame
    three = [("d1", "benchmark", _book_frame(seed=5, n=2500)),
             ("d2", "volatile", _book_frame(seed=6, n=2500)),
             ("d3", "mwcb", _book_frame(seed=7, n=2500))]
    t3 = pt.table_rigobon(three)
    has3 = any("over-ID" in str(i) for i in t3.df.index)
    two = three[:2]
    t2 = pt.table_rigobon(two)
    has2 = any("over-ID" in str(i) for i in t2.df.index)
    ok = has3 and not has2
    print("(3) Rigobon over-ID row present with 3 labels (%s), absent with 2 (%s) : %s"
          % (has3, not has2, ok))
    return ok


def check_realbar_lag_diagnostic():
    r = sp.run([sys.executable, "run_table9_both_ways.py", "--source", "demo", "--n-demo", "4",
                "--n-demo-bars", "3000", "--n-boot", "0", "--corr-window", "40",
                "--n-lags", "bic", "--pmax", "6", "--bar-seconds", "30",
                "--no-dcc", "--no-mean-group"],
               capture_output=True, text=True, timeout=540, cwd=HERE)
    ok = r.returncode == 0 and "RealBar diagnostic" in r.stdout and "gap" in r.stdout
    print("(4) Table 9 CLI prints the RealBar own-lag MA(1) diagnostic under a criterion "
          "(rc=%d) : %s" % (r.returncode, ok))
    return ok


def check_gram_conditioning():
    import correlation_svar as cs
    rng = np.random.default_rng(9)
    Z = rng.normal(size=(4000, 4)) * np.array([1.0, 1e-6, 1e6, 1.0])
    b_true = np.array([[0.5], [2e5], [3e-6], [-1.0]])
    y = Z @ b_true + rng.normal(0, 1e-3, size=(4000, 1))
    b_ls = np.linalg.lstsq(Z, y, rcond=None)[0]
    B, _rss = cs._gram_solve(Z.T @ Z, Z.T @ y, y.T @ y)
    # relative to each coefficient's own scale, the equilibrated Gram solve must match
    # lstsq to float precision despite the 1e12 spread in column norms
    rel = np.abs(B - b_ls) / np.maximum(np.abs(b_ls), 1e-12)
    ok = bool(np.max(rel) < 1e-6)
    print("(5) Jacobi-equilibrated Gram solve matches lstsq to %.1e max relative error on "
          "a 1e6-column-scale-spread design : %s" % (float(np.max(rel)), ok))
    return ok


def check_rank_sample():
    r = sp.run([sys.executable, "rank_sample.py", "--selftest"],
               capture_output=True, text=True, timeout=180, cwd=HERE)
    ok = r.returncode == 0 and "rank_sample checks -> True" in r.stdout
    print("(6) rank_sample --selftest passes (rc=%d) : %s" % (r.returncode, ok))
    return ok


def check_garch_scale_invariance():
    """v0.9.65 audit fix: garch_x_fit standardizes internally, so raw log-return scale
    (~4e-9 variance at 1s) no longer pins omega to its absolute bound or floors h."""
    import dcc_garch as dg
    rng = np.random.default_rng(6)
    T = 4000; om_t, al_t, be_t = 8.4e-11, 0.08, 0.90         # uncond var ~4.2e-9 (1s scale)
    e = np.empty(T); h = om_t / (1 - al_t - be_t)
    for t in range(T):
        if t:
            h = om_t + al_t * e[t - 1] ** 2 + be_t * h
        e[t] = np.sqrt(h) * rng.normal()
    raw = dg.garch_x_fit(e)
    scl = dg.garch_x_fit(e * 1e4)
    # exact bit-equality across scaling is NOT attainable: var(e*1e4) rounds differently
    # from var(e)*1e8 in the last ulp, and L-BFGS numerical gradients amplify that into
    # slightly different optima. The fix's guarantee is that both scales land on the SAME
    # model: parameters within a couple percent, omega scaling by ~1e8, near-identical h
    # paths -- versus the old raw-scale fit whose omega pinned to its absolute bound and
    # whose persistence collapsed outright.
    ab_close = (abs(raw["alpha"] - scl["alpha"]) < 0.02 and abs(raw["beta"] - scl["beta"]) < 0.02)
    om_ratio = scl["omega"] / raw["omega"]
    om_scales = abs(om_ratio - 1e8) / 1e8 < 0.05
    h_track = float(np.corrcoef(raw["h"], scl["h"])[0, 1]) > 0.999
    persistent = raw["alpha"] + raw["beta"] > 0.8            # the old raw-scale fit collapsed this
    ok = ab_close and om_scales and h_track and persistent
    print("(7) GARCH scale invariance at raw 1s magnitudes: alpha/beta agree across x1e4 "
          "scaling (%s), omega scales by ~1e8 (ratio/1e8 = %.4f, %s), h paths track (%s), "
          "persistence recovered a+b=%.3f (%s) : %s"
          % (ab_close, om_ratio / 1e8, om_scales, h_track,
             raw["alpha"] + raw["beta"], persistent, ok))
    return ok


def check_dcc_masked_alignment():
    """v0.9.65 audit fix: _dcc_corr scatters the clean-row rho back onto the full grid, so a
    halt-masked session no longer crashes build_svar_frame(corr_method='dcc')."""
    import correlation_svar as cs
    from test_panel_svar import _book_frame
    df = _book_frame(seed=5, n=1500)
    for c in df.columns:
        df.loc[df.index[400:460], c] = np.nan                # a masked halt window
    rho = cs._dcc_corr(df, None, None)
    full_len = len(rho) == len(df)
    masked_nan = np.all(~np.isfinite(rho[401:461]))          # returns shift by one
    some_finite = np.isfinite(rho).sum() > 800
    try:
        X, names, ci = cs.build_svar_frame(df, "standard", 3, None, "dcc", 40, "cost_to_fill")
        no_raise = True
    except ValueError:
        no_raise = False
    ok = full_len and masked_nan and some_finite and no_raise
    print("(8) _dcc_corr on a masked session: full-length rho (%s), NaN inside the halt (%s), "
          "finite elsewhere (%s), build_svar_frame(dcc) no longer raises (%s) : %s"
          % (full_len, masked_nan, some_finite, no_raise, ok))
    return ok


def check_is_relative_guard():
    """v0.9.65 audit fix: the IS degeneracy guards are RELATIVE, so fine-grid magnitudes no
    longer silently return a fake 0.5/0.5 tie -- and the shares are scale-invariant."""
    import price_discovery_shares as pds
    import markov_switching_vecm as msv
    alpha = np.array([-3.9e-3, 3.1e-4])
    Om = np.array([[4.0e-11, 2.8e-11], [2.8e-11, 4.1e-11]])  # 10ms log-return scale
    _lo, _hi, mid = pds.hasbrouck_is(alpha, Om)
    not_tie = abs(mid[0] - 0.5) > 0.05
    _lo2, _hi2, mid2 = pds.hasbrouck_is(alpha, Om * 1e8)
    invariant = np.allclose(mid, mid2, atol=1e-9)
    z = pds.hasbrouck_is(np.zeros(2), Om)[2]
    degenerate_still = np.allclose(z, [0.5, 0.5])
    # the MS-VECM sibling: signs kept -> matches pds on SAME-SIGNED alphas, and scale-invariant
    a_same = np.array([0.052, 0.111])
    Om1 = np.array([[1.0, 0.77], [0.77, 1.21]])
    m_p = pds.hasbrouck_is(a_same, Om1)[2]
    m_m = msv.hasbrouck_is_from_alpha(a_same, Om1)[2]
    siblings_agree = np.allclose(m_p, m_m, atol=1e-6)
    m_m_small = msv.hasbrouck_is_from_alpha(a_same, Om1 * 4e-9)[2]
    msv_invariant = np.allclose(m_m, m_m_small, atol=1e-6)
    ok = not_tie and invariant and degenerate_still and siblings_agree and msv_invariant
    print("(9) relative IS guards: 10ms-scale system is NOT a fake tie (%s), scale-invariant "
          "(%s), true degeneracy still 0.5 (%s); MS-VECM sibling keeps signs and matches pds "
          "on same-signed alphas (%s), scale-invariant (%s) : %s"
          % (not_tie, invariant, degenerate_still, siblings_agree, msv_invariant, ok))
    return ok


def check_romano_wolf_degenerate():
    """v0.9.65 audit fix: a zero-bootstrap-variance coefficient gets adj_p=1, never ***; and
    an impossible block length no longer collapses every draw to the identical sample."""
    import inference as inf
    rng = np.random.default_rng(2)
    point = np.array([0.1, 0.5, -0.2, 0.0])
    boot = rng.normal(size=(200, 4)) * 0.1 + point
    boot[:, 3] = 0.0                                         # degenerate column
    rw = inf.romano_wolf_from_boot(point, boot)
    degen_ok = rw["adj_p"][3] == 1.0 and not rw["reject"][3]
    others_ok = np.all(np.isfinite(rw["adj_p"][:3]))
    x = rng.normal(size=200)
    draws = inf.moving_block_bootstrap(x, lambda d: d.mean(), n_boot=50, block_len=500, seed=0)
    varied = float(np.std(draws)) > 0
    ok = degen_ok and others_ok and varied
    print("(10) Romano-Wolf: degenerate column adj_p=1, no reject (%s), others finite (%s); "
          "block_len >= T produces varied draws, not 50 identical copies (%s) : %s"
          % (degen_ok, others_ok, varied, ok))
    return ok


def check_memo_soundness():
    """v0.9.65 audit fix: the attrs memos carry a data fingerprint, extra_fn builds are not
    cached, masked copies do not serve the unmasked X, and persistence strips the memos."""
    import correlation_svar as cs
    import market_halts as mh
    import derive_frames as dvf
    from test_panel_svar import _book_frame
    idx_kw = dict(seed=5, n=1500)
    df = _book_frame(**idx_kw)
    df.attrs["halt_windows_SPY"] = [(df.index[400], df.index[460])]
    df.attrs["halt_windows_ES"] = [(df.index[400], df.index[460])]
    X0, _n, _c = cs.build_svar_frame(df, "standard", 3, None, "rolling", 40, "cost_to_fill")
    memo_set = "_svar_frame_memo" in df.attrs
    masked, rep = mh.mask_frame(df)
    stripped_on_copy = "_svar_frame_memo" not in masked.attrs
    X1, _n1, _c1 = cs.build_svar_frame(masked, "standard", 3, None, "rolling", 40, "cost_to_fill")
    different = len(X1) < len(X0)                            # halt rows fell out of the design
    quiet = _book_frame(seed=6, n=1500)
    out_q, _rep_q = mh.mask_frame(quiet)
    untouched = out_q is quiet and "halt_masked" not in quiet.attrs
    import ab_halt_mask as ab
    zero_dict = _book_frame(seed=7, n=300)
    zero_dict.attrs["halt_masked"] = {"SPY": 0, "ES": 0}
    ab_ok = not ab._masked_at_rest([("d", "benchmark", zero_dict)])
    strip = cs.strip_memo_attrs(df)
    stripped = "_svar_frame_memo" not in strip.attrs
    ok = memo_set and stripped_on_copy and different and untouched and ab_ok and stripped
    print("(11) memo soundness: memo set on build (%s); masked copy carries no memo (%s) and "
          "rebuilds a smaller design (%s); no-halt mask leaves the original unannotated (%s); "
          "ab_halt_mask ignores zero-count dicts (%s); strip_memo_attrs strips (%s) : %s"
          % (memo_set, stripped_on_copy, different, untouched, ab_ok, stripped, ok))
    return ok


def main():
    checks = [check_gfevd, check_weighted_mean_group, check_rigobon_overid_row,
              check_realbar_lag_diagnostic, check_gram_conditioning, check_rank_sample,
              check_garch_scale_invariance, check_dcc_masked_alignment,
              check_is_relative_guard, check_romano_wolf_degenerate, check_memo_soundness]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback
            traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("improvement checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
