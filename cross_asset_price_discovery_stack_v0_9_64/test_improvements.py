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


def main():
    checks = [check_gfevd, check_weighted_mean_group, check_rigobon_overid_row,
              check_realbar_lag_diagnostic, check_gram_conditioning, check_rank_sample]
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
