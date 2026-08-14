#!/usr/bin/env python3
"""test_golden_numbers.py -- frozen numeric outputs on deterministic inputs.

The other gates pin BEHAVIORS (a fix stays fixed, a contract holds). Nothing pinned
NUMBERS: a silent numeric drift -- a library upgrade changing a default, a refactor
flipping a sign convention, an EPS moved -- would pass every behavioral gate and change
every table. This module freezes five results computed on fully deterministic inputs
(fixed seeds; numpy guarantees default_rng stream stability across versions):

  (A) Hasbrouck / Gonzalo-Granger shares on a hand-fixed (alpha, Omega)
  (B) pds.estimate_day on a seeded synthetic ECM pair (alphas, kappa, IS, CS)
  (C) tandem dependence_summary on a hand-fixed 3x3 state matrix (log OR family)
  (D) inference.wild_cluster_bootstrap p-value on seeded clustered data (Webb weights)
  (E) the pooled FE correlation IRF on seeded book frames (end-to-end through
      build_svar_frame -> panel Grams -> orthogonalized IRF)

Golden values were computed at v0.9.65 (numpy stream-stable). Tolerance is rtol=1e-6 --
loose enough for BLAS reduction-order differences across platforms, tight enough that any
formula change fails. If a check fails after an INTENTIONAL methodology change, re-derive
the constants and say so in the CHANGELOG; a failure after anything else is a regression.

Run: python test_golden_numbers.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

OK = True


def _close(name, got, want, rtol=1e-6, atol=1e-9):
    global OK
    good = np.allclose(np.asarray(got, float), np.asarray(want, float), rtol=rtol, atol=atol)
    if not good:
        OK = False
    print("    %-28s got %s  want %s : %s" % (name, np.round(got, 8), want, good))
    return good


def check_hasbrouck_closed_form():
    import price_discovery_shares as pds
    alpha = np.array([-0.10, 0.20]); Om = np.array([[1.0, 0.5], [0.5, 2.0]])
    _lo, _hi, mid = pds.hasbrouck_is(alpha, Om)
    gg = pds.gonzalo_granger(alpha)
    print("(A) Hasbrouck / GG on fixed (alpha, Omega)")
    ok = _close("IS_mid", mid, [0.609375, 0.390625])
    ok &= _close("CS", gg, [0.6666666667, 0.3333333333])
    return ok


def check_estimate_day():
    import price_discovery_shares as pds
    from test_audit_fixes import _ecm_pair
    m1, m2 = _ecm_pair(8000, 42)
    d = pds.estimate_day(m1, m2, n_lags=5)
    print("(B) estimate_day on the seeded ECM pair")
    ok = _close("alpha_spy", d["alpha_spy"], -0.0777149)
    ok &= _close("alpha_es", d["alpha_es"], -0.0091831)
    ok &= _close("kappa", d["kappa"], 0.0685318)
    ok &= _close("IS_mid_SPY", d["IS_mid_SPY"], 0.00324929)
    ok &= _close("CS_ES", d["CS_ES"], 0.8943232)
    return ok


def check_tandem_summary():
    import tandem_order_flow as tof
    M = pd.DataFrame([[8.0, 12.0, 4.0], [10.0, 30.0, 12.0], [3.0, 13.0, 8.0]],
                     index=tof.DIR3, columns=tof.DIR3)
    d = tof.dependence_summary(M, n_bars=23400)
    print("(C) dependence_summary on a fixed 3x3 matrix")
    ok = _close("PCMOF", d["PCMOF"], 16.0)
    ok &= _close("log_OR", d["log_OR"], 1.67397643)
    ok &= _close("log_OR_z", d["log_OR_z"], 28.0509705)
    ok &= _close("lor_sell", d["lor_sell"], 0.69314718)
    ok &= _close("corner_asym", d["corner_asym"], 0.26236426)
    return ok


def check_wild_cluster():
    import inference as inf
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(8), 40)
    x = rng.normal(size=320); u = rng.normal(size=320) + rng.normal(size=8)[g]
    y = 0.3 * x + u
    X = np.column_stack([np.ones(320), x])
    w = inf.wild_cluster_bootstrap(y, X, g, test_idx=1, n_boot=499, seed=1)
    print("(D) wild cluster bootstrap on seeded clustered data")
    ok = _close("beta", w["beta"], 0.39586517)
    ok &= _close("p", w["p"], 0.004, rtol=0, atol=1e-12)   # exact: counting statistic
    ok &= w["weights"] == "webb6"
    print("    weights scheme webb6 : %s" % (w["weights"] == "webb6"))
    return ok


def check_pooled_fe_irf():
    import correlation_svar as cs
    from test_panel_svar import _book_frame
    sessions = [("d1", "volatile", _book_frame(seed=5)), ("d2", "volatile", _book_frame(seed=6))]
    tbl = cs.correlation_irf(sessions, spec="standard", n_levels=3, n_lags=2,
                             corr_method="rolling", corr_window=40, min_obs=100, panel="fe")
    print("(E) pooled FE correlation IRF on seeded book frames (end-to-end)")
    ok = _close("Spread_ES", tbl.loc["Spread_ES", "volatile"], 0.01770332)
    ok &= _close("Spread_SPY", tbl.loc["Spread_SPY", "volatile"], -0.01140504)
    return ok


def main():
    checks = [check_hasbrouck_closed_form, check_estimate_day, check_tandem_summary,
              check_wild_cluster, check_pooled_fe_irf]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback
            traceback.print_exc()
            res.append(False)
        print()
    ok = all(res) and OK
    print("golden-number checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
