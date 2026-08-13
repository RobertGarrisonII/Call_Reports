#!/usr/bin/env python3
"""test_copula_tables.py -- the v0.9.70 copula exhibits behave as specified.

  (1) the new Joe and Frank densities integrate to ~1 (they are densities, not scores)
  (2) family recovery on simulated truths: Clayton data -> Clayton wins the full menu
      with lambda_L recovered and the survival rotation ranked worse; Gumbel/Joe data ->
      an upper-tail family wins with lambda_U - lambda_L large; Frank data -> a
      no-tail-dependence family wins with both lambdas ~0; rotated-Clayton data -> the
      selected family carries the upper tail
  (3) a planted sign-dependent common factor in RETURNS (down-loading doubled) produces a
      positive BB1 lower-minus-upper contrast in table_copula_regimes with a small
      day-level sign-flip p; a symmetric DGP stays clean
  (4) table_copula_liquidity runs end-to-end on varying-depth books and does NOT invent a
      thin-book effect under a depth-independent DGP
  (5) a planted asymmetric tandem-flow DGP (exact OFI injection) produces a positive
      contrast in table_copula_flows with a small sign-flip p
  (6) driver/runner wiring: STAGE 5c calls run_copula.py; the parser carries the
      documented flags; the self-test jobs exist in build_all_tables

Run: python test_copula_tables.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NY = "America/New_York"


def check_density_normalization():
    import copula_garch as cg
    # np.trapezoid arrived in numpy 2.0 (np.trapz removed there); support both
    trapz = getattr(np, "trapezoid", None) or np.trapz
    g = np.linspace(0.001, 0.999, 400)
    U, V = np.meshgrid(g, g)
    ok = True
    for name, fn, par in (("joe", cg._joe_logpdf, 2.0), ("frank", cg._frank_logpdf, 5.0)):
        I = float(trapz(trapz(np.exp(fn(U, V, par)), g, axis=1), g))
        print(f"  {name} density integral {I:.4f}")
        ok &= abs(I - 1.0) < 0.02
    return ok


def check_family_recovery():
    import copula_garch as cg
    rng = np.random.default_rng(11)
    ok = True

    U = cg.simulate_copula("clayton", 4000, {"theta": 2.0}, rng)
    sel = cg.select_copula(U, is_uniform=True)
    tab = sel["table"].set_index("copula")
    best = sel["fits"][sel["best"]]
    ok1 = (sel["best"] in ("clayton", "bb1", "sjc") and abs(best["lambda_L"] - 0.707) < 0.10
           and tab.loc["clayton", "bic"] < tab.loc["clayton180", "bic"])
    print(f"  clayton data: best={sel['best']} lL={best['lambda_L']:.3f} (true 0.707); "
          f"rotation ranked worse: {tab.loc['clayton', 'bic'] < tab.loc['clayton180', 'bic']}")

    Ug = cg.simulate_copula("gumbel", 4000, {"theta": 2.0}, rng)
    sg = cg.select_copula(Ug, is_uniform=True)
    bg = sg["fits"][sg["best"]]
    ok2 = sg["best"] in ("gumbel", "joe", "bb1", "sjc", "clayton180") and \
        (bg["lambda_U"] - bg["lambda_L"]) > 0.15
    print(f"  gumbel data: best={sg['best']} lU-lL={bg['lambda_U'] - bg['lambda_L']:.3f}")

    Uj = cg.simulate_copula("joe", 4000, {"theta": 2.5}, rng)
    fj = cg._fit_copula("joe", Uj[:, 0], Uj[:, 1])
    sj = cg.select_copula(Uj, is_uniform=True)
    bj = sj["fits"][sj["best"]]
    ok3 = abs(fj["params"]["theta"] - 2.5) < 0.5 and (bj["lambda_U"] - bj["lambda_L"]) > 0.15
    print(f"  joe data: theta_hat={fj['params']['theta']:.2f} (true 2.5), best={sj['best']}")

    Uf = cg.simulate_copula("frank", 4000, {"theta": 5.0}, rng)
    sf = cg.select_copula(Uf, is_uniform=True)
    bf = sf["fits"][sf["best"]]
    ok4 = sf["best"] in ("frank", "gaussian", "t") and \
        max(bf["lambda_L"], bf["lambda_U"]) < 0.12
    print(f"  frank data: best={sf['best']} lL={bf['lambda_L']:.3f} lU={bf['lambda_U']:.3f}")

    Ur = cg.simulate_copula("clayton180", 4000, {"theta": 2.0}, rng)
    sr = cg.select_copula(Ur, is_uniform=True)
    br = sr["fits"][sr["best"]]
    ok5 = br["lambda_U"] > 0.3 and br["lambda_L"] < 0.15
    print(f"  rotated clayton: best={sr['best']} lL={br['lambda_L']:.3f} lU={br['lambda_U']:.3f}")

    return ok and ok1 and ok2 and ok3 and ok4 and ok5


def _asym_pair(T, rng, down_mult=2.0, rho_load=0.7):
    """Two series driven by a common factor whose loading doubles when the factor is
    negative -- joint crashes tighter than joint rallies (down_mult=1 is symmetric)."""
    f = rng.standard_normal(T)
    load = np.where(f < 0, rho_load * down_mult, rho_load)
    e1 = load * f + np.sqrt(1 - rho_load**2) * rng.standard_normal(T)
    e2 = load * f + np.sqrt(1 - rho_load**2) * rng.standard_normal(T)
    return e1, e2


def _ret_frame(T, seed, down_mult=1.0, n_levels=10):
    """Book frames whose mids follow correlated (optionally down-asymmetric) random walks
    and whose level-1 depth varies -- for the return-copula and liquidity tables."""
    rng = np.random.default_rng(seed)
    e1, e2 = _asym_pair(T, rng, down_mult=down_mult)
    p1 = 500.0 * np.exp(np.cumsum(e1) * 1e-4)
    p2 = 5000.0 * np.exp(np.cumsum(e2) * 1e-4)
    idx = pd.date_range("2024-08-05 09:30", periods=T, freq="s", tz=NY)
    cols = {}
    for a, p, tick in (("SPY", p1, 0.01), ("ES", p2, 0.25)):
        for l in range(1, n_levels + 1):
            cols[f"{a}_bidprice_{l}"] = p - l * tick
            cols[f"{a}_askprice_{l}"] = p + l * tick
            cols[f"{a}_bidquantity_{l}"] = 150.0 + 50.0 * np.abs(rng.standard_normal(T))
            cols[f"{a}_askquantity_{l}"] = 150.0 + 50.0 * np.abs(rng.standard_normal(T))
    return pd.DataFrame(cols, index=idx)


def check_return_asymmetry():
    import copula_garch as cg
    fams = ("gaussian", "frank", "clayton", "gumbel", "joe")
    days_a = [(f"2024-08-{d + 1:02d}", "volatile" if d % 2 else "benchmark",
               _ret_frame(4000, 70 + d, down_mult=2.0)) for d in range(8)]
    ta, _ = cg.table_copula_regimes(days_a, fams, min_obs=500, trim_min=2.0, n_flip=2000)
    da = float(ta.loc["all days", "mean tail asym (L-U)"])
    pa = float(ta.loc["all days", "sign-flip p"])
    days_s = [(f"2024-08-{d + 1:02d}", "volatile" if d % 2 else "benchmark",
               _ret_frame(4000, 90 + d, down_mult=1.0)) for d in range(8)]
    ts, _ = cg.table_copula_regimes(days_s, fams, min_obs=500, trim_min=2.0, n_flip=2000)
    ds = float(ts.loc["all days", "mean tail asym (L-U)"])
    ps = float(ts.loc["all days", "sign-flip p"])
    ok = da > 0.05 and pa < 0.10 and ps > 0.05 and abs(ds) < abs(da)
    print(f"  planted: dlam {da:+.3f} (p {pa:.4f}); symmetric: dlam {ds:+.3f} (p {ps:.4f})")
    return ok


def check_liquidity_table():
    import copula_garch as cg
    days = [(f"2024-08-{d + 1:02d}", "volatile" if d % 2 else "benchmark",
             _ret_frame(4000, 110 + d, down_mult=1.5)) for d in range(8)]
    t, _ = cg.table_copula_liquidity(days, min_obs=500, trim_min=2.0, n_flip=2000)
    ok = (not t.empty) and "all days" in t.index
    if ok:
        d = float(t.loc["all days", "mean delta (thin-deep)"])
        p = float(t.loc["all days", "sign-flip p"])
        # depth is independent of the return DGP here: the split must NOT invent an effect
        ok = abs(d) < 0.15 and p > 0.05
        print(f"  depth-independent DGP: delta {d:+.3f} (p {p:.4f}) -- no invented effect: {ok}")
    return ok


def check_flow_asymmetry():
    import copula_garch as cg
    from test_flow_correlation import _designed_frame
    rng = np.random.default_rng(31)
    days = []
    for d in range(8):
        T = 4000
        u1, u2 = _asym_pair(T, rng, down_mult=2.0)
        fa = np.zeros(T)
        fb = np.zeros(T)
        for t in range(1, T):                              # mild own-persistence
            fa[t] = 0.3 * fa[t - 1] + u1[t]
            fb[t] = 0.3 * fb[t - 1] + u2[t]
        days.append((f"2024-08-{d + 1:02d}", "volatile" if d % 2 else "benchmark",
                     _designed_frame(fa, fb)))
    fams = ("gaussian", "frank", "clayton", "gumbel", "joe")
    t, _ = cg.table_copula_flows(days, fams, min_obs=500, trim_min=2.0, n_flip=2000)
    d = float(t.loc["all days", "mean tail asym (L-U)"])
    p = float(t.loc["all days", "sign-flip p"])
    ok = d > 0.05 and p < 0.10
    print(f"  planted tandem-selling: dlam {d:+.3f} (sign-flip p {p:.4f})")
    return ok


def check_wiring():
    import run_copula as rc
    drv = open(os.path.join(HERE, "run_paper_replication.sh")).read()
    ok1 = "run_copula.py" in drv and "STAGE 5c" in drv
    hp = rc.build_parser().format_help()
    ok2 = all(f in hp for f in ("--families", "--min-obs", "--trim-min", "--n-flip",
                                "--tag", "--out-dir"))
    pt_src = open(os.path.join(HERE, "paper_tables.py")).read()
    ok3 = "copula_regimes" in pt_src and "_copula_table" in pt_src
    print(f"  driver wired: {ok1}; runner flags: {ok2}; self-test jobs: {ok3}")
    return ok1 and ok2 and ok3


def main():
    warnings.simplefilter("ignore")
    checks = [("joe/frank densities integrate to 1", check_density_normalization),
              ("family recovery on simulated truths", check_family_recovery),
              ("return-copula asymmetry: planted found, symmetric clean", check_return_asymmetry),
              ("liquidity table: end-to-end, no invented effect", check_liquidity_table),
              ("flow-copula asymmetry: planted tandem selling found", check_flow_asymmetry),
              ("driver/runner/self-test wiring", check_wiring)]
    rc = 0
    for i, (name, fn) in enumerate(checks, 1):
        try:
            ok = fn()
        except Exception as e:
            ok = False
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
        print(f"{'ok' if ok else 'FAIL':>4}  {i}. {name}")
        rc |= 0 if ok else 1
    print("test_copula_tables:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
