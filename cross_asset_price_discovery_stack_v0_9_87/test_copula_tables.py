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
  (7) nonparametric estimator vs a KNOWN answer: planted Clayton -> empirical lambda_L
      at q=0.95 lands on the closed-form Clayton tail concentration C(u,u)=(2u^-th-1)^
      (-1/th) evaluated at that level; upper concentration small; L beats U decisively
  (8) planted Gumbel -> the mirror image (empirical U on its closed-form diagonal, U>L)
  (9) planted symmetric t data -> the empirical asymmetry is ~0: the np estimator must
      not manufacture asymmetry where none exists (it exists to REMOVE the constraint
      channel, not to add a biased one)
  (10) the *_np columns appear in the rendered regime/liquidity tables and the existing
      parametric columns are byte-identical with and without the np record fields

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


def _clayton_diag(w, th):
    """Closed-form Clayton copula on the diagonal: C(w,w) = (2 w^-theta - 1)^(-1/theta)."""
    return (2.0 * w**(-th) - 1.0)**(-1.0 / th)


def _gumbel_diag(w, th):
    """Closed-form Gumbel copula on the diagonal: C(w,w) = w^(2^(1/theta))."""
    return w**(2.0**(1.0 / th))


def check_np_clayton():
    """WHY: the empirical estimator must land on a KNOWN answer, and the known answer at
    fixed q is the level-q tail CONCENTRATION (not the asymptotic lambda) -- Clayton's
    diagonal is closed-form, so both targets are exact by construction. Clayton is the
    crash-only family: L must beat U decisively, which is exactly the asymmetry the
    winning t family is structurally unable to report."""
    import copula_garch as cg
    th, q, n = 2.0, 0.95, 20000
    rng = np.random.default_rng(202)
    U = cg.pseudo_obs(cg.simulate_copula("clayton", n, {"theta": th}, rng))
    e = cg.empirical_tail_dependence(U[:, 0], U[:, 1], q)
    p = 1.0 - q
    lo_true = _clayton_diag(p, th) / p                     # 0.7075 (limit lambda_L 0.7071)
    up_true = (1.0 - 2.0 * q + _clayton_diag(q, th)) / p   # 0.1368 (limit lambda_U 0)
    ok1 = abs(e["lambda_L"] - lo_true) < 0.08
    ok2 = abs(e["lambda_U"] - up_true) < 0.08 and e["lambda_U"] < 0.30
    ok3 = e["lambda_L"] > e["lambda_U"] + 0.30
    # tail_concentration must agree with the point estimator on its grid
    tc = cg.tail_concentration(U[:, 0], U[:, 1], qs=(0.90, 0.95, 0.99))
    row = tc[np.isclose(tc["q"], q)].iloc[0]
    ok4 = (abs(float(row["lower"]) - e["lambda_L"]) < 1e-12
           and abs(float(row["upper"]) - e["lambda_U"]) < 1e-12
           and list(tc.columns) == ["q", "lower", "upper"])
    print(f"  clayton(th={th}): lL_np={e['lambda_L']:.3f} (closed-form {lo_true:.3f}), "
          f"lU_np={e['lambda_U']:.3f} (closed-form {up_true:.3f}); grid consistent: {ok4}")
    return ok1 and ok2 and ok3 and ok4


def check_np_gumbel():
    """WHY: the mirror image of the Clayton check -- a rally-tail family must produce
    U > L in the empirical estimates, with both landing on Gumbel's closed-form diagonal
    concentration. Also pins the documented finite-q bias in the honest direction: at
    q=0.95 Gumbel's LOWER concentration is ~0.29 even though its limit lambda_L is 0 --
    body dependence leaks into any fixed-level count, which is why the columns are
    reported at fixed q and never extrapolated."""
    import copula_garch as cg
    th, q, n = 2.0, 0.95, 20000
    rng = np.random.default_rng(203)
    U = cg.pseudo_obs(cg.simulate_copula("gumbel", n, {"theta": th}, rng))
    e = cg.empirical_tail_dependence(U[:, 0], U[:, 1], q)
    p = 1.0 - q
    lo_true = _gumbel_diag(p, th) / p                      # 0.289 (limit lambda_L 0)
    up_true = (1.0 - 2.0 * q + _gumbel_diag(q, th)) / p    # 0.601 (limit lambda_U 0.586)
    ok = (abs(e["lambda_U"] - up_true) < 0.08 and abs(e["lambda_L"] - lo_true) < 0.08
          and e["lambda_U"] > e["lambda_L"] + 0.20)
    print(f"  gumbel(th={th}): lU_np={e['lambda_U']:.3f} (closed-form {up_true:.3f}), "
          f"lL_np={e['lambda_L']:.3f} (closed-form {lo_true:.3f}); U>L: "
          f"{e['lambda_U'] > e['lambda_L'] + 0.20}")
    return ok


def check_np_symmetric_t():
    """WHY: the np estimator exists because the t family FORCES lambda_L == lambda_U; it
    is only a fair referee if it reports ~zero asymmetry when the truth IS symmetric.
    On planted t data (radially symmetric by construction) the empirical L-minus-U must
    stay inside sampling noise -- the estimator must not manufacture the asymmetry it was
    brought in to adjudicate."""
    import copula_garch as cg
    rng = np.random.default_rng(204)
    U = cg.pseudo_obs(cg.simulate_copula("t", 20000, {"rho": 0.5, "nu": 4.0}, rng))
    e = cg.empirical_tail_dependence(U[:, 0], U[:, 1], 0.95)
    asym = e["lambda_L"] - e["lambda_U"]
    ok = abs(asym) < 0.05 and e["lambda_L"] > 0.1 and e["lambda_U"] > 0.1
    print(f"  t(rho=0.5, nu=4): lL_np={e['lambda_L']:.3f} lU_np={e['lambda_U']:.3f} "
          f"asym={asym:+.4f} (|asym| < 0.05: {abs(asym) < 0.05})")
    return ok


def check_np_columns_and_backcompat():
    """WHY: the np columns are only credible BESIDE the parametric menu if adding them
    changed nothing else -- so (a) the rendered regime table carries the five np columns
    after the parametric block, (b) stripping the np fields from the per-day records
    reproduces the previous table byte-identically (same columns, same values, same
    order), and (c) the liquidity table carries its np thin/deep twin."""
    import copula_garch as cg
    import paper_tables as pt
    fams = ("gaussian", "frank", "clayton", "gumbel", "joe")
    days = [(f"2024-08-{d + 1:02d}", "volatile" if d % 2 else "benchmark",
             _ret_frame(3000, 70 + d, down_mult=2.0)) for d in range(4)]
    recs = cg.day_copula_records(days, fams, min_obs=500, trim_min=2.0)
    t_new = cg._group_copula_rows(recs, n_flip=2000)
    np_cols = ["median lambda_L_np", "median lambda_U_np", "mean np tail asym (L-U)",
               "se(np asym)", "sign-flip p (np)"]
    ok1 = all(c in t_new.columns for c in np_cols)
    stripped = [{k: v for k, v in r.items() if k not in ("lL_np", "lU_np", "dlam_np")}
                for r in recs]
    t_old = cg._group_copula_rows(stripped, n_flip=2000)
    old_cols = [c for c in t_new.columns if c not in np_cols]
    ok2 = (list(t_old.columns) == old_cols
           and list(t_new.columns)[:len(old_cols)] == old_cols
           and t_new[old_cols].equals(t_old[old_cols]))
    md = pt.Table("np check", t_new, "n").to_markdown()
    ok3 = "lambda_L_np" in md and "sign-flip p (np)" in md
    tl, _ = cg.table_copula_liquidity(days, min_obs=500, trim_min=2.0, n_flip=2000)
    liq_np = ["median lambda_L_np thin", "median lambda_L_np deep",
              "mean np delta (thin-deep)", "se(np delta)", "sign-flip p (np)"]
    liq_old = ["median lambda_L thin", "median lambda_L deep", "mean delta (thin-deep)",
               "se(delta)", "sign-flip p", "days"]
    ok4 = (all(c in tl.columns for c in liq_np)
           and list(tl.columns)[:len(liq_old)] == liq_old
           and np.isfinite(float(tl.loc["all days", "mean np delta (thin-deep)"])))
    print(f"  regime np cols present: {ok1}; parametric block byte-identical without np "
          f"fields: {ok2}; markdown renders np: {ok3}; liquidity np twin: {ok4}")
    return ok1 and ok2 and ok3 and ok4


def main():
    warnings.simplefilter("ignore")
    checks = [("joe/frank densities integrate to 1", check_density_normalization),
              ("family recovery on simulated truths", check_family_recovery),
              ("return-copula asymmetry: planted found, symmetric clean", check_return_asymmetry),
              ("liquidity table: end-to-end, no invented effect", check_liquidity_table),
              ("flow-copula asymmetry: planted tandem selling found", check_flow_asymmetry),
              ("driver/runner/self-test wiring", check_wiring),
              ("np tail dependence: Clayton closed-form recovered", check_np_clayton),
              ("np tail dependence: Gumbel mirror image", check_np_gumbel),
              ("np tail dependence: symmetric t stays symmetric", check_np_symmetric_t),
              ("np columns rendered; parametric block untouched",
               check_np_columns_and_backcompat)]
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
