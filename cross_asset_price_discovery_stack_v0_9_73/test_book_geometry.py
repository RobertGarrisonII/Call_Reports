#!/usr/bin/env python3
"""Gate: the ||L|| (arc-length) book-geometry check behaves as derived.

The measure ships AS A CHECK: tau (normalized index-square arc length) is
predicted to be a pure concentration index -- permutation-invariant in the
levels, rank-equivalent to the Herfindahl, and worth ~zero incremental R^2 over
{total depth, centroid, HHI}. This gate pins the derivation facts the claim
rests on, so a regression in any of them would silently change what the check
means:

  1. endpoints -- uniform depth -> tau = 0 exactly; all depth at ONE level ->
     tau = 1 exactly, whichever level it is, and the discrete max L_max(n) is
     used (normalizing by the continuum bound 2 would cap tau at ~0.86 at n=10)
  2. permutation invariance -- shuffling the levels leaves tau and HHI
     unchanged to machine precision while the CENTROID moves (location lives
     there, by construction)
  3. the sweep identity -- centroid c$ equals VWAP(full sweep) - touch exactly,
     including on a book with price gaps
  4. rank equivalence -- on Dirichlet-random share profiles, Spearman(tau, HHI)
     is high (the Schur-convex family claim, verified numerically)
  5. hygiene -- zero-depth rows give NaN (not fake-uniform 0), partial-NaN
     levels follow the _usable_levels convention
  6. the runner writes the per-day table and verdict
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import book_geometry as bg


def _book_frame(qmat_bid, qmat_ask=None, base=500.0, tick=0.01, gap_level=None):
    """Wide frame with contiguous levels (or one price gap at `gap_level`)."""
    qb = np.asarray(qmat_bid, float)
    qa = np.asarray(qmat_ask if qmat_ask is not None else qmat_bid, float)
    T, n = qb.shape
    idx = pd.date_range("2024-07-24 09:30:00", periods=T, freq="s", tz="America/New_York")
    cols = {}
    for i in range(1, n + 1):
        off = i + (3 if (gap_level is not None and i >= gap_level) else 0)
        cols[f"SPY_bidprice_{i}"] = np.full(T, base - tick * off)
        cols[f"SPY_askprice_{i}"] = np.full(T, base + tick * off)
        cols[f"SPY_bidquantity_{i}"] = qb[:, i - 1]
        cols[f"SPY_askquantity_{i}"] = qa[:, i - 1]
    return pd.DataFrame(cols, index=idx)


def check_endpoints():
    n = 10
    uni = np.full((1, n), 700.0)
    ok_uni = abs(float(bg.tau(uni)[0])) < 1e-12
    ok_one = True
    for lvl in range(n):
        q = np.zeros((1, n)); q[0, lvl] = 5000.0
        ok_one &= abs(float(bg.tau(q)[0]) - 1.0) < 1e-12
    lmax = bg.tau_L_max(n)
    ok_max = abs(lmax - (np.sqrt(1 + 1 / n**2) + (n - 1) / n)) < 1e-15 and lmax < 2.0
    print("(1) uniform -> tau=0 (%s); all-at-one-level -> tau=1 at every level (%s);"
          % (ok_uni, ok_one))
    print("    discrete L_max(10)=%.5f < 2 (continuum bound not used) : %s" % (lmax, ok_max))
    return ok_uni and ok_one and ok_max


def check_permutation_invariance():
    rng = np.random.default_rng(3)
    q = rng.gamma(2.0, 300.0, size=(200, 10))
    perm = rng.permutation(10)
    t0, t1 = bg.tau(q), bg.tau(q[:, perm])
    h0, h1 = bg.herfindahl(q), bg.herfindahl(q[:, perm])
    ok_tau = np.nanmax(np.abs(t0 - t1)) < 1e-12
    ok_hhi = np.nanmax(np.abs(h0 - h1)) < 1e-12
    px = 500.0 - 0.01 * np.arange(1, 11)[None, :] * np.ones((200, 1))
    touch = np.full(200, 500.0 - 0.01)
    c0, _ = bg.centroid_concession(px, q, touch)
    c1, _ = bg.centroid_concession(px, q[:, perm], touch)
    ok_cen = np.nanmax(np.abs(c0 - c1)) > 1e-6          # centroid MUST move
    print("(2) level shuffle: tau invariant (%s), HHI invariant (%s), centroid moves (%s)"
          % (ok_tau, ok_hhi, ok_cen))
    return bool(ok_tau and ok_hhi and ok_cen)


def check_sweep_identity():
    rng = np.random.default_rng(7)
    ok = True
    for gap in (None, 4):                                # contiguous AND gapped book
        q = rng.gamma(2.0, 200.0, size=(50, 10)) + 1.0
        df = _book_frame(q, gap_level=gap)
        for side in ("bid", "ask"):
            px, qty = bg._side_matrices(df, "SPY", side, 10)
            touch = df[f"SPY_{side}price_1"].to_numpy(float)
            c, _ = bg.centroid_concession(px, qty, touch)
            vwap = np.nansum(px * qty, axis=1) / qty.sum(axis=1)
            ok &= np.nanmax(np.abs(np.abs(vwap - touch) - c)) < 1e-9
    print("(3) centroid == |VWAP(full sweep) - touch| exactly, gapped book included : %s" % ok)
    return bool(ok)


def check_rank_equivalence():
    """Family membership, not identity: tau and HHI are both symmetric
    Schur-convex in the shares, so they must rank profiles the same way up to
    the higher-moment differences tau carries. On maximally-variable Dirichlet
    draws the measured rho is 0.93-0.96 (real books, with far less shape
    variation, sit higher); the pin is > 0.90 on every concentration."""
    rng = np.random.default_rng(11)
    rho_all = []
    for conc in (0.5, 1.0, 3.0):                         # spiky to diffuse profiles
        s = rng.dirichlet(np.full(10, conc), size=4000)
        q = s * 1000.0
        rho = bg._spearman(bg.tau(q), bg.herfindahl(q))
        rho_all.append(rho)
    ok = all(r > 0.90 for r in rho_all)
    print("(4) Spearman(tau, HHI) on Dirichlet profiles (conc 0.5/1/3): %s (> 0.90 each) : %s"
          % (", ".join("%.3f" % r for r in rho_all), ok))
    return ok


def check_hygiene():
    q = np.array([[0.0] * 10, [np.nan] * 10, [100.0] * 10])
    t = bg.tau(q)
    ok_nan = np.isnan(t[0]) and np.isnan(t[1]) and np.isfinite(t[2])
    px = np.array([[500.0 - 0.01 * i for i in range(1, 11)]] * 3)
    px[2, 5] = np.nan                                     # torn level: price NaN, size present
    import liquidity_curve_metrics as lcm
    P, Q = lcm._usable_levels(px, np.full((3, 10), 50.0))
    ok_torn = Q[2, 5] == 0.0                              # carries no depth anywhere
    print("(5) zero/NaN-depth rows -> tau NaN not fake-uniform (%s); torn level carries "
          "no depth (%s)" % (ok_nan, ok_torn))
    return bool(ok_nan and ok_torn)


def check_runner():
    import run_book_geometry as rbg
    with tempfile.TemporaryDirectory() as td:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rbg.main(["--source", "demo", "--n-demo", "4", "--n-demo-bars", "600",
                           "--horizon", "20", "--out-dir", td])
        files = os.listdir(td)
        per_day = [f for f in files if "book_geometry_per_day" in f]
        ok_files = len(per_day) == 1
        ok_cols = False
        if per_day:
            t = pd.read_csv(os.path.join(td, per_day[0]))
            ok_cols = ({"tau_bid", "hhi_bid", "centroid_asym", "spearman_tau_hhi_bid",
                        "tau_increment"} <= set(t.columns)) and len(t) > 0
        out = buf.getvalue()
        ok_verdict = "VERDICT" in out
    print("(6) runner: rc=%s, per-day CSV written (%s), verdict printed (%s), columns (%s)"
          % (rc, ok_files, ok_verdict, ok_cols))
    return rc == 0 and ok_files and ok_verdict and ok_cols


def main():
    checks = [check_endpoints, check_permutation_invariance, check_sweep_identity,
              check_rank_equivalence, check_hygiene, check_runner]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("book-geometry checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
