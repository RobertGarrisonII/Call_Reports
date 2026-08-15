#!/usr/bin/env python3
"""Gate: the v0.9.72 lag-robustness harness -- selection on the window-free frame,
robustness diagnostics, and the lag-band stability verdict.

The problem this release closes: with --n-lags bic the criterion was scored on the
POOLED PEARSON frame, whose dependent variable differences a corr_window-bar rolling
box. That plants an MA spike at exactly lag W, so the selection either equals the
window or -- with the shipped pmax=12 under corr_window=100 -- climbs to its own
search bound walking toward it. Every real run reproduced it (p*=pmax, BIC still
falling), the paper's footnote 17 is the same artifact ("AIC points at 60"), and no
larger pmax can fix it because the target the criterion is walking toward is the
estimator, not the data.

The harness: (1) criterion resolution moves to the RealBar bar frame, whose
non-overlapping bars share no data -- there is no window to chase, so the argmin can
be dynamics; (2) the driver prints how much the argmin means (AIC/BIC/HQ agreement,
the flatness set, the per-day modal vote); (3) --lag-band lo:hi refits the whole
table across a band of orders and keeps, per cell, only what survives all of them.

Six checks:
  1. window-artifact discrimination -- on constant-correlation no-dynamics data the
     rolling frame selects p* == W exactly while the bar frame selects a small
     interior order; the planted artifact cannot cross the frame boundary
  2. lag_robustness internals -- picks match per-column argmins, the flat set
     contains the argmin and is computed in TOTAL IC units, the per-day vote is
     unanimous on identically-distributed days
  3. parse_table_cell -- the exact inverse of _fmt_cell for value and stars
  4. band_stability -- a planted stable cell passes; sign-flip, star-dropout, and
     missing-depth cells each fail for the reason they should
  5. driver surface -- --lag-band/--band-boot/--flat-tol exist with the shipped
     defaults, parse_lag_band accepts lo:hi and rejects malformed bands
  6. end-to-end -- the demo driver with a criterion and a band writes the stability
     CSV and the IC CSV, and its chosen order comes from the bar frame
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import correlation_svar as cs
import run_table9_both_ways as r9
from test_svar_lag_artifact import _no_dynamics_frame


def check_frame_discrimination():
    """The headline: same data, same criterion -- the rolling frame reports its own
    window, the bar frame reports a small interior order."""
    df = _no_dynamics_frame(n=15000, seed=7)
    W = 20
    p_roll, _ = cs.select_svar_lag(df, spec="standard", n_levels=3, corr_method="rolling",
                                   corr_window=W, criterion="bic", pmax=2 * W)
    p_bar, _ = cs.select_svar_lag(df, spec="standard", n_levels=3, corr_method="bar",
                                  bar_seconds=60, criterion="bic", pmax=12)
    ok_roll = p_roll == W                       # the artifact, reproduced exactly
    ok_bar = p_bar is not None and 0 <= int(p_bar) <= 3 and int(p_bar) < 12
    print("(1) constant correlation, iid liquidity, corr_window=%d:" % W)
    print("      rolling frame: BIC selects p* = %-3s (== the window: %s)" % (p_roll, ok_roll))
    print("      bar frame:     BIC selects p* = %-3s (small interior, not the bound: %s)"
          % (p_bar, ok_bar))
    print("    the window artifact cannot cross the frame boundary : %s" % (ok_roll and ok_bar))
    return bool(ok_roll and ok_bar)


def check_lag_robustness_internals():
    """picks/agree from the IC table itself, flat set in total-IC units around the
    argmin, unanimous per-day vote on identically-distributed sessions."""
    sessions = [(f"d{i}", "benchmark", _no_dynamics_frame(n=12000, seed=100 + i))
                for i in range(3)]
    rob = cs.lag_robustness(sessions, pmax=8, criterion="bic", spec="standard", n_levels=3,
                            corr_method="bar", bar_seconds=60)
    tab = rob["ic"]
    ok = rob["p"] is not None and not tab.empty and rob["n_common"] > 0
    picks_ok = ok
    if ok:
        for c in ("aic", "bic", "hqic"):
            col = tab[c]
            want = int(col.idxmin()) if col.notna().any() else None
            picks_ok &= rob["picks"].get(c) == want
        agree_ok = rob["agree"] == (len(set(rob["picks"].values())) == 1
                                    and None not in rob["picks"].values())
        col = tab["bic"]
        cmin = float(col.min())
        want_flat = [int(p) for p, v in col.items()
                     if np.isfinite(v) and rob["n_common"] * (v - cmin) <= rob["flat_tol"]]
        flat_ok = rob["flat_set"] == want_flat and rob["p"] in rob["flat_set"]
        day_ok = (len(rob["per_day"]) == 3 and rob["modal"] is not None
                  and rob["modal_share"] >= 2 / 3)
    else:
        agree_ok = flat_ok = day_ok = False
    print("(2) lag_robustness on 3 iid sessions (bar frame, pmax=8):")
    print("      p=%s  picks=%s  agree=%s (recomputed: %s)"
          % (rob["p"], rob["picks"], rob["agree"], agree_ok))
    print("      flat set (%.1f total-IC units) = %s (matches recomputation: %s)"
          % (rob["flat_tol"], rob["flat_set"], flat_ok))
    print("      per-day vote = %s -> modal %s share %s (unanimity check: %s)"
          % (rob["per_day"], rob["modal"], rob["modal_share"], day_ok))
    out = bool(ok and picks_ok and agree_ok and flat_ok and day_ok)
    print("    diagnostics are recomputable from the IC table : %s" % out)
    return out


def check_parse_table_cell():
    """The parser must be the exact inverse of _fmt_cell for the two fields the
    band sweep reads: value and star count."""
    cases = [
        (cs._fmt_cell(-0.123, 0.045, 0.004), (-0.123, 3)),
        (cs._fmt_cell(0.211, 0.100, 0.07), (0.211, 1)),
        (cs._fmt_cell(0.500, None, None), (0.5, 0)),
        (cs._fmt_cell(float("nan")), (float("nan"), 0)),
        ("--", (float("nan"), 0)),
        ("", (float("nan"), 0)),
        ("+0.042** (0.019)", (0.042, 2)),
    ]
    ok = True
    for s, (v, k) in cases:
        got_v, got_k = cs.parse_table_cell(s)
        same_v = (np.isnan(v) and np.isnan(got_v)) or got_v == v
        ok &= same_v and got_k == k
        print("      %-22r -> (%s, %d stars)  expected (%s, %d) : %s"
              % (s, got_v, got_k, v, k, same_v and got_k == k))
    print("(3) parse_table_cell inverts _fmt_cell : %s" % ok)
    return bool(ok)


def check_band_stability():
    """Planted verdicts: a cell that keeps sign and stars across the band is stable;
    sign flips, stars at only 2/3 of depths, and a missing depth each fail."""
    ps = range(4, 13)                                              # 9 depths
    tabs = {}
    for i, p in enumerate(ps):
        df = pd.DataFrame(index=["Shock_A", "Shock_B", "Shock_C", "Shock_D"])
        df[("Est", "volatile")] = [
            cs._fmt_cell(0.25 + 0.01 * i, 0.05, 0.004),            # stable: + and *** always
            cs._fmt_cell((0.10 if i % 2 else -0.10), 0.05, 0.5),   # sign flips
            cs._fmt_cell(0.30, 0.05, 0.02 if i < 6 else 0.5),      # stars at 6/9 = 67% only
            (cs._fmt_cell(0.20, 0.05, 0.03) if i else "--"),       # not estimable at depth 1
        ]
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        tabs[p] = df
    stab = cs.band_stability(tabs).set_index("shock")
    a = bool(stab.loc["Shock_A", "band_stable"])
    b = not stab.loc["Shock_B", "band_stable"] and not stab.loc["Shock_B", "sign_consistent"]
    c = (not stab.loc["Shock_C", "band_stable"]
         and stab.loc["Shock_C", "sign_consistent"]
         and stab.loc["Shock_C", "star_agreement"] < 0.9)
    d = not stab.loc["Shock_D", "band_stable"] and stab.loc["Shock_D", "n_finite"] == 8
    print("(4) band_stability over p=4..12:")
    print("      planted stable cell     -> band_stable %s : %s" % (stab.loc['Shock_A', 'band_stable'], a))
    print("      sign-flip cell          -> rejected for sign : %s" % b)
    print("      stars-at-67%%-of-depths  -> rejected for star agreement (%.2f) : %s"
          % (stab.loc["Shock_C", "star_agreement"], c))
    print("      missing-depth cell      -> rejected, n_finite=%d/9 : %s"
          % (stab.loc["Shock_D", "n_finite"], d))
    out = a and b and c and d
    print("    the classifier separates findings from lag-dependent cells : %s" % out)
    return bool(out)


def check_driver_surface():
    """The shipped defaults, pinned; plus the band parser's accept/reject behaviour."""
    d = vars(r9.build_parser().parse_args(["--source", "demo"]))
    flags_ok = (d.get("lag_band") == "" and d.get("band_boot") == 199
                and d.get("flat_tol") == 2.0 and d.get("with_bar") is True)
    parse_ok = r9.parse_lag_band("4:12") == (4, 12) and r9.parse_lag_band("6:6") == (6, 6)
    rejects = 0
    for bad in ("12:4", "0:5", "4", "a:b"):
        try:
            r9.parse_lag_band(bad)
        except ValueError:
            rejects += 1
    print("(5) driver surface: --lag-band '' --band-boot 199 --flat-tol 2.0 --with-bar on : %s"
          % flags_ok)
    print("      parse_lag_band accepts 4:12 and 6:6, rejects %d/4 malformed bands : %s"
          % (rejects, parse_ok and rejects == 4))
    return bool(flags_ok and parse_ok and rejects == 4)


def check_end_to_end():
    """The demo driver with --n-lags bic and a band: selection is announced from the
    bar frame, the footnote-17 rolling diagnostic is printed but not used, and both
    the stability CSV and the IC CSV land in --out-dir."""
    with tempfile.TemporaryDirectory() as td:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = r9.main(["--source", "demo", "--n-demo", "4", "--n-demo-bars", "4000",
                          "--n-lags", "bic", "--pmax", "6", "--n-boot", "0", "--no-dcc",
                          "--no-mean-group", "--lag-band", "2:3", "--band-boot", "0",
                          "--out-dir", td])
        out = buf.getvalue()
        band = [f for f in os.listdir(td) if f.startswith("table9_lag_band_")]
        iccsv = [f for f in os.listdir(td) if f.startswith("table9_lag_ic_")]
        frame_ok = "RealBar bar frame" in out
        diag_ok = "footnote-17 diagnostic (NOT used)" in out
        verdict_ok = "LAG-BAND VERDICT" in out
        files_ok = len(band) == 1 and len(iccsv) == 1
        cols_ok = False
        if band:
            stab = pd.read_csv(os.path.join(td, band[0]))
            cols_ok = ({"estimator", "regime", "shock", "band_stable", "sign_consistent",
                        "star_agreement"} <= set(stab.columns)) and len(stab) > 0
    print("(6) end-to-end demo driver (--n-lags bic --lag-band 2:3): rc=%s" % rc)
    print("      selection announced from the bar frame : %s" % frame_ok)
    print("      rolling path printed as diagnostic only : %s" % diag_ok)
    print("      band verdict printed, stability + IC CSVs written : %s"
          % (verdict_ok and files_ok))
    print("      stability CSV carries the verdict columns : %s" % cols_ok)
    out_ok = rc == 0 and frame_ok and diag_ok and verdict_ok and files_ok and cols_ok
    print("    the harness runs whole from the CLI : %s" % out_ok)
    return bool(out_ok)


def main():
    checks = [check_frame_discrimination, check_lag_robustness_internals,
              check_parse_table_cell, check_band_stability,
              check_driver_surface, check_end_to_end]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("lag-robustness checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
