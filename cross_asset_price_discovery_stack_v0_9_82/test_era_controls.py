#!/usr/bin/env python3
"""Gate: v0.9.82's era controls -- the within-pair permutation and the
market-era flags with the straddle check.

Day clustering fixes inference, not confounding: if volatile days cluster in
late-sample years while market structure drifts, the free-permutation regime
test partially measures ERA. The fixes pinned here:

  1. within-pair permutation -- on a PLANTED era confound (three eras with level
     shifts, volatile days concentrated in the hot era, no true within-pair
     effect) the free test fires falsely while the within-pair test stays quiet;
     with a planted TRUE within-pair effect the within-pair test fires. That
     double dissociation is the reason the mode exists.
  2. era flags -- post_<event> booleans switch on the documented dates (and not
     before), the retail-surge window flags only its window, and the columns
     land in the information-shares per-day table
  3. straddle check -- the BAKED 2026 pairs (post-tick-regime volatile day,
     pre-tick-regime control) are flagged; era-interior pairs are not; the
     STAGE 0 validator prints the verdict
  4. wiring -- regime_test_within_pair reaches summary.json's picks and pairs
     with missing/NaN days degrade to fewer pairs, never an exception
"""
import sys

import numpy as np
import pandas as pd

import market_eras as me
import price_discovery_shares as pds


def _confounded_per_day(true_effect=0.0, seed=0):
    """The REAL sample's exposure, planted. When every pair is era-internal and
    balanced, group means are era-matched by construction and the free test is
    merely underpowered, not biased. The bias enters through the UNPAIRED
    volatile days -- the designed sample carries 6 of them (2016 x2, 2020 x2,
    2022, 2023: no control cleared the vol screen), so the volatile group holds
    era mass with no benchmark counterweight. Planted here: three eras with
    level shifts, two balanced pairs per era, plus four unpaired volatile days
    in the hot era. Free test fires on the composition imbalance; the
    within-pair test only sees the balanced pairs. `true_effect` adds a genuine
    within-pair difference on top."""
    rng = np.random.default_rng(seed)
    eras = {"e1": ("2018-01-05", 0.20), "e2": ("2021-01-05", 0.35), "e3": ("2025-01-03", 0.60)}
    rows = []
    pairs = []
    for name, (start, level) in eras.items():
        for v in pd.bdate_range(start, periods=16, freq="7D"):
            b = v - pd.Timedelta(days=364)
            vs, bs = str(v.date()), str(b.date())
            rows.append({"date": vs, "regime": "volatile",
                         "CS_ES": level + true_effect + 0.02 * rng.standard_normal()})
            rows.append({"date": bs, "regime": "benchmark",
                         "CS_ES": level + 0.02 * rng.standard_normal()})
            pairs.append((vs, bs))
    for v in pd.bdate_range("2025-06-02", periods=48, freq="3D"):  # unpaired, hot era only
        rows.append({"date": str(v.date()), "regime": "volatile",
                     "CS_ES": 0.60 + true_effect + 0.02 * rng.standard_normal()})
    per_day = pd.DataFrame(rows).set_index("date")
    return per_day, pairs


def check_double_dissociation():
    per_day, pairs = _confounded_per_day(true_effect=0.0, seed=0)
    free = pds.compare_regimes(per_day, metric="CS_ES", n_perm=4000)
    wp = pds.compare_regimes(per_day, metric="CS_ES", n_perm=4000, pairs=pairs)
    ok_fake = free["p_perm"] < 0.05                       # era confound fires the free test
    ok_quiet = wp["p_perm"] > 0.10                        # within-pair sees through it
    per_day2, pairs2 = _confounded_per_day(true_effect=0.10, seed=1)
    wp2 = pds.compare_regimes(per_day2, metric="CS_ES", n_perm=4000, pairs=pairs2)
    ok_true = wp2["p_perm"] < 0.05 and wp2["mean_pair_diff"] > 0.05
    print("(1) planted era confound (no true effect): free p=%.4f fires falsely (%s), "
          "within-pair p=%.3f stays quiet (%s);"
          % (free["p_perm"], ok_fake, wp["p_perm"], ok_quiet))
    print("    planted TRUE within-pair effect 0.10: within-pair p=%.4f, mean diff %+.3f "
          "-> detected (%s)" % (wp2["p_perm"], wp2["mean_pair_diff"], ok_true))
    return bool(ok_fake and ok_quiet and ok_true)


def check_era_flags():
    dates = ["2019-05-03", "2019-05-06", "2020-06-15", "2019-06-15",
             "2025-10-10", "2026-01-20", "2024-05-28"]
    f = me.era_flags(dates)
    ok = (not f.loc[pd.Timestamp("2019-05-03"), "post_mes_launch"]
          and f.loc[pd.Timestamp("2019-05-06"), "post_mes_launch"]
          and f.loc[pd.Timestamp("2020-06-15"), "retail_surge"]
          and not f.loc[pd.Timestamp("2019-06-15"), "retail_surge"]
          and not f.loc[pd.Timestamp("2025-10-10"), "post_tick_regime_2025"]
          and f.loc[pd.Timestamp("2026-01-20"), "post_tick_regime_2025"]
          and f.loc[pd.Timestamp("2024-05-28"), "post_t_plus_1"])
    print("(2) era flags switch exactly on their dates; windows flag only their window : %s" % ok)
    return bool(ok)


def check_straddle():
    import re
    src = open("run_paper_replication.sh").read()
    vol = re.search(r'^VOLATILE="([^"]+)"', src, re.M).group(1).split(",")
    base = re.search(r'^BASELINE="([^"]+)"', src, re.M).group(1).split(",")
    strad = me.pair_straddle(vol, base)
    flagged = {(r["volatile"], r["baseline"]) for _i, r in strad.iterrows()}
    ok_2026 = {("2026-01-20", "2025-01-21"), ("2026-06-05", "2025-06-06")} <= flagged
    ok_only = len(flagged) == 2                           # no era-interior pair flagged
    import subprocess as sp
    res = sp.run([sys.executable, "validate_sample.py", "--volatile", ",".join(vol),
                  "--baseline", ",".join(base)], capture_output=True, text=True, timeout=120)
    ok_wire = "market-era straddle check" in res.stdout and "tick_regime_2025" in res.stdout
    print("(3) baked 2026 pairs straddle the tick break (%s); nothing else flagged (%s); "
          "STAGE 0 validator prints the verdict (%s)" % (ok_2026, ok_only, ok_wire))
    return bool(ok_2026 and ok_only and ok_wire)


def check_wiring():
    import run_analysis as ra
    s = ra._scalar_summary({"information_shares": {
        "regime_test_within_pair": {"p_perm": 0.031}}})
    ok_pick = s.get("regime_p_CS_within_pair") == 0.031
    per_day, pairs = _confounded_per_day(seed=2)
    degraded = pairs + [("1999-01-01", "1998-01-01")]      # absent days must not raise
    wp = pds.compare_regimes(per_day, metric="CS_ES", n_perm=500, pairs=degraded)
    ok_deg = wp["n_pairs"] == len(pairs)
    try:
        pds.compare_regimes(per_day, metric="CS_ES", pairs=pairs, weights="CS_ES")
        ok_excl = False
    except ValueError:
        ok_excl = True
    print("(4) summary pick (%s); absent pair members degrade to fewer pairs (%s); "
          "pairs+weights rejected (%s)" % (ok_pick, ok_deg, ok_excl))
    return bool(ok_pick and ok_deg and ok_excl)


def main():
    checks = [check_double_dissociation, check_era_flags, check_straddle, check_wiring]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("era-controls checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
