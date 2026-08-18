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


def _baked_pairs():
    import re
    src = open("run_paper_replication.sh").read()
    vol = re.search(r'^VOLATILE="([^"]+)"', src, re.M).group(1).split(",")
    base = re.search(r'^BASELINE="([^"]+)"', src, re.M).group(1).split(",")
    return vol, base


def check_ex_straddle_row():
    """(v0.9.87) The ex-straddle within-pair row: the two baked 2026 pairs straddle the
    2025-11-03 tick break, so run_analysis must emit a second within-pair test with them
    dropped. Planted DGP: ONLY the straddling pairs carry a (spurious) +0.40 pair
    difference; the clean pairs carry an exactly sign-balanced +-0.05 (23 pairs, odd
    count, so the ex-straddle sign-flip null is EXACT: every flip's |mean| >= the
    observed |mean|, p = 1). The all-pairs p must be pulled down by the straddle pairs
    (~0.25 analytically) and the ex-straddle p must be materially larger -- the row
    removes exactly the spurious contribution and nothing else."""
    import run_analysis as ra
    vol, base = _baked_pairs()
    prs = list(zip(vol, base))                             # positional, driver convention
    strad = me.pair_straddle([v for v, _ in prs], [b for _, b in prs])
    bad = {(r["volatile"], r["baseline"]) for _i, r in strad.iterrows()}
    rows, j = [], 0
    for v, b in prs:
        if (v, b) in bad:
            diff = 0.40                                    # spurious, straddle pairs only
        else:
            diff = 0.05 if j % 2 == 0 else -0.05
            j += 1
        rows.append({"date": b, "regime": "benchmark", "CS_ES": 0.50})
        rows.append({"date": v, "regime": "volatile", "CS_ES": 0.50 + diff})
    per_day = pd.DataFrame(rows).set_index("date")
    out = ra._within_pair_tests(per_day, vol, base)
    wp = out.get("regime_test_within_pair")
    ex = out.get("regime_test_within_pair_ex_straddle")
    ok_row = isinstance(ex, dict) and ex.get("mode") == "within_pair"
    ok_n = ok_row and ex["n_pairs"] == wp["n_pairs"] - 2 and ex["n_pairs_excluded"] == 2
    ok_which = ok_row and ex["excluded_pairs"] == ["2026-01-20/2025-01-21",
                                                   "2026-06-05/2025-06-06"]
    ok_p = ok_row and ex["p_perm"] > 2.5 * wp["p_perm"] and ex["p_perm"] > 0.9 \
        and wp["p_perm"] < 0.35
    ok_mean = ok_row and abs(ex["mean_pair_diff"]) < 0.005 < wp["mean_pair_diff"]
    print("(5) ex-straddle row: %d -> %d pairs, exclusions %s (right pairs: %s);"
          % (wp["n_pairs"], ex["n_pairs"] if ok_row else -1,
             ex.get("n_pairs_excluded") if ok_row else "?", ok_which))
    print("    spurious-in-straddle-only DGP: p all=%.3f ex=%.3f (materially larger: %s), "
          "mean diff %.4f -> %.4f (%s)"
          % (wp["p_perm"], ex["p_perm"] if ok_row else float("nan"), ok_p,
             wp["mean_pair_diff"], ex["mean_pair_diff"] if ok_row else float("nan"), ok_mean))
    return bool(ok_row and ok_n and ok_which and ok_p and ok_mean)


def check_within_pair_is_twin():
    """(v0.9.87) The within-pair test must cover the IS metric, not only CS. The 20260817
    run's only significant regime result was the free-permutation IS test; era-robust
    within-pair inference existed only for CS, so the IS result had no era-robust
    counterpart on the record. Planted DGP: per-day table where IS_mid_ES carries a +0.30
    pair difference and CS_ES carries none -- the IS twin must reject, the CS row must
    not, and the suffixed keys (incl. the ex-straddle twin) must all be present."""
    import numpy as np
    import pandas as pd
    import run_analysis as ra
    rng = np.random.default_rng(11)
    vol = [str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * i)).split()[0] for i in range(12)]
    ben = [str(pd.Timestamp("2023-01-02") + pd.Timedelta(days=7 * i)).split()[0] for i in range(12)]
    rows = []
    for i, (v, b) in enumerate(zip(vol, ben)):
        base_cs = 0.40 + rng.normal(0, 0.02)
        rows.append({"date": v, "regime": "volatile",
                     "CS_ES": base_cs + rng.normal(0, 0.01),
                     "IS_mid_ES": 0.45 + 0.30 + rng.normal(0, 0.02)})
        rows.append({"date": b, "regime": "benchmark",
                     "CS_ES": base_cs + rng.normal(0, 0.01),
                     "IS_mid_ES": 0.45 + rng.normal(0, 0.02)})
    per_day = pd.DataFrame(rows).set_index("date")
    out = {}
    out.update(ra._within_pair_tests(per_day, vol, ben, metric="CS_ES"))
    out.update(ra._within_pair_tests(per_day, vol, ben, metric="IS_mid_ES", key_suffix="_IS"))
    keys_ok = all(k in out for k in ("regime_test_within_pair", "regime_test_within_pair_IS",
                                     "regime_test_within_pair_ex_straddle",
                                     "regime_test_within_pair_ex_straddle_IS"))
    p_is = out.get("regime_test_within_pair_IS", {}).get("p_perm", 1.0)
    p_cs = out.get("regime_test_within_pair", {}).get("p_perm", 0.0)
    d_is = out.get("regime_test_within_pair_IS", {}).get("mean_pair_diff", 0.0)
    a = keys_ok
    b_ = p_is < 0.05 and abs(d_is - 0.30) < 0.05
    c = p_cs > 0.1
    ok = a and b_ and c
    print("(8) within-pair IS twin: suffixed keys present (%s); planted IS pair diff %.3f "
          "detected p=%.4f (%s); null CS row does not reject p=%.3f (%s) : %s"
          % (a, d_is, p_is, b_, p_cs, c, ok))
    return bool(ok)


def check_es_stale_frac():
    """(v0.9.87) The per-day ES staleness covariate: a planted frame with the ES top
    frozen on a known block of open rows must yield exactly frozen/open to 1e-6, with
    post-13:00 rows of a half day and MWCB-halt rows excluded from the DENOMINATOR --
    otherwise a closed market reads as a stale data feed on exactly the stressed and
    half-day sessions. Also checks the wiring: tier1_per_day carries the column and it
    matches the direct computation."""
    import flow_correlation as fc
    import market_halts as mh

    def _frozen_frame(idx, blocks):
        n = len(idx)
        bid = 5000.0 + 0.25 * np.arange(n)                # moves every row unless frozen
        ask = bid + 0.25
        frz = np.zeros(n, bool)
        for a, b in blocks:
            frz[a:b] = True
        for t in range(1, n):
            if frz[t]:
                bid[t] = bid[t - 1]
                ask[t] = ask[t - 1]
        return pd.DataFrame({"ES_bidprice_1": bid, "ES_askprice_1": ask}, index=idx), frz

    # half day (2024-11-29, day after Thanksgiving): 1000 frozen open rows + 600 frozen
    # rows entirely after the 13:00 close that must count NOWHERE
    idx = pd.date_range("2024-11-29 09:30", "2024-11-29 13:59:59", freq="s",
                        tz="America/New_York")
    n_open = int((pd.Timestamp("2024-11-29 13:00", tz="America/New_York")
                  - idx[0]).total_seconds())              # rows strictly before the close
    df, frz = _frozen_frame(idx, [(1000, 2000), (n_open + 100, n_open + 700)])
    got = fc.es_stale_frac(df, date="2024-11-29")
    exp = 1000.0 / n_open
    ok_half = abs(got - exp) < 1e-6
    ok_denom = abs(got - frz.sum() / len(idx)) > 1e-3     # naive all-rows frac differs
    # MWCB halt day (2020-03-09, halt 09:34:13-09:49:13): freeze the whole halt window
    # (a halted top IS frozen) + 500 open rows; only the open 500 may count
    idx2 = pd.date_range("2020-03-09 09:30", "2020-03-09 11:29:59", freq="s",
                         tz="America/New_York")
    hm = mh.halt_mask(idx2)
    h0, h1 = int(np.argmax(hm)), int(len(hm) - np.argmax(hm[::-1]))
    df2, _ = _frozen_frame(idx2, [(h0, h1), (3000, 3500)])
    got2 = fc.es_stale_frac(df2)
    exp2 = 500.0 / float((~hm).sum())
    ok_halt = abs(got2 - exp2) < 1e-6 and int(hm.sum()) == 901
    # wiring: the column lands in tier1_per_day and matches the direct call
    import test_hy_correlation as th
    sess = []
    for i in range(2):
        d, _ = th._frame(n=4000, rho=0.6, refresh=0.7, seed=50 + i)
        sess.append((f"2024-08-0{5 + i}", "benchmark", d))
    per_day = fc.tier1_per_day(sess)
    ok_wire = ("es_stale_frac" in per_day.columns and len(per_day) == 2
               and all(abs(per_day.es_stale_frac.iloc[k]
                           - fc.es_stale_frac(sess[k][2], date=sess[k][0])) < 1e-12
                       for k in range(2)))
    print("(6) es_stale_frac: half day %.6f vs planted %.6f (match %s; post-close out of "
          "the denominator %s);" % (got, exp, ok_half, ok_denom))
    print("    halt day %.6f vs planted %.6f over %d open rows (%s); tier1_per_day "
          "carries the matching column (%s)"
          % (got2, exp2, int((~hm).sum()), ok_halt, ok_wire))
    return bool(ok_half and ok_denom and ok_halt and ok_wire)


def main():
    checks = [check_double_dissociation, check_era_flags, check_straddle, check_wiring,
              check_within_pair_is_twin,
              check_ex_straddle_row, check_es_stale_frac]
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
