"""test_release_calendar.py -- guards for the NYSE trading calendar and the rule-generated 10:00
release cluster.

The failure this exists to catch is SILENT: a generated release date that lands on a non-trading day,
or whose onset falls outside the 09:30-16:00 extraction window, does not raise -- it produces an empty
pre-window, returns transmission=NaN, and the event vanishes from the surface. A backbone run would
burn full extraction cost on hundreds of sessions and quietly harvest nothing. So the checks below are
not "does the generator emit dates" but "is every emitted date a real session with a usable onset",
plus the counter-cases proving the check has teeth (08:30 and 09:30 onsets MUST come back NaN).

Holiday rules are pinned to known answers rather than recomputed, since a wrong Good Friday or a wrong
Saturday-observation rule shifts an ISM 'first business day' by a day without any symptom.
"""
import datetime as dt
import sys
import warnings

import numpy as np
import pandas as pd

import market_shocks as mk
import onset_response as onr

NY = "America/New_York"
LO, HI = 2020, 2026


def rth_frame(date_str, T=23401):
    """A session frame exactly as extraction produces it: 09:30-16:00 on a 1s grid."""
    idx = pd.date_range(f"{date_str} 09:30", periods=T, freq="s", tz=NY)
    rng = np.random.default_rng(7)
    ms = np.exp(np.r_[np.log(500.), np.log(500.) + np.cumsum(rng.normal(0, 2e-5, T - 1))])
    me = np.exp(np.r_[np.log(5000.), np.log(5000.) + np.cumsum(rng.normal(0, 2e-5, T - 1))])
    cols = {}
    for a, mid, tk in (("SPY", ms, 0.01), ("ES", me, 0.25)):
        for i in range(1, 7):
            cols[f"{a}_bidprice_{i}"] = mid - (i - .5) * tk
            cols[f"{a}_askprice_{i}"] = mid + (i - .5) * tk
            cols[f"{a}_bidquantity_{i}"] = 100.
            cols[f"{a}_askquantity_{i}"] = 100.
    return pd.DataFrame(cols, index=idx)


def main():
    warnings.simplefilter("ignore")
    ok = True

    # (A) Easter / Good Friday -- the only NYSE holiday with no fixed date and no federal counterpart
    easter_known = {2020: "2020-04-12", 2021: "2021-04-04", 2024: "2024-03-31", 2025: "2025-04-20"}
    a_ok = all(mk._easter(y).isoformat() == s for y, s in easter_known.items())
    gf_ok = (mk._easter(2024) - dt.timedelta(days=2)).isoformat() == "2024-03-29"
    ok &= a_ok and gf_ok
    print("(A) Easter known answers=%s ; Good Friday 2024 == 2024-03-29 -> %s" % (a_ok, gf_ok))

    # (B) observation rules, where an off-by-one silently shifts every 'first business day'
    cases = [
        ("2020-07-03", False, "Jul-4 2020 falls Sat -> NYSE closes Fri Jul 3"),
        ("2021-12-24", False, "Dec-25 2021 falls Sat -> NYSE closes Fri Dec 24"),
        ("2021-12-31", True,  "Jan-1 2022 falls Sat -> NYSE keeps the year's last session"),
        ("2021-06-18", True,  "Juneteenth not observed by NYSE until 2022"),
        ("2022-06-20", False, "Juneteenth 2022 falls Sun -> observed Mon Jun 20"),
        ("2025-01-09", False, "Carter national day of mourning (ad-hoc closure)"),
        ("2024-03-29", False, "Good Friday 2024"),
    ]
    b_ok = True
    for ds, want_open, why in cases:
        got = mk.is_trading_day(pd.Timestamp(ds).date())
        if got != want_open:
            b_ok = False
            print("    MISMATCH %s: is_trading_day=%s want=%s (%s)" % (ds, got, want_open, why))
    ok &= b_ok
    print("(B) holiday observation rules (%d cases) -> %s" % (len(cases), b_ok))

    # (C) THE core invariant: every generated date must be a real trading session
    gen = {c: mk.rule_release_days(c, LO, HI) for c in mk.RULE_RELEASE_CATEGORIES}
    bad = [(c, d.isoformat()) for c, ds in gen.items() for d in ds if not mk.is_trading_day(d)]
    counts_ok = all(len(ds) == 12 * (HI - LO + 1) for ds in gen.values())
    c_ok = not bad and counts_ok
    ok &= c_ok
    print("(C) all generated dates are trading days: %d non-sessions | 12/yr per series=%s -> %s"
          % (len(bad), counts_ok, c_ok))

    # (D) business-day rule spot checks (1st / 3rd trading day, holiday- and weekend-shifted)
    spot = [("ISM_MFG", 2024, 1, "2024-01-02"),   # Jan 1 is a holiday
            ("ISM_MFG", 2024, 9, "2024-09-03"),   # Sep 2 is Labor Day
            ("ISM_MFG", 2024, 6, "2024-06-03"),   # Jun 1 is a Saturday
            ("ISM_SVC", 2021, 4, "2021-04-06"),   # Apr 2 is Good Friday -> 3rd bday slips to Apr 6
            ("ISM_SVC", 2024, 1, "2024-01-04")]
    d_ok = True
    for cat, y, m, want in spot:
        rule, _ = mk._RULE_RELEASES[cat]
        got = mk._rule_release_date(rule, y, m).isoformat()
        if got != want:
            d_ok = False
            print("    MISMATCH %s %d-%02d: got %s want %s" % (cat, y, m, got, want))
    ok &= d_ok
    print("(D) business-day rule spot checks (%d) -> %s" % (len(spot), d_ok))

    # (E) END-TO-END: a 10:00 onset must yield usable windows on an RTH frame, and the excluded
    #     release times MUST come back NaN -- the counter-case that proves the check has teeth.
    df = rth_frame("2024-05-01")
    r10 = onr.event_onset_estimate(df, pd.Timestamp("2024-05-01 10:00", tz=NY), pre_steps=600, post_steps=600)
    ten_ok = r10["n_pre"] >= 600 and r10["n_post"] >= 600 and np.isfinite(r10["transmission"])
    dropped = {}
    for lbl, t in (("08:30 (CPI/NFP block)", "08:30"), ("09:30 (OPEX, frame edge)", "09:30")):
        rr = onr.event_onset_estimate(df, pd.Timestamp(f"2024-05-01 {t}", tz=NY), pre_steps=600, post_steps=600)
        dropped[lbl] = (rr["n_pre"], np.isfinite(rr["transmission"]))
    teeth = all(n == 0 and not fin for n, fin in dropped.values())
    e_ok = ten_ok and teeth
    ok &= e_ok
    print("(E) 10:00 onset usable (n_pre=%d n_post=%d)=%s ; excluded times return NaN=%s -> %s"
          % (r10["n_pre"], r10["n_post"], ten_ok, teeth, e_ok))
    for lbl, (n, fin) in dropped.items():
        print("      %-26s n_pre=%d finite=%s" % (lbl, n, fin))

    # (F) every generated event in the frame carries a 10:00 time (no 08:30 leakage into the backbone)
    f = mk.shocks_frame(include_scheduled=True, scheduled_categories=mk.RULE_RELEASE_CATEGORIES,
                        start=f"{LO}-01-01", end=f"{HI}-12-31")
    gf = f[f["category"].isin(mk.RULE_RELEASE_CATEGORIES)]
    f_ok = len(gf) > 0 and set(gf["event_et"].astype(str)) == {"10:00"}
    ok &= f_ok
    print("(F) all %d generated events stamped 10:00 (no pre-open leakage) -> %s" % (len(gf), f_ok))

    # (G) collision detector finds multi-onset dates (one session per DATE resolves to one arbitrary
    #     onset, so these must be surfaced rather than silently resolved by sort order)
    fc = mk.shocks_frame(include_scheduled=True,
                         scheduled_categories=("FOMC",) + mk.RULE_RELEASE_CATEGORIES,
                         start=f"{LO}-01-01", end=f"{HI}-12-31")
    fc = fc[fc["impact_blind"]]
    col = mk.release_collisions(fc)
    g_ok = isinstance(col, dict) and len(col) > 0
    ok &= g_ok
    print("(G) collision detector: %d multi-onset dates found -> %s" % (len(col), g_ok))
    for d in sorted(col)[:4]:
        print("      %s  %s" % (d, ", ".join("%s@%s" % (c, t) for c, t in col[d])))

    print("\nrelease-calendar checks ->", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
