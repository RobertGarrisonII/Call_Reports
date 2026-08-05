#!/usr/bin/env python3
"""test_ssr_confound.py -- Rule 201 as a confound: measured, diagnosed, and honestly not 'controlled'.

The stack measures SSR everywhere (market_state), flags it in the QC table, and warns at
extraction. What it deliberately does NOT do is put an SSR dummy into a regression, and the reason
is identification, not laziness: the sample contains exactly ONE restricted session, 2020-03-16 --
which is also an MWCB day that opened one second into its circuit-breaker halt on a limit-down gap.
A dummy on that is a relabelled day effect, collinear with everything else that made the day
extreme, and calling it "the SSR control" would claim an identification the sample cannot deliver.

What replaces it, both pinned here:

  * **corner_asym** -- Rule 201 is ONE-SIDED (short sales cannot execute at or below the NBB, so
    aggressive ETF selling is suppressed while buying is untouched), which gives it a signature
    genuine tandem trading has no reason to produce: the sell-side association collapsing while
    the buy side holds. The statistic is the difference of Neutral-referenced local log odds
    ratios, NOT corner mass over its independence expectation -- suppressing a corner also moves
    the marginals, so P/E self-normalises (measured: a 60% cut to the Sell-Sell cell moved P/E by
    ~1% and the local log OR by its full ln 0.4 = -0.916).

  * **the ex-SSR panel** -- STAGE 4 rebuilds the MWCB panel without its restricted session(s), so
    the below-baseline MWCB dependence result can be read against the version that cannot carry
    the SSR channel. Direction matters: SSR pushes log_OR DOWN, the same direction as that result.

And one substantive finding, pinned so it cannot drift: on the paper's PUBLISHED matrices the
asymmetry is +0.005 / +0.011 / +0.066 -- symmetric. The pooled published panels do not carry the
SSR fingerprint; whether the single restricted session does is a question for the per-session
real-data run.

Run: python test_ssr_confound.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import mstbook_loader as ml
import tandem_order_flow as tof

TZ = "America/New_York"

PUBLISHED = {"A baseline": [[21.24, 12.01, 7.73], [4.82, 6.97, 4.88], [7.86, 12.07, 21.50]],
             "B volatile": [[24.55, 11.12, 3.52], [5.73, 9.66, 4.04], [4.65, 13.63, 20.99]],
             "C MWCB":     [[9.96, 20.89, 6.00], [4.94, 18.06, 6.77], [2.86, 17.46, 10.68]]}


def _m(rows):
    return pd.DataFrame(np.array(rows, float), index=tof.DIR3, columns=tof.DIR3)


def check_asymmetry_recovers_a_known_suppression():
    """Cut the Sell-Sell cell by 60% and the statistic must return ln(0.4), exactly."""
    sym = _m([[20, 10, 5], [5, 10, 5], [5, 10, 20]])
    d0 = tof.dependence_summary(sym)
    ssr = sym.copy()
    ssr.loc["Sell", "Sell"] *= 0.4
    d1 = tof.dependence_summary(ssr)
    zero = abs(d0["corner_asym"]) < 1e-9
    exact = abs(d1["corner_asym"] - np.log(0.4)) < 1e-9
    down = d1["log_OR"] < d0["log_OR"]
    ok = zero and exact and down
    print("(1) symmetric table -> corner_asym = %+0.3f (%s); Sell-Sell cut to 40%% -> %+0.3f = "
          "ln(0.4) exactly (%s)" % (d0["corner_asym"], zero, d1["corner_asym"], exact))
    print("    and log_OR falls with it (%.2f -> %.2f) -- SSR pushes the headline statistic DOWN, "
          "the same direction as the below-baseline MWCB result (%s) : %s"
          % (d0["log_OR"], d1["log_OR"], down, ok))
    return ok


def check_published_panels_are_symmetric():
    """The finding: no SSR fingerprint in the paper's pooled matrices."""
    asyms = {k: tof.dependence_summary(_m(M))["corner_asym"] for k, M in PUBLISHED.items()}
    small = all(abs(v) < 0.10 for v in asyms.values())
    ordered = abs(asyms["A baseline"]) < 0.01 and abs(asyms["B volatile"]) < 0.02
    ok = small and ordered
    print("(2) published panels: %s"
          % ", ".join("%s %+0.3f" % (k, v) for k, v in sorted(asyms.items())))
    print("    all under 0.10 in absolute value -- symmetric dependence, no one-sided suppression "
          "at the pooled level. Against a ln(0.4)=-0.92 scale for a 60%% cut, these are zero : %s"
          % ok)
    return ok


def check_session_ssr_helper():
    """Restricted, unrestricted and UNKNOWN must be three distinguishable states."""
    idx = pd.date_range("2020-03-16 09:30:00", periods=50, freq="s", tz=TZ)
    on = pd.DataFrame({"SPY_bidprice_1": 240.0}, index=idx)
    on.attrs["market_state"] = {"ssr_known": 1.0, "ssr_frac_on": 1.0}
    off = pd.DataFrame({"SPY_bidprice_1": 240.0}, index=idx)
    off.attrs["market_state"] = {"ssr_known": 1.0, "ssr_frac_on": 0.0}
    col = pd.DataFrame({"SPY_bidprice_1": 240.0, "SPY_ssr": 1.0}, index=idx)
    unk = pd.DataFrame({"SPY_bidprice_1": 240.0, "SPY_ssr": np.nan}, index=idx)
    r_on, r_off, r_col, r_unk = (ml.session_is_ssr(x) for x in (on, off, col, unk))
    ok = (r_on == (True, True) and r_off == (False, True) and r_col == (True, True)
          and r_unk == (False, False))
    print("(3) attrs-restricted %s, attrs-clear %s, column-restricted %s, source-silent %s -- "
          "unknown is (False, False), never 'unrestricted' : %s"
          % (r_on, r_off, r_col, r_unk, ok))
    return ok


def check_ex_ssr_panel_logic():
    """The STAGE 4 split: the restricted MWCB session leaves the exSSR panel, others stay."""
    rng = np.random.default_rng(1)

    def frame(day, ssr):
        n = 1200
        idx = pd.date_range(f"{day} 09:30:00", periods=n, freq="s", tz=TZ)
        f = pd.DataFrame(index=idx)
        for a, lam in (("SPY", 505.0), ("ES", 112.0)):
            f[f"{a}_trade_buy"] = rng.poisson(lam / 2, n).astype(float)
            f[f"{a}_trade_sell"] = rng.poisson(lam / 2, n).astype(float)
            f[f"{a}_trade_px"] = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
        f.attrs["market_state"] = {"ssr_known": 1.0, "ssr_frac_on": 1.0 if ssr else 0.0}
        return f

    sess = [("2020-03-12", "volatile", frame("2020-03-12", False)),
            ("2020-03-16", "volatile", frame("2020-03-16", True)),
            ("2020-03-18", "volatile", frame("2020-03-18", False))]
    mw = {"2020-03-12", "2020-03-16", "2020-03-18"}
    tagged = [(d, "C MWCB" if d in mw else r, f) for d, r, f in sess]
    ssr_days = [d for d, _r, f in sess if ml.session_is_ssr(f)[0]]
    by = tof.table5_from_sessions(tagged, ml.counts_from_frame)
    t2 = [(d, "C MWCB exSSR", f) for d, r, f in tagged if r == "C MWCB" and d not in ssr_days]
    by.update(tof.table5_from_sessions(t2, ml.counts_from_frame))
    full = by["C MWCB"].get("n_sessions") == 3
    ex = by["C MWCB exSSR"].get("n_sessions") == 2
    named = ssr_days == ["2020-03-16"]
    ok = full and ex and named
    print("(4) MWCB panel pools 3 sessions (%s); the exSSR panel pools 2 (%s); the excluded one is "
          "2020-03-16 by its own recorded state (%s) : %s" % (full, ex, named, ok))
    print("    a DUMMY for it is not identifiable: one restricted session, which is also the day "
          "that opened one second into its circuit-breaker halt -- exclusion is the treatment the "
          "sample can support.")
    return ok


def main():
    checks = [check_asymmetry_recovers_a_known_suppression, check_published_panels_are_symmetric,
              check_session_ssr_helper, check_ex_ssr_panel_logic]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("SSR-confound checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
