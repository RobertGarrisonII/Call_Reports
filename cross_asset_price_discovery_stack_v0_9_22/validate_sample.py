#!/usr/bin/env python3
"""validate_sample.py -- check the date universe BEFORE spending hours extracting it.

Extraction is the longest stage in the pipeline (10-25 minutes of vendor I/O per session, hours for
a full sample). A date that cannot produce a session -- a weekend, an exchange holiday, a half day
padded out to a full 23,401-row grid -- is therefore the most expensive kind of typo in the stack,
and it does not announce itself: the loader will happily return a full-length frame of NaNs for a
day the exchange was closed, and the median-price diagnostic prints `nan` in the middle of a
multi-hour log where nobody is watching.

That is not hypothetical. The 2026-07-30 run included 2026-01-19 -- Martin Luther King Jr. Day, NYSE
closed -- and the log shows exactly what it produced: `median SPY=nan ES=6913.75`. SPY was empty
because the equity market was shut; ES had data because CME Globex runs an abbreviated holiday
session. One leg of a cross-asset study, from a day with no cross to measure.

Checks
------
  * parseable, unique within and across groups
  * not a weekend
  * not an exchange holiday (incl. one-off closures: Sandy 2012, the Bush/Ford/Carter funerals)
  * not an early-close half day (13:00 ET) -- a WARNING, not an error: the data are real, but a
    09:30-16:00 grid is then two-thirds empty and the session is not comparable to a full one
  * not in the future
  * volatile <-> baseline pairing: same weekday, ~one year apart (the paper's matching rule)

Exit codes: 0 = clean (warnings allowed);  1 = at least one error;  2 = could not run.

    python validate_sample.py --volatile 2024-12-18,... --baseline 2023-12-20,... --mwcb 2020-03-09,...
    python validate_sample.py --volatile ... --baseline ... --allow-bad-dates   # warn, do not fail

The holiday rules below are the standard NYSE calendar; for a date near an edge case confirm against
the exchange's published calendar rather than trusting this file.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday)


class NYSECalendar(AbstractHolidayCalendar):
    """Regular annual NYSE closures. Juneteenth became a holiday in 2022 (first observed 2022-06-20)."""
    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


# Unscheduled / one-off full closures the rule-based calendar cannot know about.
_ONE_OFF = {
    "2001-09-11": "9/11 (market closed 09-11 through 09-14)",
    "2001-09-12": "9/11 closure",
    "2001-09-13": "9/11 closure",
    "2001-09-14": "9/11 closure",
    "2004-06-11": "Reagan state funeral",
    "2007-01-02": "Ford state funeral",
    "2012-10-29": "Hurricane Sandy",
    "2012-10-30": "Hurricane Sandy",
    "2018-12-05": "Bush state funeral",
    "2025-01-09": "Carter state funeral",
}


def _early_close(d: pd.Timestamp) -> str:
    """NYSE 13:00 ET half days. Returns a reason, or '' for a full session."""
    if d.month == 11 and d.weekday() == 4 and 23 <= d.day <= 29:
        return "day after Thanksgiving"
    if d.month == 7 and d.day == 3 and d.weekday() <= 3:      # Fri Jul 3 is the observed holiday
        return "July 3"
    if d.month == 12 and d.day == 24 and d.weekday() <= 3:    # Fri Dec 24 is the observed holiday
        return "Christmas Eve"
    return ""


def check_dates(groups: dict, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """``groups`` = {'volatile': [...], 'baseline': [...], 'mwcb': [...]}.
    -> one row per date with 'level' in {'ok', 'WARN', 'ERROR'} and a reason."""
    today = today or pd.Timestamp.today().normalize()
    all_dates = [d for lst in groups.values() for d in lst]
    span = [pd.Timestamp(d) for d in all_dates if d]
    lo = min(span) - pd.Timedelta(days=5) if span else pd.Timestamp("2000-01-01")
    hi = max(span) + pd.Timedelta(days=5) if span else pd.Timestamp("2030-01-01")
    hol = {}                                   # date -> holiday name, so the message can say WHICH one
    for rule in NYSECalendar.rules:
        for dt in AbstractHolidayCalendar(rules=[rule]).holidays(lo, hi):
            hol[dt.date()] = rule.name
    seen: dict = {}
    rows = []
    for grp, lst in groups.items():
        for s in lst:
            s = str(s).strip()
            if not s:
                continue
            level, reason = "ok", ""
            try:
                d = pd.Timestamp(s)
            except Exception:
                rows.append({"group": grp, "date": s, "dow": "?", "level": "ERROR",
                             "reason": "unparseable date"})
                continue
            if s in seen:
                level, reason = "ERROR", f"duplicate (already in '{seen[s]}')"
            elif d.weekday() >= 5:
                level, reason = "ERROR", f"{d.strftime('%A')} -- market closed"
            elif s in _ONE_OFF:
                level, reason = "ERROR", f"market closed: {_ONE_OFF[s]}"
            elif d.date() in hol:
                level, reason = "ERROR", ("market HOLIDAY (%s) -- the equity leg is empty; CME "
                                          "Globex still runs an abbreviated session, so the futures "
                                          "leg looks fine and the day extracts to a one-legged frame"
                                          % hol[d.date()])
            elif d > today:
                level, reason = "ERROR", "in the future"
            elif _early_close(d):
                level, reason = "WARN", (f"{_early_close(d)} -- 13:00 ET close, so a 09:30-16:00 grid "
                                         f"is ~65% empty and the session is not comparable to a full day")
            seen.setdefault(s, grp)
            rows.append({"group": grp, "date": s, "dow": d.strftime("%a"), "level": level,
                         "reason": reason})
    return pd.DataFrame(rows)


def check_pairing(volatile, baseline, lo_days: int = 330, hi_days: int = 400) -> pd.DataFrame:
    """The paper matches each volatile day to the same weekday roughly one year earlier. Positional
    pairing of the two lists; a mismatch is a WARNING, since the matching rule is a convention."""
    rows = []
    for i, (v, b) in enumerate(zip(volatile, baseline)):
        try:
            dv, db = pd.Timestamp(v), pd.Timestamp(b)
        except Exception:
            continue
        gap = (dv - db).days
        dow_ok = dv.weekday() == db.weekday()
        gap_ok = lo_days <= gap <= hi_days
        why = []
        if not dow_ok:
            why.append(f"weekday {dv.strftime('%a')} vs {db.strftime('%a')}")
        if not gap_ok:
            why.append(f"gap {gap}d outside [{lo_days},{hi_days}]")
        rows.append({"#": i + 1, "volatile": v, "dow_v": dv.strftime("%a"), "baseline": b,
                     "dow_b": db.strftime("%a"), "gap_days": gap,
                     "level": "ok" if not why else "WARN", "reason": "; ".join(why)})
    if len(volatile) != len(baseline):
        rows.append({"#": 0, "volatile": f"[{len(volatile)} dates]", "dow_v": "",
                     "baseline": f"[{len(baseline)} dates]", "dow_b": "", "gap_days": 0,
                     "level": "WARN", "reason": "the two lists are different lengths, so the "
                                                "positional pairing is not well defined"})
    return pd.DataFrame(rows)


def _split(s: str) -> list:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a date universe before extracting it")
    ap.add_argument("--volatile", default="")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--mwcb", default="")
    ap.add_argument("--allow-bad-dates", action="store_true",
                    help="report problems but exit 0 (for a deliberate one-off run)")
    ap.add_argument("--no-pairing", action="store_true", help="skip the volatile<->baseline check")
    ap.add_argument("--out", default="", help="also write the report here")
    a = ap.parse_args(argv)

    groups = {"volatile": _split(a.volatile), "baseline": _split(a.baseline), "mwcb": _split(a.mwcb)}
    if not any(groups.values()):
        print("validate_sample: nothing to check (pass --volatile / --baseline / --mwcb)", file=sys.stderr)
        return 2
    tbl = check_dates(groups)
    lines = ["date universe: %d volatile + %d baseline + %d MWCB = %d session(s)"
             % (len(groups["volatile"]), len(groups["baseline"]), len(groups["mwcb"]),
                sum(len(v) for v in groups.values())), ""]
    bad = tbl[tbl["level"] != "ok"]
    if len(bad):
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 120)
        lines += [bad.to_string(index=False), ""]
    else:
        lines += ["every date is a full trading session", ""]

    if not a.no_pairing and groups["volatile"] and groups["baseline"]:
        pr = check_pairing(groups["volatile"], groups["baseline"])
        off = pr[pr["level"] != "ok"]
        if len(off):
            lines += ["volatile <-> baseline matching (same weekday, ~1 year prior):",
                      off.to_string(index=False),
                      "", "Every other pair satisfies the rule, so an exception needs a sentence in "
                      "the appendix saying why -- otherwise it reads as an error.", ""]
            has_pair_warn = True
        else:
            lines += ["volatile <-> baseline matching: all pairs same weekday, ~1 year apart", ""]
            has_pair_warn = False
    else:
        has_pair_warn = False

    n_err = int((tbl["level"] == "ERROR").sum())
    n_warn = int((tbl["level"] == "WARN").sum()) + (1 if has_pair_warn else 0)
    if n_err:
        lines += ["%d date(s) CANNOT produce a usable session. Extraction would spend 10-25 minutes "
                  "each to return an empty or one-legged frame." % n_err,
                  "Fix the universe (or pass --allow-bad-dates to proceed anyway)."]
    elif n_warn:
        lines.append("%d warning(s) -- usable, but they need a sentence in the sample appendix." % n_warn)
    else:
        lines.append("sample OK")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return 1 if (n_err and not a.allow_bad_dates) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"validate_sample: FAILED to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
