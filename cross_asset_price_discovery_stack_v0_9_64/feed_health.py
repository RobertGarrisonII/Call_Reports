#!/usr/bin/env python3
"""feed_health.py -- did the CAPTURE lose data, or did the REPLAY get it wrong?

Every crossed-book diagnostic so far has had to infer this from symptoms. The tape answers it
directly, and the stack was never asking:

  mt_missing_product_messages  the venue's OWN gap report -- expected vs received sequence number,
                               i.e. the multicast packets that never arrived. Missing packets mean
                               missing adds, so the cancels and trades that reference them are
                               orphans and the levels they should have removed rest forever. That is
                               a crossed book caused by DATA LOSS, and no code change can fix it.
  mt_error                     feed decoder errors (bad packet, unparseable field), same implication.
  mt_clear_orders /            venue feed RESETS. A reset inside the session is not an error, but any
  mt_clear_price_levels        replay that ignores one carries the disowned orders to the close.

An empty result for all four is the strong statement: the capture is complete, so a crossed book is
the replay's fault. A non-empty one localizes the damage to a feed and a time.

    python feed_health.py --date 20200309 --product SPY
    python feed_health.py --date 20200309 --product SPY --session 09:30-16:00 --out health.txt
    python feed_health.py --date 20200309 --product ESH0 --product-type futures

Exit codes: 0 = clean;  2 = gaps/errors found;  1 = could not run (e.g. no MayStreet binary).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

_TYPES = ("mt_missing_product_messages", "mt_error", "mt_clear_orders", "mt_clear_price_levels")
_STATUS = "mt_product_status"


def _fetch(date_str, product, ptype, mtype, data_source, tz, clock):
    import mstbook_loader as ml
    try:
        return ml._fetch_messages(date_str, product, ptype, mtype, data_source, tz=tz, clock=clock), ""
    except Exception as exc:                       # a type the venue does not publish is not a failure
        return pd.DataFrame(), str(exc).splitlines()[0][:200]


def _window(df, date_str, session, tz):
    """-> (in_session, total). A reset before 09:30 still matters -- the replay starts from the first
    message of the day, not from the open -- so both counts are reported."""
    if df.empty or not session:
        return df, df
    d = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
    a, b = session.split("-")
    lo = pd.Timestamp(f"{d.date()} {a}", tz=tz)
    hi = pd.Timestamp(f"{d.date()} {b}", tz=tz)
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = pd.to_datetime(idx).tz_localize(tz)
    return df[(idx >= lo) & (idx <= hi)], df


def report(date_str, product, ptype="direct", data_source="apu", tz="America/New_York",
           clock="receipt", session="09:30-16:00") -> tuple:
    """-> (text, n_bad). n_bad counts gap/error rows; resets are reported but are not themselves a
    fault (the question is whether the replay applies them, which it does from v0.9.23)."""
    lines = [f"feed health: {product} ({ptype}) on {date_str}   session={session}   clock={clock}", ""]
    n_bad = 0
    halt_windows = []
    for mt in _TYPES:
        df, err = _fetch(date_str, product, ptype, mt, data_source, tz, clock)
        if err:
            lines.append(f"{mt:32s} fetch failed: {err}")
            continue
        if df.empty:
            lines.append(f"{mt:32s} (none)")
            continue
        insess, allday = _window(df, date_str, session, tz)
        feeds = (allday["f"].astype(str).value_counts().to_dict() if "f" in allday.columns else {})
        lines.append(f"{mt:32s} {len(allday):>8,d} row(s)   {len(insess):>8,d} in session")
        for f, n in sorted(feeds.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {f:38s} {n:>8,d}")
        if mt in ("mt_missing_product_messages", "mt_error"):
            n_bad += len(allday)
            if mt == "mt_missing_product_messages" and {"expectedproductsequencenumber",
                                                        "currentproductsequencenumber"} <= set(allday.columns):
                exp = pd.to_numeric(allday["expectedproductsequencenumber"], errors="coerce")
                cur = pd.to_numeric(allday["currentproductsequencenumber"], errors="coerce")
                lost = (cur - exp).where(lambda s: s > 0)
                if lost.notna().any():
                    lines.append("    messages never delivered: %s (max gap %s in one break)"
                                 % (f"{int(lost.sum()):,d}", f"{int(lost.max()):,d}"))
        else:
            for ts, row in allday.iterrows():
                mark = "IN SESSION" if len(insess) and ts in insess.index else "outside session"
                lines.append(f"    {ts}  {row.get('f', '?')}  [{mark}]")
    # The venue state stream: where the circuit-breaker halt actually is. A crossed book inside one
    # is correct, so the boundary belongs in the health report next to the gap counts.
    try:
        stat, serr = _fetch(date_str, product, ptype, _STATUS, data_source, tz, clock)
        if serr:
            lines.append(f"{_STATUS:32s} fetch failed: {serr}")
        elif stat.empty:
            lines.append(f"{_STATUS:32s} (none)")
        else:
            import market_halts as mh
            h = mh.windows_from_status(stat, tz=tz)
            halt_windows = h["windows"]
            if h["windows"]:
                lines.append(f"{_STATUS:32s} {len(stat):>8,d} row(s)   HALT on this date:")
                for a, b in h["windows"]:
                    lines.append("    %s -> %s  (%.0f s, %d venue(s): %s)"
                                 % (a.strftime("%H:%M:%S"), b.strftime("%H:%M:%S"),
                                    (b - a).total_seconds(), len(h["by_feed"]),
                                    ", ".join(sorted(h["reasons"]))))
                lines.append("    A halted market does not match, so a crossed book HERE is correct.")
                lines.append("    Exclude these snapshots from every estimate: no valid midpoint.")
            else:
                lines.append(f"{_STATUS:32s} {len(stat):>8,d} row(s)   no market-wide halt")
            for f_, a, b in h.get("venue_only", []):
                lines.append("    venue-level status only: %s halted %s-%s (%.0f s). NOT a market-wide"
                             % (f_, a.strftime("%H:%M:%S"), b.strftime("%H:%M:%S"),
                                (b - a).total_seconds()))
                lines.append("    halt -- the other venues kept matching, so crossing here is NOT excused.")
            # The two REGULATORY constraints on quoting, both confounds for this paper. Coverage is
            # reported, not assumed: a control that is silently all-NaN is worse than none, because
            # the regression still runs and the coefficient means nothing.
            try:
                import market_state as msx
                cov = msx.coverage(stat)
                lines.append("    regulatory state: SSR reported on %.0f%% of rows, LULD bands on "
                             "%.0f%%" % (100 * cov.get("ssr", 0.0), 100 * cov.get("luld_upper", 0.0)))
                sv = msx.state_series(stat)["ssr"].dropna()
                if len(sv) and sv.max() > 0:
                    lines.append("    RULE 201 SHORT SALE RESTRICTION was in effect on this date --")
                    lines.append("    short sales cannot execute or display at or below the NBB, so")
                    lines.append("    sell-side flow is mechanically constrained. Control for it before")
                    lines.append("    reading signed order flow (Tables 5 and 7) as tandem trading.")
                elif len(sv):
                    lines.append("    Rule 201 SSR: not in effect at any point on this date")
                if cov.get("luld_upper", 0.0) == 0.0:
                    lines.append("    LULD bands are NOT carried by these feeds (they come from the")
                    lines.append("    SIP); the band columns will be all-NaN -- do not use as a control.")
            except Exception as exc:
                lines.append("    regulatory state: could not read (%s)" % type(exc).__name__)
            if "marketsession" in stat.columns:
                ses = sorted(set(stat["marketsession"].astype(str).str.strip()) - {"", "\\N", "nan"})
                if ses:
                    lines.append("    market sessions seen: %s" % ", ".join(ses))
    except Exception as exc:
        lines.append(f"{_STATUS:32s} could not read: {type(exc).__name__}: {exc}")

    lines.append("")
    if n_bad:
        pass
    if n_bad:
        lines += [f"VERDICT: the capture is INCOMPLETE ({n_bad:,d} gap/error record(s)).",
                  "Missing packets mean missing adds, so the removals that reference them are orphans",
                  "and the levels they should have taken out rest for the remainder of the session.",
                  "A crossed book on this date is DATA loss, not a replay fault -- re-request the day",
                  "from the vendor or drop it; do not tune the reconstruction against it."]
    elif halt_windows:
        lines += ["VERDICT: the capture is COMPLETE -- no gaps, no decoder errors.",
                  "This date HALTED, so the book is legitimately crossed for %d s of it. Judge the"
                  % int(sum((b - a).total_seconds() for a, b in halt_windows)),
                  "replay on the crossed rate OUTSIDE the halt (qc_frames reports it as crossed_open);",
                  "crossing outside it is the replay's fault and is fixable in code."]
    else:
        lines += ["VERDICT: the capture is COMPLETE -- no gaps, no decoder errors.",
                  "A crossed book on this date is therefore the REPLAY's fault and is fixable in code."]
    return "\n".join(lines), n_bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Capture-completeness check for one product-day")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--product", default="SPY")
    ap.add_argument("--product-type", default="direct", choices=["direct", "futures"])
    ap.add_argument("--data-source", default="apu")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--clock", default="receipt", choices=["receipt", "exchange"])
    ap.add_argument("--session", default="09:30-16:00")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    text, n_bad = report(a.date.replace("-", ""), a.product, a.product_type, a.data_source,
                         a.tz, a.clock, a.session)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return 2 if n_bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"feed_health: FAILED to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
