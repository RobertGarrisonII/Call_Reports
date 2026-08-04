"""market_halts.py -- market-wide circuit breaker (MWCB) halt windows, and why the book crosses in them.

During a Level 1 halt the exchanges stop matching but they do NOT cancel resting orders, and new
orders keep arriving. A limit order priced through the last trade therefore sits in the book,
unmatched, for the full fifteen minutes. **A correctly reconstructed book is crossed during a halt.**
That is not a replay fault -- it is what a halted market looks like, and it is the phenomenon the
MWCB event study is about.

This matters because the crossed-book invariant is the stack's primary data-integrity gate, and on
exactly the four sessions the paper cares most about, the invariant is *supposed* to be violated for
900 of 23,401 snapshots. The arithmetic on 2020-03-09 is unambiguous:

    crossed fraction              3.88%  of 23,401 snapshots  =  908 snapshots
    a 15-minute Level 1 halt at 1s                            =  900 snapshots
    crossed rate by session sixth  23.1% 0.0% 0.0% 0.0% 0.1% 0.1%
    23.1% of the first sixth (3,900 snapshots)                =  901 snapshots

and the halt that day ran 09:34:13-09:49:13 ET, inside that first sixth. The residual crossing that
three rounds of diagnosis treated as an open defect is the circuit breaker.

Two consequences the paper has to state, not just the code:

  * A halted market has no valid midpoint. Snapshots inside a halt must be excluded from any
    lead-lag, correlation or information-share estimate -- the "prices" on both legs are stale
    quotes that cannot trade against each other. Including them does not add noise, it adds a
    mechanical cross-asset comovement that has nothing to do with price discovery.
  * The reopening auction is the interesting object (see mwcb_event_study.py, which already keys
    off the 2020-03-09 release at 09:49:13), and it needs the halt boundary to be exact.

Times are ET, from the exchange halt notices. Each MWCB Level 1 halt is exactly 15 minutes.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

TZ = "America/New_York"

# date -> [(halt_start, halt_end)] in ET. Level 1 (7% S&P 500 decline) halts are 15 minutes.
# 2020-03-16 triggered at the opening bell, so its halt starts at 09:30:00 exactly.
# Three of the four are now VERIFIED against mt_product_status.haltreason and corrected to it:
# 2020-03-12 was 7 s later than the published notice time and 2020-03-16 1 s later. 2020-03-18 has
# not been checked against the tape yet and is still the notice time -- the derived window wins
# wherever the status stream is available, so this only matters as an offline fallback.
MWCB_HALTS = {
    "2020-03-09": [("09:34:13", "09:49:13")],     # tape: 09:34:13.078 -> 09:49:13.103
    "2020-03-12": [("09:35:44", "09:50:44")],     # tape: 09:35:44     -> 09:50:44  (was 09:35:37)
    "2020-03-16": [("09:30:01", "09:45:01")],     # tape: 09:30:01     -> 09:45:01  (was 09:30:00)
    "2020-03-18": [("12:56:11", "13:11:11")],     # CORROBORATED: the ESH0/ESM0 Velocity Logic pause
                                                  # lands at 12:57:39.7, +88.7s -- in family with the
                                                  # +61.1s and +54.0s on 03-12 and 03-16. Not proof
                                                  # (it is the other leg), but a materially wrong
                                                  # equity time would not bracket it.
}


# ── deriving the halt from the tape, which is where it actually lives ────────────────────────────
# mt_product_status carries `haltreason` per venue. On 2020-03-09 the five xdp_* feeds all publish
# MarketWideCircuitBreakerLevel1 at 09:34:13.0787 ET and clear it at 09:49:13.0787 -- 900.0 s to the
# tenth of a millisecond, matching the table below exactly. So the table is a FALLBACK, not the
# source of truth: reading the tape covers every date rather than four, is per venue, and cannot be
# wrong the way a hand-entered time can.
_HALT_NULL = ("", "nan", "<NA>", "None", "NaN", "null", "NULL", "\\N")

# `haltreason` is NOT a halt flag -- it is a status-reason field, and on CME most of its values are
# routine session bookkeeping. ESZ4 on 2024-12-18 carries eighteen `GroupSchedule` and nineteen
# `MarketEvent` rows whose `tradingevent` is `NoEvent` / `NoCancel` /
# `ChangeOfTradingSessionResetStatistics`; treating them as halts produced spans of 843 s, 5,914 s
# and 43,910 s on days the future never stopped trading.
#
# So halts are WHITELISTED, not blacklisted, and the direction of that choice is deliberate. A false
# halt EXCUSES crossing and can hide a replay fault; a missed halt merely flags a session that turns
# out to be fine, which is visible and investigable. Unrecognized reasons are therefore treated as
# NOT halts and returned in `unknown_reasons` so they can be classified rather than silently assumed.
_HALT_TOKENS = ("circuitbreaker", "suspended", "halt", "regulatory", "limitstate", "limitup",
                "limitdown", "pause", "reasonnotavailable")
_SCHEDULE_TOKENS = ("groupschedule", "marketevent")


def _is_halt_reason(v: str) -> bool:
    t = str(v).strip().lower().replace(" ", "").replace("_", "")
    if not t or t in [x.lower() for x in _HALT_NULL]:
        return False
    return any(tok in t for tok in _HALT_TOKENS)


def windows_from_status(df, tz: str = TZ, min_seconds: float = 1.0, min_venues: int = 2,
                        activity=None) -> dict:
    """Derive halt windows per feed from an ``mt_product_status`` frame (no I/O -- pass the frame).

    A halt OPENS on the first row for a feed whose ``haltreason`` is populated and CLOSES on that
    feed's next row where it is not.

    Two filters, both learned from real days:

    ``min_seconds``  drops instantaneous status rows. It applies to an UNCLOSED span too: on
                     2024-12-18 `memoir_ltse_depth_l3` publishes `RegulatoryConcern` once, at
                     06:28:02, and never clears it, which produced a zero-length "halt" on an
                     ordinary day. The floor is ONE SECOND, not thirty: the CME status flag is set
                     for only 5-10 s even when the market is down for a quarter of an hour (see
                     ``activity``), so a 30 s floor discarded exactly the cross-asset events this
                     paper is about. The reason whitelist, not a duration threshold, is what filters
                     routine churn.

    ``min_venues``   is the one that matters. A MARKET-WIDE halt stops every venue at once -- on
                     2020-03-09 six feeds report within 30 ms of each other. ONE venue reporting a
                     halt reason is a venue-level event, and the consolidated book keeps matching on
                     the other fifteen. Excusing crossing as "halt" on that basis would let a
                     single venue's hour-long regulatory status hide a real replay fault for an
                     hour. Spans that miss quorum are returned under ``venue_only`` rather than
                     discarded, so nothing disappears silently.

    ``activity``     timestamps at which the product actually TRADED (see
                     ``lob_reconstruct.reconstruct_session``, which records them for free from the
                     tape it already fetched). Pass it and the window END is taken from the resumption
                     of trading rather than from the status flag clearing.

                     **This is not a refinement; without it the CME windows are wrong by two orders
                     of magnitude.** On 2020-03-18 ESM0:

                         last trade before the stop   12:57:39.713
                         haltreason SET               12:57:39.716   <- 3 ms later: the ONSET is exact
                         haltreason CLEARED           12:57:45.549   <- 5.83 s: reads as a brief pause
                         first trade after the stop   13:11:17.008   <- 817.3 s of no trading at all

                     Zero contracts for 13.6 minutes, then 1,221 in the first print. The flag is a
                     transient NOTIFICATION, not a duration: it marks the stop to the millisecond and
                     says nothing about the resume. Trusting the clear excluded 6 s of a session
                     where 817 s of the futures leg had no market -- 811 snapshots of a stale book
                     entering every pair estimate as though it were live.

                     The equity feeds do not have this problem: their MWCB spans clear at exactly
                     900.0 s, matching the published halt durations, so the flag there IS the
                     duration. Hence a parameter rather than a global change of rule.

    -> {'windows': [(start, end)] market-wide, 'venue_only': [(feed, start, end)],
        'by_feed': {feed: [(start, end)]}, 'reasons': set, 'flag_windows': [(start, end)],
        'extended': [{'flag_end', 'resume', 'extra_seconds', 'quiet_ratio'}]}

    ``flag_windows`` always holds the unextended spans, so nothing is lost by passing ``activity``."""
    out = {"windows": [], "venue_only": [], "by_feed": {}, "reasons": set(), "unknown_reasons": set(),
           "flag_windows": [], "extended": []}
    if df is None or len(df) == 0 or "haltreason" not in df.columns:
        return out
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = pd.DatetimeIndex(idx).tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    reason = df["haltreason"].astype(str).str.strip()
    halted = reason.map(_is_halt_reason)
    for v in reason.unique():                     # anything neither null, halt, nor known-routine
        t = str(v).strip().lower().replace(" ", "").replace("_", "")
        if t and t not in [x.lower() for x in _HALT_NULL] and not _is_halt_reason(v) \
                and not any(tok in t for tok in _SCHEDULE_TOKENS):
            out["unknown_reasons"].add(str(v).strip())
    feeds = df["f"].astype(str) if "f" in df.columns else pd.Series("?", index=df.index)
    # Quorum cannot exceed the number of venues that publish status at all. CME is a SINGLE venue,
    # so a fixed min_venues=2 would suppress every real futures halt -- including the
    # SuspendedBySurveillance on 2020-03-12 ESH0.
    n_venues = int(feeds.nunique())
    min_venues = max(1, min(int(min_venues), n_venues))

    for feed in sorted(set(feeds)):
        m = (feeds == feed).to_numpy()
        t, h, r = idx[m], halted.to_numpy()[m], reason.to_numpy()[m]
        order = np.argsort(t.values, kind="stable")
        t, h, r = t[order], h[order], r[order]
        spans, start, why = [], None, ""
        for k in range(len(t)):
            if h[k] and start is None:
                start, why = t[k], r[k]
            elif (not h[k]) and start is not None:
                if (t[k] - start).total_seconds() >= min_seconds:
                    spans.append((start, t[k]))
                    out["reasons"].add(why)
                start = None
        if start is not None and (t[-1] - start).total_seconds() >= min_seconds:
            spans.append((start, t[-1]))            # halted with no resume row in the fetch
            out["reasons"].add(why)
        if spans:
            out["by_feed"][feed] = spans

    # Consolidate: a market-wide halt stops every venue, so the union across feeds is the window the
    # book is unmatched over. Overlapping per-feed spans (they differ by microseconds) merge into one.
    flat = sorted((a, b, f) for f, v in out["by_feed"].items() for a, b in v)
    merged = []
    for a, b, f in flat:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged[-1][2] | {f})
        else:
            merged.append((a, b, {f}))
    for a, b, feeds in merged:
        if len(feeds) >= min_venues:
            out["windows"].append((a, b))
        else:
            out["venue_only"].append((sorted(feeds)[0], a, b))
    out["flag_windows"] = list(out["windows"])
    if activity is not None and len(out["windows"]):
        out["windows"] = _extend_to_resume(out["windows"], activity, tz, out["extended"])
    return out


def _extend_to_resume(windows, activity, tz, log_into: list) -> list:
    """Replace each window's end with the first trade after its start (see ``activity`` above).

    ``quiet_ratio`` is the extension divided by the MEDIAN inter-trade gap in the ten minutes before
    the halt. It is the honesty check: extending to the next trade is only sound where trading was
    dense enough that a long silence cannot be ordinary. On 2020-03-18 ESM0 the pre-halt median gap
    is a fraction of a second against an 817 s extension, so the ratio is enormous and the inference
    is safe. On a thin product it would be small, the extension would be indistinguishable from a
    quiet spell, and the number says so rather than hiding it."""
    act = pd.DatetimeIndex(pd.to_datetime(activity))
    act = act.tz_localize(tz) if act.tz is None else act.tz_convert(tz)
    act = act.sort_values()
    if not len(act):
        return list(windows)
    vals = act.values
    out = []
    for a, b in windows:
        nxt = np.searchsorted(vals, np.datetime64(a.tz_convert("UTC").tz_localize(None)), side="right")
        if nxt >= len(vals):                       # halted into the close: nothing resumes it
            out.append((a, b))
            continue
        resume = act[nxt]
        if resume <= b:                            # the flag already covered it (the equity case)
            out.append((a, b))
            continue
        pre = act[(act >= a - pd.Timedelta(minutes=10)) & (act < a)]
        gaps = np.diff(pre.values).astype("timedelta64[ns]").astype(float) / 1e9 if len(pre) > 1 else []
        med = float(np.median(gaps)) if len(gaps) else float("nan")
        extra = (resume - b).total_seconds()
        log_into.append({"flag_end": b, "resume": resume, "extra_seconds": extra,
                         "pre_halt_median_gap_s": med,
                         "quiet_ratio": (extra / med) if med and np.isfinite(med) and med > 0 else float("inf")})
        out.append((a, resume))
    return out


def cross_asset_summary(spy_windows, es_windows, tz: str = TZ) -> dict:
    """Line the two legs' halt windows up against each other. This is a RESULT, not a diagnostic.

    The futures do not keep trading through an equity circuit breaker. They halt too, about a minute
    later, and the two markets reopen together. On 2020-03-18, measured against ESM0's own trade
    tape:

        12:56:11        SPY MWCB Level 1 halt begins
        12:56:11-12:57:39.7   ES trades ON, 3,927 contracts in 88.7 s -- the ONLY price venue
        12:57:39.713    ES last trade
        12:57:39.716    ES haltreason set (3 ms later: the onset is exact)
        12:57:45.549    ES haltreason CLEARS -- but nothing trades
        13:11:11        SPY reopens
        13:11:17.008    ES first trade, 1,221 contracts (+6.0 s)

    So the interesting object is **the 88.7 seconds of solitude**, not a 15-minute divergence: a
    short, sharp window in which the futures are the sole price-discovery venue, bounded at both
    ends. CME halts equity-index futures in coordination with the primary market, as its rules
    require; the ~1 minute is the relay, and the consistent +54 to +89 s across 2020-03-12, 03-16
    and 03-18 is that relay, not a volatility threshold being crossed.

    Getting this right REQUIRES passing ``activity`` to :func:`windows_from_status`. Read off the
    status flag alone, 2020-03-18 looks like a 5.83 s pause inside a 900 s equity halt -- the futures
    apparently trading through almost all of it. The tape says they were down for 817.3 s. The two
    readings support opposite claims about which market leads, so the flag-only version is not a
    rough answer, it is the wrong one.

    Two consequences:

      * For estimation, the union of the two windows is what has to be excluded -- a pair estimate
        needs BOTH legs matching, and during the ES stop neither leg has a tradeable midpoint.
        `_extract_one_session` stores the union in ``df.attrs["halt_windows"]`` for exactly this.
      * For the event study, the ES halt is its own onset nested inside the equity one, and the
        88.7 s before it is the sole-venue window. Treating the equity reopening as the only event
        on these days misses both.

    ``lag_s`` is measured from the START of the containing equity window, so it is directly
    comparable across days with different halt times.

    -> {'overlaps': [{...}], 'es_only': [(a, b)], 'spy_only': [(a, b)], 'n_nested': int}"""
    def _norm(ws):
        out = []
        for a, b in (ws or []):
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            a = a.tz_localize(tz) if a.tz is None else a.tz_convert(tz)
            b = b.tz_localize(tz) if b.tz is None else b.tz_convert(tz)
            out.append((a, b))
        return sorted(out)

    spy, es = _norm(spy_windows), _norm(es_windows)
    rep = {"overlaps": [], "es_only": [], "spy_only": [], "n_nested": 0}
    matched_spy = set()
    for a, b in es:
        hit = next(((x, y) for x, y in spy if x <= a <= y or x <= b <= y or (a <= x and y <= b)), None)
        if hit is None:
            rep["es_only"].append((a, b))
            continue
        matched_spy.add(hit)
        nested = hit[0] <= a and b <= hit[1]
        rep["n_nested"] += int(nested)
        rep["overlaps"].append({"spy": hit, "es": (a, b), "nested": nested,
                                "lag_s": (a - hit[0]).total_seconds(),
                                "es_seconds": (b - a).total_seconds(),
                                "spy_seconds": (hit[1] - hit[0]).total_seconds()})
    rep["spy_only"] = [w for w in spy if w not in matched_spy]
    return rep


def describe_cross_asset(rep: dict) -> str:
    lines = []
    for o in rep["overlaps"]:
        lines.append("ES paused %s-%s (%.2fs) %s the equity halt %s-%s (%.0fs), starting %+.1fs in"
                     % (o["es"][0].strftime("%H:%M:%S.%f")[:-3], o["es"][1].strftime("%H:%M:%S.%f")[:-3],
                        o["es_seconds"], "INSIDE" if o["nested"] else "overlapping",
                        o["spy"][0].strftime("%H:%M:%S"), o["spy"][1].strftime("%H:%M:%S"),
                        o["spy_seconds"], o["lag_s"]))
    for a, b in rep["es_only"]:
        lines.append("ES paused %s-%s (%.2fs) with the equity market OPEN -- a futures-only event"
                     % (a.strftime("%H:%M:%S"), b.strftime("%H:%M:%S"), (b - a).total_seconds()))
    for a, b in rep["spy_only"]:
        lines.append("equity halted %s-%s (%.0fs) with ES matching throughout"
                     % (a.strftime("%H:%M:%S"), b.strftime("%H:%M:%S"), (b - a).total_seconds()))
    if rep["overlaps"]:
        lines.append("Exclude the UNION from any pair estimate: during the ES pause neither leg has "
                     "a tradeable midpoint. The nested pause is its own event onset.")
    return "\n".join(lines) if lines else "no halt on either leg"


def halt_windows(date, tz: str = TZ) -> list:
    """-> [(start, end)] tz-aware Timestamps for that date, or [] if the date had no MWCB halt."""
    key = str(date)[:10].replace("/", "-")
    if len(key) == 8 and key.isdigit():                      # accept YYYYMMDD too
        key = f"{key[:4]}-{key[4:6]}-{key[6:]}"
    out = []
    for a, b in MWCB_HALTS.get(key, []):
        out.append((pd.Timestamp(f"{key} {a}", tz=tz), pd.Timestamp(f"{key} {b}", tz=tz)))
    return out


def halt_mask(index: pd.DatetimeIndex, date=None, tz: str = TZ, windows=None) -> np.ndarray:
    """Boolean mask, True where the timestamp falls inside a halt on that date.

    ``date`` defaults to the index's own first date, so a session frame needs no extra argument.
    ``windows`` overrides the built-in table -- pass what ``windows_from_status`` derived from the
    tape, which is authoritative and covers every date rather than the four recorded here."""
    if index is None or len(index) == 0:
        return np.zeros(0, bool)
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if date is None:
        d0 = index[0]
        date = (d0.tz_convert(tz) if getattr(index, "tz", None) is not None else d0).strftime("%Y-%m-%d")
    wins = list(windows) if windows else halt_windows(date, tz=tz)
    m = np.zeros(len(index), bool)
    if not wins:
        return m
    idx = index.tz_localize(tz) if getattr(index, "tz", None) is None else index.tz_convert(tz)
    for a, b in wins:
        m |= np.asarray((idx >= a) & (idx <= b), dtype=bool)
    return m


def is_halt_date(date) -> bool:
    return bool(halt_windows(date))


def describe(date, tz: str = TZ) -> str:
    w = halt_windows(date, tz=tz)
    if not w:
        return f"{date}: no MWCB halt on record"
    return "%s: MWCB Level 1 halt %s" % (
        date, ", ".join(f"{a.strftime('%H:%M:%S')}-{b.strftime('%H:%M:%S')} ET "
                        f"({int((b - a).total_seconds())}s)" for a, b in w))


def _selftest() -> bool:
    ok = []
    idx = pd.date_range("2020-03-09 09:30:00", "2020-03-09 16:00:00", freq="s", tz=TZ)
    m = halt_mask(idx)
    n = int(m.sum())
    # 09:34:13-09:49:13 inclusive of both endpoints = 901 one-second snapshots
    a = n == 901
    print("(1) 2020-03-09 halt covers %d of %d 1s snapshots (%.2f%% -- the session's crossed rate "
          "was 3.88%%) : %s" % (n, len(idx), 100 * n / len(idx), a))
    ok.append(a)

    b = int(halt_mask(pd.date_range("2020-03-10 09:30", "2020-03-10 16:00", freq="s", tz=TZ)).sum()) == 0
    print("(2) an ordinary date has no halt window : %s" % b)
    ok.append(b)

    c = (len(halt_windows("20200318")) == 1 and len(halt_windows("2020-03-18")) == 1
         and halt_windows("2020-03-18")[0][0].strftime("%H:%M:%S") == "12:56:11")
    print("(3) YYYYMMDD and YYYY-MM-DD both resolve; 2020-03-18 halts at 12:56:11 (afternoon, not "
          "the open) : %s" % c)
    ok.append(c)

    # every MWCB halt is 15 minutes
    d = all(int((y - x).total_seconds()) == 900 for k in MWCB_HALTS for x, y in halt_windows(k))
    print("(4) all four Level 1 halts are exactly 900s : %s" % d)
    ok.append(d)

    # naive index (no tz) must still work -- session frames are not always tz-aware
    e = int(halt_mask(pd.date_range("2020-03-09 09:30", "2020-03-09 16:00", freq="s"),
                      date="2020-03-09").sum()) == 901
    print("(5) a tz-naive index is localized rather than silently mismatched : %s" % e)
    ok.append(e)

    print("\nmarket-halts checks -> %s" % all(ok))
    return all(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
