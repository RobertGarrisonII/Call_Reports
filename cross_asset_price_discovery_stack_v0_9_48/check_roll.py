#!/usr/bin/env python3
"""check_roll.py -- measure the front-month VOLUME SHARE on any date, instead of assuming it.

Three different things had been conflated, and the replication log made them look like one:

  * **Which contract is used** is a calendar rule -- `get_front_month_contract`, third Friday minus
    `rollover_days`. It runs on EVERY session, all 24 of them.
  * **Whether a session sits in a roll window** is also computed per session -- `roll_window_days`,
    signed, on every date.
  * **What share the chosen contract actually carries** has been measured on exactly FOUR dates:
    2020-03-09, 03-12, 03-16 and 03-18. One roll, in a crisis week.

The extractor's roll warning quoted those four measurements on every session that fell in a roll
window, including 2023-03-09, 2024-12-18 and 2025-06-13, where nothing has been measured at all.
That reads as a fact about the session and is an extrapolation from a different roll three to five
years earlier -- and a crisis-week roll at that, where positions were being moved under duress. An
ordinary quarter may roll far more cleanly, in which case the warning is alarmist; it may roll
worse; the honest statement is that it is unknown.

This tool makes it knowable cheaply. `mt_product_statistics` carries a cumulative `volume` counter
and an `openinterest` level, and BOTH discriminators are present in the first rows of the stream --
the pre-open rows carry the previous session's closing totals -- so a small `--limit` fetch settles
a date without pulling the ~400 MB the full day costs.

    python check_roll.py --dates 2023-03-09,2024-12-18,2025-06-13
    python check_roll.py --dates 2020-03-12 --full        # true RTH volume, slow

Exit codes: 0 = the calendar rule picked the leading contract everywhere it could be measured;
2 = it did not, on at least one date; 1 = nothing could be measured.
"""
from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
import tempfile

import numpy as np
import pandas as pd

import mstbook_loader as ml

TZ = "America/New_York"
STATS = "mt_product_statistics"


def _fetch_head(date_ymd: str, product: str, limit: int = 40, timeout: int = 1800):
    """-> DataFrame of the first ``limit`` statistics rows, or None. Cheap by construction."""
    cmd = ["mstwx-lakequery", "--date", str(date_ymd), "-s", "futures", "-p", product,
           "-m", STATS, "--print-headers", "--format", "csv"]
    if limit:
        cmd += ["--limit", str(limit)]
    fd, tmp = tempfile.mkstemp(prefix="roll_", suffix=".csv")
    os.close(fd)
    try:
        with open(tmp, "w") as fh:
            res = sp.run(cmd, stdout=fh, stderr=sp.PIPE, text=True, timeout=timeout, check=False)
        if res.returncode != 0:
            return None
        df = pd.read_csv(tmp, low_memory=False)
        return df if len(df) else None
    except FileNotFoundError:
        raise RuntimeError("mstwx-lakequery not on PATH -- run this on the workbench")
    except (sp.TimeoutExpired, Exception):
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _num(df, col):
    if df is None or col not in df.columns:
        return np.nan
    v = pd.to_numeric(df[col].replace("\\N", np.nan), errors="coerce").dropna()
    return float(v.iloc[-1]) if len(v) else np.nan


def measure(date_str: str, symbol: str = "ES", rollover_days: int = 8, full: bool = False) -> dict:
    """Measure the share of futures volume carried by each candidate contract on one date."""
    ymd = date_str.replace("-", "")
    d = ml._parse_yyyymmdd(ymd)
    prev, front, nxt = ml.adjacent_contracts(symbol, d, rollover_days)
    off = ml.roll_window_days(d, rollover_days)
    # Which contract the front month is competing with depends on the side of the boundary.
    # AT the boundary (offset 0) the rule still returns the OLD contract, so the rival is the
    # deferred month -- not the already-expired one. Getting this wrong on 2020-03-12 paired ESH0
    # against ESZ9, a contract that expired the previous December, and the date came back
    # "not measured" instead of 72.8%.
    rival = nxt if off <= 0 else prev
    rep = {"date": date_str, "front": front, "rival": rival, "roll_offset_days": off,
           "measured": False, "front_share": float("nan"), "note": ""}

    vols, ois = {}, {}
    for c in (front, rival):
        df = _fetch_head(ymd, c, limit=0 if full else 40)
        vols[c] = _num(df, "volume")
        ois[c] = _num(df, "openinterest")
    rep["volume"] = vols
    rep["open_interest"] = ois
    # Open interest is reported ALONGSIDE volume, not instead of it, because the two disagree
    # exactly where the answer matters. On 2020-03-16 open interest still favoured ESH0
    # (2.69 M vs 1.43 M) while volume had already moved to ESM0 (3.03 M vs 1.99 M) -- an
    # open-interest rule would have put that session on the contract carrying 39.6% of the trading.
    # OI is a STOCK of positions settled once a day; volume is the flow of transactions that price
    # discovery is made of. For a book-level study the flow is the criterion; OI is reported so the
    # disagreement is visible rather than assumed away.
    otot = np.nansum([ois.get(front, np.nan), ois.get(rival, np.nan)])
    rep["oi_share"] = (float(ois[front] / otot)
                       if np.isfinite(ois.get(front, np.nan)) and np.isfinite(ois.get(rival, np.nan))
                       and otot > 0 else float("nan"))
    tot = np.nansum([vols.get(front, np.nan), vols.get(rival, np.nan)])
    if np.isfinite(vols.get(front, np.nan)) and np.isfinite(vols.get(rival, np.nan)) and tot > 0:
        rep["measured"] = True
        rep["front_share"] = float(vols[front] / tot)
        rep["rule_agrees"] = bool(vols[front] >= vols[rival])
        rep["note"] = ("volume from the first %s rows -- the pre-open rows carry the PREVIOUS "
                       "session's closing cumulative total, which is the cheap discriminator; pass "
                       "--full for this session's own RTH volume" % ("(all)" if full else "40"))
    else:
        rep["note"] = "could not read volume for both contracts"
    return rep


def describe(reps) -> str:
    lines = ["front-month check: is the contract the calendar rule picks the one with the volume?",
             ""]
    lines.append("%-12s %-7s %-7s %-6s %13s %13s %7s %7s %s"
                 % ("date", "front", "rival", "offset", "front vol", "rival vol", "vol%", "OI%",
                    "rule ok"))
    for r in reps:
        v, o = r.get("volume", {}), r.get("open_interest", {})
        fv, rv = v.get(r["front"], np.nan), v.get(r["rival"], np.nan)
        lines.append("%-12s %-7s %-7s %+6d %13s %13s %7s %7s %s"
                     % (r["date"], r["front"], r["rival"], r["roll_offset_days"],
                        "{:,}".format(int(fv)) if np.isfinite(fv) else "-",
                        "{:,}".format(int(rv)) if np.isfinite(rv) else "-",
                        ("%.1f%%" % (100 * r["front_share"])) if r["measured"] else "-",
                        ("%.1f%%" % (100 * r["oi_share"])) if np.isfinite(r.get("oi_share", np.nan)) else "-",
                        ("yes" if r.get("rule_agrees") else "NO") if r["measured"] else "not measured"))
    bad = [r for r in reps if r["measured"] and not r.get("rule_agrees")]
    split = [r for r in reps if r["measured"] and r["front_share"] < 0.90]
    lines.append("")
    if bad:
        lines.append("THE CALENDAR RULE PICKED THE WRONG CONTRACT on %s -- the rival carries more "
                     "volume. Re-extract those sessions with the other contract."
                     % ", ".join(r["date"] for r in bad))
    if split:
        lines.append("SPLIT sessions (front month under 90%% of the two-contract volume): %s."
                     % ", ".join("%s %.1f%%" % (r["date"], 100 * r["front_share"]) for r in split))
        lines.append("  The single-contract ES leg is not the whole futures market on those days.")
        lines.append("  Put the measured share in the sample appendix; do NOT splice the contracts,")
        lines.append("  which carry a 10-12 point calendar spread that jumps at the seam.")
    if not bad and not split and any(r["measured"] for r in reps):
        lines.append("Every measured session is concentrated (>=90%) in the contract the rule picks.")
    disagree = [r for r in reps if r["measured"] and np.isfinite(r.get("oi_share", np.nan))
                and (r["front_share"] >= 0.5) != (r["oi_share"] >= 0.5)]
    if disagree:
        lines.append("")
        lines.append("VOLUME AND OPEN INTEREST DISAGREE on %s."
                     % ", ".join("%s (vol %.1f%% vs OI %.1f%%)"
                                 % (r["date"], 100 * r["front_share"], 100 * r["oi_share"])
                                 for r in disagree))
        lines.append("  Open interest rolls LATER than volume -- a position must be closed to move,")
        lines.append("  and it is settled once a day, so OI is a day-stale stock while volume is the")
        lines.append("  flow. Price discovery happens where the trading is, so VOLUME is the")
        lines.append("  criterion here; an OI rule would put these sessions on the quieter contract.")
    unmeasured = [r["date"] for r in reps if not r["measured"]]
    if unmeasured:
        lines.append("NOT MEASURED (and therefore not verified): %s" % ", ".join(unmeasured))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", required=True, help="comma-separated YYYY-MM-DD or YYYYMMDD")
    ap.add_argument("--symbol", default="ES")
    ap.add_argument("--rollover-days", type=int, default=8)
    ap.add_argument("--full", action="store_true",
                    help="drop the row limit and read the session's own totals (SLOW: ~400 MB per "
                         "contract-day). The limited read is enough to rank the two contracts.")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    reps = []
    for d in [x.strip() for x in a.dates.split(",") if x.strip()]:
        try:
            reps.append(measure(d, a.symbol, a.rollover_days, a.full))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    text = describe(reps)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    if not any(r["measured"] for r in reps):
        return 1
    return 2 if any(r["measured"] and not r.get("rule_agrees") for r in reps) else 0


if __name__ == "__main__":
    sys.exit(main())
