#!/usr/bin/env python3
"""probe_es_2020.py -- why the four 2020 ES sessions came back empty, settled in one run.

The failure, from the replication log:

    extracted 2020-03-09 (ES=ESH0, ...): 23401 rows, flow=True | median SPY=279.25 ES=nan (ES/SPY=nan)
    extracted 2020-03-12 (ES=ESH0, ...): 23401 rows, flow=True | median SPY=254.19 ES=nan (ES/SPY=nan)
    extracted 2020-03-16 (ES=ESM0, ...): 23401 rows, flow=True | median SPY=248.84 ES=nan (ES/SPY=nan)
    extracted 2020-03-18 (ES=ESM0, ...): 23401 rows, flow=True | median SPY=237.96 ES=nan (ES/SPY=nan)

Every 2022-2026 session is fine, so this is specific to the 2020 futures capture. Three hypotheses
are consistent with what we have, and they need different fixes:

  (A) WRONG ERA, WRONG TYPES.  ES is fetched MBO-only -- `mt_add_order` / `mt_cancel_order` /
      `mt_modify_order` / `mt_trade` -- on the strength of one observation: ESZ4 on 2024-12-18
      returns header-only for every `mt_price_level_*` type. That is a fact about the 2024 capture
      that was written down as a fact about CME. If MayStreet's 2020 CME capture is price-level
      (MBP) instead, the MBO fetch returns zero rows and the replay builds nothing -- which is
      exactly the observed shape.
  (B) WRONG CONTRACT.  ESH0 expires 2020-03-20, so `rollover_days=8` rolls the front month on
      2020-03-12 and 03-16/03-18 resolve to ESM0. If the lake keys 2020 rows differently, or the
      real volume roll sat elsewhere, the code asks for a contract with no messages on that date.
  (C) WRONG SYMBOLOGY.  The 2020 rows differ from 2024 in ways that show the conventions moved:
      `marketparticipant` is XCME in 2020 and empty in 2024, and `mic` is empty in 2020 and XCME in
      2024. A product-code or venue-filter difference would return an empty set with a clean exit.

We already know (C) is not total: `--date 20200312 --mtype mt_product_status --product ESH0
--source futures` returns 43 rows. The product code and the source are right and the date is in the
lake. So the question is narrowed to WHICH MESSAGE TYPES carry the 2020 ES book.

This probe answers that by counting rows for every candidate type, on both candidate contracts, on
each 2020 date, with a known-good 2024 date as the control. It uses `--limit` so each query is
cheap: the question is presence, not volume. Nothing is reconstructed and nothing is written to the
dataset.

    python probe_es_2020.py --out es_2020_probe.txt

Add `--full-counts` to drop the limit and get true row counts (slow: the crash-day trade tape is
millions of rows). Presence is enough to choose the fix.
"""
from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
import tempfile

DATES = ["20200309", "20200312", "20200316", "20200318"]
CONTROL = "20241218"

# both sides of the March 2020 roll, plus the deferred month in case the lake keys it differently
CONTRACTS_2020 = ["ESH0", "ESM0"]
CONTRACTS_CONTROL = ["ESZ4", "ESH5"]

# Everything that could carry a book, plus the oracles that say whether the capture is intact.
# Grouped, because which GROUP is populated is the actual answer.
GROUPS = [
    ("MBO  (order-by-order -- what the extractor asks for today)",
     ["mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade"]),
    ("MBP  (price-level -- header-only for ESZ4 2024, the (A) hypothesis for 2020)",
     ["mt_price_level_update", "mt_modify_price_level", "mt_delete_price_level"]),
    ("ladder / stats  (a fallback book source and the volume discriminator for the roll)",
     ["mt_aggregated_price_update", "mt_product_statistics", "mt_product_status"]),
    ("resets + oracles  (is the capture even intact for this date?)",
     ["mt_clear_orders", "mt_clear_price_levels", "mt_missing_product_messages", "mt_error"]),
]
ALL_TYPES = [m for _, v in GROUPS for m in v]


def probe(date, product, mtype, limit=5, timeout=900):
    """-> (n_data_rows, header_or_error). n = -1 means the query itself failed.

    A header line with no data rows is the signature that matters: the type EXISTS in the schema for
    this product and era and is empty, which is a statement about the capture. A non-zero exit is a
    statement about the query and must not be confused with it."""
    cmd = ["mstwx-lakequery", "--date", str(date), "-s", "futures", "-p", product,
           "-m", mtype, "--print-headers", "--format", "csv"]
    if limit:
        cmd += ["--limit", str(limit)]
    fd, tmp = tempfile.mkstemp(prefix="probe_", suffix=".csv")
    os.close(fd)
    try:
        with open(tmp, "w") as fh:
            res = sp.run(cmd, stdout=fh, stderr=sp.PIPE, text=True, timeout=timeout, check=False)
        if res.returncode != 0:
            return -1, "rc=%d %s" % (res.returncode, (res.stderr or "").strip().splitlines()[0][:120]
                                     if res.stderr else "(no stderr)")
        with open(tmp) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        if not lines:
            return -1, "(no output at all -- not even a header)"
        hdr = lines[0]
        rows = lines[1:]
        feeds = ""
        if rows:                                   # which feed publishes it -- (C), the symbology check
            cols = hdr.split(",")
            if "f" in cols:
                i = cols.index("f")
                seen = []
                for r in rows:
                    p = r.split(",")
                    if len(p) > i and p[i] and p[i] not in seen:
                        seen.append(p[i])
                feeds = "  feed=" + ",".join(seen[:3])
        return len(rows), ("%d col(s)%s" % (len(hdr.split(",")), feeds))
    except sp.TimeoutExpired:
        return -1, "TIMEOUT after %ds" % timeout
    except FileNotFoundError:
        return -1, "mstwx-lakequery not on PATH -- run this on the workbench"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def run(dates, contracts, limit, out):
    lines = []

    def w(s=""):
        print(s)
        lines.append(s)
        if out:
            with open(out, "a") as fh:
                fh.write(s + "\n")

    for date in dates:
        for product in contracts:
            w("=" * 96)
            w("%s  %s  (source=futures%s)" % (date, product,
                                              "" if limit else ", FULL COUNTS"))
            w("=" * 96)
            any_book = False
            for title, types in GROUPS:
                w("  " + title)
                for mt in types:
                    n, note = probe(date, product, mt, limit=limit)
                    if n < 0:
                        mark = "QUERY FAILED"
                    elif n == 0:
                        mark = "empty (header only)"
                    else:
                        mark = ("%d+ row(s)" % n) if limit else ("%d row(s)" % n)
                        if mt in ("mt_add_order", "mt_price_level_update",
                                  "mt_aggregated_price_update"):
                            any_book = True
                    w("    %-34s %-22s %s" % (mt, mark, note))
                w()
            w("  -> a book CAN be built for %s on %s: %s" % (product, date, any_book))
            w()
    return lines


def roll_evidence(dates, contracts, out):
    """Which contract is the FRONT MONTH, by volume rather than by a calendar rule.

    The status stream cannot answer this. ESH0 and ESM0 both publish status, price limits and halts
    on every 2020 date checked, and their limits differ only by a constant calendar spread (12.00
    index points on 2020-03-16, 10.00 on 03-18) -- real for both, decisive for neither. Meanwhile
    `rollover_days=8` puts 03-16 and 03-18 on ESM0 purely because ESH0 expires on the 20th, and
    March 2020 is not a week to trust a calendar rule about where the volume went.

    `mt_product_statistics` is a handful of rows per contract per day and carries the session's own
    totals, so it is fetched in FULL and dumped verbatim -- reading the volume field off the real
    row beats guessing which column name the vendor uses."""
    def w(s=""):
        print(s)
        if out:
            with open(out, "a") as fh:
                fh.write(s + "\n")

    w("=" * 96)
    w("ROLL EVIDENCE -- mt_product_statistics in full (small), to pick the front month by VOLUME")
    w("=" * 96)
    for date in dates:
        for product in contracts:
            cmd = ["mstwx-lakequery", "--date", str(date), "-s", "futures", "-p", product,
                   "-m", "mt_product_statistics", "--print-headers", "--format", "csv"]
            fd, tmp = tempfile.mkstemp(prefix="roll_", suffix=".csv")
            os.close(fd)
            try:
                with open(tmp, "w") as fh:
                    res = sp.run(cmd, stdout=fh, stderr=sp.PIPE, text=True, timeout=1800, check=False)
                if res.returncode != 0:
                    w("  %s %s: QUERY FAILED rc=%d %s" % (date, product, res.returncode,
                                                          (res.stderr or "").strip()[:120]))
                    continue
                lines = [ln for ln in open(tmp).read().splitlines() if ln.strip()]
            except FileNotFoundError:
                w("  mstwx-lakequery not on PATH -- run this on the workbench")
                return
            except sp.TimeoutExpired:
                w("  %s %s: TIMEOUT" % (date, product))
                continue
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            w("  --- %s %s : %d row(s) ---" % (date, product, max(0, len(lines) - 1)))
            for ln in lines[:40]:
                w("    " + ln)
            if len(lines) > 40:
                w("    ... %d more" % (len(lines) - 40))
            w()
    w("Read the volume/turnover column off these rows: the contract with the larger RTH volume is")
    w("the front month for that date, whatever rollover_days says.")
    w()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", default=",".join(DATES),
                    help="comma-separated YYYYMMDD (default: the four 2020 MWCB dates)")
    ap.add_argument("--contracts", default=",".join(CONTRACTS_2020),
                    help="comma-separated contract codes to try (default: ESH0,ESM0)")
    ap.add_argument("--control", default=CONTROL,
                    help="known-good date probed last for contrast ('' to skip)")
    ap.add_argument("--full-counts", action="store_true",
                    help="drop --limit for true row counts (SLOW: millions of rows on a crash day)")
    ap.add_argument("--no-roll", action="store_true",
                    help="skip the mt_product_statistics dump that picks the front month by volume")
    ap.add_argument("--out", default="", help="also write the report here")
    a = ap.parse_args()

    if a.out and os.path.exists(a.out):
        os.unlink(a.out)
    limit = 0 if a.full_counts else 5

    dates = [d.strip() for d in a.dates.split(",") if d.strip()]
    contracts = [c.strip().upper() for c in a.contracts.split(",") if c.strip()]
    run(dates, contracts, limit, a.out)
    if not a.no_roll:
        roll_evidence(dates, contracts, a.out)
    if a.control:
        print("\n### CONTROL: a date the extractor handles correctly, for contrast ###\n")
        run([a.control], CONTRACTS_CONTROL, limit, a.out)

    print("\nWhat the answer means:")
    print("  MBO populated, MBP empty          -> same as 2024; the problem is the CONTRACT (B) or")
    print("                                       the reconstruction, not the message types.")
    print("  MBO empty, MBP populated          -> hypothesis (A): the 2020 CME capture is")
    print("                                       price-level. ES needs the MBP types for this era,")
    print("                                       exactly as SPY already does.")
    print("  both empty, ladder populated      -> build the 2020 ES leg from")
    print("                                       mt_aggregated_price_update (validate_aggregated.py")
    print("                                       already reads it; it becomes a source, not just a")
    print("                                       benchmark).")
    print("  everything empty on BOTH          -> the date/product pair is genuinely absent from the")
    print("  contracts, on a date whose           futures capture and this is a vendor question, not")
    print("  mt_product_status DOES return        a code one.")
    print("  rows")
    print("  one contract populated, the other -> hypothesis (B): fix the roll, not the fetch.")
    print("  empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
