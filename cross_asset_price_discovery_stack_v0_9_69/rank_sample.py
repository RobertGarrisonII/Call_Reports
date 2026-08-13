#!/usr/bin/env python3
"""rank_sample.py -- make the volatile-day selection DERIVABLE instead of an input.

The replication sample's construction rule (Appendix A.1, re-based onto 2022-2026) is
"volatile = the largest intraday-range days in the window; baseline = the same weekday
roughly one year prior to each". validate_sample.py has always checked the PAIRING rules,
but nothing in the repo computed the RANKING -- the stack pulls intraday book data for the
chosen days only, not the daily bars the ranking needs. That made the volatile list an
input a referee could not re-derive.

This script closes the gap from a user-supplied DAILY OHLC file (any source: CRSP, vendor
export, public daily bars for SPY):

    python rank_sample.py --csv spy_daily.csv                       # rank + suggest pairs
    python rank_sample.py --csv spy_daily.csv --window 2022-01-01:2026-08-05 --top 10
    python rank_sample.py --csv spy_daily.csv --compare             # where does the shipped
                                                                    # list rank under YOUR data?
    python rank_sample.py --selftest                                # synthetic end-to-end check

CSV format: columns date, high, low, close (case-insensitive; extra columns ignored).
Metric: intraday range (high - low) / close, in percent -- the paper's "largest intraday
range" criterion. --metric range_prevclose divides by the PREVIOUS close instead (gap days
rank higher); the shipped default list was constructed under range/close.

Pairing rule (unchanged from the driver): baseline = same weekday, 350-371 calendar days
earlier, must itself be a trading day in the file; the candidate closest to 364 days wins.
Days whose window has no valid candidate are reported as UNPAIRABLE -- replace them, since
the level-matched control design needs the pair.

Exit codes: 0 ok; 1 bad input (unreadable csv, missing columns, empty window).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

# the shipped default sample (run_paper_replication.sh); --compare ranks these
DEFAULT_VOLATILE = ["2024-12-18", "2026-06-05", "2025-10-10", "2024-09-03", "2025-04-03",
                    "2024-08-05", "2024-07-24", "2025-01-27", "2023-03-09", "2025-08-01"]
DEFAULT_BASELINE = ["2023-12-20", "2025-06-13", "2024-10-18", "2023-09-05", "2024-04-04",
                    "2023-08-07", "2023-07-19", "2024-01-29", "2022-03-24", "2024-08-09"]

PAIR_MIN_D, PAIR_MAX_D, PAIR_TARGET_D = 350, 371, 364


def load_daily(path: str) -> pd.DataFrame:
    """-> DataFrame indexed by normalized date with float columns high, low, close."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in ("date", "high", "low", "close") if c not in cols]
    if missing:
        raise SystemExit("csv %s lacks column(s) %s (have: %s)"
                         % (path, missing, list(df.columns)))
    out = pd.DataFrame({
        "high": pd.to_numeric(df[cols["high"]], errors="coerce").to_numpy(),
        "low": pd.to_numeric(df[cols["low"]], errors="coerce").to_numpy(),
        "close": pd.to_numeric(df[cols["close"]], errors="coerce").to_numpy(),
    }, index=pd.to_datetime(df[cols["date"]], errors="coerce").dt.normalize())
    out = out[out.index.notna()].sort_index()
    out = out[(out[["high", "low", "close"]] > 0).all(axis=1)
              & (out["high"] >= out["low"])]
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def rank_days(daily: pd.DataFrame, window=("2022-01-01", "2026-12-31"),
              metric="range_close") -> pd.DataFrame:
    """Every trading day in the window ranked by intraday range %, rank 1 = largest."""
    d = daily.loc[str(window[0]): str(window[1])].copy()
    if d.empty:
        raise SystemExit("no rows in window %s..%s" % window)
    if metric == "range_prevclose":
        base = d["close"].shift(1)
    else:
        base = d["close"]
    d["range_pct"] = 100.0 * (d["high"] - d["low"]) / base
    d = d.dropna(subset=["range_pct"]).sort_values("range_pct", ascending=False)
    d["rank"] = np.arange(1, len(d) + 1)
    d["weekday"] = d.index.day_name()
    return d


def suggest_pair(daily: pd.DataFrame, vol_day) -> tuple:
    """-> (baseline date or None, gap days or None): same weekday, 350-371 d earlier,
    a trading day in the file, closest to 364."""
    v = pd.Timestamp(vol_day).normalize()
    cands = [(abs((v - b).days - PAIR_TARGET_D), b) for b in daily.index
             if PAIR_MIN_D <= (v - b).days <= PAIR_MAX_D and b.dayofweek == v.dayofweek]
    if not cands:
        return None, None
    _, best = min(cands)
    return best, (v - best).days


def build_report(daily, window, top, metric, compare):
    ranked = rank_days(daily, window, metric)
    lines = ["ranked intraday-range days, window %s..%s, metric %s (n=%d trading days)"
             % (window[0], window[1], metric, len(ranked)), ""]
    lines.append("%-4s %-12s %-10s %9s   suggested baseline pair" % ("rank", "date", "weekday", "range%"))
    sel = ranked.head(top)
    pairs = []
    for ts, row in sel.iterrows():
        b, gap = suggest_pair(daily, ts)
        pairs.append((ts.date(), None if b is None else b.date()))
        tag = ("%s (%dd, %s)" % (b.date(), gap, b.day_name()) if b is not None
               else "UNPAIRABLE in %d-%dd same-weekday window -- replace this day"
               % (PAIR_MIN_D, PAIR_MAX_D))
        lines.append("%-4d %-12s %-10s %9.3f   %s"
                     % (row["rank"], ts.date(), row["weekday"], row["range_pct"], tag))
    lines.append("")
    lines.append("--volatile %s" % ",".join(str(v) for v, _b in pairs))
    if all(b is not None for _v, b in pairs):
        lines.append("--baseline %s" % ",".join(str(b) for _v, b in pairs))
    else:
        lines.append("(baseline list omitted: unpairable day(s) above)")
    if compare:
        lines += ["", "shipped default list under YOUR data (rank 1 = largest range):"]
        by_date = {ts.date().isoformat(): int(r) for ts, r in ranked["rank"].items()}
        for v in DEFAULT_VOLATILE:
            r = by_date.get(v)
            lines.append("  %-12s rank %s" % (v, r if r is not None else
                                              "-- (not in file/window)"))
        in_top = sum(1 for v in DEFAULT_VOLATILE
                     if by_date.get(v) is not None and by_date[v] <= top)
        lines.append("  %d/%d of the shipped days are in your top %d. Divergence is not an "
                     "error by itself" % (in_top, len(DEFAULT_VOLATILE), top))
        lines.append("  (different daily source, window edges, or the metric variant), but the "
                     "paper must cite the file this script ran on.")
    return "\n".join(lines)


def _selftest() -> int:
    print("rank_sample self-test")
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2022-01-03", "2026-08-05")
    close = 400 * np.exp(np.cumsum(rng.normal(0, 0.008, len(idx))))
    spread = np.abs(rng.normal(0.006, 0.002, len(idx)))
    planted = {"2024-12-18": 0.09, "2025-04-03": 0.07, "2024-08-05": 0.11}
    for d, s in planted.items():
        spread[idx.get_loc(pd.Timestamp(d))] = s
    daily = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"),
                          "high": close * (1 + spread / 2),
                          "low": close * (1 - spread / 2),
                          "close": close})
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "daily.csv"); daily.to_csv(p, index=False)
        loaded = load_daily(p)
        ranked = rank_days(loaded, ("2022-01-01", "2026-08-05"))
        top3 = [ts.date().isoformat() for ts in ranked.head(3).index]
        ok1 = sorted(top3) == sorted(planted)
        print("(1) the three planted extreme-range days are exactly the top 3: %s (%s)"
              % (ok1, top3))
        b, gap = suggest_pair(loaded, "2024-12-18")
        ok2 = (b is not None and b.dayofweek == pd.Timestamp("2024-12-18").dayofweek
               and PAIR_MIN_D <= gap <= PAIR_MAX_D)
        print("(2) pairing: same weekday (%s), gap %s in [%d, %d]: %s"
              % (b.day_name() if b is not None else None, gap, PAIR_MIN_D, PAIR_MAX_D, ok2))
        rep = build_report(loaded, ("2022-01-01", "2026-08-05"), 5, "range_close", True)
        ok3 = "--volatile" in rep and "shipped default list" in rep and "2024-08-05" in rep
        print("(3) report renders the CLI lists and the --compare block: %s" % ok3)
    ok = ok1 and ok2 and ok3
    print("rank_sample checks -> %s" % ok)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="", help="daily OHLC csv (date, high, low, close)")
    ap.add_argument("--window", default="2022-01-01:2026-08-05",
                    help="START:END (default the revised sample window)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--metric", choices=["range_close", "range_prevclose"],
                    default="range_close")
    ap.add_argument("--compare", action="store_true",
                    help="also rank the shipped default volatile list under this data")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not a.csv:
        ap.error("--csv is required (or --selftest)")
    window = tuple(a.window.split(":", 1)) if ":" in a.window else (a.window, "2099-01-01")
    print(build_report(load_daily(a.csv), window, a.top, a.metric, a.compare))
    return 0


if __name__ == "__main__":
    sys.exit(main())
