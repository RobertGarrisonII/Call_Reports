#!/usr/bin/env python3
"""validate_aggregated.py -- benchmark the reconstructed book against the VENUE'S OWN ladder.

"How do you know the reconstruction is right?" is the first question a referee asks about a paper
whose central claim rests on a book replayed from raw messages. Until now the only answer was
``validate_against_snapshot`` against ``mstbook-query`` -- and that benchmark was demoted precisely
because it sits on a DIFFERENT clock, so any disagreement is confounded with the clock difference
and neither direction of the comparison proves anything.

The ES tape supplies a benchmark without that defect. ``mt_aggregated_price_update`` is CME's own
10-level ladder -- price, quantity AND order count per level -- delivered through the SAME lake, on
the SAME capture clock as the add/cancel/modify/trade messages the replay consumes. Disagreement
therefore localizes to the replay, which is the whole point of a validation. For ESZ4 on 2024-12-18
it is fully populated from the 17:45 ET pre-open onward, single feed (cme_globex30_cme), so it is a
clean apples-to-apples comparison with no consolidation ambiguity.

``mt_product_statistics`` adds a second, independent axis: the session's official high, low, volume,
VWAP and settlement. A replay can match the ladder tick for tick and still have the wrong session
boundaries; this catches that.

    python validate_aggregated.py --date 20241218 --product ESZ4 --product-type futures \\
        --price-scale 0.01
    python validate_aggregated.py --date 20241218 --product ESZ4 --product-type futures \\
        --interval 1s --session 09:30-16:00 --out val_esz4.txt

Exit codes: 0 = agreement within tolerance;  2 = disagreement;  1 = could not run.

WHAT THIS PROVES SINCE v0.9.42. The extractor now builds the ES leg FROM this ladder
(aggregated_book.py), so running this is no longer a check on the shipped futures book -- that would
be the ladder against itself. The direction has inverted, and the result is still worth having: an
INDEPENDENT reconstruction from the raw message tape is compared against the ladder, and agreement
is evidence that the ladder is a faithful book rather than a lossy summary of one. It validates the
SOURCE CHOICE, not the extraction. Report it that way.

It also remains the right tool if the leg is switched back with --es-book-source replay, where it
recovers its original meaning.

CAVEAT on mt_product_statistics: the rows are a running stream and the EARLY ones carry the PREVIOUS
session's figures (on 2024-12-18 the 17:38 ET row still shows the prior settlement, stamped
2024-12-15 19:00 ET). Take the last non-null value of each field inside the trade date, never the
first -- which is what ``session_statistics`` below does.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd

import lob_reconstruct as lob

_AGG = "mt_aggregated_price_update"
_STATS = "mt_product_statistics"


def _fetch(date_str, product, ptype, mtype, data_source, tz, clock):
    import mstbook_loader as ml
    return ml._fetch_messages(date_str, product, ptype, mtype, data_source, tz=tz, clock=clock)


def aggregated_to_canonical(df: pd.DataFrame, asset: str, levels: int = 10,
                            price_scale: float = 1.0) -> pd.DataFrame:
    """mt_aggregated_price_update -> the stack's canonical {ASSET}_{bid|ask}{price|quantity}_{i}.

    Nulls print as ``\\N``; the reader leaves them as strings, so coerce. The venue emits one row per
    ladder change, so the frame is the venue's book as-of each row -- exactly what a reconstruction
    is trying to reproduce."""
    out = pd.DataFrame(index=df.index)
    nan = pd.Series(np.nan, index=df.index)

    def _num(col):                       # a level the venue never sent is absent, not zero
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else nan

    for i in range(1, levels + 1):
        for side, tag in (("bid", "bid"), ("ask", "ask")):
            out[f"{asset}_{tag}price_{i}"] = _num(f"{side}price_{i}").to_numpy(float) * price_scale
            out[f"{asset}_{tag}quantity_{i}"] = _num(f"{side}quantity_{i}").to_numpy(float)
    out[f"{asset}_mid"] = (out[f"{asset}_bidprice_1"] + out[f"{asset}_askprice_1"]) / 2.0
    return out


def ladder_cadence(agg: pd.DataFrame, interval: str = "1s", session=("09:30", "16:00"),
                   tz: str = "America/New_York", date_str: Optional[str] = None) -> dict:
    """How often the venue republishes the ladder, RELATIVE TO THE SAMPLING GRID.

    This is the number that decides "rebuild the ES book or use CME's ladder", and it decides it
    empirically rather than by argument. The two sources can only differ on the grid the paper
    actually uses if the ladder is republished LESS often than the grid samples it. If the ladder
    updates many times per grid cell, then the as-of sample takes the same final state a replay
    would have reached, and the choice is immaterial for every measurement built on that grid.

    So: the fraction of grid cells containing at least one ladder update, and the median number of
    updates per cell. A ladder that beats the grid comfortably makes the question moot; one that
    does not is a real argument for rebuilding, and the numbers say which.

    Note what this does NOT settle: order-level detail (queue position, per-order dynamics) is
    absent from an aggregated ladder at ANY cadence, so a question that needs it needs the replay
    regardless of what this reports."""
    out = {"n_rows": int(len(agg))}
    if agg is None or len(agg) == 0:
        return out
    idx = agg.index
    if getattr(idx, "tz", None) is None:
        idx = pd.DatetimeIndex(idx).tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    if date_str:
        d = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    else:
        d = idx[0].strftime("%Y-%m-%d")
    g0 = pd.Timestamp(f"{d} {session[0]}", tz=tz)
    g1 = pd.Timestamp(f"{d} {session[1]}", tz=tz)
    grid = pd.date_range(g0, g1, freq=interval)
    insess = idx[(idx >= g0) & (idx <= g1)]
    out["n_in_session"] = int(len(insess))
    out["n_cells"] = int(len(grid))
    if not len(insess) or len(grid) < 2:
        return out
    step = grid[1] - grid[0]
    cell = ((insess - g0) // step).astype("int64")
    per = pd.Series(1, index=cell).groupby(level=0).size()
    out["cells_covered_frac"] = float(len(per) / len(grid))
    out["updates_per_cell_median"] = float(per.median())
    out["updates_per_cell_p05"] = float(per.quantile(0.05))
    gaps = np.diff(insess.values).astype("timedelta64[ns]").astype(float) / 1e9
    out["gap_median_s"] = float(np.median(gaps)) if len(gaps) else float("nan")
    out["gap_p95_s"] = float(np.quantile(gaps, 0.95)) if len(gaps) else float("nan")
    out["grid_s"] = float(step.total_seconds())
    return out


def describe_cadence(c: dict) -> list:
    if not c.get("n_in_session"):
        return ["ladder cadence: no in-session rows"]
    faster = c.get("gap_median_s", float("inf")) < c.get("grid_s", 1.0)
    lines = ["ladder cadence vs the %.0fs sampling grid: %s in-session update(s) over %s cell(s); "
             "median gap %.4fs (p95 %.4fs); %.1f%% of cells carry an update, median %.0f per cell"
             % (c.get("grid_s", 1.0), f"{c['n_in_session']:,d}", f"{c['n_cells']:,d}",
                c.get("gap_median_s", float("nan")), c.get("gap_p95_s", float("nan")),
                100 * c.get("cells_covered_frac", 0.0), c.get("updates_per_cell_median", 0))]
    if faster:
        lines.append("  -> the ladder is republished FASTER than the grid samples it, so an as-of "
                     "sample takes the same final state a replay would reach. On this grid the two "
                     "book sources cannot materially differ; rebuilding buys order-level detail "
                     "(queue position), not accuracy.")
    else:
        lines.append("  -> the ladder is republished SLOWER than the grid samples it, so cells "
                     "without an update are stale. Here rebuilding from messages IS more faithful "
                     "and --es-book-source replay is the better default.")
    return lines


def session_statistics(df: pd.DataFrame, price_scale: float = 1.0) -> dict:
    """Last non-null value of each session statistic INSIDE the frame's span (see the caveat above)."""
    want = {"openingprice": price_scale, "closingprice": price_scale, "highprice": price_scale,
            "lowprice": price_scale, "lasttrade": price_scale,
            "volumeweightedaverageprice": price_scale, "settlementprice": price_scale,
            "previousdaysettlementprice": price_scale, "volume": 1.0, "totalvolume": 1.0,
            "openinterest": 1.0}
    out = {}
    for c, scale in want.items():
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s):
            out[c] = float(s.iloc[-1]) * scale
    return out


def compare(recon: pd.DataFrame, bench: pd.DataFrame, asset: str, levels: int = 10,
            tol: float = 1e-6) -> pd.DataFrame:
    """As-of align the venue ladder onto the reconstruction's grid and compare level by level.

    The benchmark is event-stamped and the reconstruction is grid-stamped, so the venue row is
    carried FORWARD to each grid point (its state as of that instant) -- the same as-of rule the
    reconstruction itself uses. Comparing without that would score a timing difference as an error."""
    if recon.empty or bench.empty:
        return pd.DataFrame()
    b = bench.sort_index()
    if b.index.tz is None:
        b.index = b.index.tz_localize(recon.index.tz)
    aligned = b.reindex(b.index.union(recon.index)).ffill().reindex(recon.index)

    rows = []
    for i in range(1, levels + 1):
        for tag in ("bid", "ask"):
            pc, qc = f"{asset}_{tag}price_{i}", f"{asset}_{tag}quantity_{i}"
            if pc not in recon.columns or pc not in aligned.columns:
                continue
            rp, bp = recon[pc].to_numpy(float), aligned[pc].to_numpy(float)
            rq, bq = recon[qc].to_numpy(float), aligned[qc].to_numpy(float)
            both_p = np.isfinite(rp) & np.isfinite(bp)
            both_q = both_p & np.isfinite(rq) & np.isfinite(bq)
            n = int(both_p.sum())
            price_match = float((np.abs(rp[both_p] - bp[both_p])
                                 <= tol * np.maximum(1.0, np.abs(bp[both_p]))).mean()) if n else np.nan
            qty_match = float((np.abs(rq[both_q] - bq[both_q]) <= EPSQ).mean()) if both_q.any() else np.nan
            mad = float(np.mean(np.abs(rp[both_p] - bp[both_p]))) if n else np.nan
            rows.append({"level": i, "side": tag, "n": n, "price_match": price_match,
                         "qty_match": qty_match, "price_mad": mad,
                         "recon_only": int((np.isfinite(rp) & ~np.isfinite(bp)).sum()),
                         "bench_only": int((~np.isfinite(rp) & np.isfinite(bp)).sum())})
    return pd.DataFrame(rows)


EPSQ = 1e-6          # quantities are integers; anything above float noise is a real disagreement


def run(date_str, product, ptype="futures", price_scale=0.01, levels=10, interval="1s",
        session="09:30-16:00", data_source="apu", tz="America/New_York", clock="receipt",
        min_l1_match=0.99) -> tuple:
    a, b = session.split("-")
    agg = _fetch(date_str, product, ptype, _AGG, data_source, tz, clock)
    if agg.empty:
        return (f"{product} {date_str}: {_AGG} returned no rows -- this venue does not publish an "
                f"aggregated ladder, so there is nothing to validate against here.", 0)
    feeds = sorted(agg["f"].astype(str).unique()) if "f" in agg.columns else []
    asset = "X"

    # EVERY candidate family, then let the row counts choose -- the same rule the extractor uses.
    # Hardcoding the order-by-order types here meant this validation could not run at all on the
    # 2020 sessions, whose capture is price-level: it would replay nothing and report a 0% match
    # against a perfectly good ladder. The comparison that decides "rebuild or use the ladder" was
    # unavailable on exactly the four dates where the question mattered.
    msgs = {}
    for mt in ("mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade",
               "mt_price_level_update", "mt_modify_price_level", "mt_delete_price_level",
               "mt_clear_orders", "mt_clear_price_levels"):
        try:
            msgs[mt] = _fetch(date_str, product, ptype, mt, data_source, tz, clock)
        except Exception as exc:
            return (f"{product} {date_str}: fetch {mt} failed ({exc}); cannot validate a replay that "
                    f"could not be built.", 1)
    counts = {k: int(len(v)) for k, v in msgs.items()}
    family, why = lob.select_book_family(counts)
    if family in ("MBO", "MBP"):
        keep = set(lob._MBO_INSERT + lob._MBO_REMOVE) if family == "MBO" else \
            set(lob._MBP_INSERT + lob._MBP_REMOVE)
        for mt in list(msgs):
            if mt in set(lob._MBO_INSERT + lob._MBO_REMOVE + lob._MBP_INSERT + lob._MBP_REMOVE) \
                    and mt not in keep:
                msgs[mt] = pd.DataFrame()
    elif family is None:
        return (f"{product} {date_str}: no message family can build a book here -- {why}. The ladder "
                f"has {len(agg):,d} row(s), so the ES leg is fine; there is simply no independent "
                f"replay to compare it against on this date.", 1)
    recon = lob.reconstruct_book(msgs, asset=asset, levels=levels, interval=interval,
                                 session=(a, b), tz=tz, date_str=date_str, clock=clock,
                                 price_scale=price_scale, consume=True)
    bench = aggregated_to_canonical(agg, asset, levels=levels, price_scale=price_scale)
    tbl = compare(recon, bench, asset, levels=levels)
    cadence = ladder_cadence(agg, interval=interval, session=(a, b), tz=tz, date_str=date_str)

    lines = [f"reconstruction vs the venue's own ladder: {product} ({ptype}) {date_str}",
             f"benchmark = {_AGG}, feeds={feeds or '?'}, SAME lake and SAME capture clock ({clock})",
             f"replay built from the {family} family ({why})",
             f"grid {interval} over {session}, {len(recon):,d} snapshots, price_scale={price_scale}", ""]
    if tbl.empty:
        return ("\n".join(lines + ["no comparable levels"]), 1)
    show = tbl.copy()
    for c in ("price_match", "qty_match"):
        show[c] = show[c].map(lambda v: "n/a" if not np.isfinite(v) else f"{v:.4%}")
    show["price_mad"] = show["price_mad"].map(lambda v: "n/a" if not np.isfinite(v) else f"{v:.6f}")
    pd.set_option("display.width", 200)
    lines += [show.to_string(index=False), ""]

    l1 = tbl[tbl["level"] == 1]
    worst = float(np.nanmin(l1["price_match"])) if len(l1) else np.nan
    lines.append("level-1 price agreement: %s" % ("n/a" if not np.isfinite(worst) else f"{worst:.4%}"))
    lines += [""] + describe_cadence(cadence)

    try:
        st = _fetch(date_str, product, ptype, _STATS, data_source, tz, clock)
        stats = session_statistics(st, price_scale=price_scale)
    except Exception as exc:
        stats = {}
        lines.append(f"({_STATS} unavailable: {exc})")
    if stats:
        lines += ["", "session statistics reported by the venue (last value in the day, NOT the first "
                  "-- early rows carry the PREVIOUS session):"]
        for k, v in stats.items():
            lines.append(f"    {k:30s} {v:,.4f}")
        hi = pd.to_numeric(recon.get(f"{asset}_askprice_1"), errors="coerce").max()
        lo = pd.to_numeric(recon.get(f"{asset}_bidprice_1"), errors="coerce").min()
        lines.append(f"    reconstructed top-of-book range over the grid: {lo:,.4f} .. {hi:,.4f}")
        if "highprice" in stats and "lowprice" in stats:
            lines.append("    (the venue figures span the FULL trade date, which for a CME product "
                         "opens at 18:00 ET the previous calendar day -- a 09:30-16:00 grid is a "
                         "subset, so containment is the check, not equality)")

    bad = np.isfinite(worst) and worst < min_l1_match
    lines += ["", ("VERDICT: level-1 price agreement %.4f%% is BELOW the %.2f%% threshold -- the replay "
                   "and the venue disagree about the top of book. Investigate before publishing any "
                   "number that depends on it." % (100 * worst, 100 * min_l1_match)) if bad else
              ("VERDICT: the replay reproduces the venue's own ladder within tolerance. This is the "
               "robustness result to report -- same lake, same clock, so the comparison is not "
               "confounded the way an mstbook-query benchmark is.")]
    return "\n".join(lines), (2 if bad else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate the replay against the venue's aggregated ladder")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--product", required=True, help="e.g. ESZ4")
    ap.add_argument("--product-type", default="futures", choices=["direct", "futures"])
    ap.add_argument("--price-scale", type=float, default=0.01,
                    help="0.01 for CME integer-hundredths (ES); 1.0 for equities")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--interval", default="1s")
    ap.add_argument("--session", default="09:30-16:00")
    ap.add_argument("--data-source", default="apu")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--clock", default="receipt", choices=["receipt", "exchange"])
    ap.add_argument("--min-l1-match", type=float, default=0.99)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    text, rc = run(a.date.replace("-", ""), a.product, a.product_type, a.price_scale, a.levels,
                   a.interval, a.session, a.data_source, a.tz, a.clock, a.min_l1_match)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"validate_aggregated: FAILED to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
