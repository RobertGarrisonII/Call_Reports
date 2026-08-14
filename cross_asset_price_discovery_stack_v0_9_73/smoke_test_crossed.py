#!/usr/bin/env python3
"""smoke_test_crossed.py -- audit reconstructed books for the invariant violation found on Liberation
Day: a book whose best bid >= best ask (crossed/locked). A matching engine CANNOT cross, so a crossed
reconstruction is a hard bug -- resting orders that should have left the book are pinning the top
(traced on 04-03 to the trade-consumption fallback: trades without a resting-order reference reduce the
level on the trade's side, which strands the resting order if that side is the aggressor's).

For each date it reconstructs SPY (consolidated multi-venue hybrid MBO+MBP) and ES (single-venue CME
order-by-order MBO) with the SAME parameters the contagion pipeline uses (mstbook_loader._extract_one_
session), then reports per (date, asset):

  crossed_frac       share of 1s grid points with best_bid >  best_ask  -- THE invariant (0 if correct)
  locked_frac        share with best_bid == best_ask                    -- also impossible for one venue
  trade_no_ref_rate  trades whose print carried no resting ref / trades -- the suspected mechanism
  open_frozen        is {ASSET}_mid constant over 09:30:30-09:35:00      -- the auction-linkage symptom
  mid_distinct_frac  distinct mids / grid points over the whole day      -- low => degenerate mid
  cancel_no_order, resting_eod, events, median_mid                       -- sanity / scale

Two questions this answers: (1) is the crossing UNIVERSAL across dates or a Liberation-Day aberration,
and (2) -- the high-stakes one -- is SPY compromised too, or is the bug ES-only? It also gives the
pre-fix BASELINE the fix must drive to ~0 (so a clean post-fix re-run proves a fix, not a mask).

Run on the workbench from the package dir (re-queries lakequery per date/asset). One asset/day at a
time, so peak RAM is a single reconstruction. Crash days take a few minutes; the full 12 x 2 ~ 1-2 h.

  python smoke_test_crossed.py                          # the 12-date window, SPY+ES, 1s  (the default)
  python smoke_test_crossed.py 2025-04-03 2025-04-04    # an explicit subset
  python smoke_test_crossed.py --assets ES              # ES only
  python smoke_test_crossed.py --interval 100ms --out audit_100ms.csv
"""
import argparse
import time
import numpy as np
import pandas as pd

DEFAULT_DATES = ["2025-03-24", "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31",
                 "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07", "2025-04-08"]


def audit_book(frame: pd.DataFrame, prefix: str) -> dict:
    """Compute the crossed/degeneracy metrics from a reconstructed frame + its df.attrs['lob_stats'].
    Pure function of the frame -- unit-testable without lakequery."""
    stats = dict(frame.attrs.get("lob_stats", {}))
    n_grid = len(frame)
    rec = {"n_grid": n_grid, "events": int(stats.get("events", 0)),
           "resting_eod": frame.attrs.get("resting_orders_eod", np.nan)}
    bp, ap, mid = (frame.get(f"{prefix}_bidprice_1"), frame.get(f"{prefix}_askprice_1"),
                   frame.get(f"{prefix}_mid"))
    if bp is None or ap is None or mid is None:
        rec["error"] = f"missing {prefix}_bidprice_1/_askprice_1/_mid (have: {list(frame.columns)[:6]}...)"
        return rec
    bp, ap = bp.to_numpy(float), ap.to_numpy(float)
    both = np.isfinite(bp) & np.isfinite(ap)
    n2 = int(both.sum())
    rec["crossed_frac"] = float(np.mean(bp[both] > ap[both])) if n2 else float("nan")   # bid > ask: impossible
    rec["locked_frac"] = float(np.mean(bp[both] == ap[both])) if n2 else float("nan")   # bid == ask
    rec["stat_crossed_frac"] = (stats.get("consolidated_crossed", 0) / n_grid) if n_grid else float("nan")
    nt = int(stats.get("n_trade", 0))
    rec["trade_no_ref"] = int(stats.get("trade_no_ref", 0))
    rec["trade_no_ref_rate"] = (stats.get("trade_no_ref", 0) / nt) if nt else float("nan")
    rec["cancel_no_order"] = int(stats.get("cancel_no_order", 0))
    m = mid.dropna()
    rec["mid_distinct_frac"] = (m.nunique() / len(m)) if len(m) else float("nan")
    rec["median_mid"] = float(m.median()) if len(m) else float("nan")
    win = mid.between_time("09:30:30", "09:35:00").dropna()                              # the auction symptom
    rec["open_frozen"] = bool(win.nunique() <= 1) if len(win) else None
    return rec


def reconstruct_one(date: str, asset: str, a) -> tuple:
    """Reconstruct one (date, asset) standalone -- NOT via the joined session frame, because the join
    drops df.attrs (lob_stats / resting_orders_eod). Mirrors mstbook_loader._extract_one_session."""
    import lob_reconstruct as lob
    import mstbook_loader as ml
    ymd = date.replace("-", "")
    if asset == "ES":
        contract = ml.get_front_month_contract(a.es_symbol, as_of_date=ml._parse_yyyymmdd(ymd),
                                                rollover_days=a.rollover_days)
        frame = lob.reconstruct_session(
            ymd, contract, "futures", levels=a.levels, interval=a.interval, round_lot=a.round_lot,
            odd_lot_inclusive=True, session=(a.start_time, a.end_time), tz=a.tz,
            data_source=a.data_source, clock=a.clock, price_scale=a.futures_scale,
            message_types=("mt_add_order", "mt_cancel_order", "mt_modify_order", "mt_trade"))
        prefix = ml.canonical_root(contract, "futures")
    else:                                                # SPY: default message_types (hybrid MBO+MBP)
        frame = lob.reconstruct_session(
            ymd, "SPY", "direct", levels=a.levels, interval=a.interval, round_lot=a.round_lot,
            odd_lot_inclusive=True, session=(a.start_time, a.end_time), tz=a.tz,
            data_source=a.data_source, clock=a.clock)
        prefix = "SPY"
    return frame, prefix


def _verdict(df: pd.DataFrame) -> None:
    if "crossed_frac" not in df.columns:
        print("no successful reconstructions to summarize."); return
    print("\nPER-ASSET SUMMARY")
    for asset in sorted(df["asset"].dropna().unique()):
        d = df[(df["asset"] == asset) & df["crossed_frac"].notna()]
        if d.empty:
            continue
        n = len(d); n_x = int((d["crossed_frac"] > 0.5).sum())
        print(f"  {asset}: {n_x}/{n} dates >50% crossed | crossed_frac mean={d['crossed_frac'].mean():.3f} "
              f"max={d['crossed_frac'].max():.3f} min={d['crossed_frac'].min():.3f} | "
              f"trade_no_ref_rate mean={d['trade_no_ref_rate'].mean():.3f} | "
              f"open_frozen {int(d['open_frozen'].fillna(False).sum())}/{n}")
    print("\nVERDICT")
    for asset, label_bad, label_ok in (("ES", "UNIVERSAL (not a Liberation-Day aberration)", "conditional"),
                                       ("SPY", "COMPROMISED too -- the consolidated SPY book crosses, so "
                                               "the WHOLE study is on bad data", "CLEAN -- bug is ES-only; "
                                               "the SPY consolidated book is sound")):
        d = df[(df["asset"] == asset) & df["crossed_frac"].notna()]
        if d.empty:
            print(f"  {asset}: no data"); continue
        frac_bad = float((d["crossed_frac"] > 0.5).mean())
        thr = 0.5 if asset == "ES" else 0.1                  # SPY: even a little crossing implicates the study
        print(f"  {asset}: crossed on {100*frac_bad:.0f}% of dates -> {label_bad if frac_bad > thr else label_ok}")


def main() -> None:
    p = argparse.ArgumentParser(description="Audit reconstructed books for crossed/locked tops (the "
                                            "Liberation-Day invariant violation).")
    p.add_argument("dates", nargs="*", help="dates YYYY-MM-DD (default: the 12-date Liberation-Day window)")
    p.add_argument("--assets", default="both", choices=["both", "ES", "SPY"])
    p.add_argument("--interval", default="1s")
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--round-lot", type=int, default=100, dest="round_lot")
    p.add_argument("--futures-scale", type=float, default=0.01, dest="futures_scale")
    p.add_argument("--clock", default="receipt")
    p.add_argument("--tz", default="America/New_York")
    p.add_argument("--data-source", default="apu", dest="data_source")
    p.add_argument("--start-time", default="9:30", dest="start_time")
    p.add_argument("--end-time", default="16:00", dest="end_time")
    p.add_argument("--es-symbol", default="ES", dest="es_symbol")
    p.add_argument("--rollover-days", type=int, default=8, dest="rollover_days")
    p.add_argument("--out", default="crossed_book_audit.csv")
    a = p.parse_args()

    dates = a.dates or DEFAULT_DATES
    assets = ["SPY", "ES"] if a.assets == "both" else [a.assets]
    print(f">>> crossed-book audit: {len(dates)} date(s) x {assets} @ {a.interval} "
          f"(clock={a.clock}); reconstructing one asset/day at a time")
    rows = []
    for date in dates:
        for asset in assets:
            t0 = time.time()
            try:
                frame, prefix = reconstruct_one(date, asset, a)
                if frame is None or len(frame) == 0:
                    rows.append({"date": date, "asset": asset, "error": "empty reconstruction"})
                    print(f"  {date} {asset}: EMPTY", flush=True); continue
                rec = audit_book(frame, prefix)
                rec.update({"date": date, "asset": asset, "secs": round(time.time() - t0, 1)})
                rows.append(rec)
                if "error" in rec:
                    print(f"  {date} {asset}: {rec['error']}", flush=True)
                else:
                    print(f"  {date} {asset}: crossed={rec['crossed_frac']:.3f} locked={rec['locked_frac']:.3f} "
                          f"trade_no_ref_rate={rec['trade_no_ref_rate']:.3f} open_frozen={rec['open_frozen']} "
                          f"resting_eod={rec['resting_eod']} ({rec['secs']}s)", flush=True)
                del frame
            except Exception as exc:                         # one bad date must not sink the sweep
                rows.append({"date": date, "asset": asset, "error": repr(exc)})
                print(f"  {date} {asset}: ERROR {exc!r}", flush=True)

    df = pd.DataFrame(rows)
    cols = ["date", "asset", "crossed_frac", "locked_frac", "stat_crossed_frac", "trade_no_ref_rate",
            "trade_no_ref", "open_frozen", "mid_distinct_frac", "cancel_no_order", "resting_eod",
            "events", "median_mid", "n_grid", "secs", "error"]
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df.to_csv(a.out, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\n" + "=" * 110 + "\n" + df.to_string(index=False) + "\n" + "=" * 110)
    _verdict(df)
    print(f"\nwrote {a.out}")


# ── self-test of the metric logic (no lakequery): a crossed book and a clean book audit correctly ──
def _selftest() -> bool:
    idx = pd.date_range("2025-04-03 09:30:00", "2025-04-03 09:36:00", freq="1s", tz="America/New_York")
    n = len(idx)
    crossed = pd.DataFrame({"ES_bidprice_1": np.full(n, 5560.75), "ES_askprice_1": np.full(n, 5507.25),
                            "ES_mid": np.full(n, (5560.75 + 5507.25) / 2)}, index=idx)
    crossed.attrs["lob_stats"] = {"events": 1000, "n_trade": 400, "trade_no_ref": 360,
                                  "cancel_no_order": 5, "consolidated_crossed": n}
    crossed.attrs["resting_orders_eod"] = 25092
    rc = audit_book(crossed, "ES")
    ok_crossed = (abs(rc["crossed_frac"] - 1.0) < 1e-9 and rc["open_frozen"] is True
                  and abs(rc["trade_no_ref_rate"] - 0.9) < 1e-9 and rc["mid_distinct_frac"] < 0.01
                  and abs(rc["stat_crossed_frac"] - 1.0) < 1e-9)

    drift = 5507.25 + 0.25 * np.arange(n)                     # a clean, moving, uncrossed book
    clean = pd.DataFrame({"ES_bidprice_1": drift - 0.25, "ES_askprice_1": drift + 0.25,
                          "ES_mid": drift}, index=idx)
    clean.attrs["lob_stats"] = {"events": 1000, "n_trade": 400, "trade_no_ref": 0, "consolidated_crossed": 0}
    clean.attrs["resting_orders_eod"] = 20000
    cl = audit_book(clean, "ES")
    ok_clean = (cl["crossed_frac"] == 0.0 and cl["open_frozen"] is False
                and cl["trade_no_ref_rate"] == 0.0 and cl["mid_distinct_frac"] > 0.9)

    print("smoke_test_crossed self-test")
    print(f"  crossed book: crossed_frac={rc['crossed_frac']:.3f} open_frozen={rc['open_frozen']} "
          f"trade_no_ref_rate={rc['trade_no_ref_rate']:.3f} -> {ok_crossed}")
    print(f"  clean   book: crossed_frac={cl['crossed_frac']:.3f} open_frozen={cl['open_frozen']} "
          f"mid_distinct_frac={cl['mid_distinct_frac']:.3f} -> {ok_clean}")
    ok = ok_crossed and ok_clean
    print("checks:", ok)
    return ok


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    main()
