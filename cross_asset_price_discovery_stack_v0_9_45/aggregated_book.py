"""aggregated_book.py -- the ES leg from CME's own published ladder, on every date in the sample.

Why this rather than a replay
-----------------------------
The futures leg had been reconstructed from raw messages, the same way as SPY. That is the right
choice for equities -- a consolidated NBBO across sixteen venues does not exist anywhere as a
published object, so it has to be built. It is the wrong choice for ES, for three reasons that only
became clear once the 2020 capture was examined:

  * **CME is one venue, and its ladder IS the book.** There is nothing to consolidate. Replaying
    messages to rebuild a ladder the venue already publishes, on the same lake and the same capture
    clock, adds a failure mode and buys nothing.
  * **The message families change shape between eras.** 2024 ES is order-by-order; 2020 ES is
    price-level via `mt_modify_price_level` / `mt_delete_price_level`; and `mt_price_level_update`
    is empty in both. Two separate hardcoded assumptions about "what CME publishes" were wrong, one
    of them silently emptying four sessions. `mt_aggregated_price_update` is populated in EVERY era
    checked, so the ES leg stops depending on which family the vendor happened to capture.
  * **One mechanism across the panel.** A paper whose sample spans 2020-2026 should not build its
    futures book two different ways depending on the year. The aggregated path is identical on every
    date; the replay is not.

What is given up, stated plainly
--------------------------------
The venue ladder is CME's aggregated depth, so the ES leg carries price and size per level but not
order-level detail: no per-order queue position, no order counts beyond what the ladder publishes,
and depth limited to the levels CME disseminates (10). Nothing in the paper's ES measurements needs
more than that -- they are top-of-book and 10-level depth on a 1-second grid -- but a future
question about futures queue dynamics would need the replay back, which is why
`lob_reconstruct.reconstruct_session(..., select_family=True)` is kept and still tested.

The two agree where both exist
------------------------------
This is not a leap of faith. `validate_aggregated.py` compares the MBO replay against this same
ladder on 2024-12-18 and reports level-1 price agreement; the replication driver runs it as a gate.
So the switch is to the benchmark that the replay was already being measured against, not to an
unvalidated source.

Halts and staleness
-------------------
The ladder is a step function: the venue's last published state stands until it changes. During a
halt it therefore goes stale rather than empty, which is correct -- resting orders do not vanish
when matching stops -- and is the same behaviour the replay has. Halt snapshots are excluded by the
halt-window machinery (`market_halts`), not by pretending the book disappeared.
"""
from __future__ import annotations

from typing import Optional

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

AGG = "mt_aggregated_price_update"


def canonical_from_aggregated(agg: pd.DataFrame, asset: str, levels: int = 10,
                              price_scale: float = 1.0) -> pd.DataFrame:
    """Ladder rows -> canonical columns. Delegates the column mapping to validate_aggregated so the
    BENCHMARK and the BOOK cannot drift apart -- if they parsed the ladder differently, the
    validation would be comparing the replay against something the extractor never builds."""
    import validate_aggregated as va
    return va.aggregated_to_canonical(agg, asset, levels=levels, price_scale=price_scale)


def sample_on_grid(canon: pd.DataFrame, grid: pd.DatetimeIndex, asset: str,
                   levels: int = 10) -> pd.DataFrame:
    """As-of sample: for each grid point take the LAST ladder row at or before it.

    Forward only. A grid point that precedes every ladder row gets NaN, not the first row's values --
    that would be look-ahead, and it is the difference between "the book was not published yet" and
    "the book looked like this". The pre-open rows are exactly why the fetch is not restricted to
    the session: ES publishes from ~17:45 ET the previous day, so the ladder is already warm at
    09:30 and the first snapshot is a real book rather than an empty one filling up."""
    out = pd.DataFrame(index=grid)
    if canon is None or len(canon) == 0:
        for i in range(1, levels + 1):
            for tag in ("bid", "ask"):
                out[f"{asset}_{tag}price_{i}"] = np.nan
                out[f"{asset}_{tag}quantity_{i}"] = np.nan
        out[f"{asset}_nbbo_bid"] = np.nan
        out[f"{asset}_nbbo_ask"] = np.nan
        out[f"{asset}_mid"] = np.nan
        return out

    src = canon.sort_index()
    pos = np.searchsorted(src.index.values, grid.values, side="right") - 1
    take = np.clip(pos, 0, None)
    unknown = pos < 0
    for c in src.columns:
        v = pd.to_numeric(src[c], errors="coerce").to_numpy(float)[take]
        out[c] = np.where(unknown, np.nan, v)

    # A level the venue never published is absent, not zero depth -- but the replay emits 0.0 for a
    # level that does not exist, and downstream depth math sums these columns. Match the replay so
    # the two book sources are interchangeable: quantity 0.0 exactly where the price is NaN.
    for i in range(1, levels + 1):
        for tag in ("bid", "ask"):
            pc, qc = f"{asset}_{tag}price_{i}", f"{asset}_{tag}quantity_{i}"
            if pc in out.columns and qc in out.columns:
                out[qc] = np.where(~np.isfinite(out[pc].to_numpy(float)), 0.0,
                                   out[qc].to_numpy(float))

    # CME has no odd lots and no consolidation, so the round-lot NBBO IS the top of the ladder.
    # The columns exist so the ES frame has the same schema as the SPY one.
    out[f"{asset}_nbbo_bid"] = out[f"{asset}_bidprice_1"]
    out[f"{asset}_nbbo_ask"] = out[f"{asset}_askprice_1"]
    out[f"{asset}_mid"] = (out[f"{asset}_bidprice_1"] + out[f"{asset}_askprice_1"]) / 2.0
    return out


def session_from_aggregated(date_str: str, product: str, product_type: str = "futures",
                            levels: int = 10, interval: str = "1s",
                            session=("09:30", "16:00"), tz: str = "America/New_York",
                            data_source: str = "apu", clock: str = "receipt",
                            price_scale: float = 0.01, asset: Optional[str] = None,
                            with_activity: bool = True, strict: bool = True,
                            progress_cb=None) -> pd.DataFrame:
    """Build one session's ES book from the venue ladder. Same schema as ``reconstruct_session``.

    ``with_activity`` also fetches ``mt_trade`` -- not for the book, which does not need it, but for
    ``attrs['activity_seconds']``: the instants at which the product traded are the only reliable end
    for a CME halt window, because the status flag marks the stop and then clears while the market
    stays down (see market_halts). Pass False to skip it where the halt boundary is not needed.

    ``strict`` refuses a session whose ladder never yields a finite top, for the same reason the
    replay does: a full-length all-NaN leg is worse than an error, because nothing downstream
    notices it."""
    import mstbook_loader as ml
    a = asset or ml.canonical_root(product, product_type)
    if progress_cb is not None:
        progress_cb(f"{product} {AGG}")
    agg = ml._fetch_messages(date_str, product, product_type, AGG, data_source, tz=tz, clock=clock)
    n_agg = int(len(agg))

    d = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
    grid = pd.date_range(pd.Timestamp(f"{d.date()} {session[0]}", tz=tz),
                         pd.Timestamp(f"{d.date()} {session[1]}", tz=tz), freq=interval)
    canon = canonical_from_aggregated(agg, a, levels=levels, price_scale=price_scale) if n_agg else None
    out = sample_on_grid(canon, grid, a, levels=levels)

    act = []
    if with_activity:
        try:
            if progress_cb is not None:
                progress_cb(f"{product} mt_trade")
            tr = ml._fetch_messages(date_str, product, product_type, "mt_trade", data_source,
                                    tz=tz, clock=clock)
            if len(tr):
                ti = tr.index
                ti = pd.DatetimeIndex(ti).tz_localize(tz) if getattr(ti, "tz", None) is None else ti
                act = pd.DatetimeIndex(ti).floor("s").unique().sort_values()
        except Exception as exc:          # the halt boundary degrades; the book does not
            log.warning("%s %s: trade tape for halt boundaries unavailable (%s); halt windows will "
                        "fall back to the status flag, which UNDERSTATES a CME stop by ~130x",
                        date_str, product, str(exc).splitlines()[0][:120])

    bid = pd.to_numeric(out.get(f"{a}_bidprice_1"), errors="coerce")
    ask = pd.to_numeric(out.get(f"{a}_askprice_1"), errors="coerce")
    both = bid.notna() & ask.notna()
    n_top = int(both.sum())
    crossed = int((bid > ask + 1e-9).sum())
    out.attrs["book_source"] = "aggregated"
    out.attrs["book_family"] = "LADDER"
    out.attrs["message_counts"] = {AGG: n_agg}
    out.attrs["activity_seconds"] = list(act)
    out.attrs["lob_stats"] = {"source": AGG, "n_ladder_rows": n_agg, "n_grid": int(len(grid)),
                              "n_finite_top": n_top, "crossed": crossed,
                              "warm_at_open": bool(both.iloc[0]) if len(both) else False}
    if n_top == 0 and strict:
        raise ml.MessageFetchError(
            "%s %s: %s returned %d row(s) and the ladder yields a finite top on 0 of %d snapshots -- "
            "the leg is MISSING, not thin. Returning it would put a full-length all-NaN column into "
            "the dataset. Check the contract code and the product type before falling back to the "
            "message replay (lob_reconstruct.reconstruct_session(..., select_family=True))."
            % (date_str, product, AGG, n_agg, len(grid)))
    if crossed:
        # The venue's own ladder should never cross. If it does, it is CME's publication during a
        # halt or an auction, not a replay fault -- there is no replay here. Worth seeing, not fatal.
        log.info("%s %s: the VENUE ladder is crossed on %d of %d snapshot(s) -- not a replay fault "
                 "(there is no replay on this leg); check against the halt windows.",
                 date_str, product, crossed, n_top)
    log.info("%s %s: book from %s -- %d ladder row(s) -> %d/%d snapshots with a finite top%s",
             date_str, product, AGG, n_agg, n_top, len(grid),
             "" if (len(both) and both.iloc[0]) else " (NOT warm at the open: the pre-open ladder "
             "rows are missing, so the first snapshots are unknown rather than stale)")
    return out


def _selftest() -> bool:
    TZ = "America/New_York"
    ok = []
    # a ladder that starts in the pre-open and changes twice during the session
    idx = pd.DatetimeIndex(["2020-03-12 08:00:00", "2020-03-12 09:30:30", "2020-03-12 09:31:10"]
                           ).tz_localize(TZ)
    agg = pd.DataFrame({"bidprice_1": [254600, 254650, 254700], "bidquantity_1": [10, 40, 20],
                        "askprice_1": [254700, 254700, 254750], "askquantity_1": [15, 30, 25],
                        "bidprice_2": [254550, 254600, 254650], "bidquantity_2": [5, 8, 9],
                        "askprice_2": [254750, 254750, 254800], "askquantity_2": [6, 7, 8]},
                       index=idx)
    canon = canonical_from_aggregated(agg, "ES", levels=2, price_scale=0.01)
    grid = pd.date_range("2020-03-12 09:30:00", "2020-03-12 09:32:00", freq="30s", tz=TZ)
    out = sample_on_grid(canon, grid, "ES", levels=2)

    a = abs(float(out["ES_bidprice_1"].iloc[0]) - 2546.00) < 1e-9        # warm from the pre-open row
    print("(1) the 08:00 pre-open row makes 09:30 a real book (%.2f), not an empty one : %s"
          % (out["ES_bidprice_1"].iloc[0], a))
    ok.append(a)

    b = (abs(float(out["ES_bidprice_1"].loc[grid[1]]) - 2546.50) < 1e-9
         and abs(float(out["ES_bidprice_1"].loc[grid[2]]) - 2546.50) < 1e-9
         and abs(float(out["ES_bidprice_1"].loc[grid[3]]) - 2547.00) < 1e-9)
    print("(2) step function, forward only: 09:30:30 -> 2546.50 held at 09:31:00, 09:31:30 -> "
          "2547.00 : %s" % b)
    ok.append(b)

    c = abs(float(out["ES_mid"].iloc[0]) - (2546.00 + 2547.00) / 2) < 1e-9
    print("(3) mid derived from the top of the ladder : %s" % c)
    ok.append(c)

    d = (float(out["ES_nbbo_bid"].iloc[0]) == float(out["ES_bidprice_1"].iloc[0])
         and float(out["ES_nbbo_ask"].iloc[0]) == float(out["ES_askprice_1"].iloc[0]))
    print("(4) CME has no odd lots and nothing to consolidate, so nbbo == top of ladder : %s" % d)
    ok.append(d)

    # a grid point BEFORE any ladder row must be NaN, never the first row (look-ahead)
    early = pd.date_range("2020-03-12 07:00:00", "2020-03-12 07:01:00", freq="30s", tz=TZ)
    e = not np.isfinite(sample_on_grid(canon, early, "ES", levels=2)["ES_bidprice_1"].to_numpy(float)).any()
    print("(5) a grid point preceding every ladder row is NaN, not back-filled : %s" % e)
    ok.append(e)

    # an absent level: price NaN, quantity 0.0 -- matching the replay so depth math agrees
    agg2 = agg.drop(columns=["bidprice_2", "bidquantity_2"])
    o2 = sample_on_grid(canonical_from_aggregated(agg2, "ES", levels=2, price_scale=0.01),
                        grid, "ES", levels=2)
    f = (not np.isfinite(o2["ES_bidprice_2"].to_numpy(float)).any()
         and float(o2["ES_bidquantity_2"].iloc[0]) == 0.0)
    print("(6) a level the venue never published: price NaN, quantity 0.0 (as the replay emits) : %s" % f)
    ok.append(f)

    print("\naggregated-book checks -> %s" % all(ok))
    return all(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
