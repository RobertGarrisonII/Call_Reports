# What the SPY tape says, and what the replay was ignoring

A sample pull of **every** `mt_*` type for SPY on 2024-12-18 (`mstwx-lakequery --print-headers
--limit 20`) settled several questions that had been inferred from symptoms. The reconstruction was
consuming five message types and taking its semantics for two of them from assumptions the tape
contradicts. Three of the omissions change the book.

Read the timestamps carefully: `1734512400` is **04:00 ET**, not 09:30. The pull starts at the
pre-market open, and the replay processes every message from the first one of the day — only the
snapshot **grid** is restricted to 09:30–16:00. Pre-market state is therefore carried into the
session, which is what makes finding 1 matter.

---

## 1. `mt_clear_orders` / `mt_clear_price_levels` — feed resets, never fetched

```
mt_clear_orders,1734505200018350611,3396,...,xdp_nyse_integrated,...        02:00:00 ET
mt_clear_orders,1734517544663671426,5,...,miax_pearl_equities_dom,...       05:25:44 ET
mt_clear_orders,1734574666870010018,247476821,...,memoir_depth,...          21:17:46 ET
```

A clear is the venue saying *discard everything I have told you and rebuild from my next message*.
It is issued on a line failover, a gap recovery, or a session-state transition, and it carries no
price or quantity — the point is that the prior state is void.

**Not applying one is unrecoverable.** The venue never cancels the orders it just disowned, because
as far as it is concerned they no longer exist. Every one of them rests in the consolidated ladder
for the remainder of the session while the venue rebuilds its book under fresh reference numbers.
A stale pre-market bid pinned that way sits above the current ask for hours — precisely "resting
orders that should have left are pinning the top."

The 02:00–02:04 clears land on an empty book and are routine session init. The **05:25:44
miax_pearl_equities_dom** one does not: it is 85 minutes into the pre-market, with the replay
already holding that feed's state.

`test_feed_reset.py` demonstrates the mechanism end to end: with the reset ignored, a stale 605.50
bid pins the top against a 604.90 ask and the book is **100% crossed**; with it applied the same
messages give a clean 604.80 / 604.90 and other venues are untouched.

Fixed: both types are fetched and replayed. Within a sequence tie a reset ranks **first**, not last
— the venue clears and then rebuilds, so a clear tied with adds in the same packet must precede
them, or it would wipe the very re-adds it exists to make room for.

## 2. `mt_trade.leavesquantity` — the venue's own post-trade size, pruned away

```
605,2,...,3189381831070546706,1198,Ask     # trade 2  -> 1198 remain
605,4,...,3189381831070546706,1194,Ask     # trade 4  -> 1194
605,2,...,3189381831070546706,1192,Ask     # trade 2  -> 1192
```

The arithmetic is exact, so the semantics are not in doubt: `leavesquantity` is what is left on the
**resting** order after the execution. It is authoritative where a decrement is merely arithmetic —
immune to a missed earlier partial, a duplicate print, or a size mis-parsed on the add. The case
that matters is `leaves = 0`: a fully filled order that is not removed rests forever and pins the
top.

It was not in `_MSG_NEEDED_COLS`, so `usecols` dropped it at CSV read — **the same failure mode as
the original `sequencenumber` bug**, in the same function. Now read, and the replay assigns the
venue's figure rather than decrementing (a corrected drift is counted in
`lob_stats["trade_leaves_corrected"]`).

This also settles the `side` question. The tape has a **separate** `aggressorside` column (blank for
equities — the stack already falls back to the tick rule), while `side` accompanies
`orderreferencenumber` and `leavesquantity` and describes the **resting** order. The three trades
above are `side=Ask` at 605.00 with the bid at 604.87, so the resting order is genuinely an ask.

## 3. Refless trades are mostly *hidden liquidity*, and were being treated as faults

```
605.02,7,...,RegularTrade,,,,Printable,,,,...,EarlyOpening,...   # no orderreferencenumber
605.02,1,...,OddLotTrade,NonPrintable,,,...,Hidden,...           # no orderreferencenumber
604.54,4,...,OddLotTrade,Printable,3189381831070637292,,,Hidden  # ref, no side/leaves
```

An execution against non-displayed liquidity has **no order reference by construction** — there is
no displayed resting order to reference. The tape marks it: `executionattribute='Hidden'` and/or
`printable='NonPrintable'`.

Two consequences, both bad:

* The diagnostic counted these in `trade_no_ref`, so the reported "11–17% of trades have no
  reference" on the MWCB days conflates an expected property of the SPY tape with a
  reference-matching fault. The number that indicates a fault is the **displayed** refless rate.
* Worse, the fallback consumed displayed size at the print's price. A hidden execution at a price
  that is also displayed then deletes liquidity that is still resting.

Fixed: `executionattribute` is read (it was pruned too), undisplayed prints consume nothing, and
the two rates are counted and reported separately.

## 4. `mt_missing_product_messages` and `mt_error` — the oracle nobody was asking

```
mtype,receipttimestamp,sequencenumber,dt,f,marketparticipant,product,
      expectedproductsequencenumber,currentproductsequencenumber
(no rows for 2024-12-18)
```

This is the venue's own gap report: expected vs received sequence number, i.e. the multicast packets
that never arrived. Missing packets mean missing **adds**, so the cancels and trades referencing
them are orphans and the levels they should have removed rest forever — a crossed book caused by
data loss, which no code change can fix.

Empty on 2024-12-18 (a clean capture). `feed_health.py` runs it, `mt_error`, and the reset inventory
for any product-day and returns a verdict: **capture incomplete → DATA**, or **capture complete →
the replay's fault, fixable in code**. This is the decisive test for the residual ~3.9% crossing on
the four MWCB days, and it costs one query per day:

```bash
python feed_health.py --date 20200309 --product SPY
```

STAGE 3 now runs it automatically on any session the gate flags, before `debug_crossing`.

## 5. The gate was diagnosing a different book than the one it saved

`debug_crossing` was invoked with `--clock exchange` while extraction runs on `receipt`. Two
problems: the root cause then describes a book that was never written to disk, and the tape shows
why exchange is the weaker choice for this feed — many consecutive `bats_edgx` messages carry the
**same** `exchangetimestamp` (`1734512400000047000`) while their receipt timestamps differ by
milliseconds, so exchange-clock ordering is degenerate across those bursts. Now `--clock receipt`,
matching the extraction.

---

## Confirmed correct, no change needed

* **`aggressorside` is blank for equities** and populated for CME futures — the stack's documented
  tick-rule fallback for equities is right.
* **Per-feed sequence namespaces.** `total_view` 282783, `bats_edgx` 5163, `xdp_arca` 7634 at the
  same instant. Ordering keyed on `(feed, sequencenumber)` is correct; a global sort would be
  meaningless.
* **`total_view` re-IDs on modify** (`orderreferencenumber=4616` from `previousorderreferencenumber=24`),
  `bats_edgx` keeps the reference. Both handled.
* **MBP price-level updates key on price, not `level`.** `iex_deep` sends `level` empty and
  `numberoforders=0`, so anything keyed on `level` would fail; the stack keys on price. `quantity=0`
  as a level delete is handled.
* **`mt_bbo_quote` / `mt_nbbo_quote` are empty for SPY**, and `mt_aggregated_price_update` returns
  `\N` for the early rows. The stack builds its own NBBO from the message stream and depends on
  none of them — the right call, and now demonstrably so.
* **`mt_trade_break` / `mt_trade_correction`** are empty for this date; both are already consumed by
  the trade scrubber.

## Still unfetched, deliberately

`mt_retail_price_improvement` and `mt_index_update`. Neither affects the book.

---

# The ES side (ESZ4, same date)

The futures tape is a different shape, and it settles the ES leg's design.

## 6. CME publishes **no** price-level types — the MBO-only path is necessary, not a choice

`mt_price_level_update`, `mt_modify_price_level`, `mt_delete_price_level` all return **header only**
for ESZ4. So does `mt_bbo_quote`, `mt_nbbo_quote` and `mt_order_imbalance`. The stack already
assumed this (`_extract_one_session` passes an MBO-only message list for ES and the docstring says
the price-level types are empty for futures) — it is now verified rather than asserted, and
`test_validate_aggregated.py` pins that an ES replay still builds correctly with every price-level
frame empty.

Two consequences worth stating in the paper: the ES book has no MBP supplement to fall back on if
the MBO stream is incomplete, and **any auction-imbalance feature is equity-only** — `auction_imbalance`
will silently produce nothing for ES because the venue sends no imbalance messages.

## 7. `mt_aggregated_price_update` **is** populated for ES — a benchmark with no clock confound

```
mt_aggregated_price_update,1734475500134856451,...,cme_globex30_cme,...,ESZ4,One,None,One,false,true,
  605175,9,4, 605150,12,5, 605125,11,2, ...   |  605225,2,1, 605250,11,4, ...
```

CME's own 10-level ladder — price, quantity **and order count** per level — delivered through the
same lake, on the same capture clock as the add/cancel/modify/trade messages the replay consumes.
Single feed (`cme_globex30_cme`), fully populated from the pre-open onward.

This matters because "how do you know the reconstruction is right?" previously had a weak answer.
`validate_against_snapshot` compares against `mstbook-query`, which was demoted *precisely* because
it sits on a different clock — so any disagreement is confounded and neither direction proves
anything. This benchmark has no such defect: disagreement localizes to the replay.

`validate_aggregated.py` runs the comparison level by level (as-of aligning the event-stamped venue
ladder onto the reconstruction's grid), and STAGE 3 runs it automatically on the first volatile
session of an extract run. **This is the robustness table the paper needs**, and it is now a
one-line command:

```bash
python validate_aggregated.py --date 20241218 --product ESH5 --product-type futures --price-scale 0.01
```

Note the contract: `get_front_month_contract` returns **ESH5** for 2024-12-18, not the ESZ4 you
queried, because the December contract expires 2024-12-20 and the 8-day rollover has already moved
the front month to March. That is the right choice for a liquidity-based price-discovery study — but
validate the contract the pipeline actually uses, or the comparison is against a book nobody built.

Prices confirm the scale convention: `605175` = 6051.75 index points, so `price_scale=0.01`, and
6051.75 × 10 ≈ the SPY level of ~604.87 that morning.

## 8. `mt_product_statistics` — a second validation axis, with a trap

For ES it is rich: `openingprice`, `highprice`, `lowprice`, `lasttrade`,
`volumeweightedaverageprice`, `volume`, `openinterest`, `settlementprice`,
`previousdaysettlementprice`, `indicativeopeningprice`. A replay can match the ladder tick for tick
and still have the wrong session boundaries; these catch that.

**The trap:** the rows are a running stream and the early ones carry the **previous** session. The
17:38 ET row on 2024-12-18 reports `settlementprice=608050` stamped `2024-12-15 19:00 ET`, and a
`highprice`/`lowprice` (6079.25 / 6040.75) that are last session's, not the FOMC day's. A reader
that takes the first value reports last week's numbers. `session_statistics()` takes the last
non-null value of each field, and `test_validate_aggregated.py` pins it.

## 9. The ES trade date starts at 18:00 ET the **previous calendar day**

The ESZ4 stream opens at 17:38–17:45 ET on Dec 17 (the Globex pre-open), with a
`mt_product_statistics` row at exactly 18:00:00.002 ET marking the session open. The replay ingests
all of it and only the snapshot **grid** is 09:30–16:00, which is what you want — the book is warm
at the open rather than being rebuilt from scratch at 09:30.

It also means the venue's session statistics span a window the RTH grid is a strict subset of, so
containment — not equality — is the correct check against a 09:30–16:00 reconstruction.
`validate_aggregated.py` says so in its output rather than leaving it to be misread.

## 10. Clean capture on both legs

`mt_missing_product_messages`, `mt_error`, `mt_clear_orders` and `mt_clear_price_levels` are all
**empty for ESZ4** on 2024-12-18 — no packet loss, no decoder errors, no Globex resets. Combined
with the same result on the SPY side, 2024-12-18 is a clean capture on both legs, which makes it a
good control day for the MWCB comparison.

The clear types exist in the futures schema even though they are empty here, so ES now fetches them
too: a Globex reset on a crash day would otherwise be invisible, and an empty query costs nothing
against a 10–25 minute session.

## 11. `mt_product_status` on the futures leg — three equity assumptions that do not hold

The ES status stream is not a smaller version of the SPY one. Every difference below produced a
plausible number rather than an error, which is why each needed the real rows to find.

**CME publishes price limits; the equity feeds do not.** `luldlowerlimit` / `luldupperlimit` are
empty on every row of all four SPY dates checked (2020-03-09/12/16, 2024-12-18) — the NMS bands come
from the SIP, not the direct venue feeds. They are **populated on every ES date checked**, in the
same integer-hundredths convention as CME prices:

    ESZ4 2024-12-18   563025 / 647725      ->  5630.25 / 6477.25 index points

So the futures leg has a price-band control today. Anything written about "the bands are absent" is
a statement about the equity feeds only.

**`9223372036.8547758070` is INT64_MAX, meaning "no limit this side".** It is finite when parsed, so
it silently became a price: the ES band width came out at 1.5e+10 bps. Values at or beyond 1e9 are
now NaN. On 2020-03-12 the sentinel sits on the **upper** limit for most of the session while the
**lower** ratchets down through the crash — 2594.00 → 2601.00 → 2546.50 → 2382.00 → 2190.00 →
2332.50 — i.e. the contract is limit-**down** constrained, one side only. That asymmetry is the
economics of the day, and it was being averaged into a nonsense width.

**`haltreason` is a status-reason field, not a halt flag.** On ESZ4 most of its non-null values are
routine session bookkeeping:

| value | rows | `tradingevent` |
|---|---|---|
| `GroupSchedule` | 18 | `NoEvent`, `NoCancel`, `ChangeOfTradingSessionResetStatistics` |
| `MarketEvent` | 19 | `NoEvent` |

Treated as halts these produced spans of 843 s, 5,914 s, 43,910 s and 20,055 s on days the future
never stopped trading. Halt reasons are therefore **whitelisted**, not blacklisted. The asymmetry is
deliberate: a false halt *excuses* crossing and can hide a replay fault, whereas a missed halt only
flags a session that turns out to be fine. Unrecognized values are reported in `unknown_reasons`.

**One venue is the whole market.** A market-wide equity halt stops every venue at once — on
2020-03-09 six feeds report within 30 ms — so a venue quorum is what stops one venue's long
regulatory status from excusing an hour of crossing on a book the other fifteen kept matching. CME
has no second venue, so a fixed quorum of two suppressed every futures halt. Quorum now caps at the
number of venues that publish status at all.

What those last two recover, on a day already in the volatile panel:

    2020-03-12  SPY   09:35:44 → 09:50:44   900.0 s   MarketWideCircuitBreakerLevel1
    2020-03-12  ESH0  09:36:45 → 09:36:51     6.4 s   SuspendedBySurveillance

A CME Velocity Logic pause, 61 s into the equity circuit-breaker halt. Velocity Logic pauses run
5–10 s by design, so the 30 s minimum-duration filter was discarding exactly the class of
cross-asset event this paper studies; the floor is now 1 s.

Consequence for the QC: each book is judged against **its own** leg's halt windows, and the pair
against their union. A CME pause cannot excuse a crossed SPY top — NYSE never stopped matching —
and an equity halt cannot excuse a crossed ES top.

`shortsaleindicator` is `\N` throughout on ES, as it should be: Rule 201 is an equity rule.

## 12. The March-2020 roll file — three things ESH0 vs ESM0 settles

`mt_product_status` for **both** contracts on 2020-03-16 and 2020-03-18.

**`sequencenumber` is a CHANNEL counter, not a per-product one — demonstrated, not inferred.** On
2020-03-16, all 69 of ESM0's sequence numbers are also ESH0's, at *identical receipt timestamps*,
carrying *different prices*:

    seq 1821  receipt=1584307805348763972  ESH0  exchange=1584307776613206983  limits 2567.50/2838.50
    seq 1821  receipt=1584307805348763972  ESM0  exchange=1584307757763848469  limits 2555.50/2826.50

One CME packet, several instruments. So a per-product fetch sees a **sparse subset** of a
channel-wide counter, and gaps in it are the other products rather than lost messages — which is
what `debug_crossing` CHECK 4 reports as `not-ours`, and what made pruning by `sequencenumber` the
original crossed-book bug. It also shows the `exchangetimestamp` is **per instrument** inside a
shared packet and can be **18.8 s older** than its packet-mates', so the receipt clock is the only
one that orders the packet itself.

**Velocity Logic pauses the ES GROUP, not a contract.** ESH0 and ESM0 stop at the same nanosecond
(09:30:54.973158575 on 2020-03-16). The halt window therefore does not depend on getting the front
month right — a roll ambiguity cannot silently move it.

**The futures halt too, about a minute later — they do not trade through.** On every MWCB day where
both status streams exist, the ES `haltreason` is SET 54–89 s into the equity halt:

| date | SPY MWCB Level 1 | ES halt onset | lag into the halt |
|---|---|---|---|
| 2020-03-12 | 09:35:44–09:50:44 | 09:36:45.10 | **+61.1 s** |
| 2020-03-16 | 09:30:01–09:45:01 | 09:30:54.97 | **+54.0 s** |
| 2020-03-18 | 12:56:11–13:11:11 | 12:57:39.72 | **+88.7 s** |

CME halts equity-index futures in coordination with the primary market, as its rules require; the
~1 minute is the relay, not a volatility threshold being crossed independently.

**The flag is not the duration — see §13.** Those spans clear after 5–7 s, which reads as a brief
pause. Measured against ESM0's own tape on 2020-03-18, the actual stop is **817.3 s**.

`market_halts.cross_asset_summary()` produces the table.

It also **corroborates the 2020-03-18 equity halt time**, which was the one entry in `MWCB_HALTS`
never checked against a tape: a materially wrong 12:56:11 would not bracket a 12:57:39 futures
pause that sits in family with the other two days' lags. That is the other leg speaking, not proof.

2020-03-18 carries a **fourth** pause, 2.09 s at 09:24:58 — pre-open, with no equity halt near it.
`cross_asset_summary` returns it under `es_only` rather than dropping it for failing to match.

**What this does NOT settle: the roll.** Both contracts publish status, price limits and halts on
every 2020 date checked, and their limits differ by a constant calendar spread (12.00 index points
on 03-16, 10.00 on 03-18) — real for both, decisive for neither. `rollover_days=8` puts 03-16 and
03-18 on ESM0. Only **volume** can confirm that, which needs `mt_product_statistics` or a trade
count, not the status stream.


## 13. `haltreason` marks the stop, not the duration — and the 88.7 seconds it was hiding

`ESM0_Product_Statistics_20200318` (976,175 rows) carries the cumulative `volume` counter, so the
instants at which ES actually traded can be read directly. Against them, 2020-03-18:

    last trade before the stop   12:57:39.713
    haltreason SET               12:57:39.716    <- 3 ms later: the ONSET is exact
    haltreason CLEARED           12:57:45.549    <- 5.83 s: reads as a brief pause
    first trade after the stop   13:11:17.008    <- 817.3 s with ZERO contracts traded

Zero contracts for 13.6 minutes, then 1,221 in the first print. Thirty-second buckets through the
window show 1,277 contracts in 12:57:30–12:58:00 and then a flat run of zeros to 13:11:00.

The status flag is a transient **notification**. It marks the stop to the millisecond and says
nothing about the resume. Reading the clear as a resume understated this halt by **140×**, and the
two readings are not "rough" versus "precise" — they support opposite claims. Flag-only says the
futures traded through almost the entire equity halt (a 5.8 s pause inside 900 s). The tape says
they were down for all but the first 89 seconds of it.

So halt ends come from the resumption of trading. `windows_from_status(..., activity=...)` takes the
trade instants that `reconstruct_session` now records for free from the tape it already fetches, and
extends each window to the first trade after the onset. The unextended spans stay under
`flag_windows`, and each extension carries a `quiet_ratio` — the extension divided by the pre-halt
median inter-trade gap — because extending to the next trade is only sound where trading was dense
enough that a long silence cannot be ordinary. Here that gap is 2 ms against 811 s of silence.

The equity feeds do **not** need this. Their MWCB spans clear at exactly 900.0 s, matching the
published durations, so there the flag *is* the duration. Hence a parameter rather than a new rule.

**What the correction reveals.** The interesting object is not a 15-minute divergence between the
legs but a short, sharp, bounded one:

    12:56:11              SPY MWCB Level 1 halt begins
    12:56:11–12:57:39.7   ES trades on — 3,927 contracts in 88.7 s, the ONLY price venue
    12:57:39.7            ES halts
    13:11:11 / 13:11:17   SPY reopens; ES follows 6.0 s later

Eighty-eight seconds of solitude, bounded at both ends, on a day in the volatile panel. Both legs
reopen within six seconds of each other.

**Other facts from this file.** `bidprice` / `askprice` / `volumeweightedaverageprice` /
`lasttradevolume` are **empty** in `mt_product_statistics`, so it is not a top-of-book fallback for
the 2020 ES leg. `volume` (99.99% populated) and `openinterest` (90.2%) are cumulative and reset at
18:00 ET; sorted by `sequencenumber` the counter is monotone with exactly one step down, the session
reset. Sorted by receipt time **with a stable sort** it is also monotone — an unstable sort scrambles
the many exact timestamp ties and manufactures thousands of spurious inversions, which is worth
knowing because the replay's own ordering depends on `kind="stable"` (it uses it).

The file is **424 MB for one contract-day**, which corrects an assumption in `probe_es_2020.py`:
`mt_product_statistics` is not small. The front-month discriminators — the prior session's closing
`volume` and the `openinterest` — are both in the FIRST rows, so a `--limit` fetch answers the roll
question without pulling a gigabyte.

## 14. Four ES tapes: the roll settled, and the flag gap confirmed on a second day

`mt_product_statistics` for ESH0 and ESM0 on both 2020-03-16 and 2020-03-18.

**The roll: ESM0, on both dates — the calendar rule was right.**

| date | ESH0 RTH volume | ESM0 RTH volume | front-month share |
|---|---|---|---|
| 2020-03-16 | 1,986,076 | **3,027,078** | 60.4% |
| 2020-03-18 | 732,903 | **2,605,122** | 78.0% |

`rollover_days=8` picks ESM0 for both, and the tape agrees. But **the roll is a week, not a switch.**
On an ordinary session the front month is essentially all of the volume; here the single-contract ES
leg misses **22–40%** of futures activity, on two sessions that are *in* the volatile panel. "Right
contract" and "the whole market" are different claims and only the second is what a price-discovery
estimate assumes.

Splicing is not the fix — the contracts carry a 10–12 index-point calendar spread, so a stitched
series manufactures a jump at the seam. It is a sample fact to report, so `roll_window_days()`
measures the distance to the nearest roll boundary **in either direction** and the extractor warns
inside ±7 days. Direction matters: 2020-03-16 is four days *past* the March boundary, and a
forward-only measure calls it 87 days from the June roll — silent on exactly the session that needs
it. All four MWCB dates land inside the roll week (3, 0, 4 and 6 days); so does 2024-12-18 (6 days).

Open interest rolls later than volume: ESH0 2.69 M vs ESM0 1.43 M on 03-16, then ESH0 1.59 M vs
ESM0 2.96 M on 03-18.

**The flag/tape gap, on a second date and both contracts.** §13 measured one day. Adding 03-16:

| date | flag span | actual stop | understated | sole-venue window | ES reopens vs SPY |
|---|---|---|---|---|---|
| 2020-03-16 | 7.27 s | **846.06 s** | 116× | 53.9 s | **+0.012 s** |
| 2020-03-18 | 5.83 s | **817.30 s** | 140× | 88.7 s | +6.0 s |

The onset is exact on both — the flag is set 24 ms and 11 ms after the respective last trades — and
both contracts stop and resume at the same instants, confirming the halt is group-level. Only the
CLEAR is meaningless.

**2020-03-16 cross-validates the equity side.** `MWCB_HALTS` puts that day's SPY halt end at
09:45:01, derived from the equity status tape. ES's first trade back is **09:45:01.012**. Two
independent feeds, 12 ms apart, agreeing on a boundary that was hand-entered before either was
checked.

Contracts traded while ES was the only venue: 3,698 (ESH0) + 4,132 (ESM0) on 03-16 in 53.9 s;
1,288 + 3,927 on 03-18 in 88.7 s.

## 15. 2020-03-09 completes the set — four MWCB days, one pattern

`mt_product_status` and `mt_product_statistics` for ESH0 and ESM0 on 2020-03-09, the date that was
missing.

**The flag/tape gap on every day we can measure:**

| date | flag span | actual stop | understated | sole-venue window | ES resumes vs SPY reopen |
|---|---|---|---|---|---|
| 2020-03-09 | 6.38 s | **834.12 s** | 131× | 65.9 s | **+0.010 s** |
| 2020-03-16 | 7.27 s | **846.06 s** | 116× | 53.9 s | **+0.012 s** |
| 2020-03-18 | 5.83 s | **817.30 s** | 140× | 88.7 s | +6.0 s |

Both contracts halt and resume at identical instants on every date, confirming the halt is
group-level. The flag ONSET tracks the last trade closely on the front month (+452 ms on 03-09,
+24 ms on 03-16, +11 ms on 03-18) and less closely on the back month (+5.96 s for ESM0 on 03-09,
which trades a fraction as much) — as expected, since the last trade is only a proxy for the stop in
proportion to how densely the contract trades.

**A second cross-validation of the equity halt table.** ES's first trade back is 09:49:13.010 on
03-09 and 09:45:01.012 on 03-16, against `MWCB_HALTS` ends of 09:49:13 and 09:45:01 derived from the
SPY status tape. Two dates, +10 ms and +12 ms, from an independent feed.

**The futures do not absorb the flow — they nearly stop.** The natural prior is that when equities
halt, trading concentrates into the futures. On the two days with a valid intraday baseline:

| date | RTH lots/s before the halt | during the sole-venue window | change |
|---|---|---|---|
| 2020-03-09 | 61.8 | 3.0 | **−95%** |
| 2020-03-18 | 105.7 | 44.2 | **−58%** |

Then zero for the rest of the halt, then 144–368 lots/s on the joint reopen. **2020-03-16 is
excluded**: its equity halt begins one second after the 09:30 open, so there is no RTH trading to
compare against, and using the overnight session as a baseline yields +1128% — an artifact of
comparing an opening print to Globex overnight, not a finding.

**The roll, with 03-09 added.** Share of RTH volume held by the contract the calendar rule picks:

| date | roll offset | picked | share |
|---|---|---|---|
| 2020-03-09 | −3 days | ESH0 | **93.7%** |
| 2020-03-16 | +4 days | ESM0 | 60.4% |
| 2020-03-18 | +6 days | ESM0 | 78.0% |

The two sides of the boundary are not alike. **Before** it the front month is still the old contract
and holds nearly everything; **after** it the new contract leads while the old one keeps a large
share as its open interest unwinds (ESH0 open interest is still 2.69 M on 03-16 against ESM0's
1.43 M). So `roll_window_days()` is signed — negative before, positive after — and the extractor
warns on the post-roll window while only noting the pre-roll one.

**Still unmeasured: 2020-03-12**, the roll boundary itself (offset 0), where the code picks ESH0.
It is the session most likely to be near 50/50, and the only one of the four whose front-month
choice has not been checked against volume.

## 16. 2020-03-12 closes the set — all four MWCB days measured

`mt_product_status` and `mt_product_statistics` for ESH0 and ESM0 on the roll boundary itself.

**The roll boundary: ESH0 at 72.8%.** The calendar rule picks ESH0 at offset 0, and the tape agrees —
so `rollover_days=8` is correct on **all four** MWCB days. The full roll curve, as share of RTH
volume held by the chosen contract:

| date | roll offset | picked | share |
|---|---|---|---|
| 2020-03-09 | −3 days | ESH0 | **93.7%** |
| 2020-03-12 | **0 (the boundary)** | ESH0 | **72.8%** |
| 2020-03-16 | +4 days | ESM0 | **60.4%** ← trough |
| 2020-03-18 | +6 days | ESM0 | **78.0%** |

Concentrated before, dropping through the boundary, troughing four days after, recovering by six.
The single-contract ES leg misses 6–40% of futures volume depending where in the week a session
sits. ESH0 traded 3,136,474 RTH lots on 03-12 — the largest single-contract figure of the four days.

**The flag/tape gap, complete:**

| date | flag span | actual stop | understated | sole-venue window | ES resumes vs SPY reopen |
|---|---|---|---|---|---|
| 2020-03-09 | 6.38 s | 834.12 s | 131× | 65.9 s | **+0.010 s** |
| 2020-03-12 | 6.38 s | **839.88 s** | **132×** | **60.1 s** | **+0.006 s** |
| 2020-03-16 | 7.27 s | 846.06 s | 116× | 53.9 s | **+0.012 s** |
| 2020-03-18 | 5.83 s | 817.30 s | 140× | 88.7 s | +6.0 s |

Four days, both contracts each, 116–140×. The ES down-time is strikingly stable at **817–846 s**
against a 900 s equity halt, because ES halts about a minute in and resumes with the cash market.

**Third cross-validation of `MWCB_HALTS`.** ES's first trade back on 03-12 is 09:50:44.006 against a
table end of 09:50:44 — +6 ms, joining +10 ms (03-09) and +12 ms (03-16). Three independent
confirmations of boundaries hand-entered before any of them was checked.

**The volume collapse, on a third day with a valid baseline:**

| date | RTH lots/s before the halt | during the sole-venue window | change |
|---|---|---|---|
| 2020-03-09 | 61.8 | 3.0 | −95% |
| 2020-03-12 | **240.3** | **65.8** | **−73%** |
| 2020-03-18 | 105.7 | 44.2 | −58% |

2020-03-12 has the busiest pre-halt tape of the four (240 lots/s) and still falls 73%. 2020-03-16
remains excluded — no RTH baseline exists when the halt begins one second after the open.

## 17. The 2020 ES leg: the capture is market-by-PRICE

`probe_es_2020.py` across all four MWCB dates and both contracts, with 2024-12-18 as control:

| message type | 2020 ES | 2024 ES |
|---|---|---|
| `mt_add_order` | **1–4 rows for a whole session*** | populated |
| `mt_cancel_order` | **EMPTY** | populated |
| `mt_modify_order` | **EMPTY** | populated |
| `mt_price_level_update` | **EMPTY** | **EMPTY** |
| `mt_modify_price_level` | **populated** | EMPTY |
| `mt_delete_price_level` | **populated** | EMPTY |
| `mt_trade` | populated | populated |
| `mt_aggregated_price_update` | populated | populated |

\* under a probe limit of five, so those are *complete* counts — stray messages, not a stream.

**The 2020 CME capture is market-by-price; the 2024 capture is market-by-order.** The extractor
asked for order-by-order on every date, so on the four 2020 sessions it fetched a handful of orphan
adds, no cancels and no modifies, and built nothing. That is the whole of `median ES=nan`.

`mt_missing_product_messages` and `mt_error` are empty on every 2020 date and both contracts, so the
capture is intact. Nothing was lost; the wrong question was asked.

**Why a direct check missed it.** `mt_price_level_update` — the one MBP type the ES fetch list did
carry — is empty in **both** eras. Testing it against 2024 returned "header only", which read as
*"CME publishes no price-level types"* and was written into the code as a fact about the venue. It
was a fact about the one MBP type CME never populates. §6 recorded that conclusion and it was wrong.

**The fix is deliberately not "2020 uses MBP".** That would hardcode a second era observation one
release after the first one broke, and a third capture change would break it again. Every candidate
type is now fetched and `lob_reconstruct.select_book_family()` picks from the row counts, requiring
both an insert source **and** a removal source. That requirement is what rules out 2020's four
orphan adds: adds with no cancels do not build a thin book, they build a book nothing ever leaves —
depth growing monotonically, top crossed on most snapshots, the exact signature of the original
crossed-book bug. A naive "use whatever has rows" fallback would have produced it.

A consolidated equity book is genuinely hybrid, so when both families are populated the verdict is
`BOTH` and nothing is dropped — taking "the denser family" on SPY would silently remove whole venues
from the NBBO.

**Still open: `mt_aggregated_price_update` is populated in both eras** — the only book source uniform
across the whole sample. Using it for the ES leg throughout would remove the futures replay entirely
and make the ES methodology identical on every date, at the cost of CME's published depth rather
than a reconstructed one. `validate_aggregated.py` already reads it and already shows the 2024 MBO
replay agreeing with it. That is a methodology decision, not a defect.