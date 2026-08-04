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
