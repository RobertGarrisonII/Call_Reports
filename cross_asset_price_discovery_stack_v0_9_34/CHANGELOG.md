# Changelog

## v0.9.34 -- the futures status stream, and a correction to v0.9.33

The ES `mt_product_status` tables (ESZ4 2024-12-18, ESH0 2020-03-12) break three assumptions the
halt and market-state code had absorbed from SPY. Each broke SILENTLY, producing a number rather
than an error.

**The correction first.** v0.9.33 concluded, from four SPY dates with empty band columns, that the
absence of LULD was *"conclusively a property of the direct venue feeds."* That is true of the
direct **equity** feeds and false as a general claim. **CME populates the price-limit columns for
ES on every date checked** -- ESZ4 2024-12-18 carries `563025` / `647725`, i.e. 5630.25 / 6477.25
index points after the same integer-hundredths scale the ES book uses. The futures leg has a usable
price-band control today; the equity leg still does not.

* **INT64_MAX is the "no limit this side" sentinel.** `9223372036.8547758070` is 2^63-1 scaled.
  Parsed as a price it is finite, so nothing raised -- it turned the ES band width into
  **1.5e+10 bps**: a plausible-looking column in a regression table that is pure sentinel. Anything
  at or beyond 1e9 is now NaN. What it was hiding is the actual economics of the crash: on
  2020-03-12 the ESH0 **lower** limit ratchets 2594.00 -> 2601.00 -> 2546.50 -> 2382.00 -> 2190.00
  -> 2332.50 while the **upper** is unbounded. The contract is limit-DOWN constrained, one side
  only, and that is now visible instead of averaged into a nonsense width.
* **`haltreason` is a status-reason field, not a halt flag.** ESZ4 carries eighteen `GroupSchedule`
  and nineteen `MarketEvent` rows -- routine session bookkeeping (`NoEvent`, `NoCancel`,
  `ChangeOfTradingSessionResetStatistics`). Read as halts they produced spans of 843 s, 5,914 s,
  43,910 s and 20,055 s on days the future never stopped trading. Halt reasons are now
  **whitelisted**, and the direction is deliberate: a false halt EXCUSES crossing and can hide a
  replay fault, while a missed halt merely flags a session that turns out to be fine. Unrecognized
  values are returned in `unknown_reasons` and logged, so they get classified rather than assumed.
* **CME is one venue.** The `min_venues=2` quorum added in v0.9.33 -- correct for a sixteen-venue
  consolidated equity book -- suppressed **every** real futures halt, because there is no second
  futures venue to agree. Quorum is now capped at the number of venues that publish status at all.
* **The 30 s duration floor was discarding the events this paper is about.** CME Velocity Logic
  pauses equity-index futures for 5-10 s by design. The floor is now 1 s; the reason whitelist, not
  a duration threshold, does the filtering.

What that recovers:

    2020-03-12  SPY   09:35:44 -> 09:50:44   900.0 s   MarketWideCircuitBreakerLevel1
    2020-03-12  ESH0  09:36:45 -> 09:36:51     6.4 s   SuspendedBySurveillance

**The futures stopped matching 61 seconds into the equity circuit-breaker halt.** That is a
cross-asset event, on a day in the volatile panel, and it was invisible three times over: the
extractor never fetched the ES status stream, the quorum rejected it, and the duration floor
discarded it.

Also in this release:

* `_extract_one_session` fetches `mt_product_status` for **both** legs and attaches
  `{ES,SPY}_ssr`, `{ES,SPY}_luld_lower/_upper/_band_bps/_dist_*_bps/_luld_binding`. The futures
  bands go through `price_scale` (0.01), the same scale as the ES book.
* `df.attrs` gains `halt_windows_SPY` / `halt_windows_ES`; `halt_windows` is now their **union**.
  `session_qc` judges each book against **its own** leg's windows and the pair against the union --
  a CME pause must not excuse a crossed SPY top, and vice versa. An empty per-leg list is a positive
  statement from the tape ("this leg did not halt") and no longer falls back to the built-in table.
* The STAGE 3 root-cause loop in `run_paper_replication.sh` now runs `feed_health` on the ES
  contract as well as SPY -- the gate fails on either leg, so diagnosing only SPY could answer the
  wrong question.
* `test_es_product_status.py` (7 checks) pins all of the above to the verbatim vendor rows.
* The venue-quorum check in `test_halt_aware_qc.py` used a fixture containing one venue's rows and
  called it "one venue out of sixteen". With quorum capped at what publishes, that fixture *is* a
  single-venue market. It now carries the other venues' rows, which is what a real consolidated
  status stream looks like, plus the CME case the cap exists for.

## v0.9.33 -- a clean day found a false halt, and a quorum rule the MWCB days could not have

2024-12-18 is a volatile day with no circuit breaker -- exactly the control the halt code needed,
and it failed two ways at once. The date carries a single halt row:

    06:28:02  memoir_ltse_depth_l3  haltreason=RegulatoryConcern     (never cleared)

* **A zero-length halt on a clean day.** The unclosed-span branch appended `(start, last_row)`
  without the `min_seconds` filter the closed branch applies. With one row, start == last row, so
  `feed_health` reported *"HALT on this date: 06:28:02 -> 06:28:02"* on a session that never halted.
* **No venue quorum -- the one that would have mattered.** A market-wide halt stops every venue at
  once: on 2020-03-09 six feeds report within 30 ms. ONE venue out of sixteen reporting a status is
  a venue-level event, and the consolidated book keeps matching on the other fifteen. Without a
  quorum, a single venue's hour-long regulatory status would have excused an hour of crossing as
  "the halt" -- hiding a genuine replay fault behind a rule that does not apply to the book being
  measured. The three MWCB days could never have surfaced this, because on all of them every venue
  halts together.

`windows_from_status` now takes `min_venues=2`, applies `min_seconds` to unclosed spans, and returns
sub-quorum spans under `venue_only` rather than discarding them -- `feed_health` prints them with
"NOT a market-wide halt -- the other venues kept matching, so crossing here is NOT excused."

Verified against all four days now available:

| date | market-wide window | venues | venue-only |
|---|---|---|---|
| 2024-12-18 | **none** | - | none (the LTSE row is sub-`min_seconds`) |
| 2020-03-09 | 09:34:13-09:49:13 (900 s) | 6 | none |
| 2020-03-12 | 09:35:44-09:50:44 (900 s) | 6 | none |
| 2020-03-16 | 09:30:01-09:45:01 (900 s) | 6 | none |

SSR on 2024-12-18: **not restricted** (27% coverage, all NotInEffect). LULD: 0% coverage again --
four dates, four sources, no bands. That is now conclusively a property of the direct venue feeds.

One thing NOT to build on: `marketsession` is unreliable as a session boundary. On 2024-12-18
`bats_edgx` reports `CoreSession` at 01:42 ET, and `xdp_chicago_integrated` replays
PreOpen -> EarlySession -> CoreSession -> LateSession -> Closed in **0.3 seconds** at 22:26 ET
during its end-of-day symbol clear. It is fine as a descriptive summary, which is all it is used
for; it is not a source for the 09:30-16:00 grid.

## v0.9.32 -- 2020-03-16 was short-sale restricted for the whole session

The 2020-03-12 and 2020-03-16 status streams answer the question v0.9.31 raised, and they answer it
unevenly:

| date | halt (from the tape) | Rule 201 SSR | LULD |
|---|---|---|---|
| 2020-03-09 | 09:34:13 - 09:49:13 | not restricted | 0% coverage |
| 2020-03-12 | 09:35:44 - 09:50:44 | **not restricted** | 0% coverage |
| 2020-03-16 | 09:30:01 - 09:45:01 | **IN EFFECT, 100% of the session** | 0% coverage |

**2020-03-16 is a short-sale-restricted session.** SSR turns on at 09:30:00.044 -- at the opening
bell, on a limit-down gap -- and all thirteen venues confirm within 1.1 seconds. So one of the four
MWCB days carries a one-sided constraint on sell-side flow for its entire length, and that day is in
the volatile panel. Tables 5 and 7 count signed order flow. This has to be in the sample appendix
and, ideally, in the specification.

Reading it correctly needed two fixes, both from the real rows:

* **`Activated`.** IEX spells the restriction that way (`iex_deep` at 09:30:00.057) while the other
  twelve say `ShortSaleRestrictionInEffect`. Unmapped it returned NaN -- one venue's view of a
  market-wide restriction silently became "unknown". Now mapped, with `Deactivated` as its pair.
* **Latching.** Rule 201 is a DAY-level state: once triggered it holds for the remainder of that day
  and all of the next, and does not switch off intraday. Venues report it with a lag, so the raw
  stream churns -- `xdp_national`/`nyse`/`chicago` still say NotInEffect at 09:30:00.011-.012, and
  `xdp_american` says NotInEffect at .050 before catching up at .069. Last-value-wins would have made
  the session's state depend on which venue published last. `state_series` now latches ON per
  calendar day; `latch_ssr=False` inspects the raw disagreement.

**The hardcoded halt table was wrong by 7 seconds on 2020-03-12** (09:35:44 actual vs 09:35:37 from
the published notice) and by 1 second on 2020-03-16. Both corrected, and the entries now carry the
tape times they were verified against. 2020-03-18 is still the notice time and is marked as not yet
verified. This only matters as an offline fallback -- the derived window wins wherever the status
stream is available -- but it is exactly the class of error that reading the data removes.

The 09:30:01 start on 2020-03-16 also means the market traded for one second before the breaker
fired, so that session's first snapshot is genuinely open: its head window is 99.6% halt, not 100%.

**LULD remains 0% covered on all three dates.** The direct venue feeds do not carry the bands on any
of them, which now looks like a property of the source rather than of the date.

## v0.9.31 -- LULD bands and Rule 201 as session columns, with coverage measured not assumed

Both fields live in `mt_product_status`, next to the halt. Neither is a result; both are mechanical
constraints on quoting, and both are confounds for this paper in particular.

**Rule 201 (SSR) is the one that bites.** After a 10% decline a covered security cannot have short
sales executed or displayed at or below the national best bid, that day and the next. The constraint
is one-sided by construction -- sell side yes, buy side no -- and Tables 5 and 7 count SIGNED order
flow. A restricted session therefore has mechanically asymmetric flow that looks exactly like
directional tandem trading. March 2020 is full of such days, and the paper currently does not
mention the rule.

**LULD bands** bound where a quote may be displayed, and a quote sitting at a band edge for 15
seconds triggers a Limit State and then a pause. As the mid approaches a band, spread, depth and
cancellation rates change *because they must*. On a volatility day that is a large share of the
session, and the paper reads those movements as responses to information.

`market_state.py` attaches both to every session frame:

    {A}_ssr              1 while restricted, 0 while not, NaN if unknown -- never 0 for unknown
    {A}_luld_lower/_upper, {A}_luld_indicator
    {A}_luld_band_bps    band width relative to the mid
    {A}_dist_upper_bps / {A}_dist_lower_bps    room the quote has before each band
    {A}_luld_binding     1 where the best ask is at/through the upper band (or bid through the lower)

State is carried FORWARD only -- a change at 11:00 does not colour 10:00. The columns ride on the
frame, so they reach `final_dataset.parquet` and any regression without further plumbing.
`qc_frames` now lists short-sale-restricted sessions, and `feed_health` reports the state per date.

### Coverage is measured, because one of the two is not actually there

On the 2020-03-09 SPY status stream `shortsaleindicator` is populated on **every** venue row
(`ShortSaleRestrictionNotInEffect` throughout, so that session was NOT restricted), while
`luldlowerlimit` / `luldupperlimit` / `limituplimitdownindicator` are **empty on every row** -- the
bands are disseminated by the SIP, not by the direct venue feeds.

So SSR is usable today from data already fetched, and LULD is wired end to end but will produce
all-NaN columns until a source carries it. `coverage()` reports the fraction of rows that actually
had each field, and both `describe()` and `qc_frames` say outright *"do NOT use them as a control"*
when the answer is zero. **A control that is silently all-NaN is worse than no control, because the
regression still runs and the coefficient means nothing.** For the same reason an unknown SSR is
NaN, never 0 -- absence of the field is not evidence of an unrestricted session.

To get the bands, the LULD messages need a SIP source (CTA/UTP) in the lake; if one exists, point
`mt_product_status` at it and every derived column populates itself with no further change.

## v0.9.30 -- the halt is in the tape, so stop hardcoding it

`mt_product_status` carries `haltreason` per venue, and on 2020-03-09 SPY it says exactly this:

    09:34:13.0787  xdp_arca / xdp_nyse / xdp_national / xdp_chicago / xdp_american
                   haltreason = MarketWideCircuitBreakerLevel1   (iex_deep: ReasonNotAvailable)
    09:49:13.0787  haltreason clears, marketsession = CoreSession
                   -> 900.0 s, to the tenth of a millisecond

That is the same window `market_halts.py` has been asserting from a table I typed in by hand --
confirmed to under a tenth of a second, which is the good news. The better news is that it makes
the table unnecessary: `windows_from_status()` derives halt windows from the status stream, per
venue, for **any** date rather than the four recorded.

* Extraction fetches `mt_product_status` per session (one tiny query) and records the derived
  windows on the frame as `df.attrs["halt_windows"]`. `session_qc` prefers those over the table,
  so a halt on a date nobody anticipated is handled correctly and a hand-entered time can no longer
  be wrong.
* `feed_health.py` reports the halt boundary next to the gap counts, and its clean verdict now says
  *"this date HALTED, so the book is legitimately crossed for 900 s of it -- judge the replay on the
  rate OUTSIDE the halt"* rather than the flat "any crossing is the replay's fault", which was
  misleading on exactly the four sessions that matter most.
* A status blip shorter than 30 s is not treated as a halt: a single-symbol LULD pause is not a
  market-wide event and must not blank a session.

Two incidental confirmations from the same pull. The `SymbolClear` trading events at 00:18:12.6816
and 00:24:21.6455 carry the **same nanosecond timestamps** as that date's `mt_clear_orders` rows, so
those resets are session-init symbol clears rather than error recoveries -- consistent with them
landing on an empty book. And `marketsession` gives authoritative session boundaries
(PreOpen / EarlySession / CoreSession / LateSession / Closed; CoreSession opens 09:30:00.6 on ARCA,
Closed at 16:00:00.01 on NYSE), which is a better source than a hardcoded 09:30-16:00 grid if the
paper ever needs the exact boundary.

Also available in that stream and not yet used: `luldlowerlimit` / `luldupperlimit` /
`limituplimitdownindicator` (the LULD bands) and `shortsaleindicator` (Rule 201 state, reported as
ShortSaleRestrictionNotInEffect throughout 2020-03-09). Both are natural controls for a
volatility-day study.

## v0.9.29 -- STAGE 6 had never run, and a dry run could not have told you

A clean v0.9.28 demo: STAGE 0 through STAGE 5 pass, all eleven gate tests pass, Tables 5/7/9 are
produced. Then STAGE 6 dies.

    + python3 run_analysis.py --source demo --legacy --output-dir ... --n-jobs 128
    run_analysis.py: error: unrecognized arguments: --n-jobs 128

`run_analysis.py` has no `--n-jobs`; it sizes its bootstrap from `autoscale.cpu_jobs()`, which reads
the `BOOT_WORKERS` env var. **Both** STAGE 6 invocations passed the flag -- demo and non-demo -- so
that stage had never completed on ANY source. It survived because every previous run failed earlier:
the first extraction died in STAGE 2, the second in STAGE 3, and the partial runs used
`--stages 0,1,4`.

* Fixed: STAGE 6 no longer passes `--n-jobs` to `run_analysis`, and the driver exports
  `BOOT_WORKERS="$NJ"` so the bootstrap width still comes from the same number.
* A failing stage now prints the **last 15 lines of the log** to the console. "FAILED (see the log)"
  made the reader open a file to discover it was an argparse error.

**The general fix: `check_driver_flags.py`, run as a STAGE 0 gate.** It reads the shell script,
extracts every `$PY <tool>.py ... --flag` it would run, asks each tool for its own `--help`, and
fails if any flag is undefined. Neither existing safety net covers this class of bug: a dry run
prints a command without parsing it, so a rejected flag looks identical to an accepted one, and a
unit test exercises the Python while the mistake is in the shell. It costs one `--help` per tool
(under a second) and it runs before anything expensive -- against three hours on the extract path.

It correctly attributes a wrapped command's flags to the wrapped tool rather than the wrapper
(`autoscale.py measure -- $PY run_analysis.py --flag`), joins backslash continuations, and reports
flags hidden behind a shell variable as UNCHECKED rather than implying they passed. Pinned by
`test_driver_flags.py`, which injects the exact mistake and requires a non-zero exit.

The full demo now runs STAGE 0 through STAGE 7 to completion.

Watch item, not a defect: STAGE 5's lag selection reported *12 of 13 candidate orders could not be
scored (singular design at that p)*, so p=2 was the minimum over a single scorable candidate. That
is expected on 6,000-bar synthetic frames and the output says so. If it appears on the REAL frames,
"chosen by BIC" would be hollow -- the criterion would have picked the only order it could fit.

## v0.9.28 -- 2020-03-16 opens INTO its halt, and two more checks were reading that as a fault

The 2020-03-16 report (v0.9.25 build) is the same story as 2020-03-12, plus one failure mode unique
to this date.

**Same reset bug, already fixed in v0.9.27.** `debug_crossing` reports 4.01% crossed; the extraction
reported 11.9%. `debug_crossing` did not fetch the clears, the extraction did -- and `feed_health`
shows an out-of-band reset at **21:04:46 ET on xdp_american_integrated**, five hours after the
close, exactly the shape that froze 2020-03-12. Re-measure on v0.9.27; both should now sit at the
halt rate.

**New: this is the only one of the four that halted at the OPENING BELL.** Limit-down premarket,
Level 1 halt 09:30:00-09:45:00, so the session's first 234 snapshots are 100% halt. Two checks read
that as a replay fault:

* `crossed in first 234 snapshots: 99.1%` fed the structural test -- "crossed from the very first
  snapshots with a stable book => side/scale/column parsing fault" -- which fired CODE/SEVERE. The
  other three MWCB dates halted later in the day and show 0.0% on the same line, which is why this
  surfaced on exactly one date. The test now runs on the first N **open** snapshots and says so.
* CHECK 7's single-venue crossing rate was still raw while `cross_rate` had become halt-excluded in
  v0.9.26, so the two were not comparable and the ratio test was meaningless. Both are now measured
  outside the halt.

That is the fourth and fifth false positive in this diagnostic. With all of them gone the 2020-03-16
report reads: capture complete, 0 resurrections, 0 wrong-side orders at the close, 0 of 353,106
orphans are CODE, sixth-profile 23.3% / 0.2 / 0.3 / 0.2 / 0.1 / 0.0 -- the halt, again.

## v0.9.27 -- the 97.3% was a reset ordered by its sequence number instead of its clock

Two runs of 2020-03-12 disagreed, and the disagreement was the whole diagnosis:

    debug_crossing.py, CHECK 9 sequence ordering   :  3.90% crossed
    the extraction pipeline, same date, same clock : 97.30% crossed

`debug_crossing` fetched five message types; the extraction fetched seven. The only difference was
`mt_clear_orders`.

**Root cause, and it is mine (v0.9.23).** That tape carries a reset at 22:25:47 ET on
`xdp_nyse_integrated` -- six hours after the close. A reset voids the venue's entire previous state
*including its sequence namespace*, so the clear's sequence number is not comparable to the stream
around it. Threading it into the per-feed clock cummax placed it at a **mid-day sequence position**
while it still carried a 22:25 timestamp. The replay's grid pointer only moves forward, so on
reaching that event it flushed EVERY remaining grid point in one step, and the whole consolidated
book froze at 09:40:31 -- **inside** that day's 09:35:37-09:50:37 halt. Every later snapshot then
showed the frozen, correctly-crossed halt book. The arithmetic is exact: 100% - 97.3% = 632
snapshots, and 632 seconds after 09:30 is 09:40:31.

Three fixes, in increasing order of generality:

* **A reset is placed by its CLOCK, never by its sequence.** It no longer contributes to the
  per-feed running maximum and keeps its own timestamp, so the 22:25 clear now sorts last, where it
  belongs. Event timestamps are monotone again.
* **The grid pointer no longer trusts that timestamps are sorted.** It advances on the running
  maximum, which is identical when the input is sorted and bounded when it is not, and counts
  inversions in `lob_stats["clock_inversions"]`. One stray late stamp can no longer flush the
  session -- this was a latent hazard independent of the clears, and it produces exactly the
  failure this stack keeps hitting: a full-length frame, no error.
* **`debug_crossing` now replays the same seven message types the extraction does.** A diagnostic
  that replays a different book than the pipeline cannot diagnose the pipeline; the two-run
  disagreement was luck, not design.

The rest of the 2020-03-12 report reads clean once the v0.9.26 false positives are removed: capture
complete (`mt_missing_product_messages` empty), 0 resurrections, 0 wrong-side orders at the close,
0 of 404,473 orphans are CODE, and the sixth-profile (23.1%, 0.1, 0, 0.1, 0, 0.1) is the halt --
23.1% of the first sixth is 901 snapshots, and the halt is 900. So 2020-03-12's true crossed rate
is the same ~3.9% halt as 2020-03-09's, and 2020-03-16's 11.9% should be re-measured on this build.

## v0.9.26 -- the residual crossing on the MWCB days is the halt, and three findings were false positives

The 2020-03-09 diagnostic is a clean read, and it settles the "residual ~3.9% crossing" that had
survived three rounds of investigation as an open defect. It is the circuit breaker.

    crossed overall                3.88% of 23,401 snapshots      =  908 snapshots
    MWCB Level 1 halt 09:34:13-09:49:13 ET, at 1s                 =  901 snapshots
    crossed rate by session sixth  23.1% 0.0% 0.0% 0.0% 0.1% 0.1%
    23.1% of the first sixth (3,900)                              =  901 snapshots

Exchanges stop matching during a halt but do NOT cancel resting orders, and new orders keep
arriving, so a limit order priced through the last trade sits unmatched for the full fifteen
minutes. **A correctly reconstructed book is crossed during a halt.** The invariant is what was
wrong: it assumes a market that is always matching, and on the four sessions the paper cares most
about it flagged the paper's own subject matter as a data fault.

* `market_halts.py` -- the four MWCB Level 1 halt windows, `halt_mask()`, and the reasoning. A
  table rather than a heuristic because 2020-03-16 halted AT the open and 2020-03-18 in the
  afternoon, so no "skip the first N minutes" rule covers both.
* `session_qc` and `qc_frames` now report `crossed_frac` AND `crossed_frac_ex_halt`, and judge on
  the second. A session crossed only during its halt passes with a note; one crossed outside it
  still fails. `debug_crossing` CHECK 6 splits the rate inside/outside the halt.
* **For the paper, not just the code:** halt snapshots must be excluded from every lead-lag,
  correlation and information-share estimate. A halted market has no valid midpoint, and two legs
  of stale quotes that cannot trade against each other produce mechanical comovement, not price
  discovery. Including them does not add noise, it adds a spurious result.

### Three diagnostic findings were false positives, and they outvoted the truth

The session returned `VERDICT: DATA (14 DATA findings, 1 CODE finding)` -- telling you to re-fetch
or drop 13 feeds -- on a session whose capture the venue itself reports as complete.

* **CHECK 4 inferred packet loss from sequence gaps.** 13 FATAL findings, up to "missing 98.8%".
  But the sequence is a FEED-level counter covering every symbol on the multicast line, while the
  fetch is per PRODUCT (`-p SPY`), so the gaps are the other symbols and near-100% is normal on a
  busy consolidated feed. `mt_missing_product_messages` -- the venue's own gap report, which
  `feed_health.py` reads -- was empty. The column is now labelled `not-ours`, explained, and raises
  nothing; genuine loss is reported by the venue, not inferred here.
* **CHECK 5 counted hidden executions as missing messages.** "350,275 orphaned removals, 100% add
  ABSENT -> DATA" is exactly the 6 displayed + 350,269 undisplayed prints the replay counted that
  day. Executions against non-displayed liquidity carry no order reference BY CONSTRUCTION. Now
  excluded and reported separately, the same fix the replay got in v0.9.23.
* **The single-venue crossing finding was the halt** -- CHECK 7's 3.85% single-venue vs 3.88%
  consolidated is one venue's book crossing because matching stopped, not because its replay is
  wrong.

What the session actually shows, with those removed: capture complete, reference matching exact,
replay correct, and the book crossed for precisely the fifteen minutes the market was halted.

## v0.9.25 -- the run reached the gate, and the gate had no file to read

Second real extraction: **24 of 24 sessions landed, none failed, 85 minutes** (the previous attempt
lost 22 finished days to one vendor error after 2h50m). The log shows the retry doing its job --
`mstwx-lakequery rc=1 on attempt 1/3 ... retrying in 5s`, then `extracted 2024-12-18`. It also shows
the refless-print reclassification was right: **6 displayed refless prints out of 2,463,095 trades**,
where the old diagnostic reported 350,275. Reference matching is essentially exact; the rest was
hidden liquidity all along.

Then STAGE 3 died on `no pickle matched .../frames.pkl` and took the run with it.

**`run_analysis --only extract` never wrote a session-frames pickle.** It writes
`final_dataset.parquet` (flat) and `analysis_objects.pkl` (results, not frames) into a TIMESTAMPED
SUBDIRECTORY, and the driver globbed `${OUT}/*aggregated*.pkl` -- a name from a legacy path, at the
wrong level. Now `run_analysis` writes `frames_<interval>.pkl` in the `List[(date, regime, df)]`
shape every downstream driver loads (`--save-frames`, on by default), and STAGE 2 searches
recursively for it, failing loudly with an explanation rather than handing STAGE 3 a path to
nothing. The flat dataset is not a substitute: it discards the per-session split.

**Four sessions were emptied silently and still counted.** `_align_books` keeps only rows where
EVERY asset has a quote, so the four March-2020 days -- whose ES leg is entirely absent -- were
trimmed `23401 -> 0 rows` at INFO level, while the summary said "24 usable session(s)" and the
dataset held 468,020 rows (= 20 x 23,401). Emptying a session is now an ERROR that names the leg
responsible, with a closing summary. The MWCB panel currently contributes NOTHING, which is a
sample-size fact the paper cannot state wrongly.

**Two March-2020 sessions got WORSE when the v0.9.23 replay rules went in** (2020-03-16
4.0% -> 11.9%, 2020-03-12 3.9% -> 97.3%; the other two unchanged). Three rules changed at once, so
which one is responsible is an empirical question:

* Each rule is now independently switchable via `reconstruct_book(rules=...)`:
  `apply_clears`, `use_leaves`, `consume_undisplayed`.
* `ab_book_rules.py` fetches a session ONCE and replays it under all eight combinations, prints the
  crossed fraction for each, and names the rule that moves it -- with the interpretation, which is
  not symmetric. If `consume_undisplayed=on` restores the old number, the pre-v0.9.23 behaviour was
  **masking** the crossing rather than avoiding it: letting a hidden print delete displayed size is
  wrong in principle, but it erased stale levels as a side effect. A lower crossed fraction from a
  spurious deletion is not a better book.
* `set_remaining` now refuses to GROW a resting order. A trade cannot add liquidity, so
  `leavesquantity > tracked size` means either our size is wrong or that feed populates the field
  from the aggressor (it is verified only on bats_edgx). Counted as `trade_leaves_gt_size` and
  reported, instead of inflating a level -- an inflated bid level IS a crossed book.
* The crossed-book warning now prints every deciding counter: resets applied, orders and levels
  cleared, leaves corrections, leaves rejections, and which rules were active.

Measured on the run: peak RSS of the largest extraction worker is **58.4 GiB**, so
`PEAK_GB_GUESS=76` -> 5 workers on a 495 GiB node. The run used 13 and survived, because worker
peaks do not coincide -- but 13 x 58 GiB is 759 GiB of exposure against 495 available. Treat 13 as
lucky rather than sized.

## v0.9.24 -- the ES tape carries its own benchmark, and the paper needs it

The ESZ4 sample for 2024-12-18 settles the futures leg's design and hands over the robustness
result the paper was missing. Detail in `TAPE_SEMANTICS.md` sections 6-10.

**CME publishes no price-level types for ES.** `mt_price_level_update`, `mt_modify_price_level`,
`mt_delete_price_level`, `mt_bbo_quote`, `mt_nbbo_quote` and `mt_order_imbalance` are all
header-only for ESZ4. The MBO-only path was already assumed; it is now verified, and
`test_validate_aggregated.py` pins that an ES replay builds correctly with every price-level frame
empty. Two things follow for the text: the ES book has no MBP fallback if its MBO stream is
incomplete, and **any auction-imbalance feature is equity-only** -- `auction_imbalance` silently
produces nothing for ES because the venue sends no imbalance messages.

**`mt_aggregated_price_update` IS populated for ES, and it is a benchmark with no clock confound.**
CME's own 10-level ladder -- price, quantity and order count per level -- from the same lake, on the
same capture clock as the messages the replay consumes. "How do you know the reconstruction is
right?" previously had only `validate_against_snapshot` against `mstbook-query`, which was demoted
*precisely* because it sits on a different clock, so disagreement there is confounded and proves
nothing in either direction. This one localizes disagreement to the replay.

* `validate_aggregated.py` -- level-by-level comparison, as-of aligning the event-stamped venue
  ladder onto the reconstruction's grid (comparing without that would score a timing difference as
  an error), plus the venue's session statistics as a second axis.
* STAGE 3 runs it on the first volatile session of an extract run, after the invariant gate. The
  invariant says the book is self-CONSISTENT; this says it is RIGHT.
* Fetching it at all required two fixes: `mt_aggregated_price_update` names its clocks
  `last{receipt,exchange}timestamp` (it is a ladder snapshot after a burst, not one event), so
  without the `_CLOCK_COLS` aliases the type raised `ValueError`; and it plus
  `mt_product_statistics` were absent from `_MSG_QTY_COL`.

**`mt_product_statistics` has a recency trap.** The rows are a running stream and the early ones
carry the PREVIOUS session -- the 17:38 ET row on 2024-12-18 still reports the 2024-12-15
settlement and last session's high/low. `session_statistics()` takes the last non-null value of each
field, never the first.

**ES fetches the clear types too.** Empty for ESZ4 on this date, but they exist in the futures
schema, and a Globex reset on a crash day would otherwise be invisible. An empty query costs nothing
against a 10-25 minute session.

Also recorded: the ES trade date opens at 18:00 ET the PREVIOUS calendar day (the stream starts
17:38 ET Dec 17), so the venue's session statistics span a window the 09:30-16:00 grid is a strict
subset of -- containment, not equality, is the correct check. And 2024-12-18 is a clean capture on
BOTH legs (no gaps, no decoder errors, no resets), which makes it a good control day against the
MWCB sessions.

## v0.9.23 -- the replay now reads the rest of the tape

A sample pull of every `mt_*` type for SPY on 2024-12-18 showed the reconstruction was consuming
five message types and taking its semantics for two of them from assumptions the tape contradicts.
Three of the omissions change the book. Full detail in `TAPE_SEMANTICS.md`.

**Feed resets were never fetched.** `mt_clear_orders` / `mt_clear_price_levels` are the venue saying
*discard everything I have sent and rebuild from my next message* -- a line failover or gap
recovery. Not applying one is unrecoverable: the venue never cancels the orders it just disowned, so
every one rests in the consolidated ladder to the close while the venue rebuilds under fresh
reference numbers. The 2024-12-18 tape carries one at 05:25:44 ET on `miax_pearl_equities_dom`, 85
minutes into the pre-market, with the replay already holding that feed's state (the replay starts
from the first message of the day; only the snapshot GRID is 09:30-16:00). `test_feed_reset.py`
shows the mechanism: reset ignored -> a stale 605.50 bid pins the top against a 604.90 ask, 100%
crossed; reset applied -> a clean 604.80/604.90, other venues untouched. Within a sequence tie a
reset ranks FIRST, not last, or it would wipe the re-adds it exists to make room for.

**`mt_trade.leavesquantity` was being pruned away** -- the same `_MSG_NEEDED_COLS` failure mode as
the original `sequencenumber` bug, in the same function. It is the venue's own remaining size on the
resting order after the execution (the tape proves it: ref ...546706 prints 1198 -> 1194 -> 1192 on
trades of 2, 4, 2), so it is authoritative where a decrement is only arithmetic, and `leaves=0`
removes a filled order deterministically instead of leaving it to pin the top. Now assigned rather
than decremented, with drift counted in `lob_stats["trade_leaves_corrected"]`.

**Refless trades are mostly hidden liquidity, and were treated as faults.** An execution against
non-displayed liquidity has no `orderreferencenumber` by construction; the tape marks it
`executionattribute='Hidden'` / `printable='NonPrintable'`. The old code counted these in
`trade_no_ref` -- so the "11-17% of trades have no reference" figure on the MWCB days conflated a
property of the SPY tape with a reference-matching fault -- and consumed displayed size at their
price, deleting liquidity that was still resting. Now `executionattribute` is read (also previously
pruned), undisplayed prints consume nothing, and the displayed vs undisplayed rates are reported
separately.

**`feed_health.py`** asks the question every crossed-book diagnostic had been inferring from
symptoms: `mt_missing_product_messages` is the venue's own packet-gap report, and missing packets
mean missing adds, hence orphaned removals and a crossed book that no code change can fix. With
`mt_error` and the reset inventory it returns a verdict -- capture incomplete (DATA) or capture
complete (the replay's fault, fixable). Empty for 2024-12-18. This is the decisive test for the
residual ~3.9% crossing on the four MWCB days, at one query per day. STAGE 3 runs it on any flagged
session before `debug_crossing`.

**STAGE 3 was diagnosing a different book than it saved.** `debug_crossing` ran with
`--clock exchange` while extraction runs on `receipt`; the tape also shows many consecutive
`bats_edgx` messages sharing one `exchangetimestamp` while receipt times differ, which makes that
ordering degenerate for those bursts. Now `--clock receipt`, matching the extraction.

Confirmed correct and unchanged: blank `aggressorside` for equities (tick-rule fallback), per-feed
sequence namespaces, `total_view` re-IDing on modify, MBP levels keyed on price rather than the
empty `level` field, and building the NBBO from messages rather than the empty `mt_nbbo_quote`.

## v0.9.22 -- the sample is checked against the exchange calendar before it costs three hours

The 2022-2026 re-sample (chosen to make the paper current) is now the driver's default; the
published 2014-2017 universe is `--paper-sample`. Validating it turned up two dates:

* **2026-01-19 is Martin Luther King Jr. Day.** NYSE closed. This is precisely why that session
  extracted to `median SPY=nan ES=6913.75` -- no equity session, while CME Globex ran its
  abbreviated holiday session, so the futures leg looked healthy and only SPY was missing. A
  cross-asset study cannot use a day with one leg, and the frame is full length either way.
* **Pair 8 breaks the matching rule.** 2025-01-07 (Tue) is matched to 2024-01-29 (Mon), 344 days
  apart; every other pair is same-weekday at 350-371 days. The rule-consistent match is 2024-01-09.

`validate_sample.py` checks a universe in milliseconds: weekends, exchange holidays, one-off
closures (Sandy, the Bush/Ford/Carter funerals), duplicates and future dates are errors; 13:00 ET
half days and broken volatile/baseline pairs are warnings. STAGE 0 runs it for `--source extract`
and refuses to start otherwise, since a bad date costs 10-25 minutes of vendor I/O to return an
empty frame. `--allow-bad-dates` overrides deliberately. Pinned by `test_validate_sample.py`.

`SAMPLE_UNIVERSE.md` documents the new sample, the pairing table, and what Appendix A.1 and the
text now have to say -- including that the March-2020 MWCB days are now ~6 years *before* the rest
of the sample rather than ~4 years after it, which is a different market-structure era and needs a
sentence.

## v0.9.21 -- extraction survives a bad day, and refuses to ship a book it invented

From the first real `--source extract` run: 24 sessions, 2h50m, **zero output**. The log contains
three distinct defects with one shared signature, all in the extraction path.

**1. One transient vendor error aborted the batch.** `mstwx-lakequery` exited 1 on the 23rd
session. joblib's default is fail-fast, so the exception came out of the generator and every one of
the 22 finished days was discarded. A day failing is a fact about that day, not a reason to lose
the others.

* `_run_mstwx_lakequery_to_file` retries a non-zero exit (`MST_LAKEQUERY_RETRIES=3`,
  `MST_LAKEQUERY_BACKOFF=5`, doubling). Vendor rc=1 is usually throttling or a dropped connection,
  and the same query typically succeeds on a retry.
* `_guarded_session` wraps each session: a failure returns `df=None` with the exception text
  instead of propagating. `extract_sessions` reports and drops it, and raises only if EVERY session
  fails (an empty universe downstream is indistinguishable from a small one).
* `--extract-cache DIR` memoizes each good session; `resume` (default) reuses it. A day is 10-25
  minutes of vendor I/O, so a re-run after a partial failure now pays only for the days it is
  missing. Degraded sessions are deliberately NOT cached -- they are worth retrying, not memoizing.

**2. A failed fetch was swallowed into an empty frame, and the fabricated book was saved.**
`reconstruct_session` logged `fetch mt_trade failed for SPY` -- without the date, so under 4-way
parallelism you cannot even tell which session it belongs to -- and carried on with an empty frame.
An *empty* mt_add_order stream is a claim about the day; a *failed* one is a claim about the query.
Replaying without the adds leaves cancels and trades referencing orders that were never inserted;
replaying without the trades leaves executed size resting forever. Both produce a full-length frame
with the right columns whose top crosses on up to 100% of snapshots. The run's log has exactly
that: `CROSSED on 100.0% of 23401 snapshots (trade_no_ref=0 of 0 trades)` immediately after a
swallowed SPY `mt_trade` failure, and `ES ... CROSSED on 43.8% (trade_no_ref=119383 of 119383)`.

* A failed fetch of a **critical** (MBO) type now raises `MessageFetchError` naming the date,
  product and type. The MBP price-level types stay optional (they are empty for futures by
  construction) and only warn, recorded in `df.attrs["fetch_failed"]`. `strict=False` for forensic
  replays of a known-partial day.
* `attach_flow` attributes a trade-tape failure to its date and product rather than surfacing a
  bare `RuntimeError` from three frames down.

**3. Nothing checked the frame that was about to be written.** Four sessions were saved with
`median ES=nan` -- no ES leg at all -- and went on toward a cross-asset lead-lag estimate.

* `mstbook_loader.session_qc(df)` runs at extraction time on the frame about to be saved: is each
  leg present, and does its top violate the no-crossing invariant (`crossed_tol=0.5%` -- consolidated
  multi-venue books do cross briefly at sub-second scale, so the tolerance is not zero). Stored in
  `df.attrs["qc"]`.
* `--qc-action warn|drop|raise` decides what happens to a session that fails it.
* `run_analysis` writes `extract_report.txt` next to the results: requested vs usable sessions,
  what failed, what is degraded. The sample is a result; a universe that quietly shrank from 24 days
  to 22 produces tables indistinguishable from a clean run's.

**STAGE 3 now gates on the saved frames, not on a re-fetch.** `qc_frames.py` checks the pickle in
seconds. `debug_crossing.py` re-pulls raw messages, so gating with it meant a SECOND multi-hour pass
over the whole sample -- and it judged a freshly fetched book rather than the one on disk, so a
session could pass the gate and still be estimated from a broken frame. It now runs only on the
sessions `qc_frames` flags, which is what it is good at: root cause.

`test_extract_resilience.py` pins all of it (retry / batch isolation / strict fetch / QC / cache),
and is part of the STAGE 1 gate.

## v0.9.20 -- the replication driver sizes itself from the machine, and the memory guess is measurable

Concurrency was already system-detected in `autoscale.py` -- but `run_paper_replication.sh` never
asked it, and one library bypassed it. Both fixed, plus the one number in the sizing model that was
a placeholder is now measurable.

### The replication driver now sizes from autoscale

It only forwarded `--n-jobs` if you exported `N_JOBS`. Unset, it passed nothing and the fallback
was `inference.py`'s `os.cpu_count()`. STAGE 0 now resolves cores, RAM, `extraction_workers` and
`cpu_jobs` from `autoscale.py`, prints them, and threads them through -- extraction gets
`--max-workers` from the MEMORY budget while the bootstrap gets all cores, which is the split that
matters. The manifest records what was actually used.

### `inference.parallel_bootstrap` no longer calls `os.cpu_count()`

`cpu_count()` ignores CPU affinity, so inside a container pinned to a subset of a large node it
reports the NODE's cores and the pool oversubscribes every one of them -- precisely the trap
`autoscale`'s docstring was written about, in the one module that wasn't using it. Now
`autoscale.cpu_jobs()` (max across cpu_count / /proc/cpuinfo / sched_getaffinity, honouring
`BOOT_WORKERS`), with the old call retained only as a fallback if the import fails.

### PEAK_GB_GUESS is now measured rather than guessed

`extraction_workers = (RAM - reserve) / PEAK_GB_GUESS`, and that 32 GiB was a placeholder for "one
day of raw messages". It is the binding constraint on the longest stage: too large and you burn
wall clock, too small and you OOM the node. It cannot be derived -- it depends on the venue mix and
message volume of the actual sessions -- so measure it.

* `autoscale.observed_peak_gb()` / `recommend_peak_gb()` -- high-water RSS of the largest single
  child (`RUSAGE_CHILDREN`, which is the per-worker figure the budget divides by, not the sum),
  plus 30% headroom rounded UP (the worker count is a floor division, so rounding the peak down is
  the direction that OOMs; and the budget must survive the worst session, not the measured one).
* `autoscale.py measure -- <cmd>` -- runs a command and reports its peak plus the
  `PEAK_GB_GUESS` to export. Uses `RUSAGE_CHILDREN` rather than GNU `/usr/bin/time`, which is
  absent on minimal images.
* STAGE 2 wraps the extraction in it, so a real run tells you the number for the next one.
* STAGE 0 says outright when extraction is capped by memory rather than cores, so the 14-of-128
  case is visible instead of silently costing throughput.

On a 128 vCPU / 512 GiB node with 24 sessions: cores=128, cpu_jobs=128, extraction_workers=14 at
the built-in guess -- and 24 (the session count, the natural cap) once a measured
`PEAK_GB_GUESS=6` replaces it.

### Fixed while testing

`autoscale.py`'s `_main` returned the exit code but `__main__` discarded it, so
`measure -- <failing cmd>` exited 0. STAGE 2 gates on that status, so a failed extraction would
have reported success -- the same silent-failure shape as the rest of this series. Verified: child
exit 3 now surfaces as 3.

## v0.9.19 -- the SVAR lag order is chosen by criterion, once, and reported

The paper's p=6 was never a choice, it was a constraint. Footnote 17: AIC pointed at 60 lags and
6 was used because the full model would not run there. That leaves the lag as an unreported
researcher degree of freedom sitting under every number in Table 9. It is now selected by
information criterion, ONCE, on the same pooled frame Eq. (5) is fitted on, and printed.

`--n-lags` accepts `bic` (the new default), `aic`, `hq`, or a fixed integer. `--n-lags 6`
reproduces the paper exactly; `criterion=None` is byte-for-byte what it always was.

**BIC not AIC, deliberately.** AIC is not consistent for lag order, and on a ~23k-bar intraday
sample it chases the effective sample straight into unusable lengths -- the paper's own 60 is that
failure, not a finding. BIC's log(T) penalty is what makes the choice finite. On a synthetic run
the difference is visible in the printed table: AIC keeps falling monotonically to the edge of the
search while BIC has an interior minimum.

### What was wired

* `correlation_svar.select_svar_lag()` -- the single source of truth. Scores candidates on the
  actual stacked SVAR frame (not a proxy), all on the common sample of the largest candidate.
* `correlation_svar.resolve_n_lags()` -- the adapter that lets every CLI take `6` or `bic`.
* Threaded `criterion`/`pmax` through `correlation_irf_inference`,
  `paper_tables.table_correlation_irf_both_ways`, and `run_table9_both_ways.py`; the resolved p is
  returned in the inference dict and written into the table note.
* `run_paper_replication.sh` gains STAGE 4c, which resolves the lag once and reuses that integer
  in stages 5 and 6, and records both the criterion and the resolved p in the manifest.

### Three correctness points that came out of doing it

* **Selection is now POOLED across regimes by default** (`lag_pooled=True`). It used to happen
  inside the per-regime loop, so benchmark could be fitted at one order and volatile at another --
  and Table 9 compares the same shock across those columns. A difference would then be partly a
  model difference. Same reason the paired Pearson/HY table resolves p once and hands the same
  value to both blocks: the table exists to isolate the estimator, so nothing else may vary.
* **The lag is resolved once and held fixed across bootstrap replicates.** Re-selecting per
  replicate sounds more honest but is not what the reported table is -- the point estimate is a
  VAR(p*) fit, so the resampling distribution must be of that estimator. It would also let a
  replicate land on a different p, changing the number of shocks and breaking the Romano-Wolf
  alignment across draws.
* **A failed selection no longer masquerades as p=1.** `pds.select_lag_var` returns `max(1, pmin)`
  when every candidate is NaN, which is indistinguishable from a real selection at the call site.
  `select_svar_lag` now drops exactly-constant columns first (one degenerate regressor -- a spread
  that never moves, a microprice deviation identically zero on a symmetric book -- makes the design
  singular and NaNs the whole table) and returns `None` if nothing scored, so callers fall back
  loudly. The CLI also warns when the criterion picks p == pmax, which is a BOUND rather than an
  optimum, and reports how many candidate orders could not be scored.

### Also fixed

`run_paper_replication.sh` set `FRAMES` inside STAGE 2, so any `--stages` subset that skipped
stage 2 under `--source load` pointed every later stage at a pickle that was never written. It is
now resolved from `--pickle` at startup. Found by running `--stages 5` against a real pickle.

## v0.9.18 -- per-session process pool + the GARCH recursion off numpy; mean_variance 115s -> 36s

Two changes, both bit-identical in output. The measurements below drove the design, and one of
them overturned my own recommendation.

### 1. `run_mean_variance.run_panel` runs sessions in a process pool

Sessions are independent -- each is its own GARCH marginal pair plus a cDCC fit, sharing no state.
PROCESSES not threads: the cost is the GARCH and cDCC recursions, which are Python-level loops
holding the GIL, so a thread pool would serialize them.

`n_jobs=None` (default) sizes the pool from the machine; `n_jobs=1` forces the original serial
loop for debugging or where a worker cannot be forked; a process pool that fails to start falls
back to serial with a warning. Output is order- and value-identical either way -- the panel is
reassembled by original index, and the per-session keep-going-on-error behaviour is preserved.

**`autoscale.panel_workers(n_items)`** is the new sizing function, a THIRD budget alongside
`extraction_workers` and `cpu_jobs`, because neither fits: `cpu_jobs` ignores memory (fine when
data is resident and shared, wrong when each worker gets its own pickled frame), and
`extraction_workers` is calibrated for a worker holding a full day of raw MESSAGES (32 GB/worker
against a >=16 GB reserve) -- borrowing it collapses to ONE worker on any node under ~48 GB, i.e.
it would silently serialize the loop being parallelised. An aggregated 1-second frame is ~7.5 MB.
Size = min(cores, n_items, RAM-derived at ~1 GB/worker leaving 30% headroom). Overrides:
`PANEL_WORKERS`, `PANEL_PEAK_GB`. CLI: `python autoscale.py panel --sessions 24`.

### 2. `_garch_filter` runs on Python floats -- 4.7x, and it is the bigger lever

I recommended parallelism over this rewrite. Measurement said the opposite, and the reason is the
interesting part. On a box that scales 106% on a sustained pure-CPU benchmark, per-fit time inside
the pool degraded almost linearly with worker count:

    1 fit alone 18.3 s  |  2 workers 29.2 s each  |  4 workers 57.7 s each

That is a saturated SHARED resource, and since CPU scaled fine it is memory/allocator traffic:
indexing a numpy array element-wise boxes a fresh Python float per access, and `_garch_filter` --
65% of a `mean_variance` run, ~2,200 calls per fit from the L-BFGS numerical gradients -- did that
in its inner loop. So the pool was fighting the very thing that made the loop slow. Hoisting the
square out as one vectorised op and running the recursion on lists removes the boxing, which
speeds up the serial path AND unblocks the parallel one. Arithmetic and its ordering are
unchanged: `np.array_equal` holds exactly, with and without the X covariates.

### Measured

    _garch_filter (T=20k)      14.1 ms -> 3.0 ms          (4.7x, bit-identical)
    mean_variance self-test        115 s -> 36 s
    run_panel, 8 sessions serial   185.7 s -> 72.8 s
    run_panel, 8 sessions pooled   123.6 s -> 58.2 s      (4 cores)

Every comparison max|diff| = 0.00e+00 against the serial pre-change path.

Parallelism alone was 1.50x on this 4-core box -- worth keeping, and it should do better on a
machine with more memory channels, but it was never the 8-16x I first estimated. The honest split
is that the loop rewrite did most of the work.

### Known open (pre-existing, found not caused)

`run_mean_variance` self-test reports `checks: False` on `vol_link`: the liquidity->variance LR
test is not significant on the synthetic fixture (min p ~0.51). Verified pre-existing by swapping
the old `_garch_filter` back in -- identical `lr_p=0.5136` and `CS_ES=0.582196` either way, so it
is a property of the fixture or the test, not the recursion. Every other check in that module
passes. It now exits non-zero rather than printing False and returning 0 (the same
invisible-to-the-smoke-loop defect fixed in `robust_prices` at v0.9.16).

## v0.9.17 -- scalar Kalman likelihood: state_space_efficient_price fits ~10x faster

The v0.9.16 optimizer fix made `state_space_efficient_price` correct but slow: ~12-29 s per fit,
because the Gaussian likelihood ran the full array Kalman filter and each evaluation cost ~27 ms,
nearly all of it numpy dispatch on 2x2 operands. Across the paper's 24 sessions that is real time.

Same treatment applied to `dcc_garch` in v0.9.16. `_loglik_scalar` runs the identical recursion --
same prediction-error decomposition, same 2x2 algebra -- carrying the state as (a1, a2) and the
covariance as (p11, p12, p22) in Python floats, and storing nothing. Two savings compound:

  * The MLE only ever reads `["loglik"]`. Allocating and filling the (n,2) and (n,2,2)
    prediction/filtering arrays on every evaluation was pure waste -- they are used exactly once,
    by the smoother, after the optimizer finishes.
  * No per-step numpy dispatch.

Used for k <= 2 (the k=1 UC model and the k=2 bid/ask case the module is built around); k > 2 still
goes through the array engine, as do all callers that need smoothed states.

    _nll                  27.3 ms -> 4.8 ms per evaluation
    fit_efficient_price   12.9-29.1 s -> 1.5 s
    module self-test      56 s -> 8 s

Fit quality is unchanged: nll 2313.4 against an oracle (true-parameter) 2312.4, RMSE 0.0959.

**Accuracy of the fast path, measured rather than asserted.** It agrees with the array engine to
machine precision under a well-conditioned prior and diverges only through the diffuse one: at the
kappa = 1e6 this model uses, the t=0 update subtracts two nearly equal ~1e6 quantities and the two
operation orders round differently. The gap is 1.0e-04 absolute / 6e-08 relative at kappa=1e6,
8.7e-09 at 1e4, and exactly 0 at 1e2 -- conditioning of the diffuse start, not the algebra. That is
orders of magnitude below the optimizer's tolerance, and the fit still reports its final loglik and
states from the array engine. New self-test check (3b) pins both ends of that: relative agreement
< 1e-6 at kappa=1e6 and < 1e-8 at kappa=1e2, so a future divergence in either implementation is
caught rather than silently moving every fit.

Verified: `state_space_efficient_price`, `efficient_price`, `mean_variance`, `test_basis_state`,
`test_crossing_qc` all pass.

## v0.9.16 -- the three defects v0.9.15 left open

All three are closed. Two were real estimator bugs, one was a portability defect; none of them
changed a published number, but the first two meant two modules did not do what they claimed.

### 1. state_space_efficient_price: the MLE optimizer never optimized

The self-test had been red: the Kalman smoother lost to a naive average of the two quotes
(RMSE 0.366 vs 0.096). Splitting estimation from filtering settled where the fault was -- run the
SAME smoother at the TRUE parameters and it scores 0.0929, beating the average. The filter and the
RTS smoother were correct all along. The fit was not: it converged 493 log-likelihood units below
the truth. Two independent causes.

  * **Every start seeded lambda_i = +1.** The likelihood is not concave in lambda and L-BFGS-B
    cannot cross a sign change, so a start with the wrong sign stays there. The module's headline
    use case is precisely where the sign is negative -- feed it [bid, ask] of one instrument and
    the bounce loads +1 on the ask, -1 on the bid. The canonical application was seeded into a bad
    local optimum. lambda is now estimated from the data: cross-sectionally demeaning kills m_t and
    leaves y_i - ybar = (lambda_i - lambdabar) c_t, so the first principal component of the demeaned
    panel IS the loading vector; both signs are also always in the start grid.
  * **L-BFGS-B stalled silently.** This likelihood comes from a Kalman recursion under a 1e6 diffuse
    prior, so at the default finite-difference step the numerical gradient is rounding noise:
    L-BFGS-B stopped after ONE iteration with success=False and a perfectly finite objective that
    the old "did it blow up" guard waved straight through. The tell was in the output all along --
    phi reported exactly its 0.3 starting value and q == sigma_v2 exactly. Replaced with Powell
    (derivative-free), preceded by a cheap likelihood screen of the candidate starts so the wider
    lambda grid costs little.

  Result: the fitted likelihood is now within ~1 unit of the true-parameter likelihood (was 493),
  parameters are recovered (q 0.371 vs 0.400, phi 0.50 vs 0.50, |lambda| 0.99 vs 1.00), and the
  oracle-vs-fit gap replaces the old pass criterion.

  **The self-test's assertion was also wrong**, and is fixed alongside. It required the smoother to
  beat the naive average on a fixture with lambda = [+1, -1] -- exactly the configuration where
  (y1+y2)/2 cancels the transitory term outright and is itself near-efficient. The oracle beats it
  by only ~3%, less than finite-sample estimation error, so that assertion was a coin flip. It now
  tests against the ORACLE (same smoother, true parameters) and keeps a beats-the-average check at
  lambda = [1, 0.30], where the average does not cancel the bounce and the comparison is
  informative: RMSE 0.291 vs 0.587.

### 2. dcc_garch: 2x2 LAPACK dispatch in the hot loop -- 14x faster, bit-identical

`mean_variance.py` could not finish. The cause was not the model but per-bar numpy dispatch inside
`dcc_garch`: profiling one fit showed **1.3M `np.outer` calls and 643k each of `np.linalg.solve`
and `slogdet`, all on 2x2 matrices**. The filter is re-run inside every likelihood evaluation and
every targeting iteration, so the per-bar constant is multiplied by hundreds of thousands.

At k=2 the state is three scalars and the Gaussian kernel is closed form -- log|R| = log(1-r^2),
z'R^-1 z = (z1^2 - 2 r z1 z2 + z2^2)/(1-r^2). Added scalar fast paths to `_dcc_filter`/`dcc_fit`
and to `_cdcc_filter`/`_cdcc_nll` (the cDCC path is the DEFAULT and was the actual bottleneck).
Verified algebraically exact, not approximate: correlation paths agree to 1.7e-16, z* to 8.9e-16,
and the likelihoods to the printed digit.

  dcc_garch_x(T=1300)   25.3 s -> 1.8 s
  dcc_garch self-test    133 s -> 15 s
  mean_variance         >600 s (timeout) -> 105 s, checks: True

### 3. paper_tables: hard-coded output directory

`_selftest()` wrote to an absolute `/mnt/user-data/outputs` and raised FileNotFoundError anywhere
else, so the reporting layer could not be smoke-tested on an ordinary checkout. The default is now
a repo-relative `output/` (still overridable with `OUT_DIR`), and `write_report` creates the
destination directory. 17/17 tables, exit 0.

### Verified

The four touched modules pass their self-tests, and so do every consumer of the changed code:
`copula_garch`, `correlation_svar`, `liquidity_stress`, `efficient_price`, `cross_flow`,
`noise_robust_cov`, `test_cdcc`, `test_basis_state`, `test_crossing_qc`, plus the v0.9.15
regression guards (`test_hy_correlation`, `test_tandem_null`, `test_crossed_root_cause`).

## v0.9.15 -- correctness pass on the paper pipeline: reconstruction, nulls, and the Epps dependent variable

Five defects, every one of them SILENT in the output tables. Grouped by what they broke.

### 1. The book reconstruction was wrong on every production session (root cause)

`mstbook_loader._read_messages_csv` builds `usecols` from `_MSG_NEEDED_COLS` and drops everything
else -- and `sequencenumber` was not in that set. The column was deleted during the CSV read, before
`lob_reconstruct` ever saw the frame. `_event_arrays(order_by='sequence')` then found no sequence
anywhere and fell through, silently, to its degraded clock-only ordering: the legacy path that applies
a Cancel before the Modify it precedes. `modify()` is remove-old + add-new, so the Modify re-creates
the just-cancelled order, the cancel is spent, the phantom is immortal, and it pins the venue's own top
crossed for the rest of the session. Observed on SPY 2024-01-03: 99.92% of snapshots crossed.

`test_reconstruct_ordering.py` already measured this at 100% crossed vs 0% -- production was running
the 100% configuration without knowing it. Every crossing test built frames directly and handed them to
`reconstruct_book`, so all of them carried a sequencenumber; the one layer that could drop the column
was the one layer no test crossed.

- **mstbook_loader**: `sequencenumber` added to `_MSG_NEEDED_COLS`.
- **lob_reconstruct**: WARN when ordering degrades, and when sequence coverage is only partial (an
  unsequenced message type sorts LAST within its feed -- if `mt_trade` were the frame missing it, every
  trade would be applied at end of day). Guard a NULL clock, which casts to `INT64_MIN` and sorts ahead
  of the entire session rather than merely being misplaced. Match `"<NA>"` in the null-token tests: the
  ref/side columns are read as pandas `string` dtype, whose missing values stringify to `"<NA>"`, not
  `"nan"`, so the existing guard stopped matching when that dtype was adopted.
- **debug_crossing**: CHECK 4 now raises CODE/FATAL naming the column instead of printing
  `(no sequencenumber column)` as an aside; per-feed clock-inversion rates; CHECK 4b timestamp health
  (null and dead clocks both collapse a feed's day onto the first grid point); **CHECK 8**, the missing
  mirror of CHECK 5 -- orphan provenance can only explain a removal that did nothing, never a crossed
  book, so CHECK 8 reports which orders are PINNING the top (resurrections, wrong-side resting orders
  with entry times, longest-held pinned prices, and the book stats that were collected but never
  printed); CHECK 9 settles the ordering question by experiment behind `--ab-ordering`. CHECK 5 now
  weighs the orphan RATE and ranks a low-rate uniform tail MINOR, which does not vote in the verdict.

### 2. The Table 5 null was rejected by the marginals, not by cross-market trading

Binomial(n, 1/2) with n = 505/112 bundles "each market's own flow is a fair coin" with "the two markets
are independent". Only the second is tandem trading, and it is the first that fails: from the paper's
own Table 5.II the ETF is directional in 68.7% of baseline seconds against a 2.4% prediction, the future
in 83.2% against 29.0%. Within-market clustering inflates all four corner cells with no cross-market
linkage at all. Re-benchmarked against independence GIVEN the observed marginals, PCMOF is 1.51x
(not ~100x) and NCMOF is 0.55x -- a DEFICIT, reversing "NCMOF levels are much greater than the
predicted values under either form of the null hypothesis".

- **tandem_order_flow**: `independence_given_marginals()`, `dependence_summary()` (ratios plus the
  corner log odds ratio with SE and z -- marginal-free, so comparable across the three panels),
  `permutation_null()` (shuffles the pairing between markets, preserving each market's exact marginals),
  and `independence_null_from_counts()` (the binomial null at the ACTUAL per-bar counts). The binomial
  null is a function of orders-per-bar, so the per-second calibration cannot be reused: it is 0.4% at
  one second, ~23% at ten milliseconds and ~50% in action time, where Table 7's observed 30.1% and
  48.4% sit at or below chance. `theoretical_null()` is unchanged, so published Table 5.I reproduces.

### 3. Delta-rho is an Epps-attenuated estimator, biased by a regressor in its own system

Grid-sampled Pearson is attenuated by asynchronous quote updating, and the severity depends on trading
intensity -- while volume, message traffic and liquidity demand are regressors in Eq. (5). On a DGP with
true correlation pinned at 0.60, a 25% refresh rate gives Pearson 0.072 vs HY 0.580; with activity
cycling and true rho constant, corr(measured rho, activity) is +0.774 under Pearson and +0.026 under HY.

- **correlation_svar**: `corr_method='hy'` in `build_svar_frame`; `_hy_rolling_corr()` using each
  asset's own mid-change times, attributing each interval-pair product to the later interval end so the
  window ending at t uses nothing dated after t, O(n+m+T) via cumsum; `epps_comparison()` reporting both
  with the difference as the artifact estimate. `'rolling'` and `'dcc'` untouched.
- **paper_tables**: `table_correlation_irf_both_ways()` (both estimator blocks carry day-cluster
  bootstrap SEs and Romano-Wolf stars), added to `build_all_tables` as `t9_both_ways`.

### 4. A silent NaN in the cluster-robust standard error

`ecm_sde._cluster_se` set the Liang-Zeger correction `G/(G-1)*(N-1)/(N-K)` to NaN whenever `G < 2`, so a
single session produced NaN SEs throughout -- and a NaN t-stat prints as a blank table cell rather than
raising. This hit `t_a1_SPY`, the coefficient on `z*S` that the module's own docstring calls "the
single-number test of liquidity-conditioning". Now falls back to Newey-West (valid for one contiguous
session) and logs the substitution. The warning also notes that with few clusters -- the MWCB sample has
G=4 -- a wild cluster bootstrap is preferable.

### 5. A self-test that failed and exited 0

`robust_prices` printed `checks: False` and exited 0, so the README's smoke loop
(`for m in ...; do python "$m.py"; done`), which reads only the exit code, showed it green. `_selftest()`
now returns a bool and `__main__` exits non-zero.

### New

- **run_paper_replication.sh** -- the whole corrected pipeline in one command, seven stages, two of
  them GATES: STAGE 1 refuses to proceed unless every correction's regression test passes, and STAGE 3
  refuses to estimate on any session whose book fails the crossed-book invariant. The paper's sample
  (Appendix Table A.1) is baked in.
- **run_table9_both_ways.py** -- CLI for the paired Table 9, writing .csv/.md/.tex and ranking the
  largest |HY - Pearson| gaps.
- **test_crossed_root_cause.py**, **test_tandem_null.py**, **test_hy_correlation.py** -- known-answer
  guards for 1, 2 and 3 above.

### Also fixed

- `paper_tables` markdown/LaTeX renderers stringified MultiIndex columns as Python tuples; and
  `to_latex(label=...)` unconditionally prefixed `tab:`, producing `\label{tab:tab:...}`.

### Known open

- `state_space_efficient_price` self-check (1) fails: the Kalman smoother does not beat a naive average
  of the two quotes (RMSE 0.366 vs 0.096, correlation 0.9991). The test's DGP uses lambda = [+1,-1],
  where the cross-sectional average cancels the transitory term exactly and is the efficient estimator,
  so that assertion is unachievable as written -- but a lambda sweep shows the smoother also loses at
  [1,-0.4] and [1,0.3], so it is not only a test artifact. A level offset was ruled out.
- `paper_tables._selftest()` writes to a hard-coded `/mnt/user-data/outputs/`.
- `mean_variance.py` does not finish in 400 s.

## v0.9.14 -- `debug_crossing.py`: localize a crossed-book violation to DATA or CODE

A crossed top is always our fault, but "our fault" splits into two different repairs, and the running
stack gives no way to tell them apart: DATA (the messages needed never reached the replay -- nothing in
`lob_reconstruct` can fix it) versus CODE (the messages were present and the replay failed to apply
them). This adds a single-session diagnostic that decides which.

- **Orphan provenance is the discriminator.** When a cancel or trade references an order the book does
  not have, the tool asks whether an ADD for that same `(feed, ref)` exists anywhere in the fetched
  messages: present => CODE (we had the data and failed to match it), absent => DATA (the creation
  never arrived). "Absent" is further split by time of day, because orphans in the first minutes are
  usually orders that were legitimately resting before the 09:30 window opened -- benign and expected --
  while orphans spread evenly across the session are genuinely missing messages.
- **Seven checks, cheapest first:** message inventory (an EMPTY removal stream is fatal on its own);
  per-feed x per-type matrix (one venue missing its removals crosses the CONSOLIDATED top even when
  every other venue is perfect); field health (side-token histogram, NaN price/qty rates, cross-feed
  price-scale ratio); per-feed sequence continuity; orphan provenance; replay dynamics (resting-order
  growth and crossing onset); per-venue versus consolidated crossing.
- **Names two silent-failure paths** that make the diagnosis necessary: `reconstruct_session()`
  substitutes an EMPTY frame on a fetch exception and continues, so a failed fetch yields a
  full-length frame of garbage rather than an error; and `_Book.add()` silently DROPS any order whose
  side is not exactly `"Bid"`/`"Ask"` or whose price/size is not finite and positive, so a venue with a
  different side encoding or a renamed quantity column contributes no liquidity at all.
- Runs on the production fetch path (`--date`), or offline from a pickled message dict
  (`--save-messages` once, then iterate). Exit codes: 0 clean, 2 DATA, 3 CODE.

Three defects were found by testing against fixtures with known faults rather than by inspection:
  1. Orphan provenance was keyed by reference alone, but `_Book` keys orders by `(feed, ref)`. An add
     on a DIFFERENT venue therefore counted as provenance and misattributed a data fault to code.
     Now keyed by `(feed, ref)`, with cross-feed reference collisions reported as their own finding.
  2. "Crossed in the first sixth => structural" mislabeled a fast leak as a parse fault. The verdict is
     now gated on resting-order growth: structural means crossed from the first snapshots WITH a stable
     book.
  3. Book growth measured as last-sixth over first-sixth read ~1x for a book that fills early and then
     plateaus -- exactly how an accumulation fault disguises itself as structural. Now peak over opening.

- **Tests.** New `test_debug_crossing.py` builds five sessions whose fault is known by construction
  (clean, empty cancel stream, unrecognized side tokens on one feed, cancels ordered before their own
  adds, and cancels for orders resting before the window) and asserts the verdict is right FOR THE RIGHT
  REASON. The decisive pair is the last two: both produce orphaned removals, only one is a code fault,
  and the test asserts they are separated. Full suite 31/31.

## v0.9.13 -- `run_ofr_improved.sh`: the improved-leg driver for the OFR WP 19-04 / JFM reproduction

A three-stage driver (`select` | `pilot` | `main` | `all`) for the revision's improved leg, following
the `run_liberation_day.sh` idiom (env-overridable config, `autoscale.py` worker sizing, `banner`,
timestamped `tee` logs, STAGE dispatch).

- **Selection is split from extraction.** The ex-ante vol signal is DAILY data, so `select` picks the
  day universe with no book access; passing the full 2014-01-01:2017-08-31 range to `--date-range`
  would extract ~920 sessions of consolidated SPY+ES MBO instead of the ~20 the design needs.
- **`select`** reads a required `VOL_CSV` (prior-close VIX or a HAR forecast -- information available
  BEFORE the session it labels), runs `stress_index.select_days`, thins to distinct episodes, era-matches
  controls 1:1, and writes `ofr_universe.txt` plus a selection report.
- **`pilot`** extracts two OFR-era sessions and GATES on the crossed-book invariant using
  `onset_response._crossed_frac`, exiting non-zero above `CROSS_MAX`. Every reconstruction fix in this
  stack was validated on 2025 tape; 2014-2017 is a different venue-protocol generation, so the crossing
  rate there is unknown until measured. `all` will not proceed to `main` if the gate fails.
- **`main`** extracts the selected universe and builds the 16-table suite, defaulting `CLOCK=exchange`
  (NOT `run_liberation_day.sh`'s `receipt`) and passing `--auto-regime` together with `--vol-csv` so the
  regime labels come from the SAME ex-ante series that chose the days; without `--vol-csv` the driver
  would fall back to per-session realized vol, silently reinstating ex-post selection.

Three design bugs were found by testing the selection stage on a synthetic OFR-era series rather than
by inspection, each of which would have quietly degraded inference:
  1. `stress_index.match_controls` matches nearest-tau WITH replacement -- correct for a DiD where one
     control may serve several treated days, but every high-tau stress day picked the SAME calm day, so
     the extraction universe held ONE control cluster. Now matched without replacement.
  2. Matching stress to calm on the vol LEVEL is tautological here: the labels are DEFINED by tau
     percentile, so a level caliper rejected every pair (0 controls). Controls are now ERA-matched on
     calendar proximity -- same venue protocol generation, tick regime, and contract cycle -- with the
     tau imbalance reported rather than hidden, and ex-ante level balance left to `event_study_driver`,
     which is where it belongs.
  3. tau is the PERSISTENT component, so one vol spike leaves a plateau of consecutive high-tau
     sessions and an unthinned top-N returned ten consecutive days of a single episode -- one
     day-cluster, not ten. `EPISODE_GAP` (default 10 days) now thins to distinct episodes.

Suite unchanged at 30/30 (the driver adds no Python module).

## v0.9.12 -- Release-date template + pre-ingest validator

The 10:00 releases with no publisher rule (JOLTS, NEW_HOME, UMICH) have to be ingested, and a wrong
date does not raise anywhere: it produces an onset that splits the session in the wrong place, or --
on a holiday or outside 09:30-16:00 -- an empty pre-window, so `event_onset_estimate` returns NaN and
the event silently vanishes. A mistyped year is worse still: a valid session, a valid-looking number,
and an observation centred on nothing. This release adds the fill-in template and, more importantly,
the validator that turns each of those silent failures into a loud one before ingest.

- **`release_template.py` (new).** `make_template()` emits one row per expected release with the
  loader-visible columns (`date`, `category`, `time_et`) plus entry aids the loader ignores
  (`ref_month`, `status`, `notes`). Two row kinds: NEEDS_DATE (blank date -- JOLTS, NEW_HOME, UMICH,
  the last with two rows per month for preliminary and final) and VERIFY (the rule-generated
  ISM_MFG / ISM_SVC / CONF_BOARD dates pre-filled for checking against the publisher calendar; an
  unchanged row is a no-op, a corrected one overrides generation for that category-month). The 08:30
  block and OPEX are deliberately absent -- under an RTH window they are always dropped, so including
  them would let a large useless ingest look successful.
- **`validate()`** checks, per row: parseable `YYYY-MM-DD`; NYSE trading day (else no session exists);
  known category; parseable time; onset leaves at least the 20-bar floor on BOTH sides (else NaN);
  full requested pre/post window (else a truncation warning); duplicate `(date, category)`; and
  surfaces dates carrying more than one release TIME, since one session per date resolves to a single
  arbitrary onset. Half-day awareness included via `early_close()` (Friday after Thanksgiving,
  Christmas Eve, July 3 when the 4th is a weekday) -- a 14:30 onset on a 13:00 close is an error, not
  a truncation.
- **CLI:** `python release_template.py make --start 2020-01-01 --end 2026-07-23 --out t.csv` and
  `python release_template.py validate filled.csv` (exit 1 on errors, so it can gate a pipeline).
- **Bug found by the guard:** pandas reads a blank CSV cell as float `NaN`, which is TRUTHY, so
  `str(v or "")` yielded the string `"nan"` and a blank date reached the parser -- crashing inside the
  holiday lookup on `NaT.year`. Fixed with a NaN/NaT-aware `_s()` coercion plus an explicit `pd.isna`
  guard after parsing.
- **Tests.** New `test_release_template.py` feeds a CSV containing ONE INSTANCE OF EACH failure mode
  (Good Friday, a Saturday, an ad-hoc closure, an 08:30 onset, an after-close onset, an unparseable
  date, an unknown category, a duplicate, a missing time, a half-day onset past the 13:00 close, and a
  genuine truncation) and asserts every one is caught ON THE RIGHT LINE, that unfilled rows are skipped
  silently, that the collision is surfaced, and that a clean file both validates and actually ingests
  through `load_release_schedule`. Full suite 30/30.

## v0.9.11 -- NYSE trading calendar + the rule-generated 10:00 release backbone (23 -> 263 RTH-valid onsets)

Extraction runs 09:30-16:00, so an onset needs RTH bars on BOTH sides. This was verified to fail
silently for most of the macro calendar: an 08:30 release (CPI, NFP, PPI, RETAIL, GDP, PCE, CLAIMS,
DURABLE, HOUSING, TRADE) and OPEX at 09:30 land on or before the frame's first bar, so `n_pre = 0`,
`transmission = NaN`, and the event disappears from the surface with no error. Loading those schedules
would have burned full extraction cost on hundreds of sessions and harvested nothing. This release
instead builds out the release cluster that IS valid under an RTH window -- 10:00 -- which raises the
impact-blind event supply over 2020-01-01..2026-07-23 from 23 (FOMC-only, 2023-2025) to **263**.

- **NYSE trading calendar (new, in `market_shocks.py`).** `nyse_holidays(year)`, `is_trading_day(d)`,
  `nth_trading_day(year, month, n)`, plus `_easter` (anonymous Gregorian), `_nth_weekday`,
  `_last_weekday_of_month`, `_observed`, and `_NYSE_ADHOC_CLOSURES` (Sandy, Bush, Carter). A TRADING
  calendar, not a federal one, is required: a federal calendar would place releases on Good Friday
  (market closed -> no session to extract) and skip Columbus/Veterans Day (market open -> a good
  session dropped). Anchoring the business-day rule to the tape guarantees every generated date has a
  session. Includes the NYSE quirk that a Saturday New Year's does NOT pull the holiday back to Dec 31.
- **Rule-generated 10:00 releases.** `ISM_MFG` (1st business day), `ISM_SVC` (3rd business day),
  `CONF_BOARD` (last Tuesday) -- 12/yr each, generated across any requested range and wired into
  `scheduled_events` and the `categories=None` expansion. Provenance is stamped honestly in the
  `source` field as rule-generated, in contrast to `FOMC_DECISION_DAYS`, which is a transcribed table.
  `JOLTS` / `NEW_HOME` / `UMICH` are also 10:00 but have no clean publisher rule and are deliberately
  NOT generated -- ingest them via `load_release_schedule`.
- **Ingest now beats generation.** `scheduled_events` dedups on `(date, category)` first-wins, so a
  rescheduled release ingested on its true date would previously have survived ALONGSIDE the generated
  date and double-counted that month. Generation is now suppressed for any category-MONTH that has a
  loaded release, so a partial ingest (e.g. authoritative 2024 only) corrects 2024 while generation
  still fills the untouched years.
- **`release_collisions(frame)` (new).** `cross_event_onset` builds one session per DATE and takes the
  first matching event row, so a date carrying two different release times resolves to one arbitrary
  onset. The detector surfaces them; on 2020-2026 with FOMC + the generated cluster there are 4
  (2023-02-01, 2023-05-03, 2023-11-01, 2024-05-01 -- each an ISM 10:00 sharing a session with an FOMC
  14:00). Which onset should win is a research decision, so it is reported, not silently resolved.
- **`run_onset_surface.py`** `--categories` help now names the RTH-valid set and states explicitly
  which categories are unusable without widening the extraction window, and why.
- **Tests.** New `test_release_calendar.py`: known-answer Easter/Good Friday, seven holiday-observation
  cases (Saturday/Sunday shifts, the Dec-31 exception, Juneteenth's 2022 NYSE start, an ad-hoc
  closure), the core invariant that every generated date is a trading day, business-day spot checks
  including a Good-Friday-shifted 3rd business day, and an END-TO-END check that a 10:00 onset yields
  usable windows on an RTH frame while 08:30 and 09:30 onsets return NaN -- the counter-case proving
  the guard has teeth. `test_release_loader.py` updated for the new precedence, with a new (B2)
  asserting an ingested date suppresses the generated one for that month. Full suite 28/28.

## v0.9.10 -- Per-event reconstruction QC on the onset panel (crossed-book contamination weight)

The FOMC backbone run surfaced crossed-book rates that vary enormously ACROSS sessions (observed: a
few percent up to ~75%), which the surface then pooled without any record of which events were built on
a broken book. A crossed top is impossible in a real matching engine, so a nonzero rate is
reconstruction error, and it contaminates every book-derived input to that event's onset observation --
the spread, the hollowness/capacity axes, the cost-to-fill, and the mid the returns come from. This
release makes that contamination a first-class, per-event column so a contaminated event can be
identified and excluded rather than silently weighted into b3.

- **`onset_response._crossed_frac` (new).** Share of grid points with `bid1 > ask1`, optionally over a
  window. Measured on the SAME frame the estimator consumes rather than inherited from reconstruction:
  the session join drops `df.attrs`, so `lob_stats` / `consolidated_crossed` is unavailable downstream,
  and a measured number cannot silently disagree with the frame that produced the estimate. Degrades to
  NaN (never raises) when the book columns are absent.
- **New panel columns.** `crossed_max` (the single sort/filter column), `crossed_resp`, `crossed_imp`
  -- all three over each event's ESTIMATION WINDOW (pre+post) -- plus `crossed_resp_sess` /
  `crossed_imp_sess` session-wide. The window-local rate is the one that bears on the estimate:
  crossing concentrates where message rates spike, i.e. AT THE ONSET, so an event can look acceptable
  session-wide while its own windows are badly crossed. Session-level rates are recorded before the
  short-window guard, so an event dropped for insufficient data still reports why its book was unusable.
- **Surface summary gains a reconstruction-QC block** (`_crossing_qc_lines`): clean / suspect /
  contaminated counts against the `_XED_CLEAN=1%` and `_XED_SUSPECT=10%` tiers, the five worst events
  with window-vs-session rates side by side, an explicit marker where the window rate exceeds twice the
  session rate (onset-local contamination), and an ACTION line to re-fit excluding contaminated events
  and compare b3.
- **Tests.** New `test_crossing_qc.py`, a known-answer guard built to DETECT the failure mode rather
  than merely run: three synthetic sessions with analytically known crossed-bar counts -- clean,
  onset-concentrated, and uniform-with-the-identical-count -- assert that (A) both rates recover the
  injected counts exactly, (B) the onset case shows window/session > 3x while the uniform control stays
  ~1x (so the gap is real localization, not an artifact of any crossing), (C) the onset event's session
  rate sits UNDER the contamination bar while its window rate sits over it, i.e. a session-level-only QC
  would have passed it through, (D) the summary flags contamination and stays quiet on a clean panel,
  and (E) missing book columns degrade to NaN. Full suite 27/27.

## v0.9.9 -- Dynamic parallelization: shared `autoscale` single-source, `run_onset_surface` worker knobs, thread-parallel onset loop

The onset launcher (`run_onset_surface.py`) previously inherited a STATIC 4-worker extraction default and ran the
per-event onset loop serially, and the runtime worker-sizing logic lived only in `run_liberation_day.sh` (bash) — so a
direct `python run_onset_surface.py` got none of it. This release makes the launcher size itself to the machine and
parallelizes the one region that warrants it, on a single sizing source shared by shell and Python.

- **`autoscale.py` (new — single source of truth).** Runtime worker sizing on TWO budgets, because the parallel regions
  differ: EXTRACTION is memory-bound (`extraction_workers = min(memory-derived, #sessions, cores)`), CPU work — the
  day-block bootstrap and the onset loop — is core-bound (`cpu_jobs = all cores`). Core detection takes the MAX across
  `os.cpu_count` / `/proc/cpuinfo` / affinity to defeat a cgroup/affinity cap that would silently serialize on a
  containerized node; RAM is `min(node MemTotal, cgroup limit)`. Every value is overridable via env
  (`CORES/RAM_GB/RESERVE_GB/WORKERS/BOOT_WORKERS/PEAK_GB_GUESS`), and a CLI (`python autoscale.py {cores,ram,reserve,
  workers,jobs}`) lets the shell driver source the same numbers.
- **`run_onset_surface.py`.** New `--max-workers` (extraction; default → autoscale, threaded into `load_sessions`) and
  `--onset-jobs` (onset loop; default → autoscale cores); both print the chosen sizing.
- **`onset_response.py`.** `cross_event_onset` / `run_onset_surface` gain `n_jobs`. Each event's estimate is
  independent, so n_jobs>1 runs the loop on a joblib THREAD pool (the per-event cost is GIL-releasing numpy: the
  pre/post covariances, the Rigobon solve, TSRV / Hayashi-Yoshida) — bit-identical to serial (order preserved, no
  shared state), gated above `_MIN_PARALLEL_EVENTS=8`. The trivial surface-fit / jackknife stage is deliberately NOT
  parallelized. Reconstruction (the event-ordering-sensitive book replay) is never parallelized within a feed; this
  fans out across already-built sessions.
- **`run_liberation_day.sh`.** The worker-sizing FORMULA is de-duplicated onto `autoscale.py` (the bash keeps its
  cgroup-aware detection and feeds RAM_GB/CORES through the environment; the "export to override" behavior is preserved
  exactly — verified identical numbers, 1/16/1 on the sandbox, 8/20/5 under override). The onset stage now passes
  `--max-workers`/`--onset-jobs`.
- **Tests.** New `test_autoscale.py` (two-budget sanity, session/core caps, reserve clamp boundaries, env overrides
  win, CLI==API). `test_onset_surface_stage.py` gains check (D): the parallel onset panel (n_jobs=4) is bit-identical
  to serial. Full suite 26/26.

Not changed: `run_contagion.py` — its static `--max-workers 4` default is a pre-existing standalone default, not the
duplicated formula (the orchestrator passes it explicit autoscaled values). Worker DETECTION now exists in both the
bash and `autoscale.py` by design — equivalent sources that agree — while only the sizing FORMULA is single-sourced.

## v0.9.8 -- Docs refresh: README + methods vignette + WP-19-04 memo brought current through the reconstruction-ordering arc

No code change (the suite and self-tests are byte-identical to v0.9.7). This release brings the three narrative
documents current through v0.9.7 and corrects passages that the v0.9.5–v0.9.7 work had made stale -- the analog of
the v0.9.4 docs refresh that brought them current to the onset spine.

- **README.** §5.31 gains *The event-ordering fix (v0.9.5–v0.9.7)* — the 100%-crossed root cause, the sequence-ordering
  fix, `verify_crossing`, and the eliminations-last within-tie tiebreak — and the crossing note now distinguishes a
  legitimate cross-venue lock (retained) from a persistent `consolidated_crossed` top (a reconstruction artifact, now
  invariant-guarded). Two stale lines fixed: caveat (iv) said ES stayed on the snapshot (both legs are reconstructed);
  the §9 "not yet run on tape" caveat now notes the first co-temporal 2025-04-03 window is in hand while the onset
  panel still awaits the multi-event run.
- **Methods vignette.** New §12.5 frames the reconstruction / event-ordering fix as a data-cleaning contribution
  upstream of every estimate on the surface; header range and the guard cross-reference updated (the latter was still
  citing v0.5.0).
- **WP-19-04 memo.** §6 data plan now states message-level reconstruction as the extraction path, the crossed-book
  fix as a data-cleaning appendix, and the in-hand first co-temporal window; the appendix notes the stack has grown
  beyond its original fourteen modules. The §5 identification section and the appendix module list are left as the
  author's editorial call (they predate the onset spine), flagged rather than rewritten.

Honest status preserved throughout: the reconstruction logic is fully synthetic-guard-verified, but the real-tape
crossed audit (`smoke_test_crossed`) on the 2025-04-03 window has not yet been run.

## v0.9.7 -- Within-tie ordering = eliminations-last: the CME packet-level residual crossing + the degraded-path artifact

v0.9.5 made intra-feed order follow `sequencenumber` (the 100%-crossed root cause). But the order key is
not always a total order, and the within-TIE fallback was the arbitrary message concatenation order
(adds, then cancels, then modifies, then trades). Two distinct failures rode on that fallback; both are
closed here by ranking liquidity removals (cancel / trade / level-delete) LAST within a tie, so the
removal is the final word in its packet/instant.

- **CME packet-level sequence (the ES residual).** A single CME `sequencenumber` covers a whole packet,
  so one value can carry an add/modify AND the cancel/trade that removes the order it references. Under
  the concatenation fallback the cancel landed before the modify that followed it; since `modify()` ==
  `remove()+add()`, the modify RE-ADDED the just-cancelled order, resurrecting a level that pinned the
  consolidated top below the bid on every later snapshot. This is the residual ES crossing that survived
  the v0.9.5 fix (≈88% on the real tape; `trade_no_ref=0` distinguishes it from the refless-trade bug).
  Eliminations-last makes the cancel win, and the resurrected level cannot form.
- **Degraded (no-`sequencenumber`) path.** When no event carries a sequence, every event in a feed ties,
  and the per-feed sequence cummax — meaningless without a sequence — was dragging a later same-instant
  event ahead of an earlier one, so a snapshot could be taken before a same-instant sweep applied. The
  degraded path now orders by the clock directly with removals last within an instant (the per-feed
  cummax is skipped, as it has no sequence to ride).

Strictly a no-op wherever `sequencenumber` is strictly increasing per feed — equities (SPY) reconstruct
identically (0% -> 0%); the change only bites on genuine ties, which is exactly the CME packet case.

- `lob_reconstruct._event_arrays`: `elim_rank` (0 = add/modify/level-set, 1 = removal) added as the
  innermost lexsort key in the sequence path; degraded sub-case routed to a direct clock sort with the
  same tiebreak.
- `test_crossed_regression.py`: new `packet_resurrection` guard — a cancel and a modify of one order
  sharing a packet `sequencenumber`; red on v0.9.6 (98.4% of snapshots crossed), green on v0.9.7. The
  existing `multilevel_sweep` (the degraded-path guard) flips red -> green on the same change.
- Full suite green (25/25) and the `lob_reconstruct` self-test passes, including the never-crossed
  invariant across the multi-venue SPY book, the ES MBP replay, and the consume / clock-switch invariances.

Recommended next real-data step: run the SPY (multi-venue) and ES (CME) reconstruction on the
2025-04-03 09:29–09:32 ET co-temporal window; expect SPY ≈0% crossed under both v0.9.6 and v0.9.7 and ES
crossed under v0.9.6 -> ≈0% under v0.9.7.

## v0.9.6 -- verify_crossing: one-command before/after for the ordering fix (data-cleaning appendix)

A single-session diagnostic that fetches the raw messages ONCE and reconstructs the consolidated book
TWICE -- legacy clock ordering vs fixed exchange-sequence ordering -- on identical events, printing the
crossed-book fraction and trade/cancel reference-miss rates side by side. This is the number for the
paper's data-cleaning appendix: the all-day-crossed artifact under the old ordering and its
disappearance under the fix.

- `verify_crossing.py`: `--date YYYYMMDD --product SPY` (or `--demo` for the synthetic resurrection, no
  feed needed). Holds the cross-feed clock fixed across the two runs so the ONLY variable is intra-feed
  ordering; reports consolidated-crossed %, crossed-frac (bid1>ask1), locked-or-crossed, trades,
  trade_no_ref %, cancel_no_order, and resting-orders-EOD for each ordering, plus a one-line summary.
  Reconstructs with consume=False so one fetch serves both orderings.
- Guard `test_verify_crossing.py`: the `--demo` path reproduces the contract -- legacy ordering crosses
  (99.6%), fixed ordering is clean (0%), identical snapshot count.

Recommended first real-data use (the v0.9.5 fix validation): run on SPY 2025-04-03, confirm crossing
collapses ~100% -> ~0% and watch whether trade_no_ref also falls; then repeat for the ES contract.


## v0.9.5 -- Reconstruction event-ordering fix: the 100%-crossed-book root cause

Roots out the all-day consolidated-crossed-book bug (best_bid > best_ask on ~100% of snapshots, SPY and
ES). The reconstructor ordered the merged multi-venue event stream by a single capture timestamp; UDP
multicast arrives out of packet order, so ~11% of adjacent same-feed events invert -- a Cancel lands
before the Modify it follows, the Modify resurrects the deleted order, and that phantom level crosses
the consolidated top forever. Confirmed on a sample (SPY 2025-04-03): identical events, sequence
ordering -> 0/46 snapshots crossed; receipt ordering -> 46/46.

- `lob_reconstruct._event_arrays` now ingests `sequencenumber` and orders events by a **k-way merge of
  per-feed streams in exchange-sequence order**, interleaved across feeds by the clock (vectorized: a
  stable sort on a per-feed sequence-monotone clock == heapq.merge, O(n log n)). `sequencenumber` is the
  venue's authoritative within-feed order; the clock now only places events on the grid.
- Default `clock` switched **receipt -> exchange**: with sequence governing intra-feed order, the clock's
  sole role is cross-feed interleaving, and the per-venue exchange clock removes the differential
  capture-latency bias receipt bakes in (measured ~0.2 ms venue-to-venue on SPY, vs sub-us GPS sync). The
  obsolete "exchange is a robustness lens, not a correction" warning is removed.
- `order_by` ("sequence" default | "clock" legacy) added to `_event_arrays` / `reconstruct_book` so the
  pre-fix behavior stays reproducible for comparison.
- NOT changed: order references are already namespaced -- the book keys orders by the `(feed, ref)`
  tuple, so cross-feed reference collisions cannot occur; no namespacing change was needed.
- Guard `test_reconstruct_ordering.py`: a one-venue stream with a receipt-vs-sequence inversion crosses
  100% under legacy clock ordering (with zero trades -- disproving the trade-matching attribution) and is
  clean under sequence ordering; same events, only `order_by` differs.


## v0.9.4 -- Docs refresh: README + methods vignette brought current to the onset spine

Documentation only (no code change). README gains the three onset modules in the Files table and a full
section 8, "Onset stress-surface -- the identified estimate of f(state)", covering the cross-event onset
identification, the 2-D liquidity-spiral surface and its identification gate, the noise-robust
transmission (the bid-ask-bounce confound and its fix), the sum/bottleneck/product cost aggregations,
the surface-excess test of the curated tail, and the run_onset_surface CLI / `onset` launcher stage,
plus two onset-specific caveats. The methods vignette gains a master-map row, section 12 "The onset
identification spine -- f(state) as a surface" (the conceptual identification narrative), three
onset-specific honest limits, and a corrected version footer.


## v0.9.3 -- excess_over_surface: the salience-curated tail tested against the fitted PLANE

The surface analog of excess_over_benchmark. The benchmark is now the 2-D bilinear surface
f(basis, holl) fit on the impact-blind scheduled backbone; the curated marquee shocks are tested as
EXCESS over that surface -- transmission beyond what an impact-blind shock at the SAME basis dislocation
AND book capacity predicts, jointly conditioned rather than along the basis line alone.

- onset_response.excess_over_surface(panel, fit): residual = transmission - f_surface(basis, holl) on
  the salience-curated events; mean-excess Ibragimov-Muller t across them. The benchmark carries the
  exogeneity, so the excess is identified even when the tail is curated on salience.
- Reports within_support_frac -- the fraction of curated points inside the benchmark's (basis, holl)
  bounding box. A bilinear surface EXTRAPOLATES outside its support, and the marquee shocks are exactly
  the points most likely to sit beyond the scheduled backbone's range, so out-of-box excess is
  extrapolation, not interpolation, and is flagged as such.
- fit_stress_surface now records basis_col / holl_col / y_col so excess_over_surface is self-contained
  (it tests the curated tail on the same axes and transmission the benchmark was fit on).
- run_onset_surface attaches the surface-excess to the co-headline cost axes (joint_cost, cost_max);
  empty (n=0) when the events frame is the pure scheduled backbone, populated once marquee shocks are
  included.
- Guard test_surface_excess.py: recovers a known injected excess against the fitted plane (IM-
  significant, within support); a clean null when the curated tail follows the surface; and the
  extrapolation flag fires (within_support_frac < 1) when curated points sit outside the benchmark box.


## v0.9.2 -- Cost-aggregation comparison: sum vs bottleneck vs product, adjudicated by the diagnostics

Resolves the "why not the multiplicative cost" question by EMITTING all three leg-aggregations and
letting interaction_identified + the jackknife band adjudicate, rather than asserting a winner.

- event_onset_estimate / cross_event_onset now emit three round-trip cost axes off the per-leg costs:
  state_cost_joint (SUM: legs stack linearly), state_cost_max (BOTTLENECK: the binding leg gates the
  arb), and state_cost_prod (PRODUCT, bps^2: the multiplicative comparator).
- run_onset_surface fits the surface on all three (plus the two single-leg hollowness axes) and the
  summary gains a loo_band (leave-one-out jackknife) column -- where the product's leverage fragility
  shows. cost_max stronger than joint_cost is flagged as a bottleneck / joint-withdrawal signature.
- Guard test_cost_aggregation.py: (A) the aggregations obey their invariants (max<=sum; sum>=2*sqrt(prod)
  by AM-GM; max^2>=prod); (B) under realistic CO-MOVEMENT of basis and leg costs, the multiplicative
  PRODUCT carries strictly more leverage/fragility (wider jackknife band) than the additive SUM --
  confirming the product is a dominated choice. Empirically the product is identified-but-fragile (not
  unidentified), and the bottleneck MAX is the strongest-but-most-fragile axis (binding-leg leverage):
  a richer, more honest picture than "the product fails outright".


## v0.9.1 -- run_onset_surface launcher stage: the leg triple on the robust transmission, one command

Wires the whole onset/surface apparatus into a runnable stage and resolves the book-leg decision into a
run-all-three-and-compare exhibit.

- Panel enriched for the LEG TRIPLE: event_onset_estimate / cross_event_onset now emit state_holl_imp
  (the impulse-leg / ES book hollowness) and state_cost_joint (the leg-neutral SPY+ES round-trip
  arbitrage cost) alongside the response-leg state_holl / state_cost.
- onset_response.run_onset_surface(sessions, events_frame, ...): runs the 2-D surface across the triple
  -- joint round-trip cost (primary), ES book (cross / contagion co-headline), SPY book (own-leg
  robustness) -- on the NOISE-ROBUST transmission, reporting b3 on robust AND naive side by side (does
  the spiral survive the bounce correction?), the interaction-identification verdict, and the LOO
  jackknife band, with a decision-legible text summary.
- fit_stress_surface now REFUSES ill-conditioned designs: when the centered design is near-singular
  (an unspanned plane / near-constant capacity axis, cond>1e9) it returns interaction=NaN with
  ill_conditioned=True rather than an exploded least-norm coefficient. The verdict belongs to
  interaction_identified, which flags exactly that case.
- run_onset_surface.py: a CLI launcher (demo + extract paths). The scheduled calendar (FOMC by default)
  drives the date universe; sessions are loaded via run_analysis's loader; prints the summary and writes
  the onset panel CSV.
- run_liberation_day.sh: new `onset` stage (STAGE=onset ./run_liberation_day.sh), tunable via
  ONSET_RANGE / ONSET_CATS / ONSET_PRE / ONSET_POST / ONSET_NOTIONAL.
- Guard test_onset_surface_stage.py: end-to-end plumbing over deep-book sessions on a spanned plane --
  the panel carries the enriched columns, all three axes are fit with the full field set, the primary
  (joint_cost) axis is well-conditioned, and degenerate single-leg axes report NaN honestly.


## v0.9.0 -- Noise-robust identification: defending the transmission against bid-ask-bounce attenuation

The single most important defense of the headline. Rigobon cancels TIME-CONSTANT microstructure noise,
but NOT noise that SHIFTS at the onset: post-release the touch thins and the BBO flips faster, so
mid-return noise variance rises, inflates the post-regime diagonal, and ATTENUATES the identified
transmission (errors-in-variables). A hollow pre-window predicts a larger such shift -- so the bounce
confound manufactures a spurious negative capacity/interaction effect that MIMICS the liquidity spiral.
This release neutralizes it and proves, on synthetic data, that it does.

- _robust_window_cov + transmission_robust in event_onset_estimate: each regime's 2x2 covariance
  re-estimated with noise-corrected TSRV variances on the diagonal (where the bounce confound lives)
  and the Hayashi-Yoshida cross-covariance (unbiased by per-leg idiosyncratic noise). id_strength_robust
  is the robust path's identification strength.
- Microstructure controls in the panel: pre_noise_bps (the BNHLS bounce LEVEL omega, which varies with
  touch fragility even when the quoted spread is tick-pinned, as in SPY/ES) and pre_spread_bps.
- state_cost: the joint round-trip arbitrage cost over the pre-window (cost_to_fill_bps) -- the
  symmetric, leg-neutral capacity axis, an alternative interacting conditioner (holl_col="state_cost").
- fit_stress_surface gains y_col and covar_cols: fit on transmission_robust and/or add centered controls.
- Guard test_noise_robust_surface.py -- the referee's objection, run and defused: a DGP with NO spiral
  but post-onset bounce scaled by basis*holl; the naive surface manufactures a significant negative b3,
  transmission_robust shrinks it to insignificance, and a PRE-window noise covariate provably cannot
  rescue it (the confound is the POST-onset shift) -- so the noise-robust covariance is the necessary fix.


## v0.8.9 — The 2-D liquidity-spiral interaction: f(basis, hollowness) + its identification gate

f(state) becomes a SURFACE. The mechanism the paper claims -- a basis dislocation transmits worse when
the book cannot absorb the arbitrage -- is multiplicative (Brunnermeier-Pedersen / limits-of-arbitrage):
the additive model basis + hollowness cannot represent it, so the interaction is the MINIMUM
specification in which the spiral can exist, not an extra. The onset estimator now carries the second
predetermined axis and the surface fit comes with a diagnostic that says whether the events identify it.

- **Second predetermined state wired into the onset.** `event_onset_estimate` / `cross_event_onset` now
  also return `state_holl` -- the pre-window book HOLLOWNESS (near-touch resiliency = arbitrage-capacity
  margin) from `liquidity_stress.stress_state`, on the response leg by default. Pass `book_asset=impulse`
  to condition on the OTHER leg's book instead (the cross-asset propagation channel, robust to
  own-instrument microstructure contamination). Level-capped and exception-safe: a shallow book yields
  NaN, never a crash.
- **`fit_stress_surface(panel, ...)`** fits the CENTERED bilinear surface
  transmission = b0 + b1(basis-b̄) + b2(holl-h̄) + b3(basis-b̄)(holl-h̄); b3 is the spiral interaction.
  Centering removes the mechanical collinearity, leaving only genuine corner-emptiness. b3 is a
  regression coefficient, so Ibragimov-Muller does NOT apply to it (IM stays for the per-event
  transmission/excess); the honest few-events companion is the leave-one-event-out jackknife band on b3,
  returned here.
- **`interaction_identified(panel, ...)`** -- the b3-analog of id_strength. If events cluster on the
  diagonal (basis and book stress together, the natural correlation), b3 is unidentified and its SE
  explodes -- you would miss a real spiral because the events don't span the plane. Reports cross-event
  corr(basis, holl), the VIF of the centered interaction, and the median-split quadrant counts, calling
  out the sparsest OFF-DIAGONAL corner (the cells that pin b3). NOTE: post-centering the VIF stays ~1
  even at corr~0.99, so the verdict keys on corr and corner coverage too, not VIF alone.
- The onset interaction identifies the STATIC complementarity the spiral predicts -- its cross-sectional
  shadow, not the dynamic feedback loop (that stays descriptive in the cascade tier). This is also the
  reason the design needs events scattered across a PLANE (30-50), not a line.
- **Guard `test_stress_surface.py`**: the diagnostic fires on diagonal-only events and passes on
  plane-spanning ones; the fit recovers a known b3 + main effects; the jackknife band brackets the
  estimate; cross_event_onset emits a finite state_holl on a deep book.


## v0.8.8 — Onset sensitivity diagnostics: validating the estimator's assumptions on real data

The onset estimator (v0.8.7) is correct where its assumptions hold; these tools profile, on real
sessions, whether they hold -- before any coefficient is trusted. None of them MAKE the design valid;
they reveal whether it is.

- **`onset_sensitivity.post_window_profile(df, onset_ts, post_grid, ...)`** -- per-event transmission as
  the post-window grows. A flat plateau near the onset is the pre-feedback estimate; a large DEPARTURE
  (either direction) signals the window has reached the basis-feedback cascade, where Rigobon's
  constant-A assumption breaks and the estimate destabilises. Read the coefficient off the plateau.
- **`window_sweep(sessions, events_frame, pre_grid, post_grid, ...)`** -- the pooled benchmark f(state)
  slope across a (pre,post) grid: stable across a sensible band = window-robust; swinging = window-driven.
- **`id_strength_profile(panel)` / `id_threshold_sweep(panel)`** -- the distribution of Rigobon
  identification strength across events (share weakly identified, quantiles, share with no onset
  variance rise), and whether the benchmark slope survives progressively dropping weak-ID events. A
  common-scale variance jump is unidentified however clean the eigendecomposition looks; this is the
  check that catches it.
- **`cholesky_bracket(df, onset_ts, ...)`** -- the order-free Rigobon coefficient next to the two
  Cholesky orderings; outside the bracket flags label-switching or weak ID.
- **Guard `test_onset_sensitivity.py`**: the diagnostics DETECT the failure modes -- post_window_profile
  fires on a within-post structural break and stays quiet on a clean session (window discrimination);
  a common-scale onset is flagged weakly identified despite a large variance ratio while a differential
  onset is strong; id_strength_profile / id_threshold_sweep act correctly; the bracket brackets.


## v0.8.7 — Cross-event onset driver: the identified estimate of f(state)

The analytical core. Produces the identified headline estimate of the stress-response function by
pooling per-event ONSET observations across the multi-event frame.

- **`onset_response.event_onset_estimate(df, onset_ts, ...)`** -- one event's onset observation: the
  contemporaneous ES->SPY transmission identified WITHIN-event off the onset variance shift (Rigobon
  2-regime heteroskedasticity ID, calm pre vs shocked post), paired with the PREDETERMINED basis
  dislocation (|basis| over the pre-window). Reports the post/pre variance ratio and the Rigobon
  eigenvalue separation (weak-ID flag).
- **`cross_event_onset(sessions, events_frame, ...)`** builds the cross-event panel
  [date, category, selection, impact_blind, state, transmission, var_ratio, id_strength, ...],
  resolving each onset from the release time (or open-gap for overnight events).
- **`fit_stress_response(panel, degree)`** fits f(state) on the IMPACT-BLIND benchmark (degree>=2 gives
  the convex spiral term); **`excess_over_benchmark`** tests the SALIENCE-CURATED tail as excess over
  the benchmark-extrapolated f with an Ibragimov-Muller t (the benchmark carries the exogeneity, so the
  excess is identified even when the tail is curated); **`onset_inference`** is IM across events.
  **`run_onset_study`** ties it together.
- Identification is at the onset (pre-feedback): the within-crisis cascade endogenizes the slope and is
  DESCRIBED (markov_switching_vecm), not fit as f(state) here.
- **Guard `test_onset_response.py`**: Rigobon onset recovers a known transmission off a differential
  variance shift; the benchmark slope is recovered on impact-blind events only; the excess test flags
  injected excess (IM-significant); end-to-end wiring recovers per-event betas (corr 1.00).


## v0.8.6 — Impact-blind unscheduled feed: selection provenance + scheduled-uncertainty stratum

Operationalizes the unscheduled stratum of the sampling design. "Impact-blind unscheduled" decomposes
into three provenances, and identification treats them differently, so every event now carries a tag.

- **Selection provenance.** Every event has a `selection`: `scheduled` (release schedule),
  `scheduled_uncertainty` (known date / unknown outcome -- elections, referendums, deadlines, OPEC,
  rulings), `news_classified` (categorized feed selected on EX-ANTE prominence), or `salience_curated`
  (the hand-picked crisis registry). `shocks_frame` exposes `selection` and a derived `impact_blind`
  boolean (true for the first three): impact-blind events carry exogeneity for the benchmark;
  salience-curated ones are read only as excess over it. The crisis registry defaults to salience_curated.
- **Scheduled-uncertainty stratum** -- the impact-blind way to sample "unscheduled" shocks (take every
  one, like FOMC). US general elections are GENERATED (Tue after 1st Mon in Nov, even years); other
  known-date events (Brexit, debt-ceiling X-date) are seeded -> extend or load a feed. Tagged overnight.
- **Loader provenance.** `load_release_schedule` / `load_ics` gain `selection=`; pass "news_classified"
  for a categorized news/event feed or "scheduled_uncertainty" for a known-date calendar. Unscheduled
  event types (election/referendum/opec/deadline/ruling/geopolitical/sanction) added to the name map;
  they default to overnight (open-to-open) unless the feed supplies an intraday time.
- **Guard `test_unscheduled_feed.py`**: election-generator dates; provenance + impact_blind partition;
  loader news_classified tagging flowing through the frame impact-blind, defaulting to overnight.

Note: when an event is in BOTH the salience-curated registry and an impact-blind feed, the registry row
currently wins the (date, category) dedup (conservative -- it stays salience_curated).


## v0.8.5 — Release-schedule loader (agency-general): ingest the real calendar

Completes the backbone-population path: the scheduled dates are DATA, so this adds a loader that
ingests an authoritative schedule rather than hand-keying ~70 dates.

- **`market_shocks.load_release_schedule(source, ...)`** ingests a CSV path, a DataFrame, or an
  iterable of rows with columns {date, category|release|name, [time_et|time]}. Release NAMES map to
  category codes via `RELEASE_NAME_TO_CATEGORY` (agency-general: FOMC; BLS NFP/CPI/PPI/JOLTS; BEA
  GDP/PCE; Census RETAIL/DURABLE/TRADE/HOUSING; ISM; sentiment; ADP; claims); a `category` column is
  used as-is. Times come from a time column or `DEFAULT_RELEASE_TIME_ET[category]` (08:30 macro,
  10:00 ISM/sentiment/JOLTS, 14:00 FOMC, 08:15 ADP). NaN cells handled; unmapped names skipped.
- **`load_ics(source, ...)`** parses an iCal export (BLS calendar / economic-calendar feed): VEVENT
  SUMMARY -> category, DTSTART -> date/time. No external deps.
- Loaded releases flow through `scheduled_events` / `shocks_frame` / `event_dates` / `release_times`
  once registered (`register_releases` / `clear_loaded`); `scheduled_events(categories=None)` now
  means all available categories including loaded ones.
- **Guard `test_release_loader.py`**: name+category mapping (incl. GDP/BEA, Retail/Census, ISM) with
  correct release times and classes; flow-through + clear-revert; iCal parsing; (date,category) dedup.


## v0.8.4 — Scheduled-event calendar: the impact-blind backbone for the multi-event design

Adds the scheduled-release frame that anchors f(state) impact-blind -- the spine of the sampling
strategy: take EVERY scheduled release over a wide, regime-dispersed span (not a contiguous window,
not a volatility filter); the ones that land during turbulent stretches supply impact-blind
HIGH-pre-state onset points.

- **`market_shocks.scheduled_events(start, end, categories=("FOMC","CPI","NFP","OPEX"))`** returns
  scheduled releases in the crisis-registry dict schema. FOMC rate-decision days (2023-2026, 14:00 ET)
  are a VERIFIED table from Fed press releases; OPEX/quad-witching (09:30 ET) is generated by the
  third-Friday rule; CPI/NFP (08:30 ET) are stored as verified tables but only SEEDED with the 2025
  shutdown-revised dates -- complete the full schedule from bls.gov.
- **Release dates are DATA, not a rule.** The 2025 federal shutdown cancelled the Oct-2025 Employment
  Situation and pushed Sep-2025 CPI to 10-24, so FOMC/CPI/NFP are tabular (verified), not generated.
- **`shocks_frame(include_scheduled=True, start, end)`** merges the backbone into the crisis registry,
  de-duplicated on (date, category) so hand-annotated crisis rows win; `event_dates` / `release_times`
  gain the same `include_scheduled` switch (default off -> existing behavior unchanged).
- **Guard `test_scheduled_calendar.py`**: third-Friday math vs known quad-witching dates; FOMC 8/yr at
  14:00; impact-blind enumeration with correct release times; merge/dedup (2024-09-18 once) with the
  default frame unchanged.


## v0.8.3 — Basis conditioning state for the stress-response function

Realizes the design decision to condition cross-asset price discovery on the SPY-ES BASIS and the
book-stress state (not realized volatility): the basis is the size of the law-of-one-price violation
-- the direct health of the arbitrage link -- whereas volatility is its symptom.

- **`liquidity_stress.basis_state(df, predetermined=True)`** assembles `ecm_sde.basis` (the demeaned
  (1,-1) cointegrating residual) into a predetermined conditioning state: signed `basis_bps` and the
  dislocation magnitude `abs_basis_bps`, lagged one step so the state at t is known at t-1 and does
  not condition on the contemporaneous outcome. Pass `abs_basis_bps` as `state=` to
  `irf.local_projection_irf` for the basis-conditioned ONSET IRF -- `theta_low`/`theta_high` then
  trace transmission as a function of how dislocated the link already is. The book-stress state
  (`stress_state`) is the second axis; `local_projection_irf` takes a 1-D state, so basis and book
  stress condition as separate axes (or a composite scalar -- a design choice).
- **Guard `test_basis_state.py`**: (A) `basis_state` recovers the lagged demeaned basis exactly and is
  genuinely predetermined; (B) fed as the state to `local_projection`, it recovers a known
  state-dependent response (level b0 and state-slope b1 within tolerance; transmission ~4x stronger at
  +1 SD dislocation), confirming the conditioner plugs into the state-dependent onset-IRF machinery.


## v0.8.2 — In-process upgraded analysis (no pickle IO); run_robust + multi-event launcher modes

Eliminates the v0.8.1 frame pickle and extends the upgraded analysis to the 100ms lens and a pooled
multi-event design.

- **No pickle round-trip.** `run_analysis.main()` is split into `load_sessions` + a new
  `run_stages(sessions, args)`. `run_contagion` gains `--upgraded`, which calls
  `run_analysis.run_stages` IN-PROCESS on the same in-memory sessions after the contagion pipeline —
  no serialization, no second extraction/load. (`--save-pickle` is kept as an opt-in for process
  isolation or inspecting frames.) Tradeoff: peak memory is higher than the sequential two-process
  approach, but the memory-heavy phase is extraction (once, inside `load_sessions`); the downstream
  analyses run on the small 1s frames.
- **`run_liberation_day.sh` MAIN now uses `--upgraded`** instead of the pickle; **ROBUST (100ms) gains
  `--upgraded --upgraded-skip dcc`** (cDCC is the expensive stage at 100ms). Upgraded output lands
  under `<out>/upgraded/`.
- **New `STAGE=multievent`** pools several 2025 shocks (tariffs 04-03/04-09 + Iran 06-13/06-23) plus a
  calm control pool via a curated `--dates` list (only the days needed, not a contiguous range), for
  enough TREATED day-clusters to make the wild-cluster bootstrap valid. Honest caveat printed at run
  time: pooling across event types assumes a common treatment effect and mixes timing regimes; set
  `ME_CLASSES=TRADE_POLICY` for homogeneous identification. Pooling buys cluster count, not cleanliness.


## v0.8.1 — Restore the contagion driver; route run_liberation_day.sh through the upgraded analysis

Fixes a regression: `run_contagion.py` (and `run_mean_variance.py` / `mean_variance.py`) were dropped
from the packages at v0.3.0, which silently orphaned `run_liberation_day.sh` — the Liberation-Day
launcher is built entirely around `run_contagion.py`, so in v0.3.0–v0.8.0 its pilot/main stages failed
with "No such file." All three modules are restored (verified: `run_contagion --selftest` passes —
DiD/RD estimators, 16 tables, 0 errors). And the launcher now routes the run through the re-floored
v0.6–0.8 analysis, not just the original contagion pipeline.

- **`run_liberation_day.sh` MAIN stage now does both, on one extraction.** `run_contagion.py` extracts
  the window once and `--save-pickle`s the (date,regime,df) frames; the launcher then runs
  `run_analysis.py --source load --pickle <frames> --legacy --event-class TRADE_POLICY` on the *same*
  frames — the day-clustered/wild-cluster inference, Hayashi-Yoshida + Rigobon leadership, the two-axis
  liquidity-stress conditioner, the TARIFF event-clock study, and the LEGACY before/after tables. No
  second tape pull.
- **`run_contagion.py` gains `--save-pickle PATH`** — after load + `--auto-regime`, it dumps the
  sessions so a downstream `run_analysis --source load` reuses them. (Both drivers already share
  `run_analysis.load_sessions`, so the frame format round-trips exactly.)
- The window still ends 2025-04-08 (excludes the 04-09 pause), so the TARIFF event study is a clean
  single overnight regime rather than the mixed overnight+intraday case.


## v0.8.0 — Event-clock study wired into the pipeline (`--events` / `--event-class`)

The 2025 exogenous events were already in the registry (Liberation Day 04-03 and the 04-09 pause,
Israel/US strikes on Iran 06-13 / 06-23, the 2024 yen-carry unwind, …) and the event-study driver was
already category-general; what was missing was a path to it from `run_analysis` and a guard proving a
NON-MWCB category flows through end-to-end. Both added.

- **`run_analysis --events TARIFF,GEOPOLITICAL` / `--event-class TRADE_POLICY,GEOPOLITICAL`** — a new,
  opt-in `events` stage resolves each event's release time (the event clock), matches controls on an
  ex-ante stress proxy, and runs the event-study profile / DiD / RD-in-time / cross-market (ES→SPY)
  spillover. Off by default; gated like `--legacy`. Requires the extracted sessions to include BOTH
  event days AND candidate control days (use `--date-range` over a window).
- **Ex-ante state, not an outcome.** Control matching uses `_daily_vol_state` — the PRIOR session's
  realized vol of the SPY mid (t-1, never t's own outcome) — so it does not condition on the dependent
  variable. Replace with a prior-close VIX series on real data.
- **Guard `test_event_study_2025.py`** drives the TARIFF category (2025-04-03 overnight + 2025-04-09
  intraday) through the whole path: the intraday boundary resolves to 13:18; 2 treated + 4 clean
  controls assemble; the panel builds; DiD/RD/spillover come back finite and correctly signed (treated
  cost-to-fill jumps post-release). The driver self-test had only covered MWCB.
- Honest caveat the guard surfaces: TARIFF mixes overnight (04-03) and intraday (04-09) timing
  (`mixed_regime=True`); a clean single-regime study analyses the two timings separately.


## v0.7.0 — Legacy before/after comparison (`--legacy`) + bash launcher

The single most persuasive referee exhibit: the paper's ORIGINAL estimators computed next to the
upgraded ones on the **same data**, so it is explicit which conclusions survive the re-flooring. New
module `legacy_tables.py`, a `--legacy` flag on `run_analysis.py`, and a `run_analysis.sh` launcher
with the comparison ON by default.

- **`legacy_tables.py` — four paired contrasts**, each best-effort (a failure in one leaves the rest):
  - *comovement* — Pearson realized correlation vs Hayashi-Yoshida (Epps-robust);
  - *inference* — pooled OLS iid t vs day-clustered t + wild-cluster-bootstrap p (the *** collapse);
  - *liquidity* — quoted-spread R² vs two-axis book-stress-state R² (+ partial R²);
  - *identification* — recursive Cholesky (futures ordered first) vs Rigobon heteroskedasticity ID:
    the Cholesky **assumes** ES←SPY = 0; Rigobon **estimates** it, exposing the ordering artifact.
- **`--legacy` / `--no-legacy`** on `run_analysis.py` (BooleanOptionalAction, default OFF for a bare
  python call). When on, a `legacy` stage emits the before/after tables into `report.md`, the per-output
  CSVs, and `summary.json`. Gated cleanly — off by default, unaffected by `--only`/`--skip`.
- **`run_analysis.sh`** launcher: legacy ON by default (`LEGACY=false ./run_analysis.sh ...` to disable);
  a bare `./run_analysis.sh` runs a self-contained demo; all other args forwarded verbatim;
  `PYTHON=python3` selects the interpreter.
- **Honest read of the demo.** The Epps gap, the iid→clustered t collapse, and the spread→state R² jump
  are data-dependent and muted on the *synchronous synthetic* demo; they show real magnitude on the
  *asynchronous* April-2025 tape. The identification contrast (assumed-zero vs estimated) is stark even
  on demo. The wild bootstrap is unreliable at the MWCB G=4 — read Ibragimov-Müller there.


## v0.6.0 — Liquidity-stress conditioner + validation battery; methods vignette

Realizes the paper's reframing — liquidity stress as the CONDITIONING variable, MWCB demoted to a
special case — with a two-axis, book-derived stress state plus the validation a referee demands that it
carries signal the quoted spread cannot. New module `liquidity_stress.py`, guarded end-to-end; the
methods vignette now travels with the code.

- **`liquidity_stress.py` — the conditioner.** `stress_state` returns two near-orthogonal, cross-asset-
  comparable, book-derived axes: `depth_illiq` (aggregate thinning) and `hollowness` (near-touch
  resiliency). `cross_stress_state` aligns both assets so one asset's price discovery can be conditioned
  on the OTHER's book stress. `cost_to_fill_bps` is the single cost-to-trade-Q summary;
  `quoted_spread_bps` / `amihud_illiquidity` are the benchmarks the state must beat.
- **Validation battery.** `incremental_content` (nested partial-R² + cluster-robust joint Wald: does the
  state beat spread + Amihud?); `fpca_alignment` (the hand-chosen axes ARE distinct data-driven FPCA
  components); `lambda_validation` (the state predicts realized impact). Trade-derived measures validate,
  never condition — preserving the book-derived exogeneity of the state.
- **Guard `test_liquidity_stress.py`** (tick-pinned synthetic book, two independent latent factors): the
  spread is dead (R²≈0); the axes recover their factors (corr 1.00 / 0.92) and are orthogonal (|corr|≈0);
  incremental content over spread+Amihud is partial-R² 0.90; the resiliency axis adds 0.69 beyond
  level+spread; FPCA recovers both axes as distinct labeled components; the state predicts realized
  impact (t=81 / 195).
- **`METHODS_VIGNETTE.md`** added to the repo: per-indicator what / why / how-it-enhances-the-paper /
  how-to-interpret, the leadership-triangulation argument, the reporting checklist, and honest limits.


## v0.5.0 — Hardening: jump-robust variation, Rigobon identifying-variation diagnostic, FPCA interpretation, cross-impact symmetry

Four hardening upgrades to already-working modules, each verified by a synthetic known-answer guard.

- **noise_robust_cov.py — jump-robust variation** (H3 / MWCB days). Adds `bipower_variation` (BNS 2004),
  `threshold_variance` (Mancini 2009 truncated RV), `tripower_quarticity`, and `jump_test` (BNS /
  Huang-Tauchen ratio test); `realized_variance` gains `'bipower'`/`'threshold'` methods. Guard
  `test_jumps.py`: on a diffusion + known jumps, naive RV inflates to ~2x IV while BV (1.05x) and TRV
  (0.96x) recover the continuous integrated variance; the test fires (z≈26) on jumps and is quiet on the
  no-jump control.

- **rigobon_id.py — identifying-variation diagnostic**. Adds `identification_diagnostic`: a variance-ratio
  + relative-eigenvalue-gap test that flags the failure mode where regimes differ only in SCALE (common
  variance factor → coincident eigenvalues → A not identified). Guard `test_rigobon_id.py`: relative
  heteroskedasticity → identified, C recovered (0.398/0.194); common-scale regimes → flagged not-identified.

- **functional_liquidity.py — component interpretation**. Adds `interpret_components`: labels each FPC as
  level / slope / curvature by loading-curve sign changes (the economic reading of the liquidity factors).
  Guard `test_fpca_interp.py`: an injected level+slope structure is recovered and labeled, variance shares
  matching the injection (0.79 / 0.20).

- **cross_impact.py — Schneider-Lillo symmetry test**. Adds `cross_impact_symmetry`: a Wald test of the
  no-dynamic-arbitrage restriction Lambda_12 = Lambda_21 using the joint cross-equation HAC covariance
  (the per-equation SEs cannot). Guard `test_xi_symmetry.py`: a symmetric DGP is not rejected (p≈0.67), an
  asymmetric one is (p<1e-4).


## v0.4.0 — Tier-1 methodology upgrades (gating inference + measured lead-lag + corrected cDCC)

Three estimator upgrades from the methodology upgrade plan, each verified by a synthetic
known-answer guard (a data-generating process with ground truth). All additive; existing
functions and module self-tests unchanged.

- **inference.py — wild cluster bootstrap + Ibragimov-Müller** (Stage-1 gating).
  Adds `cluster_robust_ols` (Liang-Zeger 1986), `wild_cluster_bootstrap` (Cameron-Gelbach-Miller
  2008, restricted WCR; Webb 2014 six-point weights when clusters ≤ 12), and `ibragimov_muller`
  (2010/2016 group t-statistic). Targets the paper's pooled inference over ~23.4M observations
  whose effective N is ~20 days / 4 MWCB events. Guard `test_wild_cluster.py`: under the null with
  G=10 clusters the iid t over-rejects at ~69%, the wild cluster bootstrap at ~5% and
  Ibragimov-Müller at ~7-8%, both retaining full power.

- **noise_robust_cov.py — HRY lead-lag** (Stage-2 gating). Adds `lead_lag` (Hoffmann-Rosenbaum-
  Yoshida 2013 shifted-HY contrast), turning the paper's ASSUMED Cholesky "futures leads" ordering
  into a measured, signed quantity (positive ⇒ first series leads). Guard `test_leadlag.py`
  recovers an injected ±50 ms lead and the no-lead null with correct sign.

- **dcc_garch.py — Aielli cDCC** (serves the named Copula-DCC-GARCH method). Adds `cdcc_fit`
  (Aielli 2013 corrected cDCC with consistent correlation targeting) alongside Engle DCC, and makes
  it the default in `dcc_garch_x` (`dcc_method="cdcc"|"engle"`). Guard `test_cdcc.py` recovers
  (a, b, S) from data generated by a true cDCC.

Deferred by design (unchanged): the Putniņš ILS in `price_discovery_shares.py` remains a documented
stub — the vanilla formula is academically contested (Shrestha-Lee 2023; Shen-Zivot 2024; corrected
Shen-Zhang-Zivot 2025 JEF) and is intentionally not implemented from memory.

## v0.3.0
Scope/spine triage; severed stray `paper_tables`→`mstbook_loader` self-test import; removed
superseded drivers (`run_contagion`, `run_mean_variance`, `mean_variance`).

## v0.2.0
Three verified fixes: `lob_reconstruct` crossed-book (price-keyed reduce); `ecm_sde` day-clustered
SE on the headline gradient; `price_discovery_shares` `panel_vecm` day-clustered SE.
