# Hand-carry staging — macro release calendar + surprises (non-options)

Staged 2026-08-22. These files are the non-options portion of the hand-carry
manifest, built as far as this (egress-restricted) session's clean sources
allow. They graduate to `sample_inputs/` when the v0.9.88 loaders land.

## Files

### `fomc_announcements.csv` — COMPLETE
Every FOMC statement 2017-01 → 2026-12: `dt, time_et, scheduled,
rate_change_bp, note`. Scheduled statements at 14:00 ET; the two 2020
unscheduled cuts carry their real times (2020-03-03 10:00, 2020-03-15 17:00
Sunday — that one lands on the 2020-03-16 open). The 2020 scheduled March
meeting (Mar 17–18) issued no separate statement and has no row.
`rate_change_bp` is filled 2017–2025 from the record; 2026 rows are blank
until `build_macro_surprises.py` fills them from DFEDTARU.

Provenance: hard-coded from record (knowledge through 2026-01), with 2025-10-29,
2025-12-10, and the full 2026 schedule spot-verified against
federalreserve.gov search results on 2026-08-22. The fill script re-verifies
every rate change against FRED before anything is used.

Cross-check against the design of record (`sample_design_20260817.txt`):
five sample sessions are FOMC statement days —
**2017-09-20, 2020-01-29, 2021-01-27, 2023-07-26, 2024-12-18** — all with the
14:00 ET statement inside the analysis grid.

### `macro_surprises.csv` — SCAFFOLD, partially filled (426 rows)
One row per (ref_month, release) for CPI / CPICORE / NFP plus one per FOMC
statement. Columns:
`ref_month, dt, release, time_et, actual, consensus_median, surprise_sd,
ff_futures_delta_bp, path_2y_bp, source_actual, status`

Filled here:
- **FOMC rows**: complete (dates, times, 2017–2025 rate changes).
- **CPI / CPICORE `actual`** (headline and core MoM %, seasonally adjusted):
  computed from the BLS dataset bundled in the `cpi` PyPI package v2.0.10
  (index levels through 2025-12). These are **latest-revised**, not first
  prints — CPI revisions are small (seasonal-factor updates) but the fill
  script cross-checks against fresh FRED data and reports drift.
- **2026 CPI release dates** for ref 2026-01/04/05/06 (from bls.gov archive
  URL slugs) and four 2026 MoM values from press coverage — the latter are
  marked `websearch_snippet_UNVERIFIED` and must be confirmed by the script.

Deliberately blank (with `status` saying why):
- CPI/NFP **release dates** 2017–2025 → filled by the script from the BLS
  archived-release indexes (URLs encode release dates).
- **NFP actuals** → script (FRED PAYEMS; ALFRED first prints if FRED_API_KEY
  is set — NFP revisions are large, prefer first prints for surprises).
- **path_2y_bp** → script (DGS2 daily close change in bp).
- **consensus_median** → Bloomberg/Reuters terminal only. `surprise_sd` is
  computed downstream once consensus is in; it is a per-series constant, not
  hand-entered per row.
- **ff_futures_delta_bp** (Kuttner) → best computed ON MIDAS from the lake:
  30-Day Fed Funds futures (ZQ, CBOT) ride the same Globex feed as ES. Probe
  `mstwx-lakequery --date 20240918 -s futures -p ZQV4 -m mt_trade`; if ZQ is
  in the capture, an intraday window around the statement beats any daily
  settlement delta and nothing needs to be hand-carried for this column.

Known anomalies (rows flagged `shutdown_anomaly`): ref 2025-10 and 2025-11 —
the Oct–Nov 2025 shutdown; the October 2025 CPI was never published (the
bundled dataset genuinely lacks 2025-M10), so both MoM values need hand
resolution from the actual BLS releases.

### `build_macro_surprises.py` — run OUTSIDE, once
Internet-side filler/verifier (stdlib only). Fetches the BLS archive indexes,
FRED (PAYEMS, DGS2, DFEDTARU, CPIAUCSL, CPILFESL), fills every blank it can,
re-verifies everything committed here, and refuses (exit 1) if a committed
value disagrees with the fresh source. Output `macro_surprises_filled.csv`
is what gets hand-carried; consensus columns are then added at the terminal.

## Not in this staging (and why)
- `rates_divs.csv`, SPY/ES/SPX option EOD files — options-scope, per the
  agreed split; the rates/dividends file only exists to support greeks.
- FINRA/CBOE off-exchange share — superseded by SIP (`-s sip`) in the lake.
- SSR trigger dates, zero-commission flag (2019-10-01) — constants that go
  straight into `macro_calendar.py` / `market_eras.py` at v0.9.88; the SSR
  list comes from the memo's exclusion analysis.
