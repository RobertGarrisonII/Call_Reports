# Hand-carry manifest — everything to produce manually

Consolidated 2026-08-22. Excludes what is already staged in this directory
(FOMC calendar, CPI actuals) and what is computable in-lake at v0.9.88 (SIP
retail flow, off-exchange share, odd-lot share, vol-ETP dollar volume).
Priority: P0 = unblocks macro typing; P1 = vol complex (small files, high
value); P2 = options surface / GEX package.

## A. One script run — internet laptop (P0)

**A1.** `python3 build_macro_surprises.py` (in this directory) →
`macro_surprises_filled.csv`. Set `FRED_API_KEY` first if you have one so
NFP actuals are FIRST PRINTS (ALFRED) rather than revised. Review anything
it reports as NOTE/PROBLEM, especially the two `shutdown_anomaly` months
(ref 2025-10, 2025-11).

## B. Terminal-only — Bloomberg/Reuters (P0)

**B1.** Fill `consensus_median` in `macro_surprises_filled.csv`:
  - CPI rows: survey median headline MoM % (SA)
  - CPICORE rows: survey median core MoM % (SA)
  - NFP rows: survey median payroll change (thousands)
  - FOMC rows: optional (surveyed expected target change, bp)
`surprise_sd` stays empty — the stack computes it per series from the filled
column.

## C. Vol complex — Cboe, free downloads (P1)

**C1.** `vix_futures_eod.csv` — CFE historical settlements, front 4 monthly
contracts, 2017-01 → present:
```
dt, contract, settle, volume, open_interest
```

**C2.** `vvix_daily.csv` — VVIX index daily close, 2017-01 → present:
```
dt, vvix
```

**C3.** `vol_etp_aum.csv` — shares outstanding + leverage for the vol ETPs
(issuer sites / terminal), 2017-01 → present:
```
dt, ticker, shares_outstanding, leverage
```
Tickers: VXX (incl. the VXX→VXXB→VXX series break Jan 2019), UVXY, SVXY,
XIV (through its 2018-02 termination). Record SVXY's leverage change
(-1.0 → -0.5, late Feb 2018) and UVXY's (2.0 → 1.5) as leverage-column
changes, not footnotes.

## D. Options EOD files — with IV (P2)

Common filters for all three: expiry ≤ 60 calendar days; strikes within
±20% of spot (or out to 10-delta); make sure ≥ 2 expiries survive per day
(a ~weekly and a ~monthly) or the term-structure slope is uncomputable.
Span 2017-06 → present. `iv` is REQUIRED (vendor IV much cheaper than
in-house de-Americanization).

**D1.** `spy_options_eod.csv`:
```
dt, expiry, strike, cp, open_interest, volume, close_price, iv
```

**D2.** `es_options_eod.csv` — CME daily bulletin / settlement file;
quarterlies + EOM + Mon/Wed/Fri weeklies. Needed even if ES option ticks
turn out to be in the lake — OI is a clearing figure, never on the tape:
```
dt, product, expiry, strike, cp, settlement, open_interest, volume
```

**D3.** `spx_options_eod.csv` — optional but recommended (majority of
index-complex gamma). Same schema as D1.

**D4.** `rates_divs.csv` — supports greeks/forwards for D1–D3:
```
dt, sofr_on, spy_div_yield
```
(EFFR for dates before SOFR publication, Apr 2018.)

## E. Lists to send (no files, go straight into code)

**E1.** SSR trigger dates for SPY (from the memo's exclusion analysis).
**E2.** Nothing else — zero-commission flag (2019-10-01) and early-close /
holiday calendars are already hard-coded.

## F. Workbench probes — MIDAS, one command each; results change the manifest

**F1.** Fed Funds futures in lake? →
`mstwx-lakequery --date 20240918 -s futures -p ZQV4 -m mt_trade`
If yes: `ff_futures_delta_bp` (Kuttner) is computed in-lake from an intraday
window around the statement — nothing to hand-carry for that column.
If no: add FF futures daily settlements to the terminal pulls.

**F2.** VIX futures in lake? →
`mstwx-lakequery --date 20240805 -s futures -p VXQ4 -m mt_trade`
If yes: C1 becomes optional (still handy as an EOD cross-check).

**F3.** SIP odd-lot A/B →  SPY `mt_trade` on one date with `-s direct` vs
`-s sip`; compare size<100 print counts. Sizes the BJZZ odd-lot blind spot;
decides whether SIP retail flow is near-complete or a lower bound.

**F4.** ES option ticks in lake? (curiosity only — D2 needed regardless) →
`mstwx-lakequery --date 20240918 -s futures -p <ES option symbol> -m mt_trade`

## Delivery

Drop files in this directory (or send them in-session) and say build —
loaders, QC gates, and per-day covariate joins land as v0.9.88 alongside
the SIP/retail package.
