#!/usr/bin/env python3
"""Fill and verify macro_surprises.csv from public sources. RUN THIS ON AN
INTERNET-CONNECTED MACHINE (laptop, not MIDAS), then hand-carry the output.

Fills, per row status:
  * CPI / CPICORE / NFP release dates (``dt``)  -- from the BLS archived-release
    indexes, whose URLs encode the release date (cpi_MMDDYYYY.htm).
  * NFP ``actual`` -- PAYEMS month-over-month change, thousands (FRED; these are
    LATEST-REVISED values -- see the first-print note below).
  * ``path_2y_bp`` -- DGS2 close on release day minus prior business-day close,
    in bp (FRED). Daily proxy: for 08:30 releases it brackets the print; for
    14:00 FOMC it includes the morning drift. Documented, not hidden.
  * FOMC ``actual`` (2026 rows) -- from DFEDTARU changes (FRED), and VERIFIES
    the hard-coded 2017-2025 rate changes the same way.
  * Verifies committed CPI actuals against fresh CPIAUCSL/CPILFESL (revisions
    move these a little every January; diffs are reported, not auto-clobbered).

Never touches: consensus_median, surprise_sd, ff_futures_delta_bp. Consensus
medians come from your terminal (Bloomberg ECO / Reuters poll). surprise_sd is
then computed downstream by the stack, not stored per row here. The Kuttner
ff_futures_delta_bp is best computed ON MIDAS from the lake itself if the
CBOT 30-Day Fed Funds futures (ZQ) are in the Globex capture -- an intraday
window around the statement beats any daily settlement delta; probe with
  mstwx-lakequery --date 20240918 -s futures -p ZQV4 -m mt_trade
before hand-carrying anything for that column.

First prints: FRED serves revised histories. CPI revisions are small (seasonal
factors); NFP revisions are NOT. If FRED_API_KEY is set in the environment,
NFP actuals are taken from ALFRED first-release vintages instead.

Usage:  python3 build_macro_surprises.py [--in macro_surprises.csv]
                                         [--out macro_surprises_filled.csv]
Writes the filled CSV plus a verification report to stdout. Exit 1 if any
committed value it can check disagrees with the fresh source.
"""
import argparse, csv, datetime as dt, io, json, os, re, sys, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (macro_surprises builder; research use)"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")

def fred_csv(series, start="2016-01-01"):
    txt = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}")
    out = {}
    for row in csv.DictReader(io.StringIO(txt)):
        k = [c for c in row if c.lower() in ("date", "observation_date")][0]
        v = row[series].strip()
        if v not in ("", "."):
            out[row[k]] = float(v)
    return out

def bls_release_dates(slug):
    """Release dates from the BLS archive index; URLs look like {slug}_MMDDYYYY.htm."""
    html = get(f"https://www.bls.gov/bls/news-release/{slug}.htm")
    dates = set()
    for m in re.finditer(slug + r"_(\d{2})(\d{2})(\d{4})\.htm", html):
        mm, dd, yyyy = m.groups()
        dates.add(dt.date(int(yyyy), int(mm), int(dd)))
    return sorted(dates)

def assign_ref_months(release_dates, lag_guess=1):
    """Map each release date to its reference month. Normal cadence: release in
    month m covers ref m-1. Anomalies (two releases in one month, skipped
    months -- e.g. the late-2025 shutdown) are flagged for hand review instead
    of guessed."""
    out, flagged = {}, []
    for d in release_dates:
        ry, rm = d.year, d.month - lag_guess
        if rm < 1:
            ry, rm = ry - 1, rm + 12
        key = f"{ry}-{rm:02d}"
        if key in out:
            flagged.append((key, out[key], d))
        else:
            out[key] = d
    return out, flagged

def prior_business_day(d, series):
    for back in range(1, 8):
        p = d - dt.timedelta(days=back)
        if p.isoformat() in series:
            return p.isoformat()
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="macro_surprises.csv")
    ap.add_argument("--out", dest="out", default="macro_surprises_filled.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.inp)))
    problems, notes = [], []

    print("fetching FRED series ...")
    dgs2 = fred_csv("DGS2")
    payems = fred_csv("PAYEMS", "2015-01-01")
    tgt = fred_csv("DFEDTARU", "2016-01-01")
    cpih = fred_csv("CPIAUCSL", "2015-01-01")
    cpic = fred_csv("CPILFESL", "2015-01-01")

    print("fetching BLS archive indexes ...")
    cpi_dt, cpi_flag = assign_ref_months(bls_release_dates("cpi"))
    nfp_dt, nfp_flag = assign_ref_months(bls_release_dates("empsit"))
    for tag, fl in (("CPI", cpi_flag), ("NFP", nfp_flag)):
        for key, first, second in fl:
            notes.append(f"{tag} ref {key}: two candidate releases {first} / {second} -- resolve by hand")

    def mom_pct(series, ref):
        y, m = int(ref[:4]), int(ref[5:7])
        py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
        cur = series.get(f"{y}-{m:02d}-01"); prev = series.get(f"{py}-{pm:02d}-01")
        return None if cur is None or prev is None else round(100 * (cur / prev - 1), 2)

    def payroll_chg(ref):
        y, m = int(ref[:4]), int(ref[5:7])
        py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
        cur = payems.get(f"{y}-{m:02d}-01"); prev = payems.get(f"{py}-{pm:02d}-01")
        return None if cur is None or prev is None else round(cur - prev, 0)

    api_key = os.environ.get("FRED_API_KEY", "")
    def nfp_first_print(ref, release_day):
        """First-release PAYEMS change via ALFRED vintages (needs FRED_API_KEY)."""
        if not api_key:
            return None
        y, m = int(ref[:4]), int(ref[5:7])
        py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
        url = ("https://api.stlouisfed.org/fred/series/observations?series_id=PAYEMS"
               f"&api_key={api_key}&file_type=json&realtime_start={release_day}"
               f"&realtime_end={release_day}&observation_start={py}-{pm:02d}-01"
               f"&observation_end={y}-{m:02d}-01")
        try:
            obs = json.loads(get(url))["observations"]
            vals = {o["date"]: float(o["value"]) for o in obs if o["value"] != "."}
            cur = vals.get(f"{y}-{m:02d}-01"); prev = vals.get(f"{py}-{pm:02d}-01")
            return None if cur is None or prev is None else round(cur - prev, 0)
        except Exception as e:
            notes.append(f"ALFRED first-print fetch failed for {ref}: {e}")
            return None

    for r in rows:
        rel, ref = r["release"], r["ref_month"]
        # --- release dates ---
        if rel in ("CPI", "CPICORE") and not r["dt"]:
            d = cpi_dt.get(ref)
            if d: r["dt"] = d.isoformat()
        elif rel == "CPI" and r["dt"] and ref in cpi_dt and cpi_dt[ref].isoformat() != r["dt"]:
            problems.append(f"CPI {ref}: committed dt {r['dt']} != BLS archive {cpi_dt[ref]}")
        if rel == "NFP" and not r["dt"]:
            d = nfp_dt.get(ref)
            if d: r["dt"] = d.isoformat()
        # --- actuals ---
        if rel == "NFP" and r["dt"] and not r["actual"]:
            v = nfp_first_print(ref, r["dt"]) if api_key else payroll_chg(ref)
            if v is not None:
                r["actual"] = v
                r["source_actual"] = "alfred_first_print" if api_key else "fred_payems_revised"
        if rel in ("CPI", "CPICORE"):
            fresh = mom_pct(cpih if rel == "CPI" else cpic, ref)
            if r["actual"] and fresh is not None and abs(float(r["actual"]) - fresh) > 0.05:
                notes.append(f"{rel} {ref}: committed {r['actual']} vs fresh revised {fresh} "
                             f"(revision drift -- review, committed value NOT overwritten)")
            if not r["actual"] and fresh is not None:
                r["actual"] = fresh
                r["source_actual"] = "fred_cpi_revised"
        if rel == "FOMC" and r["dt"]:
            day = r["dt"]; nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
            if day in tgt and nxt in tgt:
                chg = round((tgt[nxt] - tgt[day]) * 100)
                if r["actual"] != "" and int(float(r["actual"])) != chg:
                    problems.append(f"FOMC {day}: committed {r['actual']}bp != DFEDTARU change {chg}bp")
                if r["actual"] == "":
                    r["actual"] = chg
                    r["source_actual"] = "fred_dfedtaru"
        # --- 2y path ---
        if r["dt"] and not r["path_2y_bp"]:
            d = dt.date.fromisoformat(r["dt"])
            prev = prior_business_day(d, dgs2)
            if prev and r["dt"] in dgs2:
                r["path_2y_bp"] = round((dgs2[r["dt"]] - dgs2[prev]) * 100, 1)
        # --- status refresh ---
        need = []
        if not r["dt"]: need.append("needs_dt")
        if r["actual"] == "": need.append("needs_actual")
        if "shutdown_anomaly" in r["status"]: need.append("shutdown_anomaly;needs_review")
        r["status"] = ";".join(need) if need else "filled"

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nwrote {a.out} ({len(rows)} rows)")
    unfilled = [r for r in rows if r["status"] not in ("filled",)]
    print(f"rows still needing attention: {len(unfilled)}")
    for r in unfilled[:20]:
        print("  ", r["ref_month"], r["release"], r["status"])
    for n in notes:
        print("NOTE:", n)
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        sys.exit(1)

if __name__ == "__main__":
    main()
