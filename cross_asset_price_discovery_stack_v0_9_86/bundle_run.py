#!/usr/bin/env python3
"""bundle_run.py -- one archivable analysis bundle per run, readable by a human
or by a Claude Code session asked to write the findings report.

A finished replication run's tree holds two kinds of files: the EXHIBITS (tables,
reports, summaries, QC text, the manifest -- megabytes) and the DATA (frame
pickles, parquet datasets -- gigabytes). The exhibits are what the interpretive
report is written from; the data is what MIDAS keeps. This tool zips the exhibits
with the tree structure intact, excludes the data files, and injects two files at
the archive root:

  BUNDLE_MANIFEST.json   what is inside (relative paths, sizes), what was
                         EXCLUDED and why (suffix rule or size cap), the stack
                         version, run id, and the run's resolved config carried
                         over from MANIFEST.json
  ANALYSIS_GUIDE.md      the ingestion contract: a directory map, the exhibit
                         glossary, the reading conventions every number must be
                         interpreted under (day-level effective n, the RealBar
                         lag policy, GFEVD vs FEVD, kappa fragility, the DCC
                         boundary, 10ms jump caution, band stability, the DWC
                         and tau verdicts, the half-impact horizon), and the
                         findings-of-record report template. A fresh session
                         given only this archive can produce the report.

Excluded by rule: *.pkl, *.pickle, *.parquet, *.csv.gz (the frame caches and
final datasets), the extract cache, and any single file above --max-file-mb
(default 50; a note records what was skipped). Everything else rides.

Usage
    python bundle_run.py output/replication_20260901_120000
    python bundle_run.py <run_dir> --out /tmp/analysis_bundle.zip --max-file-mb 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

EXCLUDE_SUFFIXES = (".pkl", ".pickle", ".parquet", ".csv.gz")
EXCLUDE_DIRS = ("extract_cache",)


def collect(run_dir: str, max_file_mb: float = 50.0, self_name: str = ""):
    """-> (included [(rel, full, bytes)], excluded [(rel, bytes, reason)]) with
    deterministic (sorted) ordering."""
    inc, exc = [], []
    cap = float(max_file_mb) * 1024 * 1024
    for dirpath, dirnames, filenames in os.walk(run_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, run_dir)
            if self_name and os.path.abspath(full) == os.path.abspath(self_name):
                continue
            try:
                size = os.stat(full).st_size
            except OSError:
                continue
            low = fn.lower()
            if any(low.endswith(sfx) for sfx in EXCLUDE_SUFFIXES):
                exc.append((rel, size, "data file (suffix rule)"))
            elif size > cap:
                exc.append((rel, size, f"above --max-file-mb {max_file_mb:g}"))
            else:
                inc.append((rel, full, size))
    return inc, exc


def _run_config(run_dir: str) -> dict:
    try:
        with open(os.path.join(run_dir, "MANIFEST.json")) as fh:
            man = json.load(fh)
        return {"run_id": man.get("run_id"), "stack_version": man.get("stack_version"),
                "config": man.get("config", {})}
    except Exception:
        return {"run_id": os.path.basename(os.path.abspath(run_dir)),
                "stack_version": _version(), "config": {}}


def _version() -> str:
    try:
        import check_version as cv
        return cv.version()
    except Exception:
        return "unknown"


def analysis_guide(info: dict, inc, exc) -> str:
    cfg = "\n".join(f"- {k}: `{v}`" for k, v in sorted((info.get("config") or {}).items()))
    n_bytes = sum(s for _r, _f, s in inc)
    return f"""# Analysis Bundle — ingestion guide

**Run:** {info.get('run_id')}  |  **Stack version:** {info.get('stack_version')}  |
**Contents:** {len(inc)} files, {n_bytes / 1e6:.1f} MB (data files excluded: {len(exc)})

This archive is the complete exhibit set of one replication run of the
cross-asset price-discovery stack (Garrison, Jain & Paddrik, "Cross-Asset
Tandem Trading and Extraordinary Volatility"). It is self-contained: everything
needed to write the interpretive findings report is in here; the underlying
frame pickles and datasets are NOT (they stay on the cluster).

## Resolved configuration
{cfg or '- (no MANIFEST.json found in the run tree)'}

## Directory map

- `<grid>/analysis/` — run_analysis output per grid (1s, 10ms): `report.md`
  (every stage's numbers inline), `summary.json` (headline scalars),
  `tables/{{stage}}__{{key}}.csv` (every exhibit as CSV). Key tables:
  `information_shares__per_day` (per-day IS/CS/kappa + low_kappa flags),
  `information_shares__regime_test*` (CS, IS-primary, kappa-weighted),
  `jumps__per_day` + `jumps__cojump_per_day` (co-jump lead-lag, all sessions),
  `cross_impact__panel`, `ecm_sde__curve*`, `microstructure__staleness_report`,
  `legacy__*` (the method A/B exhibits).
- `<grid>/table9/` — Table 9 both-ways (Pearson / HY / DCC / RealBar) with
  `table9_lag_ic_*.csv` (criterion curves) and, when a band ran,
  `table9_lag_band_*.csv` (per-cell sign/star stability verdicts).
- `<grid>/flow/` — order-flow-correlation exhibits: tier1 regimes, asymmetry,
  mediation, Markov-switching regimes, price-discovery link.
- `<grid>/copula/` — tail-dependence tables (regimes, liquidity split, flows).
- `<grid>/geometry/` — the ||L|| / centroid / Herfindahl / decay-weighted-cost
  per-day table with the tau and DWC redundancy verdicts.
- `horizon/` — the propagation-horizon ladder: per-day and summary CSVs with
  the normalized cross-impact profile and the half-impact horizon.
- `qc/` — sample validation, per-day frame QC, feed health, crossing checks.
- `nulls/` — the corrected Table 5 / Table 7 null tables.
- `frames/extract_report.txt` — extraction hygiene notes (frames themselves
  excluded).
- `MANIFEST.json` / `RUN_NOTES.md` — the original run inventory and notes.
- `replication.log` — the full driver log (grep for WARNING / CAUTION /
  MINORITY / ACTIVITY RULE / LAG DIAGNOSIS / VERDICT lines).

## Reading conventions (apply to every number)

1. **Effective sample size is the day count.** All inference clusters by day;
   intraday rows buy within-day precision only. Read regime contrasts against
   the day-level permutation tests, not pooled t's.
2. **CS is fragile on low-kappa days; IS is not.** Use `low_kappa` flags; the
   kappa-weighted regime test is the robust variant. Quote IS as primary.
3. **Lag policy.** Criterion lags are resolved on the window-free RealBar frame;
   the Pearson-frame path is a printed diagnostic only (footnote-17 artifact).
   If a lag-band table exists, only band-stable cells are lag-robust findings.
4. **GFEVD, not FEVD**, under correlated OFI shocks (the measured innovation
   correlation is in the irf block); common tandem flow is partially attributed
   to both shocks.
5. **DCC persistence at ~0.9999 is a boundary estimate** — quote the realized
   correlation, not the conditional level.
6. **10ms jump FRACTIONS are staleness-inflated** (check the zero-return
   fractions); co-jump lead RATIOS are less exposed but need the per-day table,
   not one session.
7. **tau and DWC ship as checks**: their verdicts (rank-equivalence to HHI;
   increment over {{spread, depth, centroid, HHI}}) say whether they carry
   independent information. Expect "redundant" — a failing verdict is the
   finding.
8. **The half-impact horizon** is the ratio-to-coarsest-rung crossing 0.5; read
   each rung against its staleness column.
9. **The sample is designed, not hand-picked** (see the driver's provenance
   block / sample_inputs in the stack): threshold rule, episode caps, screened
   controls, 6 documented unpairable volatile days, MWCB as its own panel.

## The report to produce

Write a findings-of-record document (matter-of-fact, deposition style, no
exhortation) with numbered sections: (1) scope of the record — what ran, the
sample, completion status; (2) methodology in effect with MEASURED consequences
— each method choice paired with the number in this bundle that quantifies it;
(3) findings of fact — each with its statement, numbers, and source table named;
(4) qualifications — what limits each finding, stated as facts; (5) matters for
further examination. Lead with what changed relative to any prior run's report
if one is supplied alongside.
"""


def build_bundle(run_dir: str, out_path: str = "", max_file_mb: float = 50.0) -> dict:
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"not a directory: {run_dir}")
    info = _run_config(run_dir)
    out_path = out_path or os.path.join(
        run_dir, f"analysis_bundle_{info.get('run_id', 'run')}.zip")
    inc, exc = collect(run_dir, max_file_mb, self_name=out_path)
    man = {"run_id": info.get("run_id"), "stack_version": info.get("stack_version"),
           "config": info.get("config", {}),
           "n_files": len(inc), "total_bytes": int(sum(s for _r, _f, s in inc)),
           "files": [{"path": r, "bytes": int(s)} for r, _f, s in inc],
           "excluded": [{"path": r, "bytes": int(s), "reason": why} for r, s, why in exc]}
    guide = analysis_guide(info, inc, exc)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ANALYSIS_GUIDE.md", guide)
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(man, indent=2))
        for rel, full, _s in inc:
            z.write(full, rel)
    man["bundle_path"] = out_path
    man["bundle_bytes"] = int(os.stat(out_path).st_size)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="archive a run's exhibits (sans data files) "
                                             "with a Claude-readable analysis guide")
    ap.add_argument("run_dir", help="the replication_<id> directory")
    ap.add_argument("--out", default="", help="bundle path (default: inside the run dir)")
    ap.add_argument("--max-file-mb", type=float, default=50.0,
                    help="skip any single file above this size (recorded as excluded)")
    a = ap.parse_args(argv)
    man = build_bundle(a.run_dir, a.out, a.max_file_mb)
    print("bundle: %s" % man["bundle_path"])
    print("  %d files, %.1f MB uncompressed -> %.1f MB zipped; %d data file(s) excluded"
          % (man["n_files"], man["total_bytes"] / 1e6, man["bundle_bytes"] / 1e6,
             len(man["excluded"])))
    big = [e for e in man["excluded"] if "max-file-mb" in e["reason"]]
    for e in big[:5]:
        print("  NOTE oversized, skipped: %s (%.1f MB)" % (e["path"], e["bytes"] / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
