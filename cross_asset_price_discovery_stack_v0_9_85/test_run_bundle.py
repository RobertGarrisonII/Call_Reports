#!/usr/bin/env python3
"""Gate: the analysis bundle (v0.9.80) -- the run's exhibits, sans data files,
archived with the ingestion contract a fresh Claude Code session reads first.

Pins:
  1. exclusion rules -- *.pkl / *.pickle / *.parquet / *.csv.gz and the extract
     cache never enter the archive; oversized files are skipped WITH a recorded
     reason; plain csv/md/json/txt/log all ride, tree structure intact
  2. the injected root files -- BUNDLE_MANIFEST.json carries run id, stack
     version, the run config (from MANIFEST.json), the included inventory AND
     the excluded list with reasons; ANALYSIS_GUIDE.md carries the directory
     map, all nine reading conventions, and the report template
  3. round trip -- unzipping reproduces every included file byte-for-byte, and
     the bundle never contains itself
  4. driver wiring -- STAGE 7b invokes bundle_run.py on ${OUT} after the
     manifest, and the bundle lands inside the run tree
"""
import io
import json
import os
import sys
import tempfile
import zipfile

import bundle_run as br


def _fake_run(td):
    run = os.path.join(td, "replication_20260901_000000")
    for d in ("1s/analysis/tables", "1s/table9", "frames", "qc", "horizon",
              "extract_cache"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    files = {
        "1s/analysis/report.md": b"# report\n",
        "1s/analysis/summary.json": b"{}",
        "1s/analysis/tables/information_shares__per_day.csv": b"a,b\n1,2\n",
        "1s/table9/table9_both_ways_informational_w100_p5.csv": b"x\n",
        "frames/extract_report.txt": b"clean\n",
        "frames/frames_1s.pkl": b"\x80\x04data",                 # excluded: pickle
        "frames/final_dataset.csv.gz": b"\x1f\x8bdata",          # excluded: csv.gz
        "1s/analysis/final_dataset.parquet": b"PAR1",            # excluded: parquet
        "extract_cache/day.pkl": b"\x80\x04",                    # excluded: dir rule
        "qc/qc_frames.txt": b"ok\n",
        "horizon/horizon_profile_summary_10ms_6rungs.csv": b"i\n",
        "replication.log": b"log line\n",
        "RUN_NOTES.md": b"notes\n",
        "MANIFEST.json": json.dumps({"run_id": "replication_20260901_000000",
                                     "stack_version": "9.9.9",
                                     "config": {"interval": "1s", "n_boot": "499"}}).encode(),
    }
    for rel, data in files.items():
        path = os.path.join(run, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
    with open(os.path.join(run, "qc", "huge.txt"), "wb") as fh:
        fh.write(b"x" * (2 * 1024 * 1024))                       # 2 MB, cap at 1 MB below
    return run


def check_exclusions():
    with tempfile.TemporaryDirectory() as td:
        run = _fake_run(td)
        man = br.build_bundle(run, max_file_mb=1.0)
        names = set()
        with zipfile.ZipFile(man["bundle_path"]) as z:
            names = set(z.namelist())
        ok_no_data = not any(n.endswith((".pkl", ".parquet", ".csv.gz")) for n in names)
        ok_no_cache = not any(n.startswith("extract_cache") for n in names)
        ok_kept = {"1s/analysis/report.md", "1s/analysis/tables/information_shares__per_day.csv",
                   "qc/qc_frames.txt", "replication.log", "MANIFEST.json",
                   "horizon/horizon_profile_summary_10ms_6rungs.csv"} <= names
        exc = {e["path"]: e["reason"] for e in man["excluded"]}
        ok_reasons = ("frames/frames_1s.pkl" in exc and "suffix" in exc["frames/frames_1s.pkl"]
                      and os.path.join("qc", "huge.txt") in exc
                      and "max-file-mb" in exc[os.path.join("qc", "huge.txt")])
        ok_nohuge = not any("huge.txt" in n for n in names)
    print("(1) data files and cache excluded (%s, %s); exhibits + log + manifest kept (%s);"
          % (ok_no_data, ok_no_cache, ok_kept))
    print("    oversized file skipped with recorded reason (%s, absent from zip %s)"
          % (ok_reasons, ok_nohuge))
    return bool(ok_no_data and ok_no_cache and ok_kept and ok_reasons and ok_nohuge)


def check_injected_roots():
    with tempfile.TemporaryDirectory() as td:
        run = _fake_run(td)
        man = br.build_bundle(run, max_file_mb=1.0)
        with zipfile.ZipFile(man["bundle_path"]) as z:
            bm = json.loads(z.read("BUNDLE_MANIFEST.json"))
            guide = z.read("ANALYSIS_GUIDE.md").decode()
        ok_man = (bm["run_id"] == "replication_20260901_000000"
                  and bm["stack_version"] == "9.9.9"
                  and bm["config"].get("n_boot") == "499"
                  and bm["n_files"] == len(bm["files"]) and len(bm["excluded"]) >= 4)
        need = ["Directory map", "Reading conventions", "The report to produce",
                "Effective sample size is the day count", "RealBar frame",
                "GFEVD", "boundary estimate", "staleness-inflated", "half-impact horizon",
                "designed, not hand-picked", "low-kappa", "checks"]
        missing = [n for n in need if n not in guide]
        ok_guide = not missing and "9.9.9" in guide
    print("(2) BUNDLE_MANIFEST carries id/version/config/inventory/exclusions (%s);"
          % ok_man)
    print("    ANALYSIS_GUIDE carries map + all reading conventions + template (%s%s)"
          % (not missing, "" if not missing else " missing: " + ", ".join(missing)))
    return bool(ok_man and ok_guide)


def check_round_trip():
    with tempfile.TemporaryDirectory() as td:
        run = _fake_run(td)
        man = br.build_bundle(run, max_file_mb=1.0)
        out = os.path.join(td, "unpacked")
        with zipfile.ZipFile(man["bundle_path"]) as z:
            z.extractall(out)
        ok = True
        for f in man["files"]:
            src = os.path.join(run, f["path"])
            dst = os.path.join(out, f["path"])
            with open(src, "rb") as a, open(dst, "rb") as b:
                ok &= a.read() == b.read()
        # rebuilding over an existing bundle must not swallow itself
        man2 = br.build_bundle(run, max_file_mb=1.0)
        with zipfile.ZipFile(man2["bundle_path"]) as z:
            ok_self = not any(n.startswith("analysis_bundle") for n in z.namelist())
    print("(3) byte-for-byte round trip (%s); bundle never contains itself (%s)"
          % (ok, ok_self))
    return bool(ok and ok_self)


def check_driver_wiring():
    src = open("run_paper_replication.sh").read()
    ok_stage = "STAGE 7b" in src and "bundle_run.py" in src
    ok_after = src.find("bundle_run.py") > src.find("manifest_output.py")
    print("(4) STAGE 7b invokes bundle_run.py after the manifest (%s, %s)"
          % (ok_stage, ok_after))
    return bool(ok_stage and ok_after)


def main():
    checks = [check_exclusions, check_injected_roots, check_round_trip,
              check_driver_wiring]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("run-bundle checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
