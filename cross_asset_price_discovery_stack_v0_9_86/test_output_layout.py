#!/usr/bin/env python3
"""Gate: v0.9.73's single structured output tree.

The old layout scattered a run across a flat directory plus up to three opaque
run_<timestamp>/ subdirs (indistinguishable except by timestamp), and the STAGE 7
manifest was a flat `ls` that could not see the exhibits inside them. This pins:

  1. manifest_output.build_manifest inventories a tree recursively with relative
     paths, sizes, config, and the stack version -- and excludes itself
  2. write_manifest emits both MANIFEST.json and MANIFEST.md, and the md groups
     files by directory
  3. run_analysis grows --flat-output (default OFF), which removes the
     run_<timestamp>/ nesting so the replication driver owns the structure
  4. the replication script routes every stage into the structured tree: no
     stage writes flat into ${OUT} any more, the per-grid subdirs are used, the
     geometry stage (5d) exists, and STAGE 7 calls manifest_output.py
"""
import json
import os
import re
import sys
import tempfile

import manifest_output as mo


def check_build_manifest():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "1s", "table9"))
        os.makedirs(os.path.join(td, "qc"))
        for rel, size in (("1s/table9/t9.csv", 100), ("qc/qc_frames.txt", 40),
                          ("replication.log", 10)):
            with open(os.path.join(td, rel), "wb") as fh:
                fh.write(b"x" * size)
        man = mo.build_manifest(td, {"interval": "1s", "n_boot": "499"})
        paths = [f["path"] for f in man["files"]]
        ok_paths = paths == sorted(paths) and "1s/table9/t9.csv" in paths and len(paths) == 3
        ok_bytes = man["total_bytes"] == 150 and man["n_files"] == 3
        ok_cfg = man["config"]["n_boot"] == "499"
        ok_ver = re.match(r"^\d+\.\d+\.\d+$", str(man["stack_version"])) is not None
        # the manifest must not inventory itself
        mo.write_manifest(td, {})
        man2 = mo.build_manifest(td, {})
        ok_self = man2["n_files"] == 3
    print("(1) recursive inventory: sorted relative paths (%s), sizes (%s), config (%s),"
          % (ok_paths, ok_bytes, ok_cfg))
    print("    stack version stamped (%s), manifest excludes itself (%s)" % (ok_ver, ok_self))
    return bool(ok_paths and ok_bytes and ok_cfg and ok_ver and ok_self)


def check_write_manifest():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "10ms", "flow"))
        with open(os.path.join(td, "10ms", "flow", "f.csv"), "w") as fh:
            fh.write("a,b\n")
        mo.write_manifest(td, {"run_id": "test"})
        ok_json = os.path.exists(os.path.join(td, "MANIFEST.json"))
        ok_md = os.path.exists(os.path.join(td, "MANIFEST.md"))
        with open(os.path.join(td, "MANIFEST.json")) as fh:
            man = json.load(fh)
        md = open(os.path.join(td, "MANIFEST.md")).read()
        ok_md_group = "### 10ms/flow/" in md and "f.csv" in md
        ok_round = man["files"][0]["path"] == os.path.join("10ms", "flow", "f.csv")
    print("(2) MANIFEST.json (%s) + MANIFEST.md (%s) written; md groups by directory (%s)"
          % (ok_json, ok_md, ok_md_group))
    return bool(ok_json and ok_md and ok_md_group and ok_round)


def check_flat_output_flag():
    import run_analysis as ra
    a_off = ra.parse_args(["--source", "demo"])
    a_on = ra.parse_args(["--source", "demo", "--flat-output"])
    ok = a_off.flat_output is False and a_on.flat_output is True
    print("(3) run_analysis --flat-output exists, default OFF : %s" % ok)
    return bool(ok)


def check_driver_routing():
    src = open("run_paper_replication.sh").read()
    # no stage may write flat into ${OUT} any more (log, notes and manifest live at the root)
    flat_outdirs = re.findall(r'--out(?:put)?-dir\s+"?\$\{?OUT\}?"?(?:\s|"|$)', src)
    ok_no_flat = len(flat_outdirs) == 0
    need = ['${OUT}/${INTERVAL}/table9', '${OUT}/${INTERVAL}/flow', '${OUT}/${INTERVAL}/copula',
            '${OUT}/${INTERVAL}/geometry', '${OUT}/${FINE_INTERVAL}/table9',
            '${OUT}/${FINE_INTERVAL}/flow', '$OUT/${INTERVAL}/analysis',
            '$OUT/${FINE_INTERVAL}/analysis', '$OUT/frames', '${OUT}/qc/qc_frames.txt',
            '"$OUT/nulls"']
    missing = [n for n in need if n not in src]
    ok_routes = not missing
    ok_geom = "STAGE 5d" in src and "run_book_geometry.py" in src
    ok_manifest = "manifest_output.py" in src
    ok_flat_flag = src.count("--flat-output") >= 4      # extract, pull-once, 2b, 6, 6b
    print("(4) no flat ${OUT} out-dirs remain (%s); all structured routes present (%s%s);"
          % (ok_no_flat, ok_routes, "" if ok_routes else " missing: " + ", ".join(missing)))
    print("    geometry stage wired (%s); manifest call wired (%s); --flat-output used (%s)"
          % (ok_geom, ok_manifest, ok_flat_flag))
    return bool(ok_no_flat and ok_routes and ok_geom and ok_manifest and ok_flat_flag)


def main():
    checks = [check_build_manifest, check_write_manifest, check_flat_output_flag,
              check_driver_routing]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("output-layout checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
