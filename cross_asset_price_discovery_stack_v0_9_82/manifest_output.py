#!/usr/bin/env python3
"""manifest_output.py -- one machine-readable inventory per replication run.

The old STAGE 7 manifest was `ls -1 "$OUT"`: flat, so everything inside the
nested run_<timestamp>/ dirs -- the actual exhibits -- was invisible, and it
recorded neither the stack version nor the configuration that produced the
files. This walks the WHOLE run tree and writes MANIFEST.json:

    {"run_id", "stack_version", "generated_utc", "config": {...},
     "n_files", "total_bytes",
     "files": [{"path": <relative>, "bytes": ..., "mtime_utc": ...}, ...]}

plus a human-readable MANIFEST.md tree. Config comes in as KEY=VALUE args so
the shell driver can pass its resolved settings without JSON quoting gymnastics.

    python manifest_output.py <run_dir> interval=1s n_boot=499 ...
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


def _version() -> str:
    try:
        import check_version as cv
        return cv.version()
    except Exception:
        return "unknown"


def build_manifest(root: str, config: dict | None = None,
                   skip_names: tuple = ("MANIFEST.json", "MANIFEST.md")) -> dict:
    """Walk `root` and inventory every file (relative paths, bytes, mtime).
    Deterministic ordering (sorted paths) so two manifests of the same tree
    diff clean. The manifest files themselves are excluded."""
    files = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn in skip_names:
                continue
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            rel = os.path.relpath(full, root)
            files.append({"path": rel, "bytes": int(st.st_size),
                          "mtime_utc": datetime.fromtimestamp(
                              st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
            total += int(st.st_size)
    files.sort(key=lambda f: f["path"])                 # global order, not walk order
    return {"run_id": os.path.basename(os.path.abspath(root)),
            "stack_version": _version(),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": dict(config or {}),
            "n_files": len(files), "total_bytes": total, "files": files}


def render_md(man: dict) -> str:
    """Directory-grouped human view of the same inventory."""
    lines = [f"# Run manifest — {man['run_id']}",
             "",
             f"- stack version: {man['stack_version']}",
             f"- generated: {man['generated_utc']}",
             f"- files: {man['n_files']}  ({man['total_bytes'] / 1e6:.1f} MB)",
             ""]
    if man["config"]:
        lines.append("## Configuration")
        for k in sorted(man["config"]):
            lines.append(f"- {k}: `{man['config'][k]}`")
        lines.append("")
    lines.append("## Files")
    last_dir = None
    for f in man["files"]:
        d = os.path.dirname(f["path"]) or "."
        if d != last_dir:
            lines.append(f"\n### {d}/")
            last_dir = d
        kb = f["bytes"] / 1024.0
        lines.append(f"- {os.path.basename(f['path'])}  ({kb:,.1f} KB)")
    lines.append("")
    return "\n".join(lines)


def write_manifest(root: str, config: dict | None = None) -> dict:
    man = build_manifest(root, config)
    with open(os.path.join(root, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    with open(os.path.join(root, "MANIFEST.md"), "w") as fh:
        fh.write(render_md(man))
    return man


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: manifest_output.py <run_dir> [KEY=VALUE ...]", file=sys.stderr)
        return 2
    root = argv[0]
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    config = {}
    for kv in argv[1:]:
        k, sep, v = kv.partition("=")
        if sep:
            config[k] = v
    man = write_manifest(root, config)
    print("MANIFEST.json / MANIFEST.md written: %d files, %.1f MB (stack v%s)"
          % (man["n_files"], man["total_bytes"] / 1e6, man["stack_version"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
