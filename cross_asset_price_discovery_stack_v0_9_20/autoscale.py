#!/usr/bin/env python3
"""autoscale.py -- single source of truth for runtime worker sizing across the stack.

Two budgets, because the stack's parallel regions have DIFFERENT binding constraints:

  * EXTRACTION (per-session book reconstruction) is MEMORY-bound -- each worker holds one full
    day of resident messages, so workers = min(memory-derived, #sessions, cores). Oversubscribing
    here OOMs the node, which is why it is capped at the session count and the memory budget rather
    than the core count.
  * CPU work (the day-block bootstrap; the cross-event onset estimation loop) is CORE-bound -- it
    runs on already-resident frames and adds little memory, so jobs = all detected cores.

A single global worker count is therefore wrong; the two functions below size the two regions
independently, which is exactly the split run_liberation_day.sh has always made by hand. This
module is the single implementation; the shell driver sources these numbers via the CLI instead of
duplicating the arithmetic, so bash and Python agree by construction.

Detection defeats the cgroup/affinity trap: os.cpu_count() ignores CPU affinity, while
len(os.sched_getaffinity(0)) respects it -- a container can pin affinity to 1 on a 64-core node,
which would silently serialize extraction AND the bootstrap. We take the MAX across signals (plus
/proc/cpuinfo) to recover the physical online count. Every value is overridable by exporting the
matching env var (CORES, RAM_GB, RESERVE_GB, WORKERS, BOOT_WORKERS, PEAK_GB_GUESS), preserving the
driver's existing "export to override" behavior.

CLI (so the shell driver stops duplicating the logic):
    python autoscale.py cores
    python autoscale.py ram
    python autoscale.py reserve
    python autoscale.py workers --sessions 12 [--peak-gb 32]
    python autoscale.py jobs
    python autoscale.py panel --sessions 24
    python autoscale.py peak                 # PEAK_GB_GUESS implied by this process's high-water RSS
    python autoscale.py measure -- <cmd>     # run <cmd>, report its peak RSS + the PEAK_GB_GUESS to export
"""
import math
import os
import sys

_SESSION_CAP = 12          # default #sessions cap (the Liberation-Day window); callers pass their own
_PEAK_GB_GUESS = 32        # per-worker peak RSS guess for one full day of messages on the 1s grid
_PANEL_PEAK_GB = 1         # per-worker peak for an ALREADY-AGGREGATED session frame (~7.5 MB) + temporaries
_PANEL_USABLE = 0.70       # share of RAM a per-session process pool may use (rest: OS, parent, page cache)


def _env_int(name):
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _cpuinfo_count():
    try:
        with open("/proc/cpuinfo") as f:
            return sum(1 for ln in f if ln.split(":", 1)[0].strip() == "processor")
    except OSError:
        return 0


def detected_cores():
    """Physical online core count, robust to a cgroup/affinity cap (take the MAX across signals)."""
    ov = _env_int("CORES")
    if ov and ov > 0:
        return ov
    signals = [os.cpu_count() or 0, _cpuinfo_count()]
    try:
        signals.append(len(os.sched_getaffinity(0)))      # Linux only; the affinity-capped signal
    except AttributeError:
        pass
    c = max(signals)
    return c if c > 0 else 1


def _meminfo_gb():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemTotal:"):
                    return int(ln.split()[1]) // (1024 * 1024)         # kB -> GiB
    except OSError:
        pass
    try:
        return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) // (1024 ** 3)
    except (ValueError, OSError):
        return 0


def _cgroup_mem_gb():
    """The process's memory-cgroup limit in GiB, or None if unlimited/absent. A worker OOM-kill is a
    cgroup event, so on a container the binding RAM budget may be this cap, well below the node total."""
    paths = ["/sys/fs/cgroup/memory.max"]                              # cgroup v2 (unified)
    try:                                                               # the process's own nested v2 path
        with open("/proc/self/cgroup") as f:
            for ln in f:
                if ln.startswith("0::"):
                    paths.append("/sys/fs/cgroup%s/memory.max" % ln.strip().split(":", 2)[2])
    except OSError:
        pass
    paths.append("/sys/fs/cgroup/memory/memory.limit_in_bytes")        # cgroup v1
    for p in paths:
        try:
            with open(p) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            g = int(raw) // (1024 ** 3)
        except ValueError:
            continue
        if 1 <= g < 1_000_000:                                         # ignore v1 "unlimited" sentinels
            return g
    return None


def total_ram_gb():
    """Binding RAM budget in GiB = min(node MemTotal, cgroup limit). Override by exporting RAM_GB."""
    ov = _env_int("RAM_GB")
    if ov and ov > 0:
        return ov
    node, cg = _meminfo_gb(), _cgroup_mem_gb()
    if node >= 1 and cg is not None:
        return min(node, cg)
    if node >= 1:
        return node
    if cg is not None:
        return cg
    return 256                                                         # matches the shell driver's fallback


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def reserve_gb(ram=None):
    """RAM held back for OS/parent/extraction-tool/fragmentation: ~1/8 RAM, clamped to [16, 48] GB."""
    ov = _env_int("RESERVE_GB")
    if ov is not None and ov >= 0:
        return ov
    ram = total_ram_gb() if ram is None else ram
    return _clamp(ram // 8, 16, 48)


def extraction_workers(n_sessions=_SESSION_CAP, peak_gb=None, ram=None, cores=None):
    """Memory-bound extraction pool size = min(memory-derived, #sessions, cores)."""
    ov = _env_int("WORKERS")
    if ov and ov > 0:
        return ov
    ram = total_ram_gb() if ram is None else ram
    cores = detected_cores() if cores is None else cores
    peak = _env_int("PEAK_GB_GUESS") or (peak_gb if peak_gb else _PEAK_GB_GUESS)
    memw = max(1, (ram - reserve_gb(ram)) // peak)
    cap = max(1, int(n_sessions))
    return max(1, min(memw, cap, cores))


def cpu_jobs(cores=None):
    """Core-bound CPU pool size = all detected cores (day-block bootstrap; onset estimation loop)."""
    ov = _env_int("BOOT_WORKERS")
    if ov and ov > 0:
        return ov
    return detected_cores() if cores is None else cores


def observed_peak_gb():
    """Largest resident-set size reached so far, in GiB: max(this process, biggest single child).

    ``_PEAK_GB_GUESS`` is the one number in this module that is a GUESS rather than a measurement,
    and it is the binding constraint on extraction: workers = (RAM - reserve) / peak, so a placeholder
    that is 4x too large costs 4x the wall clock, and one that is too small OOMs the node. It cannot
    be derived from first principles -- it depends on the venue mix and message volume of YOUR
    sessions -- so measure it instead.

    RUSAGE_CHILDREN.ru_maxrss is the high-water mark of the largest single child, which is exactly
    the per-worker peak the budget needs (not the sum across workers). Call after an extraction run;
    `autoscale.py peak` prints it and the PEAK_GB_GUESS it implies."""
    try:
        import resource
    except ImportError:                                        # non-POSIX
        return None
    ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_kb = max(ru_self, ru_kids)                            # Linux reports kB; macOS reports bytes
    if sys.platform == "darwin":
        peak_kb = peak_kb / 1024.0
    return peak_kb / (1024.0 * 1024.0)


def recommend_peak_gb(observed=None, safety=1.30):
    """PEAK_GB_GUESS to export, from an observed per-worker peak plus headroom.

    The 30% margin covers the spread across sessions -- a crash day carries far more messages than
    a calm one, and the budget has to survive the worst session in the sample, not the one that
    happened to be measured. Rounded UP to a whole GiB: the worker count is a floor division, so
    rounding the peak down is the direction that OOMs."""
    obs = observed_peak_gb() if observed is None else float(observed)
    if not obs or obs <= 0:
        return None
    return max(1, int(math.ceil(obs * float(safety))))


def panel_workers(n_items, peak_gb=None, ram=None, cores=None):
    """PROCESS-pool size for per-session work that ships ONE ALREADY-AGGREGATED frame per worker.

    A third budget, because neither of the two above fits this shape:

      * ``cpu_jobs`` assumes the data is already resident and shared, so it ignores memory. Here
        each worker is a separate PROCESS and receives its own pickled copy of a session frame.
      * ``extraction_workers`` is right in structure but calibrated for a worker holding a full
        day of raw MESSAGES (``_PEAK_GB_GUESS`` = 32 GB) against a reserve floored at 16 GB. An
        aggregated 1-second book frame is ~7.5 MB, four orders of magnitude smaller, so borrowing
        that budget collapses to a single worker on any node under ~48 GB -- i.e. it would
        silently serialize the very loop we are parallelising.

    Processes rather than threads because the work is GIL-bound: the cost is the GARCH and cDCC
    recursions, which are Python-level loops, not numpy calls that release the GIL.

    Size = min(cores, n_items, memory-derived), where memory-derived leaves ~30% of RAM to the OS,
    the parent process and page cache. Override wholesale with PANEL_WORKERS, or just the per-worker
    footprint with PANEL_PEAK_GB."""
    ov = _env_int("PANEL_WORKERS")
    if ov and ov > 0:
        return ov
    ram = total_ram_gb() if ram is None else ram
    cores = detected_cores() if cores is None else cores
    peak = _env_int("PANEL_PEAK_GB") or (peak_gb if peak_gb else _PANEL_PEAK_GB)
    memw = max(1, int((ram * _PANEL_USABLE) // peak))
    return max(1, min(int(cores), max(1, int(n_items)), memw))


def measure(cmd):
    """Run `cmd` (a list) to completion and report the peak RSS of the largest child.

    Exists because the peak that matters belongs to the EXTRACTION SUBPROCESS, not to whatever
    Python happens to be asking. GNU /usr/bin/time would do it, but it is absent on plenty of
    minimal images, so do it with RUSAGE_CHILDREN, which is in the standard library and gives the
    same high-water mark for the largest single child -- exactly the per-worker figure the budget
    divides RAM by. Returns the child's exit code; prints the measurement to stderr so it survives
    a stdout pipe."""
    import resource
    import subprocess
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    rc = subprocess.call(cmd)
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_kb = max(after, before)
    if sys.platform == "darwin":
        peak_kb = peak_kb / 1024.0
    gb = peak_kb / (1024.0 * 1024.0)
    rec = recommend_peak_gb(gb)
    sys.stderr.write("\n[autoscale] measured peak RSS of the largest child: %.2f GiB\n" % gb)
    sys.stderr.write("[autoscale] recommended PEAK_GB_GUESS=%s (observed x1.30, rounded up)\n" % rec)
    if rec:
        sys.stderr.write("[autoscale] with that, extraction_workers = %d on this node "
                         "(was %d at the built-in guess of %d GiB)\n"
                         % (extraction_workers(_SESSION_CAP, peak_gb=rec),
                            extraction_workers(_SESSION_CAP), _PEAK_GB_GUESS))
        sys.stderr.write("[autoscale] export PEAK_GB_GUESS=%s to use it on the next run\n" % rec)
    return rc


def _main(argv=None):
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "measure":                          # `autoscale.py measure -- cmd ...`
        rest = argv[1:]
        if rest and rest[0] == "--":
            rest = rest[1:]
        if not rest:
            sys.stderr.write("usage: autoscale.py measure -- <command> [args...]\n")
            return 2
        return measure(rest)
    ap = argparse.ArgumentParser(description="runtime worker sizing (single source for shell + python)")
    ap.add_argument("field", choices=["cores", "ram", "reserve", "workers", "jobs", "panel", "peak"])
    ap.add_argument("--sessions", type=int, default=_SESSION_CAP)
    ap.add_argument("--peak-gb", type=int, default=None)
    a = ap.parse_args(argv)
    val = {"cores": detected_cores,
           "ram": total_ram_gb,
           "reserve": reserve_gb,
           "workers": lambda: extraction_workers(a.sessions, a.peak_gb),
           "jobs": cpu_jobs,
           "panel": lambda: panel_workers(a.sessions, a.peak_gb),
           "peak": lambda: (recommend_peak_gb() or 0)}[a.field]()
    print(int(val))


if __name__ == "__main__":
    # Propagate the exit code. `measure` wraps a real command (the extraction run), so swallowing
    # its status would let a failed extraction report success to the shell -- and the caller gates
    # on that status.
    sys.exit(_main() or 0)
