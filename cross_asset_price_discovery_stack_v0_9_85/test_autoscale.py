"""test_autoscale.py -- guards for the shared runtime worker-sizing module.

Checks that the two budgets return sane, related integers, that the memory/session/core caps on the
extraction pool hold, that the reserve clamp hits its [16,48] bounds, that exported env vars override,
and that the CLI matches the Python API -- the last is what lets run_liberation_day.sh source these
numbers instead of duplicating the arithmetic and still agree with the launchers by construction.
"""
import os
import subprocess
import sys

import autoscale as A


def _cli(field, *extra, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    out = subprocess.run([sys.executable, "autoscale.py", field, *extra],
                         capture_output=True, text=True, env=e)
    return out.stdout.strip()


def main():
    ok = True
    cores, ram, res = A.detected_cores(), A.total_ram_gb(), A.reserve_gb()
    w, j = A.extraction_workers(n_sessions=12), A.cpu_jobs()
    sane = (cores >= 1 and ram >= 1 and res >= 0 and w >= 1 and j >= 1
            and w <= cores and w <= 12 and j == cores)
    print("(A) cores=%d ram=%dGB reserve=%dGB workers(12)=%d jobs=%d sane=%s" % (cores, ram, res, w, j, sane))
    ok &= sane

    # extraction pool is capped by #sessions AND cores: 1 session -> 1 worker; a huge count -> <= cores
    cap_ok = (A.extraction_workers(n_sessions=1) == 1 and A.extraction_workers(n_sessions=10**6) <= cores)
    print("(B) extraction caps: workers(1)==1, workers(1e6)<=cores(%d) -> %s" % (cores, cap_ok))
    ok &= cap_ok

    # reserve clamp boundaries, driven through the RAM_GB override (64//8=8 -> 16; 1024//8=128 -> 48)
    lo, hi = int(_cli("reserve", env={"RAM_GB": "64"})), int(_cli("reserve", env={"RAM_GB": "1024"}))
    clamp_ok = (lo == 16 and hi == 48)
    print("(C) reserve clamp: ram64->%d(want 16) ram1024->%d(want 48) -> %s" % (lo, hi, clamp_ok))
    ok &= clamp_ok

    # exported env vars win
    env_ok = (int(_cli("workers", "--sessions", "12", env={"WORKERS": "7"})) == 7
              and int(_cli("jobs", env={"BOOT_WORKERS": "33"})) == 33
              and int(_cli("cores", env={"CORES": "9"})) == 9)
    # CLI matches the in-process API (so the shell driver agrees with the launchers)
    cli_ok = (int(_cli("cores")) == cores and int(_cli("jobs")) == j
              and int(_cli("ram")) == ram and int(_cli("workers", "--sessions", "12")) == w)
    print("(D) env overrides win=%s | CLI==API=%s" % (env_ok, cli_ok))
    ok &= env_ok and cli_ok

    print("\nautoscale checks ->", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
