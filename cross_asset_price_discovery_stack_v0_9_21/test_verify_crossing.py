"""test_verify_crossing.py -- the --demo path of the verifier reproduces the before/after contract:
legacy clock ordering crosses, fixed sequence ordering is clean, on identical fetched messages."""
import sys, warnings
import verify_crossing as vc

def main():
    warnings.simplefilter("ignore")
    r = vc.main(["--demo"])
    leg, seq = r["legacy"], r["fixed"]
    ok = (leg["crossed_frac"] > 0.0 and leg["consolidated_crossed"] > 0
          and seq["crossed_frac"] == 0.0 and seq["consolidated_crossed"] == 0
          and leg["snapshots"] == seq["snapshots"])
    print("  legacy crossed=%.2f (n=%d) | fixed crossed=%.2f (n=%d) -> %s"
          % (leg["crossed_frac"], leg["consolidated_crossed"], seq["crossed_frac"], seq["consolidated_crossed"], ok))
    print("verify-crossing checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
