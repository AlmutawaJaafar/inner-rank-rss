"""
verify_logs.py -- check that the archived reference logs carry the same numbers
                  as the raw `tee` output produced on the author's machine.

Run:  python3.10 verify_logs.py

The archived logs in logs_reference/ were transcribed from the reference-
environment console output.  They carry a two-line provenance header and omit
the trailing "elapsed Ns" timing line, so a byte-for-byte `diff` is noisy and
uninformative.  What matters is that every NUMBER agrees.  This script extracts
the numeric tokens from each pair of files, in order, and reports any
disagreement.

It compares, for each N in 15..18:

    logs_reference/expN.log      (archived)
    expN.txt                     (raw `tee` output, if present)

Exit status is 0 when every pair agrees and 1 otherwise, so the script can be
used as a pre-submission gate.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def numbers(path):
    """All numeric tokens in a file, in order, as strings (comments skipped)."""
    out = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):          # provenance header
            continue
        if "elapsed" in line:                      # wall-clock timing line
            continue
        out.extend(NUM.findall(line))
    return out


def compare(ref, raw):
    a, b = numbers(ref), numbers(raw)
    if a == b:
        return True, f"{len(a)} numeric tokens identical"
    # locate the first disagreement for a useful message
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            lo = max(0, i - 3)
            return False, (f"first disagreement at token {i}: "
                           f"archived {x!r} vs raw {y!r}\n"
                           f"      archived context: {a[lo:i+4]}\n"
                           f"      raw      context: {b[lo:i+4]}")
    return False, (f"same prefix but different length: "
                   f"archived {len(a)} tokens, raw {len(b)} tokens")


def main():
    ok = True
    checked = 0
    for n in (15, 16, 17, 18):
        ref = HERE / "logs_reference" / f"exp{n}.log"
        raw = HERE / f"exp{n}.txt"
        if not ref.exists():
            print(f"exp{n}: SKIP  (no archived log at {ref})")
            continue
        if not raw.exists():
            print(f"exp{n}: SKIP  (no raw output at {raw.name}; "
                  f"re-run with | tee {raw.name})")
            continue
        good, msg = compare(ref, raw)
        print(f"exp{n}: {'OK   ' if good else 'FAIL '} {msg}")
        ok &= good
        checked += 1

    print()
    if checked == 0:
        print("Nothing compared.  Place the raw exp15.txt..exp18.txt next to")
        print("this script, or re-run the four scripts with | tee expNN.txt.")
        return 1
    if ok:
        print(f"All {checked} pair(s) agree.  The archived reference logs carry")
        print("exactly the numbers produced on this machine.")
        return 0
    print("At least one pair disagrees.  Prefer the RAW output: copy it over")
    print("the archived log with")
    print("    cp expNN.txt logs_reference/expNN.log")
    print("and re-zip, so the archive holds the console output verbatim.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
