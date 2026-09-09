#!/usr/bin/env python3
"""Run every spec through the reference kernel.

Writes one envelope per spec to ``results/reference/``. With ``--golden`` the
same envelopes are written to ``fixtures/golden/``, which is the manifest the
suite compares against. Regenerating the golden manifest is a deliberate act
and shows up in the diff.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from econospec.runner import run_spec  # noqa: E402
from econospec.specs import load_all_specs  # noqa: E402


def write_envelope(directory: str, envelope: dict) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "%s.json" % envelope["spec_id"])
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(envelope, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_environment(directory: str) -> None:
    """Record the interpreter and numpy build behind the reference numbers.

    The golden manifest itself stays free of this metadata so that a version
    bump does not rewrite committed expectations. Provenance belongs next to
    the run, not inside the expectation.
    """
    import platform
    import sys as _sys

    import numpy

    from econospec import __version__

    payload = {
        "adapter": "reference-numpy",
        "language": "python",
        "runtime": _sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {"numpy": numpy.__version__, "econospec": __version__},
    }
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "environment.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", action="store_true", help="also refresh fixtures/golden")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    specs = load_all_specs(ROOT)
    out_dir = os.path.join(ROOT, "results", "reference")
    write_environment(out_dir)
    golden_dir = os.path.join(ROOT, "fixtures", "golden")
    for spec in specs:
        envelope = run_spec(spec, ROOT)
        write_envelope(out_dir, envelope)
        if args.golden:
            write_envelope(golden_dir, envelope)
        if not args.quiet:
            print("%-8s %-10s params=%d scalars=%d" % (
                envelope["spec_id"], spec["family"], len(envelope["params"]),
                len(envelope["scalars"]),
            ))
    print("ran %d specs" % len(specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
