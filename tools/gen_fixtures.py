#!/usr/bin/env python3
"""Regenerate every fixture CSV and the checksum manifest.

Run with ``--check`` in CI: it regenerates into a temporary directory and
compares checksums, so a fixture cannot drift from its generator without the
build going red.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from econospec.csvio import write_frame  # noqa: E402
from econospec.dgp import GENERATORS  # noqa: E402

FIXTURES = {
    "ols-001.csv": ("ols_nominal", ["y", "x1", "x2"]),
    "ols-002.csv": ("ols_heteroskedastic", ["y", "x1", "x2"]),
    "ols-003.csv": ("ols_near_collinear", ["y", "x1", "x2"]),
    "iv-001.csv": ("iv_strong", ["y", "d", "w", "z1", "z2"]),
    "iv-002.csv": ("iv_weak", ["y", "d", "w", "z1", "z2"]),
    "iv-003.csv": ("iv_just_identified", ["y", "d", "w", "z1"]),
    "fe-001.csv": ("panel_balanced", ["entity", "period", "y", "x1", "x2"]),
    "fe-002.csv": ("panel_confounded", ["entity", "period", "y", "x1", "x2"]),
    "fe-003.csv": ("panel_unbalanced", ["entity", "period", "y", "x1", "x2"]),
    "adf-001.csv": ("series_stationary", ["t", "y"]),
    "adf-002.csv": ("series_random_walk", ["t", "y"]),
    "adf-003.csv": ("series_trend_stationary", ["t", "y"]),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build(target_dir: str) -> dict:
    os.makedirs(target_dir, exist_ok=True)
    manifest = {}
    for filename, (generator, columns) in sorted(FIXTURES.items()):
        frame = GENERATORS[generator]()
        path = os.path.join(target_dir, filename)
        write_frame(path, frame, columns)
        manifest[filename] = {
            "generator": generator,
            "columns": columns,
            "rows": int(len(frame[columns[0]])),
            "sha256": sha256_file(path),
        }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    args = parser.parse_args()

    fixtures_dir = os.path.join(ROOT, "fixtures")
    manifest_path = os.path.join(fixtures_dir, "manifest.json")

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = build(tmp)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            committed = json.load(handle)
        problems = []
        for name, entry in fresh.items():
            if name not in committed:
                problems.append("%s is not in the committed manifest" % name)
            elif committed[name]["sha256"] != entry["sha256"]:
                problems.append(
                    "%s checksum drifted: committed %s, regenerated %s"
                    % (name, committed[name]["sha256"][:16], entry["sha256"][:16])
                )
            else:
                on_disk = sha256_file(os.path.join(fixtures_dir, name))
                if on_disk != entry["sha256"]:
                    problems.append(
                        "%s on disk does not match its generator" % name
                    )
        for name in committed:
            if name not in fresh:
                problems.append("%s has no generator" % name)
        if problems:
            for line in problems:
                print("FAIL %s" % line)
            return 1
        print("OK %d fixtures match their generators" % len(fresh))
        return 0

    manifest = build(fixtures_dir)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote %d fixtures and manifest.json" % len(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
