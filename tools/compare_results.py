#!/usr/bin/env python3
"""Compare adapter envelopes against the golden manifest.

Writes ``results/conformance.json`` and exits non zero if any comparison fails
or if a required adapter produced nothing. An adapter that silently skips a
spec is a failure, because a suite that reports green on missing evidence is
worse than no suite.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from econospec.compare import compare_flat, is_green, summarise  # noqa: E402
from econospec.runner import flatten  # noqa: E402
from econospec.specs import load_all_specs  # noqa: E402


def load_dir(path: str) -> dict:
    """Load result envelopes from a directory, ignoring environment manifests."""
    out = {}
    for file in sorted(glob.glob(os.path.join(path, "*.json"))):
        if os.path.basename(file) == "environment.json":
            continue
        with open(file, "r", encoding="utf-8") as handle:
            env = json.load(handle)
        if "spec_id" not in env:
            raise SystemExit("%s is not a result envelope: no spec_id" % file)
        out[env["spec_id"]] = env
    return out


def load_environment(path: str) -> dict:
    file = os.path.join(path, "environment.json")
    if not os.path.isfile(file):
        return {}
    with open(file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_payload(specs: dict, golden: dict, adapters: dict,
                  environments: dict, require: list) -> dict:
    """Assemble the conformance report from loaded envelopes.

    Kept separate from ``main`` so the report tests can build a payload in
    process instead of reading a file that only exists after an earlier
    pipeline step has run. A test suite whose result depends on what was run
    before it is not a test suite.
    """
    missing_adapters = [a for a in require if a not in adapters]
    records = []
    matrix = {}
    for adapter, envelopes in sorted(adapters.items()):
        matrix[adapter] = {}
        for spec_id, spec in sorted(specs.items()):
            if spec_id not in envelopes:
                matrix[adapter][spec_id] = {
                    "status": "absent", "pass": 0, "fail": 0, "missing": 0, "excluded": 0
                }
                records.append({
                    "spec_id": spec_id, "key": "*", "left": adapter, "right": "golden",
                    "status": "missing", "reason": "adapter produced no envelope",
                })
                continue
            recs = compare_flat(
                flatten(envelopes[spec_id]), flatten(golden[spec_id]), spec,
                adapter, "golden",
            )
            records.extend(recs)
            counts = summarise(recs)
            matrix[adapter][spec_id] = dict(
                counts, status="pass" if is_green(recs) else "fail"
            )

    green = all(r["status"] in ("pass", "excluded") for r in records) and not missing_adapters
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "adapters": sorted(adapters),
        "environments": environments,
        "required_adapters": list(require),
        "missing_adapters": missing_adapters,
        "specs": sorted(specs),
        "matrix": matrix,
        "totals": summarise(records),
        "comparisons": len(records),
        "green": green,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="directory of adapter directories")
    parser.add_argument("--golden", default="fixtures/golden")
    parser.add_argument("--require", nargs="*", default=[], help="adapters that must be present")
    parser.add_argument("--out", default="results/conformance.json")
    args = parser.parse_args()

    specs = {s["id"]: s for s in load_all_specs(ROOT)}
    golden = load_dir(os.path.join(ROOT, args.golden))
    if set(golden) != set(specs):
        print("FAIL golden manifest does not cover every spec")
        return 1

    results_root = os.path.join(ROOT, args.results)
    adapters = {}
    environments = {}
    if os.path.isdir(results_root):
        for name in sorted(os.listdir(results_root)):
            directory = os.path.join(results_root, name)
            if os.path.isdir(directory):
                loaded = load_dir(directory)
                if loaded:
                    adapters[name] = loaded
                    environments[name] = load_environment(directory)

    report = build_payload(specs, golden, adapters, environments, args.require)
    missing_adapters = report["missing_adapters"]
    matrix = report["matrix"]
    green = report["green"]
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("adapters compared: %s" % (", ".join(sorted(adapters)) or "none"))
    for adapter in sorted(matrix):
        for spec_id in sorted(matrix[adapter]):
            cell = matrix[adapter][spec_id]
            print("  %-28s %-8s %-7s pass=%d fail=%d missing=%d excluded=%d" % (
                adapter, spec_id, cell["status"], cell["pass"], cell["fail"],
                cell["missing"], cell["excluded"],
            ))
    for adapter in missing_adapters:
        print("FAIL required adapter %s produced no results" % adapter)
    totals = report["totals"]
    print("comparisons=%d pass=%d fail=%d missing=%d excluded=%d" % (
        report["comparisons"], totals["pass"], totals["fail"],
        totals["missing"], totals["excluded"],
    ))
    print("GREEN" if green else "RED")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
