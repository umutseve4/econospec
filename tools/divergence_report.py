"""Aggregate divergence probe lines into a table that names the guilty stage.

Reads newline delimited JSON produced by tools/divergence_probe.py and reports,
for every recorded stage and value, how many distinct results appeared across
the sample and how often each one occurred. A stage with one distinct digest is
reproducible; the first stage with two is where the divergence is born.

Exit code is 1 when the final value took more than one form, so the workflow
fails on evidence of divergence rather than on a rerun of the whole suite.
"""

from __future__ import annotations

import collections
import json
import sys


def main(path: str) -> int:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("{"):
                records.append(json.loads(line))
    if not records:
        print("no probe lines were recorded")
        return 2

    fields = collections.OrderedDict()
    for record in records:
        for stage, value in sorted(record["stage"].items()):
            fields.setdefault("stage " + stage, []).append(value)
        for name, value in sorted(record["value"].items()):
            fields.setdefault("value " + name, []).append(value)
        for name, value in sorted(record["align"].items()):
            fields.setdefault("align " + name, []).append(str(value))
    print("samples: %d" % len(records))
    print("pinned:  %s" % sorted({str(r["pinned"]) for r in records}))
    print("threads: %s" % sorted({str(r["threads"]) for r in records}))
    print("")
    print("%-28s %8s  %s" % ("field", "distinct", "counts"))
    diverged = []
    for name, values in fields.items():
        counts = collections.Counter(values)
        rendered = ", ".join(
            "%s x%d" % (value, count) for value, count in counts.most_common(4)
        )
        print("%-28s %8d  %s" % (name, len(counts), rendered))
        if len(counts) > 1 and not name.startswith("align "):
            diverged.append(name)

    print("")
    if diverged:
        print("first diverging stage: %s" % diverged[0])
        print("all diverging fields:  %s" % ", ".join(diverged))
        # Cross tabulate the final value against every alignment we recorded, so
        # a correlation with memory layout is visible instead of suspected.
        for key in sorted(records[0]["align"]):
            table = collections.Counter(
                (str(r["align"][key]), r["value"]["const_estimate"]) for r in records
            )
            print("align %s vs value: %s" % (key, dict(table)))
        return 1
    print("every recorded stage took exactly one value across the sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
