#!/usr/bin/env python3
"""One time repair of the long dash guard in tools/build_report.py.

The guard was damaged in transfer: it was written as a search for the literal
six character sequence backslash u 2 0 1 4 rather than for the character that
sequence denotes. A guard that cannot fire is worse than no guard, because it
reports safety it does not provide.

This script refuses to act unless it finds exactly one occurrence of the
damaged line, and it verifies afterwards, by parsing the repaired source, that
a real long dash character now appears as a string constant. It deletes itself
from the working tree once that check passes.
"""

from __future__ import annotations

import ast
import os
import sys

BS = chr(92)
EM = chr(8212)
EN = chr(8211)

DAMAGED = 'if "' + BS + BS + 'u2014" in page or "' + BS + BS + 'u2013" in page:'
REPAIRED = 'if "' + BS + 'u2014" in page or "' + BS + 'u2013" in page:'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "tools", "build_report.py")


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as handle:
        source = handle.read()

    hits = source.count(DAMAGED)
    if hits != 1:
        print("expected exactly one damaged guard, found %d" % hits)
        return 1

    patched = source.replace(DAMAGED, REPAIRED)
    if REPAIRED not in patched or DAMAGED in patched:
        print("replacement did not take")
        return 1

    constants = {
        node.value
        for node in ast.walk(ast.parse(patched))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if EM not in constants or EN not in constants:
        print("the repaired guard still does not hold a real long dash")
        return 1

    with open(TARGET, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(patched)
    print("repaired the long dash guard in %s" % TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
