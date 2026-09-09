"""CSV read and write with round trip guarantees.

Floats are written with ``%.17g``, which is the shortest format that always
round trips an IEEE 754 double. If fixtures were written with fewer digits the
conformance test would be measuring the CSV writer instead of the estimators.

Line endings are forced to ``\\n`` and column order is fixed, so the committed
files hash identically on every platform.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Sequence

import numpy as np

__all__ = ["write_frame", "read_frame", "format_float"]


def format_float(value: float) -> str:
    return "%.17g" % (float(value),)


def _format_cell(value) -> str:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    return format_float(value)


def write_frame(path: str, frame: Dict[str, np.ndarray], columns: Sequence[str] | None = None) -> None:
    cols: List[str] = list(columns) if columns is not None else list(frame.keys())
    lengths = {len(frame[c]) for c in cols}
    if len(lengths) != 1:
        raise ValueError("all columns must have the same length, got %r" % (lengths,))
    n = lengths.pop()
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(cols)
        for i in range(n):
            writer.writerow([_format_cell(frame[c][i]) for c in cols])


def read_frame(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]
    out: Dict[str, np.ndarray] = {}
    for j, name in enumerate(header):
        values = [row[j] for row in rows]
        try:
            out[name] = np.array([int(v) for v in values], dtype=np.int64)
        except ValueError:
            out[name] = np.array([float(v) for v in values], dtype=float)
    return out
