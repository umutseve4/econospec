"""The comparator.

This is the load bearing part of EconoSpec. If it is too permissive the whole
repository is decoration, so tests/test_conformance.py mutates known good
results and asserts that the comparator fails. A change that makes the
comparator softer must make that mutation test fail first.

Rules:

* Keys are compared on the union of both sides. A key present in one adapter
  and absent in the other is a failure, never a skip.
* Exclusions must be declared in the spec with a reason, and they are matched
  as glob patterns against the flattened key.
* Tolerances are per spec, with optional per key overrides, because a
  coefficient and a p-value do not deserve the same absolute floor.
"""

from __future__ import annotations

import fnmatch
import math
from typing import Dict, List, Optional, Tuple

__all__ = [
    "Comparison",
    "tolerance_for",
    "excluded_reason",
    "compare_flat",
    "compare_envelopes",
    "is_green",
    "summarise",
]

Comparison = Dict[str, object]


def tolerance_for(key: str, tolerance: dict) -> Tuple[float, float, str]:
    """Resolve (rtol, atol, source) for a flattened key."""
    for override in tolerance.get("key_overrides", []):
        if fnmatch.fnmatch(key, override["pattern"]):
            return float(override["rtol"]), float(override["atol"]), override["pattern"]
    return float(tolerance["rtol"]), float(tolerance["atol"]), "default"


def excluded_reason(key: str, comparison_block: dict) -> Optional[str]:
    for excl in comparison_block.get("exclude", []):
        if fnmatch.fnmatch(key, excl["key"]):
            return excl["reason"]
    return None


def _within(a: float, b: float, rtol: float, atol: float) -> Tuple[bool, float, float]:
    if math.isnan(a) or math.isnan(b):
        return (math.isnan(a) and math.isnan(b), float("nan"), float("nan"))
    diff = abs(a - b)
    allowed = atol + rtol * abs(b)
    return diff <= allowed, diff, allowed


def compare_flat(
    left: Dict[str, float],
    right: Dict[str, float],
    spec: dict,
    left_name: str,
    right_name: str,
) -> List[Comparison]:
    """Compare two flattened result maps under a spec's comparison block."""
    block = spec["comparison"]
    tolerance = block["tolerance"]
    keys = sorted(set(left) | set(right))
    records: List[Comparison] = []
    for key in keys:
        reason = excluded_reason(key, block)
        if reason is not None:
            records.append(
                {
                    "spec_id": spec["id"],
                    "key": key,
                    "left": left_name,
                    "right": right_name,
                    "status": "excluded",
                    "reason": reason,
                }
            )
            continue
        if key not in left or key not in right:
            missing = left_name if key not in left else right_name
            records.append(
                {
                    "spec_id": spec["id"],
                    "key": key,
                    "left": left_name,
                    "right": right_name,
                    "status": "missing",
                    "reason": "key absent from %s" % (missing,),
                }
            )
            continue
        rtol, atol, source = tolerance_for(key, tolerance)
        ok, diff, allowed = _within(left[key], right[key], rtol, atol)
        records.append(
            {
                "spec_id": spec["id"],
                "key": key,
                "left": left_name,
                "right": right_name,
                "status": "pass" if ok else "fail",
                "left_value": left[key],
                "right_value": right[key],
                "abs_diff": diff,
                "allowed": allowed,
                "rtol": rtol,
                "atol": atol,
                "tolerance_source": source,
            }
        )
    return records


def compare_envelopes(left: dict, right: dict, spec: dict) -> List[Comparison]:
    from .runner import flatten

    return compare_flat(
        flatten(left),
        flatten(right),
        spec,
        left.get("adapter", "left"),
        right.get("adapter", "right"),
    )


def is_green(records: List[Comparison]) -> bool:
    return all(r["status"] in ("pass", "excluded") for r in records)


def summarise(records: List[Comparison]) -> Dict[str, int]:
    out = {"pass": 0, "fail": 0, "missing": 0, "excluded": 0}
    for record in records:
        out[str(record["status"])] += 1
    return out
