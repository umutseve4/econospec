"""Loading and validation of experiment specifications.

A spec is the contract. It names the fixture, the exact model, the comparison
tolerances and the reason for every tolerance and every exclusion. Adapters may
not invent anything that is not in the spec, which is what makes a disagreement
attributable.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

__all__ = [
    "SpecError",
    "REQUIRED_TOP_LEVEL",
    "MODEL_REQUIRED",
    "CASE_TYPES",
    "FAMILIES",
    "validate_spec",
    "load_spec",
    "load_all_specs",
    "spec_paths",
]

REQUIRED_TOP_LEVEL = (
    "id",
    "family",
    "title",
    "case_type",
    "narrative",
    "fixture",
    "model",
    "comparison",
)

CASE_TYPES = ("nominal", "violation", "edge")

FAMILIES = {
    "ols": "ols",
    "iv2sls": "iv-2sls",
    "panel_fe": "panel-fe",
    "adf": "adf",
}

MODEL_REQUIRED: Dict[str, tuple] = {
    "ols": ("response", "regressors", "intercept", "vcov"),
    "iv2sls": ("response", "exog", "endog", "instruments", "intercept"),
    "panel_fe": ("response", "regressors", "entity"),
    "adf": ("series", "lags", "trend"),
}


class SpecError(ValueError):
    """Raised when a spec violates the contract."""


def validate_spec(spec: dict, root: str) -> None:
    for key in REQUIRED_TOP_LEVEL:
        if key not in spec:
            raise SpecError("spec %r is missing key %r" % (spec.get("id"), key))

    sid = spec["id"]
    family = spec["family"]
    if family not in FAMILIES:
        raise SpecError("spec %s has unknown family %r" % (sid, family))
    if spec["case_type"] not in CASE_TYPES:
        raise SpecError("spec %s has unknown case_type %r" % (sid, spec["case_type"]))
    if len(spec["narrative"].split()) < 25:
        raise SpecError(
            "spec %s narrative is too thin: a spec must explain what the case "
            "is testing and why it can fail" % (sid,)
        )

    fixture = spec["fixture"]
    for key in ("generator", "path", "sha256"):
        if key not in fixture:
            raise SpecError("spec %s fixture is missing %r" % (sid, key))
    fixture_path = os.path.join(root, fixture["path"])
    if not os.path.exists(fixture_path):
        raise SpecError("spec %s points at a missing fixture %s" % (sid, fixture["path"]))

    model = spec["model"]
    kind = model.get("kind")
    if kind != family:
        raise SpecError("spec %s model kind %r does not match family %r" % (sid, kind, family))
    for key in MODEL_REQUIRED[kind]:
        if key not in model:
            raise SpecError("spec %s model is missing %r" % (sid, key))

    comparison = spec["comparison"]
    if "tolerance" not in comparison:
        raise SpecError("spec %s has no tolerance block" % (sid,))
    tol = comparison["tolerance"]
    for key in ("rtol", "atol", "rationale"):
        if key not in tol:
            raise SpecError("spec %s tolerance is missing %r" % (sid, key))
    if len(tol["rationale"].split()) < 8:
        raise SpecError(
            "spec %s tolerance rationale is too thin: every tolerance must be "
            "justified, not merely declared" % (sid,)
        )
    for override in tol.get("key_overrides", []):
        for key in ("pattern", "rtol", "atol", "rationale"):
            if key not in override:
                raise SpecError("spec %s key_override is missing %r" % (sid, key))
        if len(override["rationale"].split()) < 8:
            raise SpecError("spec %s key_override rationale is too thin" % (sid,))
    for excl in comparison.get("exclude", []):
        for key in ("key", "reason"):
            if key not in excl:
                raise SpecError("spec %s exclusion is missing %r" % (sid, key))
        if len(excl["reason"].split()) < 8:
            raise SpecError(
                "spec %s exclusion reason is too thin: silently dropping a key "
                "is how a conformance suite becomes decoration" % (sid,)
            )


def spec_paths(root: str) -> List[str]:
    out: List[str] = []
    specs_dir = os.path.join(root, "specs")
    for family_dir in sorted(os.listdir(specs_dir)):
        full = os.path.join(specs_dir, family_dir)
        if not os.path.isdir(full):
            continue
        for name in sorted(os.listdir(full)):
            if name.endswith(".json"):
                out.append(os.path.join(full, name))
    return out


def load_spec(path: str, root: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    validate_spec(spec, root)
    return spec


def load_all_specs(root: str) -> List[dict]:
    return [load_spec(p, root) for p in spec_paths(root)]
