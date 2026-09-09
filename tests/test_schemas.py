"""The published schemas must describe the files that actually exist.

There is no schema validator in the reference environment, and adding one as a
hard dependency would put a library between the contract and its own tests.
Instead these checks read the schema as data and apply its structural rules by
hand to the real specs and the real envelopes, so a schema that drifts away
from the repository fails the suite rather than misleading a reader.
"""

from __future__ import annotations

import glob
import json
import os

from econospec.specs import CASE_TYPES, FAMILIES, load_all_specs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _schema(name):
    with open(os.path.join(ROOT, "schemas", name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _envelopes():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "fixtures", "golden", "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            out.append((os.path.basename(path), json.load(handle)))
    assert out, "no golden envelopes to check the result schema against"
    return out


def test_experiment_schema_matches_the_real_specs():
    schema = _schema("experiment.schema.json")
    allowed = set(schema["properties"])
    required = set(schema["required"])
    for spec in load_all_specs(ROOT):
        keys = set(spec)
        assert required <= keys, "%s is missing %s" % (spec["id"], sorted(required - keys))
        assert keys <= allowed, "%s has undeclared keys %s" % (
            spec["id"], sorted(keys - allowed)
        )


def test_experiment_schema_enums_track_the_code():
    schema = _schema("experiment.schema.json")
    assert set(schema["properties"]["family"]["enum"]) == set(FAMILIES)
    assert set(schema["properties"]["case_type"]["enum"]) == set(CASE_TYPES)
    assert set(schema["properties"]["model"]["properties"]["kind"]["enum"]) == set(FAMILIES)


def test_specs_satisfy_the_declared_prose_minimums():
    schema = _schema("experiment.schema.json")
    narrative_min = schema["properties"]["narrative"]["minLength"]
    tol_schema = schema["$defs"]["tolerance"]
    rationale_min = tol_schema["properties"]["rationale"]["minLength"]
    for spec in load_all_specs(ROOT):
        assert len(spec["narrative"]) >= narrative_min, spec["id"]
        tol = spec["comparison"]["tolerance"]
        assert len(tol["rationale"]) >= rationale_min, spec["id"]
        for override in tol.get("key_overrides", []):
            assert len(override["rationale"]) >= rationale_min, spec["id"]


def test_result_schema_matches_the_golden_envelopes():
    schema = _schema("result.schema.json")
    allowed = set(schema["properties"])
    required = set(schema["required"])
    param_schema = schema["properties"]["params"]["additionalProperties"]
    param_allowed = set(param_schema["properties"])
    param_required = set(param_schema["required"])
    for name, env in _envelopes():
        keys = set(env)
        assert required <= keys, "%s is missing %s" % (name, sorted(required - keys))
        assert keys <= allowed, "%s has undeclared keys %s" % (name, sorted(keys - allowed))
        assert env["params"], "%s has no parameters" % name
        assert env["scalars"], "%s has no scalars" % name
        for pname, entry in env["params"].items():
            entry_keys = set(entry)
            assert param_required <= entry_keys, "%s %s" % (name, pname)
            assert entry_keys <= param_allowed, "%s %s has %s" % (
                name, pname, sorted(entry_keys - param_allowed)
            )
        for sname, value in env["scalars"].items():
            assert isinstance(value, (int, float)), "%s scalar %s is not numeric" % (name, sname)


def test_adf_envelopes_publish_no_pvalue():
    """The schema and docs/divergences.md both promise this, so it is tested."""
    for name, env in _envelopes():
        if not name.startswith("adf-"):
            continue
        for pname, entry in env["params"].items():
            assert "pvalue" not in entry, (
                "%s reports a p-value for %s, but the ADF statistic is not "
                "Student t under the null" % (name, pname)
            )


def test_non_adf_envelopes_do_publish_a_pvalue():
    for name, env in _envelopes():
        if name.startswith("adf-"):
            continue
        for pname, entry in env["params"].items():
            assert "pvalue" in entry, "%s omits the p-value for %s" % (name, pname)
            assert 0.0 <= entry["pvalue"] <= 1.0, "%s %s" % (name, pname)
