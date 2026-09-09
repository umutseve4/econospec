"""The specs are the contract, so the contract itself is tested."""

from __future__ import annotations

import json
import os

from econospec.specs import (
    CASE_TYPES,
    FAMILIES,
    SpecError,
    load_all_specs,
    spec_paths,
    validate_spec,
)


def test_every_spec_loads_and_validates():
    specs = load_all_specs(".")
    assert len(specs) == 12


def test_ids_are_unique_and_match_their_filenames():
    ids = []
    for path in spec_paths("."):
        with open(path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        assert path.endswith("%s.json" % spec["id"]), path
        ids.append(spec["id"])
    assert len(set(ids)) == len(ids)


def test_families_live_in_their_own_directory():
    for path in spec_paths("."):
        with open(path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        parts = os.path.abspath(path).replace("\\", "/").split("/")
        directory = parts[-2]
        assert directory == FAMILIES[spec["family"]], (
            "%s declares family %s but lives in specs/%s"
            % (spec["id"], spec["family"], directory)
        )


def test_every_family_covers_every_case_type():
    grid = {}
    for spec in load_all_specs("."):
        grid.setdefault(spec["family"], set()).add(spec["case_type"])
    assert set(grid) == set(FAMILIES)
    for family, cases in grid.items():
        assert cases == set(CASE_TYPES), (family, cases)


def test_fixture_checksums_agree_with_the_manifest():
    with open("fixtures/manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for spec in load_all_specs("."):
        name = spec["fixture"]["path"].split("/")[-1]
        assert spec["fixture"]["sha256"] == manifest[name]["sha256"], spec["id"]
        assert spec["fixture"]["generator"] == manifest[name]["generator"], spec["id"]


def test_every_tolerance_carries_a_rationale():
    for spec in load_all_specs("."):
        tol = spec["comparison"]["tolerance"]
        assert tol["rtol"] > 0 and tol["rtol"] <= 1e-3, spec["id"]
        assert len(tol["rationale"].split()) >= 8, spec["id"]
        for override in tol.get("key_overrides", []):
            assert len(override["rationale"].split()) >= 8, spec["id"]


def test_exclusions_are_declared_with_a_reason():
    for spec in load_all_specs("."):
        for excl in spec["comparison"].get("exclude", []):
            assert excl["key"] and len(excl["reason"].split()) >= 8, spec["id"]


def test_validator_rejects_a_spec_with_a_thin_tolerance_rationale():
    spec = load_all_specs(".")[0]
    broken = json.loads(json.dumps(spec))
    broken["comparison"]["tolerance"]["rationale"] = "because"
    try:
        validate_spec(broken, ".")
    except SpecError:
        return
    raise AssertionError("a bare tolerance must not validate")


def test_validator_rejects_a_missing_fixture():
    spec = load_all_specs(".")[0]
    broken = json.loads(json.dumps(spec))
    broken["fixture"]["path"] = "fixtures/does-not-exist.csv"
    try:
        validate_spec(broken, ".")
    except SpecError:
        return
    raise AssertionError("a spec pointing at nothing must not validate")


def test_validator_rejects_a_model_kind_that_contradicts_the_family():
    spec = load_all_specs(".")[0]
    broken = json.loads(json.dumps(spec))
    broken["model"]["kind"] = "adf" if broken["family"] != "adf" else "ols"
    try:
        validate_spec(broken, ".")
    except SpecError:
        return
    raise AssertionError("family and model kind must agree")


def test_validator_rejects_an_undocumented_exclusion():
    spec = load_all_specs(".")[0]
    broken = json.loads(json.dumps(spec))
    broken["comparison"]["exclude"] = [{"key": "scalars.r2", "reason": "n/a"}]
    try:
        validate_spec(broken, ".")
    except SpecError:
        return
    raise AssertionError("an exclusion without a real reason must not validate")
