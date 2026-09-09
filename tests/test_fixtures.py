"""Fixture integrity: the committed CSVs must be exactly what the code makes."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from econospec.csvio import read_frame, write_frame
from econospec.dgp import GENERATORS

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import gen_fixtures  # noqa: E402


def test_manifest_covers_every_fixture_file():
    with open("fixtures/manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    on_disk = {n for n in os.listdir("fixtures") if n.endswith(".csv")}
    assert on_disk == set(manifest), on_disk ^ set(manifest)
    assert len(manifest) == 12


def test_committed_checksums_match_the_files_on_disk():
    with open("fixtures/manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for name, entry in manifest.items():
        actual = gen_fixtures.sha256_file(os.path.join("fixtures", name))
        assert actual == entry["sha256"], name


def test_regeneration_is_byte_identical():
    with tempfile.TemporaryDirectory() as tmp:
        fresh = gen_fixtures.build(tmp)
    with open("fixtures/manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for name, entry in fresh.items():
        assert entry["sha256"] == manifest[name]["sha256"], name


def test_generators_are_deterministic_across_calls():
    for name, fn in GENERATORS.items():
        first, second = fn(), fn()
        for column in first:
            assert np.array_equal(first[column], second[column]), (name, column)


def test_csv_round_trip_is_exact():
    g = np.random.Generator(np.random.PCG64(11))
    frame = {"a": g.normal(size=50) * 1e-9, "b": g.normal(size=50) * 1e12}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rt.csv")
        write_frame(path, frame, ["a", "b"])
        back = read_frame(path)
    for column in frame:
        assert np.array_equal(frame[column], back[column]), column


def test_fixtures_have_no_missing_or_infinite_values():
    for name in sorted(os.listdir("fixtures")):
        if not name.endswith(".csv"):
            continue
        frame = read_frame(os.path.join("fixtures", name))
        for column, values in frame.items():
            arr = np.asarray(values, dtype=float)
            assert np.all(np.isfinite(arr)), (name, column)


def test_fixture_shapes_are_what_the_specs_expect():
    expected_rows = {
        "ols-001.csv": 240,
        "ols-002.csv": 300,
        "ols-003.csv": 180,
        "iv-001.csv": 400,
        "iv-002.csv": 400,
        "iv-003.csv": 350,
        "fe-001.csv": 480,
        "fe-002.csv": 500,
        "adf-001.csv": 300,
        "adf-002.csv": 300,
        "adf-003.csv": 320,
    }
    for name, rows in expected_rows.items():
        frame = read_frame(os.path.join("fixtures", name))
        assert len(next(iter(frame.values()))) == rows, name
    # fe-003 is unbalanced by construction, so its row count is data dependent.
    fe3 = read_frame("fixtures/fe-003.csv")
    assert 55 * 3 <= len(fe3["y"]) <= 55 * 12


def test_line_endings_are_unix():
    for name in sorted(os.listdir("fixtures")):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join("fixtures", name), "rb") as handle:
            raw = handle.read()
        assert b"\r" not in raw, name
