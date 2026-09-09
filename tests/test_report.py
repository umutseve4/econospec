"""Tests for the report generator.

A report that silently omits a specification would let a red run look green,
so the generator is required to fail loudly instead of rendering less than it
was given.
"""

import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_report  # noqa: E402
import compare_results  # noqa: E402

from econospec.specs import load_all_specs  # noqa: E402


def _load():
    """Build a conformance payload here rather than reading one off disk.

    The earlier version opened results/conformance.json, which only exists
    after the comparator has already been run. On a clean checkout these five
    tests failed for a reason that had nothing to do with the report, and on a
    machine where the pipeline had been run they passed for a reason that had
    nothing to do with the checkout. The payload is now assembled in process by
    the same function the command line tool uses, from the committed golden
    manifest compared against itself, so the tests depend only on what is in
    the repository.
    """
    specs = {s["id"]: s for s in load_all_specs(ROOT)}
    golden = compare_results.load_dir(os.path.join(ROOT, "fixtures", "golden"))
    report = compare_results.build_payload(
        specs, golden, {"golden-self": golden}, {}, ["golden-self"]
    )
    assert report["green"], "the golden manifest does not agree with itself"
    return report, specs


def test_every_specification_appears_in_the_report():
    report, specs = _load()
    page = build_report.build(report, specs)
    for spec_id in report["specs"]:
        assert 'id="%s"' % spec_id in page, "specification %s is missing from the report" % spec_id


def test_unknown_family_is_refused_instead_of_dropped():
    report, specs = _load()
    specs = copy.deepcopy(specs)
    victim = report["specs"][0]
    specs[victim]["family"] = "quantile-regression"
    try:
        build_report.render_specs(report, specs)
    except SystemExit as exc:
        assert "quantile-regression" in str(exc)
        return
    raise AssertionError("the generator dropped a specification without complaining")


def test_report_states_the_independence_disclaimer():
    report, specs = _load()
    page = build_report.build(report, specs)
    assert "not an\nofficial product" in page or "not an official product" in page


def test_report_uses_no_long_dashes():
    report, specs = _load()
    page = build_report.build(report, specs)
    assert "\u2014" not in page and "\u2013" not in page


def test_status_word_follows_the_comparator():
    report, specs = _load()
    red = copy.deepcopy(report)
    red["green"] = False
    assert "NOT CONFORMANT" in build_report.build(red, specs)
    green = copy.deepcopy(report)
    green["green"] = True
    assert "NOT CONFORMANT" not in build_report.build(green, specs)
