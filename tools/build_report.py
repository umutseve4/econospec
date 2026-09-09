#!/usr/bin/env python3
"""Render the Conformance Observatory into ``site/``.

The report is a single static page with no external requests, no analytics and
no fonts fetched from a third party. It is generated from
``results/conformance.json`` and the specs, so it cannot claim a status the
comparator did not produce.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from econospec.specs import load_all_specs  # noqa: E402

FAMILY_TITLES = {
    "ols": "Ordinary least squares",
    "iv2sls": "Instrumental variables, two stage least squares",
    "panel_fe": "Panel fixed effects",
    "adf": "Augmented Dickey Fuller",
}
FAMILY_ORDER = ["ols", "iv2sls", "panel_fe", "adf"]

CSS = """
:root {
  color-scheme: dark;
  --bg: #0b0d10;
  --panel: #14181d;
  --line: #232a32;
  --ink: #e8edf2;
  --muted: #97a3b0;
  --accent: #ff4d4f;
  --pass: #35c46b;
  --fail: #ff4d4f;
  --warn: #e0a300;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.6 ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 1080px; margin: 0 auto; padding: 40px 20px 80px; }
h1 { font-size: 2.1rem; margin: 0 0 8px; letter-spacing: -0.02em; }
h2 { font-size: 1.3rem; margin: 48px 0 12px; }
h3 { font-size: 1.02rem; margin: 0 0 6px; }
p { margin: 0 0 12px; }
a { color: var(--ink); }
.lede { color: var(--muted); max-width: 68ch; }
.disclaimer {
  border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  background: var(--panel);
  padding: 12px 16px;
  border-radius: 6px;
  color: var(--muted);
  font-size: 0.92rem;
}
.status {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.85rem;
  letter-spacing: 0.04em;
}
.status.green { background: rgba(53,196,107,0.14); color: var(--pass); }
.status.red { background: rgba(255,77,79,0.14); color: var(--fail); }
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}
.card .n { font-size: 1.7rem; font-weight: 650; }
.card .k { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; }
table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
td.cell { font-variant-numeric: tabular-nums; }
.tag { font-size: 0.78rem; padding: 2px 8px; border-radius: 4px; border: 1px solid var(--line); }
.tag.pass { color: var(--pass); border-color: rgba(53,196,107,0.4); }
.tag.fail { color: var(--fail); border-color: rgba(255,77,79,0.4); }
.tag.absent { color: var(--warn); border-color: rgba(224,163,0,0.4); }
details {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 10px;
}
summary { cursor: pointer; font-weight: 600; }
summary::marker { color: var(--accent); }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.88em; }
.muted { color: var(--muted); }
.small { font-size: 0.88rem; }
footer { margin-top: 60px; color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--line); padding-top: 16px; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def status_tag(status: str) -> str:
    label = {"pass": "PASS", "fail": "FAIL", "absent": "ABSENT"}.get(status, status.upper())
    return '<span class="tag %s">%s</span>' % (esc(status), esc(label))


def render_matrix(report: dict, specs: dict) -> str:
    adapters = report["adapters"]
    if not adapters:
        return '<p class="muted">No adapter produced results.</p>'
    head = "".join("<th>%s</th>" % esc(a) for a in adapters)
    rows = []
    for spec_id in report["specs"]:
        spec = specs[spec_id]
        cells = []
        for adapter in adapters:
            cell = report["matrix"][adapter].get(spec_id)
            if cell is None:
                cells.append('<td class="cell">%s</td>' % status_tag("absent"))
                continue
            detail = '<div class="small muted">%d compared</div>' % (
                cell["pass"] + cell["fail"] + cell["excluded"],
            )
            cells.append('<td class="cell">%s%s</td>' % (status_tag(cell["status"]), detail))
        rows.append(
            "<tr><td><a href=\"#%s\"><code>%s</code></a><div class=\"small muted\">%s</div></td>%s</tr>"
            % (esc(spec_id), esc(spec_id), esc(spec["title"]), "".join(cells))
        )
    return (
        '<table><thead><tr><th>Specification</th>%s</tr></thead><tbody>%s</tbody></table>'
        % (head, "".join(rows))
    )


def render_environments(report: dict) -> str:
    envs = report.get("environments") or {}
    if not envs:
        return '<p class="muted">No environment manifest was recorded.</p>'
    blocks = []
    for adapter in sorted(envs):
        env = envs[adapter]
        if not env:
            blocks.append(
                '<div class="card"><div class="k">%s</div><p class="small muted">'
                'No environment manifest.</p></div>' % esc(adapter)
            )
            continue
        packages = "".join(
            "<li><code>%s</code> %s</li>" % (esc(k), esc(v))
            for k, v in sorted((env.get("packages") or {}).items())
        )
        blocks.append(
            '<div class="card"><div class="k">%s</div>'
            '<p class="small muted">%s %s<br>%s</p>'
            '<ul class="small mono" style="margin:0;padding-left:18px">%s</ul></div>'
            % (
                esc(adapter),
                esc(env.get("language", "")),
                esc(env.get("runtime", "")),
                esc(env.get("platform", "")),
                packages,
            )
        )
    return '<div class="grid">%s</div>' % "".join(blocks)


def render_failures(report: dict) -> str:
    bad = [r for r in report["records"] if r["status"] in ("fail", "missing")]
    if not bad:
        return (
            '<p class="muted">Every compared value agreed inside its documented '
            'tolerance. The list below stays empty only while that is true.</p>'
        )
    rows = []
    for record in bad[:200]:
        rows.append(
            "<tr><td><code>%s</code></td><td><code>%s</code></td><td>%s</td>"
            "<td class=\"cell mono\">%s</td><td class=\"cell mono\">%s</td>"
            "<td class=\"cell mono\">%s</td></tr>"
            % (
                esc(record["spec_id"]),
                esc(record["key"]),
                esc(record["left"]),
                esc(record.get("left_value", "")),
                esc(record.get("right_value", "")),
                esc(record.get("allowed", record.get("reason", ""))),
            )
        )
    return (
        "<table><thead><tr><th>Spec</th><th>Key</th><th>Adapter</th><th>Value</th>"
        "<th>Golden</th><th>Allowed</th></tr></thead><tbody>%s</tbody></table>"
        % "".join(rows)
    )


def render_specs(report: dict, specs: dict) -> str:
    unknown = sorted({specs[s]["family"] for s in report["specs"]} - set(FAMILY_ORDER))
    if unknown:
        raise SystemExit(
            "report would silently drop specifications from families: %s" % ", ".join(unknown)
        )
    blocks = []
    rendered = 0
    for family in FAMILY_ORDER:
        members = [s for s in report["specs"] if specs[s]["family"] == family]
        rendered += len(members)
        if not members:
            continue
        blocks.append("<h3>%s</h3>" % esc(FAMILY_TITLES.get(family, family)))
        for spec_id in members:
            spec = specs[spec_id]
            tol = spec["comparison"]["tolerance"]
            overrides = "".join(
                "<li><code>%s</code> rtol %s, atol %s<br><span class=\"muted\">%s</span></li>"
                % (esc(o["pattern"]), esc(o["rtol"]), esc(o["atol"]), esc(o["rationale"]))
                for o in tol.get("key_overrides", [])
            )
            excludes = "".join(
                "<li><code>%s</code> <span class=\"muted\">%s</span></li>"
                % (esc(e["key"]), esc(e["reason"]))
                for e in spec["comparison"].get("exclude", [])
            )
            blocks.append(
                '<details id="%s"><summary>%s <span class="muted">%s</span></summary>'
                '<p class="small">%s</p>'
                '<p class="small muted">Case type <code>%s</code>. Fixture '
                '<code>%s</code>, sha256 <code>%s</code>.</p>'
                '<p class="small"><strong>Tolerance</strong> rtol %s, atol %s<br>'
                '<span class="muted">%s</span></p>'
                '%s%s</details>'
                % (
                    esc(spec_id),
                    esc(spec_id),
                    esc(spec["title"]),
                    esc(spec["narrative"]),
                    esc(spec["case_type"]),
                    esc(spec["fixture"]["path"]),
                    esc(spec["fixture"]["sha256"][:16]),
                    esc(tol["rtol"]),
                    esc(tol["atol"]),
                    esc(tol["rationale"]),
                    ('<p class="small"><strong>Key overrides</strong></p><ul class="small">%s</ul>'
                     % overrides) if overrides else "",
                    ('<p class="small"><strong>Declared exclusions</strong></p><ul class="small">%s</ul>'
                     % excludes) if excludes
                    else '<p class="small muted">No key is excluded from comparison.</p>',
                )
            )
    if rendered != len(report["specs"]):
        raise SystemExit("rendered %d of %d specifications" % (rendered, len(report["specs"])))
    return "".join(blocks)


def build(report: dict, specs: dict) -> str:
    totals = report["totals"]
    green = report["green"]
    cards = [
        ("Comparisons", report["comparisons"]),
        ("Agreed", totals["pass"]),
        ("Disagreed", totals["fail"]),
        ("Missing", totals["missing"]),
        ("Excluded", totals["excluded"]),
        ("Specifications", len(report["specs"])),
    ]
    card_html = "".join(
        '<div class="card"><div class="n">%s</div><div class="k">%s</div></div>' % (esc(v), esc(k))
        for k, v in cards
    )
    missing_note = ""
    if report.get("missing_adapters"):
        missing_note = (
            '<p class="disclaimer">Required adapters that produced nothing: %s. '
            'A skipped adapter is treated as a failure, not as a pass.</p>'
            % esc(", ".join(report["missing_adapters"]))
        )
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EconoSpec Conformance Observatory</title>
<meta name="description" content="Cross language conformance results for reproducible econometrics.">
<style>%s</style>
</head>
<body>
<main>
<h1>EconoSpec Conformance Observatory</h1>
<p class="lede">The same econometric specification, estimated by independent
implementations, compared value by value against a golden manifest inside
tolerances that each carry a written justification.</p>
<p><span class="status %s">%s</span>
<span class="muted small">generated %s</span></p>
<p class="disclaimer">This is an independent student project. It is not an
official product of Bursa Uludag University or its Department of Econometrics,
and it makes no claim to represent them.</p>
%s
<h2>Totals</h2>
<div class="grid">%s</div>
<h2>Matrix</h2>
<p class="lede small">One row per specification, one column per adapter. A cell
is green only when every compared key agreed. A specification an adapter never
ran is marked absent and counts against the run.</p>
%s
<h2>Disagreements</h2>
%s
<h2>Specifications and tolerances</h2>
<p class="lede small">Every tolerance below is part of the contract. Loosening
one is a reviewable change, because the mutation test in the suite requires
each compared value to be load bearing.</p>
%s
<h2>Environments</h2>
<p class="lede small">Numbers without a version list are not evidence.</p>
%s
<h2>Evidence</h2>
<p class="small"><a href="conformance.json">conformance.json</a> holds every
individual comparison, including the observed difference and the budget it was
allowed. This page is rendered from that file and cannot claim more than it.</p>
<footer>
EconoSpec. Independent student project inspired by econometrics education at
Bursa Uludag University. Released under the MIT license.
</footer>
</main>
</body>
</html>
""" % (
        CSS,
        "green" if green else "red",
        "CONFORMANT" if green else "NOT CONFORMANT",
        esc(report["generated_at"]),
        missing_note,
        card_html,
        render_matrix(report, specs),
        render_failures(report),
        render_specs(report, specs),
        render_environments(report),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="results/conformance.json")
    parser.add_argument("--out", default="site")
    args = parser.parse_args()

    report_path = os.path.join(ROOT, args.report)
    if not os.path.isfile(report_path):
        raise SystemExit("no report at %s, run tools/compare_results.py first" % report_path)
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    specs = {s["id"]: s for s in load_all_specs(ROOT)}

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    page = build(report, specs)
    if "\u2014" in page or "\u2013" in page:
        raise SystemExit("report contains a long dash, which the house style forbids")
    index = os.path.join(out_dir, "index.html")
    with open(index, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    shutil.copyfile(report_path, os.path.join(out_dir, "conformance.json"))
    with open(os.path.join(out_dir, ".nojekyll"), "w", encoding="utf-8") as handle:
        handle.write("")
    print("wrote %s (%d bytes)" % (index, len(page.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
