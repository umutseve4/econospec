# Contributing

Contributions are welcome. The rules below exist because this repository makes
correctness claims, and a correctness claim is only worth what its weakest
review rule allows through.

## The one rule that outranks the others

**A pull request that makes a test weaker in order to make the run green will
be refused.** That includes deleting an assertion, widening a tolerance without
a measured justification, adding an exclusion, or marking a case as skipped. A
green check is not evidence. Evidence is a test that would have caught the
error.

If a tolerance genuinely needs to be wider, the pull request must contain the
observed divergence, the environment that produced it, and an entry in
`docs/divergences.md`.

## Adding a specification

1. Add a deterministic generator in `econospec/dgp.py` with a fixed seed.
2. Register it, regenerate fixtures with `python tools/gen_fixtures.py`, and
   commit the CSV and the updated `fixtures/manifest.json`.
3. Add the specification to `tools/make_specs.py`, which is the single source
   of truth for all specs, then regenerate with `python tools/make_specs.py`.
   Editing a file under `specs/` by hand will be overwritten.
4. Write a narrative of at least twenty five words that says what would be
   learned if this case ever disagreed. The validator enforces the length; the
   review enforces that it means something.
5. Every tolerance and every override needs a rationale of at least eight
   words.
6. Refresh the golden manifest with `python tools/run_reference.py --golden`
   and read the diff line by line before committing it.
7. Teach both adapters to run the new model kind. An adapter must not import
   the reference kernel.

## Adding an implementation

A new adapter must:

- read only the spec and the fixture CSV,
- recompute every scalar from the definitions in the contract rather than
  trusting whatever a package calls `rsquared`,
- write one envelope per specification plus an `environment.json` with its
  resolved package versions,
- produce exactly the same key set as the others. A missing key is a failure,
  never a skip.

## Before opening a pull request

```bash
python tools/gen_fixtures.py --check
pytest -q tests
python tools/run_reference.py
python tools/compare_results.py --require reference
python tools/build_report.py
```

## Style

- No em dashes and no en dashes anywhere, in code, prose or generated HTML.
  The report generator fails the build if it finds one.
- Plain declarative sentences. No slogans, no challenges to the reader, no
  claims of superiority over other software.
- Comments explain why a convention was chosen, not what a line does.

## Scope

Real personal data, university systems, scraped student records and anything
requiring credentials are out of scope permanently. See
[POSITIONING.md](POSITIONING.md).
