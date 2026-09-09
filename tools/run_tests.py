#!/usr/bin/env python3
"""Minimal test runner.

pytest is the supported entry point in CI. This runner exists so the suite can
also be executed in an environment with numpy only, which is exactly the
environment the reference kernel is designed for. It collects every ``test_``
function from ``tests/`` and reports failures with tracebacks.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def discover() -> list:
    """Find every test module on disk.

    A hand maintained list is how a test file gets forgotten, so the runner
    reads the directory instead of trusting a constant.
    """
    directory = os.path.join(ROOT, "tests")
    names = sorted(
        f[:-3] for f in os.listdir(directory)
        if f.startswith("test_") and f.endswith(".py")
    )
    if not names:
        raise SystemExit("no test modules found under tests/")
    return ["tests.%s" % n for n in names]


MODULES = discover()


def main() -> int:
    passed = 0
    failures = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        names = sorted(n for n in dir(module) if n.startswith("test_"))
        for name in names:
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print("  ok   %s.%s" % (module_name.split(".")[-1], name))
            except Exception:
                failures.append((module_name, name, traceback.format_exc()))
                print("  FAIL %s.%s" % (module_name.split(".")[-1], name))
    print("")
    for module_name, name, tb in failures:
        print("=" * 70)
        print("%s.%s" % (module_name, name))
        print(tb)
    print("%d passed, %d failed" % (passed, len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
