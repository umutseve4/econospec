"""Print one line of evidence about a single adf-001 reference run.

The golden manifest comparison in tests/test_conformance.py is bit for bit and
has been red on params.const.estimate in every sample taken so far. Pinning the
linear algebra backend to one thread did not stop it, which falsified the first
hypothesis. So this probe stops arguing and records, for one process:

* whether the thread pinning was in force,
* which cpu the process ran on, because the failures cluster by job,
* a digest of the bytes that went in, so "the inputs were identical" is a
  measurement rather than an assumption,
* the 64 byte alignment of every array the factorisation touches,
* a digest of each intermediate, in the order they are computed, so the first
  stage that disagrees between two processes is named rather than inferred,
* the final value under test, printed with repr so no digit is lost.

Run it a few hundred times and group the lines. The first field whose digest
takes more than one value is where the divergence enters.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

# Python puts the directory of the script on sys.path, which is tools/, so the
# package next to it is invisible unless the repository root is added by hand.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import econospec  # noqa: E402  imported first: it pins the backend before numpy loads

import numpy as np  # noqa: E402

from econospec.csvio import read_frame  # noqa: E402
from econospec.estimators import build_adf_design  # noqa: E402
from econospec.specs import load_all_specs  # noqa: E402

SPEC_ID = os.environ.get("PROBE_SPEC", "adf-001")


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def alignment(array: np.ndarray) -> int:
    return int(array.__array_interface__["data"][0] % 64)


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def main(root: str = ".") -> int:
    specs = [s for s in load_all_specs(root) if s["id"] == SPEC_ID]
    if not specs:
        raise SystemExit("spec %r not found" % SPEC_ID)
    spec = specs[0]
    model = spec["model"]

    frame = read_frame(os.path.join(root, spec["fixture"]["path"]))
    series = np.asarray(frame[model["series"]], dtype=float)
    design = build_adf_design(series, lags=model["lags"], trend=model["trend"])
    X = design["X"]
    lhs = design["y"]

    # The same call sequence as econospec.estimators._solve_ols, opened up so
    # each intermediate can be fingerprinted separately.
    q, r = np.linalg.qr(X)
    qty = q.T @ lhs
    beta = np.linalg.solve(r, qty)
    resid = lhs - X @ beta
    ss_resid = float(resid @ resid)
    r_inv = np.linalg.solve(r, np.eye(r.shape[0]))
    xtx_inv = r_inv @ r_inv.T

    report = econospec.thread_pinning_report()
    record = {
        "spec": SPEC_ID,
        "pinned": bool(report["single_threaded"]),
        "threads": report["variables"]["OPENBLAS_NUM_THREADS"],
        "hashseed": os.environ.get("PYTHONHASHSEED"),
        "numpy": np.__version__,
        "cpu": cpu_model(),
        "align": {
            "series": alignment(series),
            "X": alignment(X),
            "lhs": alignment(lhs),
            "q": alignment(q),
            "r": alignment(r),
        },
        "stage": {
            "1_series": digest(series),
            "2_X": digest(X),
            "3_lhs": digest(lhs),
            "4_q": digest(q),
            "5_r": digest(r),
            "6_qty": digest(qty),
            "7_beta": digest(beta),
            "8_resid": digest(resid),
            "9_xtx_inv": digest(xtx_inv),
        },
        "value": {
            "const_estimate": repr(float(beta[0])),
            "ss_resid": repr(ss_resid),
        },
    }
    sys.stdout.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
