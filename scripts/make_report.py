#!/usr/bin/env python3
"""Regenerate tables, figures, dashboard and the verdict from a saved results.json.

Useful after changing report or analysis code without re-running the experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from bwm.evaluation.verdict import evaluate_hypothesis, format_verdict
from bwm.visualization.report import write_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)

    path = os.path.join(a.results, "results.json")
    with open(path) as fh:
        results = json.load(fh)

    verdict = evaluate_hypothesis(results)
    results["verdict"] = verdict
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    write_report(results, a.results, make_plots=not a.no_plots)
    print(format_verdict(verdict))
    print(f"wrote {a.results}/RESULTS.md, tables/, figures/, dashboard.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
