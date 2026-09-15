#!/usr/bin/env python3
"""Run contamination and artifact audits over a dataset and a results file."""
from __future__ import annotations

import argparse
import json
import os
import sys

from bwm.data.audit import audit_observation_overlap, audit_split_disjointness
from bwm.data.generate import load_manifest, load_split
from bwm.evaluation.audit import audit_results, format_audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="datasets/main")
    ap.add_argument("--results", default=None, help="directory containing results.json")
    ap.add_argument("--overlap-episodes", type=int, default=12)
    a = ap.parse_args(argv)

    manifest = load_manifest(a.data)
    seed_audit = audit_split_disjointness(manifest)
    print("seed disjointness:", "OK" if seed_audit["ok"] else seed_audit["collisions"])

    train = load_split(a.data, "train", a.overlap_episodes)
    overlaps = {}
    for split in manifest["splits"]:
        if split == "train":
            continue
        other = load_split(a.data, split, a.overlap_episodes)
        o = audit_observation_overlap(train, other)
        overlaps[split] = o["overlap_frac_b"]
        print(f"  duplicate-state overlap train vs {split:<26s} {o['overlap_frac_b']:.5f}")

    if a.results:
        path = os.path.join(a.results, "results.json")
        with open(path) as fh:
            results = json.load(fh)
        results["data_audit"] = {"seed_disjoint": seed_audit["ok"],
                                 "collisions": seed_audit["collisions"],
                                 "observation_overlap": overlaps}
        audit = audit_results(results)
        print()
        print(format_audit(audit))
        results["audit"] = audit
        with open(path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        with open(os.path.join(a.results, "AUDIT.md"), "w") as fh:
            fh.write(format_audit(audit))
        if audit["by_severity"].get("high"):
            return 2
    return 0 if seed_audit["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
