#!/usr/bin/env python3
"""Generate the main trajectory dataset (all splits + held-out worlds)."""
from __future__ import annotations

import argparse
import json
import sys

from bwm.data.audit import audit_split_disjointness
from bwm.data.generate import GenSpec, generate_dataset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="datasets/main")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--n-ood", type=int, default=30)
    ap.add_argument("--episode-length", type=int, default=256)
    ap.add_argument("--n-agents", type=int, default=120)
    ap.add_argument("--epsilon", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    spec = GenSpec(n_train=a.n_train, n_val=a.n_val, n_test=a.n_test, n_ood=a.n_ood,
                   episode_length=a.episode_length, n_agents=a.n_agents,
                   epsilon=a.epsilon, out_dir=a.out, n_workers=a.workers)
    manifest = generate_dataset(spec)
    audit = audit_split_disjointness(manifest)
    print("\nseed-disjointness audit:", json.dumps(audit, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
