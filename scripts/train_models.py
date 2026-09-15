#!/usr/bin/env python3
"""Fit every predictive system defined by a config."""
from __future__ import annotations

import argparse
import sys

from bwm.experiments.train import save_bundle, train_all
from bwm.utils.config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="dotted overrides, e.g. train.epochs=5")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    out = cfg.get_path("experiment.out_dir", "results/run")
    bundle = train_all(cfg)
    save_bundle(bundle, out)
    print(f"\nsaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
