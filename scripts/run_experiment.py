#!/usr/bin/env python3
"""Train every system and run the full benchmark end to end."""
from __future__ import annotations

import argparse
import os
import sys
import time

from bwm.experiments.evaluate import run_evaluation
from bwm.experiments.train import load_bundle, save_bundle, train_all
from bwm.utils.config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--stages", default="prediction,counterfactual,control")
    ap.add_argument("--reuse-checkpoints", action="store_true",
                    help="load fitted models from out_dir instead of retraining")
    ap.add_argument("--skip-plots", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    out = cfg.get_path("experiment.out_dir", "results/run")
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()

    if a.reuse_checkpoints and os.path.exists(os.path.join(out, "train_logs.json")):
        print(f"loading fitted models from {out}", flush=True)
        bundle = load_bundle(cfg, out, verbose=True)
    else:
        bundle = train_all(cfg)
        save_bundle(bundle, out)

    stages = tuple(s.strip() for s in a.stages.split(",") if s.strip())
    results = run_evaluation(bundle, cfg, out, stages=stages)

    from bwm.visualization.report import write_report
    write_report(results, out, make_plots=not a.skip_plots)
    print(f"\ntotal {time.perf_counter()-t0:.1f}s -> {out}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
