"""python -m linkage.generate --seed 7 --out data/dev/

Writes truth.parquet, feeds/{state}.csv, offenders.parquet and manifest.json,
then runs the six validators. Exit 1 if the config is not ready or any
validator fails. Runs locally only.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from linkage import config as config_mod
from linkage import schema
from linkage.generate import corrupt, render, report, sample, validate
from linkage.generate import rng as streams


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.generate", description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corruption-seed", type=int, default=None, help="defaults to --seed")
    ap.add_argument("--n-cases", type=int, default=None, help="override corpus.yaml n_cases (recorded in manifest)")
    ap.add_argument("--cross-type-sharing", type=float, default=None,
                    help="0-1, required: how much mo_core habit is person-level (swept; no default)")
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    started = time.time()

    cfg = config_mod.load(args.config_dir)
    readiness = config_mod.check(cfg)
    if readiness.errors or readiness.unfilled:
        print("config not ready — run `python -m linkage.config -v`")
        return 1
    overrides = {}
    if args.n_cases is not None:
        cfg["corpus"]["n_cases"] = overrides["n_cases"] = args.n_cases
    if args.cross_type_sharing is not None:
        cfg["corpus"]["cross_type_sharing"] = overrides["cross_type_sharing"] = args.cross_type_sharing
    if cfg["corpus"]["cross_type_sharing"] is None:
        print("cross_type_sharing has no default: pass --cross-type-sharing (swept 0 to 1)")
        return 1
    if not 0 <= cfg["corpus"]["cross_type_sharing"] <= 1:
        print("--cross-type-sharing must be between 0 and 1")
        return 1
    cseed = args.seed if args.corruption_seed is None else args.corruption_seed

    print(f"sampling truth ({cfg['corpus']['n_cases']} cases, seed {args.seed})")
    truth, offenders = sample.generate(cfg, args.seed)
    model = sample.build_model(cfg)
    print("recording (corruption seed %d)" % cseed)
    cases = truth.merge(corrupt.record(truth, cfg, cseed, model), on="case_uid")
    print("rendering feeds")
    feeds, positions = render.feeds(cases, cfg, cfg["vocab"], args.seed)

    out = args.out
    (out / "feeds").mkdir(parents=True, exist_ok=True)
    cases.sort_values("case_id").to_parquet(out / "truth.parquet", index=False)
    offenders.to_parquet(out / "offenders.parquet", index=False)
    for code, frame in feeds.items():
        frame.to_csv(out / "feeds" / f"{code}.csv", index=False, encoding="utf-8")

    print("validating")
    vrng = streams.stream(args.seed, 99)
    results = {"1_background_marginals": validate.background_marginals(truth, cfg)}
    results["2_truth_mi"] = validate.truth_mi(truth, vrng)
    results["3_recorded_mi"] = validate.recorded_mi(cases, results["2_truth_mi"], vrng)
    results["4_cheat_detector"] = validate.cheat_detector(cases, positions, vrng)
    results["5_series_tail"] = validate.series_tail(offenders, cfg)
    results["6_recording_rates"] = validate.recording_rates(cases, cfg)

    marginals, drift = report.realised_marginals(truth, cfg)
    rr = cfg["corpus"]["repeat_rate"]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": True,
        "command": "python -m linkage.generate " + " ".join(argv if argv is not None else sys.argv[1:]),
        "seed": args.seed, "corruption_seed": cseed, "overrides": overrides,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "derived": {"alpha": config_mod.alpha(rr["value"], rr["reference_frequency"]),
                    "repeat_rate": rr["value"], "tau": cfg["corpus"]["tau"],
                    "cross_type_sharing": cfg["corpus"]["cross_type_sharing"]},
        "counts": {
            "cases": int(len(cases)), "serial_cases": int(cases["is_serial"].sum()),
            "serial_offenders": int(len(offenders)),
            "relocated_offenders": int(offenders["destination_state"].notna().sum()),
            "feed_rows": {code: int(len(f)) for code, f in feeds.items()},
        },
        "pairs": report.pair_stats(cases),
        "core_habit_transfer": report.core_agreement(cases),
        "marginal_drift_largest": drift,
        "realised_marginals": marginals,
        "validators": results,
        "limitations": report.LIMITATIONS,
        "config": cfg,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str, ensure_ascii=False),
                                      encoding="utf-8")

    print(f"\nwrote {out}  ({time.time() - started:.1f}s)")
    for name, res in results.items():
        print(f"  {'PASS' if res.get('passed') else 'FAIL'}  {name}")
    return 0 if all(r.get("passed") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
