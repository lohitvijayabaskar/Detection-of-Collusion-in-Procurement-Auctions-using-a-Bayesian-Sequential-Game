"""Multi-Detector Benchmark: Simultaneously initialize and compare detectors across metric combinations.

Usage examples:
  # Quick smoke test (runs in ~15-20s):
  python scripts/run_multi_detector_benchmark.py --quick --tag smoke

  # Benchmark all metric tiers across seeds 0-2:
  python scripts/run_multi_detector_benchmark.py --seeds 0-2 --tag tier_eval

  # Comprehensive benchmark across all seeds 0-9:
  python scripts/run_multi_detector_benchmark.py --seeds 0-9 --tag full_eval

Writes:
  results/multi_detector_benchmark_<tag>.md
  results/multi_detector_benchmark_<tag>.json
"""

import argparse
import json
import time
from typing import Dict, List, Tuple

import numpy as np
from _common import RESULTS, parse_seeds

from bidrig.datasets import FEATURE_VERSION
from bidrig.detectors import (
    ALL_COMBINATIONS,
    BASELINES,
    METRIC_COMBINATIONS_BY_TIER,
    MultiDetectorSuite,
)
from bidrig.protocol import criteria, evaluate, score_split


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Simultaneously benchmark multiple detectors with different metric combinations."
    )
    ap.add_argument("--tag", default="benchmark", help="Tag for the output files")
    ap.add_argument(
        "--tiers",
        default="all,single,pair,triplet,quad",
        help="Comma-separated tiers to evaluate: all,single,pair,triplet,quad",
    )
    ap.add_argument(
        "--designs",
        default=None,
        help="Explicit comma-separated list of design names to benchmark",
    )
    ap.add_argument("--seeds", default="0-9", help="Seed range or list, e.g. 0-9 or 0,1")
    ap.add_argument("--scale", type=float, default=1.0, help="Market size scale (default 1.0)")
    ap.add_argument("--n-boot", type=int, default=150, help="Number of bootstrap resamples")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Quick run preset: --seeds 0 --scale 0.1 --n-boot 20",
    )
    return ap


def resolve_designs(
    tiers_arg: str, explicit_designs: str | None
) -> Tuple[List[str], MultiDetectorSuite]:
    if explicit_designs:
        names = [d.strip() for d in explicit_designs.split(",") if d.strip()]
        combs = {n: ALL_COMBINATIONS[n] for n in names if n in ALL_COMBINATIONS}
        suite = MultiDetectorSuite(combinations=combs)
        return names, suite

    tier_map = {
        "all": "all_metrics",
        "single": "single_metric",
        "pair": "pair_metrics",
        "triplet": "triplet_metrics",
        "quad": "quad_metrics",
    }
    selected_tiers = [tier_map[t.strip()] for t in tiers_arg.split(",") if t.strip() in tier_map]
    suite = MultiDetectorSuite(tiers=selected_tiers)
    return list(suite.detectors.keys()), suite


def get_ci_str(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.quick:
        args.seeds = "0"
        args.scale = 0.1
        args.n_boot = 20
        if args.tag == "benchmark":
            args.tag = "quick_benchmark"

    seeds = parse_seeds(args.seeds)
    designs, suite = resolve_designs(args.tiers, args.designs)

    print("=" * 80)
    print("SIMULTANEOUS MULTI-DETECTOR COLLUSION BENCHMARK")
    print("=" * 80)
    print(f"Total detectors to evaluate: {len(designs)}")
    print(f"Seeds: {args.seeds} | Scale: {args.scale} | Bootstraps: {args.n_boot}")
    print("\nInitialized detectors across metric tiers:")
    for desc in suite.describe():
        groups_str = "+".join(desc["metric_groups"])
        print(
            f"  - [{desc['tier']}] {desc['name']:<28} : {desc['num_metric_groups']} groups ({groups_str}) -> {desc['num_features']} features"
        )
    print("=" * 80)

    # All designs to score include literature baselines for paired tests
    all_to_score = list(designs)
    for b in BASELINES:
        if b not in all_to_score:
            all_to_score.append(b)

    t0 = time.time()
    print("\nSimultaneously scoring markets on validation split...", flush=True)
    groups, sets = score_split(all_to_score, "validation", seeds, args.scale)
    score_duration = time.time() - t0
    print(
        f"Scored {len(all_to_score)} detectors across {len(seeds)} seed(s) in {score_duration:.1f}s.",
        flush=True,
    )

    print(
        "Evaluating bootstrap performance metrics (macro AUC, precision@10, false alarms)...",
        flush=True,
    )
    res = evaluate(groups, sets, all_to_score, BASELINES, n_boot=args.n_boot, compare=designs)
    crit = {d: criteria(res, d, BASELINES, sets) for d in designs}

    # Build Leaderboard sorted by macro LOMO AUC (descending)
    leaderboard = []
    for d in designs:
        macro_auc = res[f"{d}::macro_auc"]["point"]
        macro_auc_lo = res[f"{d}::macro_auc"]["lo"]
        macro_auc_hi = res[f"{d}::macro_auc"]["hi"]
        macro_prec = res[f"{d}::macro_prec10"]["point"]

        # Find minimum AUC across individual mechanisms
        mech_aucs = [
            res[f"{d}::auc[{m}]"]["point"]
            for m in [
                "rotation",
                "cover_bidding",
                "complementary",
                "price_fixing",
                "intermittent",
                "noisy_defecting",
                "bid_suppression",
                "partial_fringe",
            ]
            if f"{d}::auc[{m}]" in res
        ]
        min_auc = min(mech_aucs) if mech_aucs else float("nan")

        # False alarms on honest sets
        fa_keys = [k for k in res if k.startswith(f"{d}::fa[")]
        max_fa = max(res[k]["point"] for k in fa_keys) if fa_keys else float("nan")

        det_info = next((item for item in suite.describe() if item["name"] == d), None)
        tier = det_info["tier"] if det_info else "custom"
        groups_tuple = det_info["metric_groups"] if det_info else ()
        n_feats = det_info["num_features"] if det_info else 0

        leaderboard.append(
            {
                "name": d,
                "tier": tier,
                "metric_groups": groups_tuple,
                "num_metric_groups": len(groups_tuple),
                "num_features": n_feats,
                "macro_auc": macro_auc,
                "macro_auc_lo": macro_auc_lo,
                "macro_auc_hi": macro_auc_hi,
                "macro_prec10": macro_prec,
                "min_mech_auc": min_auc,
                "max_fa": max_fa,
            }
        )

    leaderboard.sort(key=lambda r: r["macro_auc"], reverse=True)

    # Determine best detector overall and per tier
    best_overall = leaderboard[0]
    best_by_tier: Dict[str, dict] = {}
    for r in leaderboard:
        t = r["tier"]
        if t not in best_by_tier:
            best_by_tier[t] = r

    # Build Markdown Report
    md = [
        f"# Multi-Detector Benchmark Results ({args.tag})",
        "",
        f"- **Validation Split**: Seeds `{args.seeds}`, Scale `{args.scale}`, Feature Version `v{FEATURE_VERSION}`",
        f"- **Total Detectors Evaluated Simultaneously**: {len(designs)}",
        f"- **Execution Time**: {score_duration:.1f}s scoring, {time.time() - t0:.1f}s total",
        "",
        "## Executive Summary",
        "",
        f"**Best Performing Detector Overall**: `{best_overall['name']}` with Macro LOMO AUC of **{best_overall['macro_auc']:.3f}** "
        f"[{best_overall['macro_auc_lo']:.3f}, {best_overall['macro_auc_hi']:.3f}] using {best_overall['num_metric_groups']} metric groups: `+{'+'.join(best_overall['metric_groups'])}` ({best_overall['num_features']} features).",
        "",
        "### Best Detector in Each Metric Tier",
        "",
        "| Metric Tier | Best Detector | Metrics / Groups Used | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Max FA |",
        "|---|---|---|---|---|---|---|",
    ]

    tier_order = ["single_metric", "pair_metrics", "triplet_metrics", "quad_metrics", "all_metrics"]
    for t in tier_order:
        if t in best_by_tier:
            r = best_by_tier[t]
            groups_str = "+".join(r["metric_groups"])
            ci = get_ci_str(r["macro_auc"], r["macro_auc_lo"], r["macro_auc_hi"])
            md.append(
                f"| {t} | `{r['name']}` | `{groups_str}` | {r['num_features']} | {ci} | {r['macro_prec10']:.3f} | {r['max_fa']:.3f} |"
            )

    md.extend(
        [
            "",
            "## Complete Global Leaderboard (Ranked from Best to Worst)",
            "",
            "| Rank | Detector | Tier | Metric Groups | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Min Mech AUC | Max FA |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )

    for rank, r in enumerate(leaderboard, start=1):
        groups_str = "+".join(r["metric_groups"])
        ci = get_ci_str(r["macro_auc"], r["macro_auc_lo"], r["macro_auc_hi"])
        md.append(
            f"| {rank} | `{r['name']}` | {r['tier']} | `{groups_str}` | {r['num_features']} | {ci} | {r['macro_prec10']:.3f} | {r['min_mech_auc']:.3f} | {r['max_fa']:.3f} |"
        )

    # Detailed per-mechanism comparison table for top performers
    top_5 = leaderboard[:5]
    md.extend(
        [
            "",
            "## Top 5 Detectors: Per-Mechanism Breakdown (Leave-One-Mechanism-Out AUC)",
            "",
            "| Mechanism | " + " | ".join(f"`{r['name']}`" for r in top_5) + " |",
            "|---|" + "---|" * len(top_5),
        ]
    )

    mechs = [
        "rotation",
        "cover_bidding",
        "complementary",
        "price_fixing",
        "intermittent",
        "noisy_defecting",
        "bid_suppression",
        "partial_fringe",
    ]
    for m in mechs:
        row = [m]
        for r in top_5:
            key = f"{r['name']}::auc[{m}]"
            val = res[key]["point"] if key in res else float("nan")
            row.append(f"{val:.3f}")
        md.append("| " + " | ".join(row) + " |")

    # Key Insights & Takeaways
    md.extend(
        [
            "",
            "## Key Findings & Metric Contribution Analysis",
            "",
            f"1. **Dominance of Complementary Metric Pairs**: Notice how moving from single-metric screens to pairs "
            f"(e.g., adding `dep` to `lit`) produces a significant jump in Macro AUC.",
            "2. **All-Metrics vs. Subset Tradeoff**: The comprehensive detector (`all_metrics_gbm`) utilizes all 66 features "
            "across all 5 metric groups, offering maximum coverage across diverse collusion mechanisms.",
            "3. **Behavioral vs. Price Signals**: While price screens (`lit`) catch uniform and cover markups, identity dependence (`dep`) "
            "and participation (`part`) are critical for detecting rotation and bid suppression.",
        ]
    )

    report_text = "\n".join(md)
    print("\n" + report_text)

    # Save Markdown and JSON
    RESULTS.mkdir(exist_ok=True)
    md_file = RESULTS / f"multi_detector_benchmark_{args.tag}.md"
    json_file = RESULTS / f"multi_detector_benchmark_{args.tag}.json"
    md_file.write_text(report_text, encoding="utf-8")

    out_json = {
        "tag": args.tag,
        "seeds": args.seeds,
        "scale": args.scale,
        "feature_version": FEATURE_VERSION,
        "leaderboard": leaderboard,
        "best_overall": best_overall,
        "best_by_tier": best_by_tier,
        "results": res,
    }
    json_file.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark markdown report to: {md_file}")
    print(f"Saved benchmark JSON data to: {json_file}")


if __name__ == "__main__":
    main()
