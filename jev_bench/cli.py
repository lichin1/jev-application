"""Command line: ``jev-bench <command>`` (or ``python -m jev_bench <command>``).

Commands:
    all        the full experiment: baseline, every Jev variant, stacking, report
    baseline   train and score the Kaggle solutions (no API key needed)
    jev        ask Jev, for the shot variants given with --shots
    stack      stack the Kaggle model with Jev, for the variants given with --stack-shots
    examples   save example Jev requests to results/ (no API calls)
    report     rebuild results/summary.md from the results files (no API calls)

Jev commands need the TYPESAFE_API_KEY environment variable. Answers are cached in .cache/jev/,
so re-running any command only calls the API for rows it has not answered yet.
"""

from __future__ import annotations

import argparse
import os
import sys

from typesafe_sdk.constants import API_KEY_ENV

from jev_bench import config
from jev_bench.methods import STACK_VARIANTS
from jev_bench.pipeline import FULL_EXPERIMENT_SHOTS, PipelineConfig, run
from jev_bench.tasks import TASKS

# Stages that call the Jev API.
API_STAGES = {"jev", "stack"}

# Which pipeline stages each command runs.
COMMAND_STAGES = {
    "all": ["baseline", "jev", "stack", "report"],
    "baseline": ["baseline", "report"],
    "jev": ["jev", "report"],
    "stack": ["stack", "report"],
    "examples": ["examples"],
    "report": ["report"],
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMAND_STAGES))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS), help="default: all five")
    parser.add_argument(
        "--shots", nargs="+", default=None,
        help="Jev shot variants: integers (examples per label), cluster, cluster-random "
             f"(default: 0; for 'all': {' '.join(FULL_EXPERIMENT_SHOTS)})",
    )
    parser.add_argument(
        "--stack-shots", nargs="+", default=list(STACK_VARIANTS), choices=list(STACK_VARIANTS),
        help="Jev variants used as the stacking feature (default: 0 cluster)",
    )
    parser.add_argument(
        "--scored-n", type=int, default=0,
        help="score every method on a stratified sample of this many test rows, to save Jev calls (default 0: all)",
    )
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help=f"Jev model (default {config.DEFAULT_MODEL})")
    parser.add_argument("--concurrency", type=int, default=config.DEFAULT_CONCURRENCY, help="parallel Jev requests")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    shots = args.shots or (FULL_EXPERIMENT_SHOTS if args.command == "all" else ["0"])
    cfg = PipelineConfig(
        stages=COMMAND_STAGES[args.command],
        tasks=args.tasks,
        shots=shots,
        stack_shots=args.stack_shots,
        scored_n=args.scored_n or None,
        model=args.model,
        concurrency=args.concurrency,
    )
    if API_STAGES & set(cfg.stages) and not os.environ.get(API_KEY_ENV, "").strip():
        sys.exit(f"jev-bench: '{args.command}' calls the Jev API; set {API_KEY_ENV} first (see .env.example).")
    try:
        run(cfg)
    except ValueError as err:  # invalid options, reported without a traceback
        sys.exit(f"jev-bench: {err}")


if __name__ == "__main__":
    main()
