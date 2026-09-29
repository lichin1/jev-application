"""Run the benchmark as an ordered list of stages over a list of tasks.

Stages, in the order they always run:

    baseline   train and score the classic Kaggle solution (no API key needed)
    jev        ask Jev about every scored test row, for each shot variant
    stack      stack the Kaggle score with Jev's probability, plus the Kaggle-only control
    examples   save the request each variant would send (no API calls)
    report     rebuild results/summary.md from all results files (no API calls)

Each stage merges its entries into ``results/<task>.json``, so stages can run separately and in
any combination; ``report`` always reads everything that exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from jev_bench import config, experiments, report
from jev_bench.datasets import split
from jev_bench.experiments import ShotPlanner
from jev_bench.methods import STACK_VARIANTS, is_valid_variant, jev_method, stack_method
from jev_bench.tasks import TASKS

STAGES = ("baseline", "jev", "stack", "examples", "report")
SCORING_STAGES = {"baseline", "jev", "stack"}
# Everything the published results use.
FULL_EXPERIMENT_SHOTS = ["0", "3", "cluster", "cluster-random"]


@dataclass
class PipelineConfig:
    stages: list[str]
    tasks: list[str] = field(default_factory=lambda: list(TASKS))
    shots: list[str] = field(default_factory=lambda: ["0"])  # Jev variants for the "jev" and "examples" stages
    stack_shots: list[str] = field(default_factory=lambda: list(STACK_VARIANTS))  # Jev features for "stack"
    scored_n: int | None = None  # None scores the whole test split
    model: str = config.DEFAULT_MODEL
    concurrency: int = config.DEFAULT_CONCURRENCY

    def validate(self) -> None:
        for name, allowed, values in (("stage", STAGES, self.stages), ("task", tuple(TASKS), self.tasks),
                                      ("stack variant", STACK_VARIANTS, self.stack_shots)):
            unknown = [v for v in values if v not in allowed]
            if unknown:
                raise ValueError(f"unknown {name}: {', '.join(unknown)} (choose from {', '.join(allowed)})")
        bad = [v for v in self.shots if not is_valid_variant(v)]
        if bad:
            raise ValueError(f"unknown shot variant: {', '.join(bad)} (an integer, cluster or cluster-random)")


def _load_task_results(name: str) -> dict[str, Any]:
    path = config.RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _save_task_results(name: str, result: dict[str, Any]) -> None:
    (config.RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


def _brief(entry: dict[str, Any]) -> str:
    metrics = entry.get("metrics", {})
    return ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "n") or "no scores"


def run(cfg: PipelineConfig) -> None:
    cfg.validate()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    task_stages = [s for s in STAGES if s in cfg.stages and s != "report"]

    for name in cfg.tasks if task_stages else []:
        task, data = TASKS[name], split(name, cfg.scored_n)
        print(f"\n=== {name}: train={len(data.train)} test={len(data.test)} scored={len(data.scored)}")
        result = _load_task_results(name)
        result["kaggle"] = task.kaggle
        result["pipeline"] = experiments.describe_task(task, data)
        planner = ShotPlanner(task, data)

        if "baseline" in task_stages:
            result["baseline"] = experiments.run_baseline(task, data)
            print(f"[baseline] {_brief(result['baseline'])}")
        if "jev" in task_stages:
            for variant in cfg.shots:
                method = jev_method(variant)
                result[method.key] = experiments.run_jev(task, data, planner, variant, cfg.model, cfg.concurrency)
                print(f"[{method.label}] {_brief(result[method.key])}, errors={result[method.key]['errors']}")
        if "stack" in task_stages:
            for variant in cfg.stack_shots:
                method = stack_method(variant)
                result[method.key] = experiments.run_stack(task, data, planner, variant, cfg.model, cfg.concurrency)
                print(f"[{method.label}] {_brief(result[method.key])}")
        if "examples" in task_stages:
            experiments.write_request_examples(task, data, planner, cfg.shots, cfg.model)

        _save_task_results(name, result)

    # Any stage that adds scores refreshes the report; "examples" alone does not change any score.
    if "report" in cfg.stages or set(task_stages) & SCORING_STAGES:
        report.write_summary()
