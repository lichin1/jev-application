"""One function per pipeline stage for one task; each returns the entry stored in the results file.

Every entry keeps its scores under ``metrics``, computed on the same scored test rows, so the
report can compare any two methods directly.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from jev_bench import config
from jev_bench.cluster_shots import select_cluster_shots, select_matched_random
from jev_bench.datasets import Split
from jev_bench.few_shot import few_shot_questions, select_shots
from jev_bench.jev_runner import Record, ask_jev, request_payload
from jev_bench.methods import jev_method
from jev_bench.metrics import positive_scores, score
from jev_bench.stacking import run_stacking
from jev_bench.tasks import Task

# Confidence above which a Choice answer counts as "confident" in the run details.
CONFIDENT = 0.8


class ShotPlanner:
    """The few-shot examples of every variant for one task.

    The cluster selection is computed once and reused, since "cluster-random" matches its counts
    and stacking with "cluster" needs the same examples.
    """

    def __init__(self, task: Task, data: Split) -> None:
        self.task, self.data = task, data
        self._cluster: tuple[pd.DataFrame, dict[str, Any]] | None = None

    def _cluster_selection(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        if self._cluster is None:
            self._cluster = select_cluster_shots(self.task, self.data.train)
        return self._cluster

    def examples(self, variant: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        """(example rows, or None for zero-shot; extra details to store with the results)."""
        if variant == "cluster":
            rows, clustering = self._cluster_selection()
            return rows, {"clustering": clustering}
        if variant == "cluster-random":
            return select_matched_random(self.task, self.data.train, self._cluster_selection()[0]), {}
        k = int(variant)
        return (select_shots(self.task, self.data.train, k) if k else None), {}

    def questions(self, variant: str) -> dict[str, Any]:
        rows, _ = self.examples(variant)
        return self.task.questions if rows is None else few_shot_questions(self.task, rows)


def describe_task(task: Task, data: Split) -> dict[str, Any]:
    """What the report's summary table shows: data sizes, the Jev question, and one example state."""
    question = task.question
    answers = "yes / no" if question.type == "noul" else " / ".join(question.criteria)
    return {
        "problem": task.problem,
        "n_scored": len(data.scored),
        "n_full_test": len(data.test),
        "n_train": len(data.train),
        "question_type": question.type.capitalize(),
        "question": question.instructions,
        "answers": answers,
        "state_example": task.to_state(data.scored.iloc[0]),
        "state_example_label": str(data.scored["label"].iloc[0]),
    }


def run_baseline(task: Task, data: Split) -> dict[str, Any]:
    """Train the Kaggle solution on the train split; score it on the scored rows and the whole test split."""
    model = task.make_baseline()
    start = time.perf_counter()
    model.fit(task.baseline_features(data.train), data.train["label"])
    entry: dict[str, Any] = {"model": task.baseline_desc, "fit_seconds": round(time.perf_counter() - start, 2)}
    for field, rows in (("metrics", data.scored), ("full_test_metrics", data.test)):
        X = task.baseline_features(rows)
        scores = positive_scores(model, X) if task.is_binary else None
        entry[field] = score(task, rows["label"], model.predict(X), scores)
    return entry


def _usage_details(records: list[Record]) -> dict[str, Any]:
    """Latency percentiles and total input tokens of successful answers."""
    details: dict[str, Any] = {}
    latencies = [r["latency_s"] for r in records if "latency_s" in r]
    if latencies:
        details["latency_p50_s"] = round(float(np.median(latencies)), 3)
        details["latency_p95_s"] = round(float(np.percentile(latencies, 95)), 3)
    tokens = [r.get("usage", {}).get("input_tokens") for r in records]
    tokens = [t for t in tokens if t is not None]
    if tokens:
        details["input_tokens_total"] = int(sum(tokens))
    return details


def _confidence_details(records: list[Record], correct: np.ndarray) -> dict[str, Any]:
    """For Choice answers: how many are confident, and how accurate those are."""
    confidence = np.array([r["answer"]["confidence"] for r in records])
    confident = confidence >= CONFIDENT
    return {
        f"accuracy_when_confidence_ge_{CONFIDENT}": float(correct[confident].mean()) if confident.any() else None,
        f"share_confidence_ge_{CONFIDENT}": float(confident.mean()),
    }


def run_jev(task: Task, data: Split, planner: ShotPlanner, variant: str, model: str, concurrency: int) -> dict[str, Any]:
    """Ask Jev (with the variant's examples) about every scored row and score its answers."""
    shot_rows, extra = planner.examples(variant)
    records = ask_jev(task, data.scored, model, concurrency, planner.questions(variant))
    answered = [(i, r) for i, r in enumerate(records) if "error" not in r]
    errors = [r["error"] for r in records if "error" in r]

    entry: dict[str, Any] = {"model": model, "shots": variant, "errors": len(errors)}
    if shot_rows is not None:
        entry["examples_per_label"] = {str(k): int(v) for k, v in shot_rows["label"].value_counts().sort_index().items()}
        # Training-row indices, so the exact examples can be reproduced and checked against the test set.
        entry["shot_train_rows"] = [int(i) for i in shot_rows.index]
    entry.update(extra)
    if errors:
        entry["error_examples"] = sorted(set(errors))[:3]
    if not answered:
        return entry

    ok_records = [r for _, r in answered]
    decoded = [task.decode(r["answer"]) for r in ok_records]
    y_true = data.scored["label"].iloc[[i for i, _ in answered]]
    y_pred = [label for label, _ in decoded]
    y_score = [p for _, p in decoded] if task.is_binary else None
    entry["metrics"] = score(task, y_true, y_pred, y_score)
    entry.update(_usage_details(ok_records))
    if "confidence" in ok_records[0]["answer"]:
        entry.update(_confidence_details(ok_records, np.array([p == t for p, t in zip(y_pred, y_true)])))
    return entry


def run_stack(task: Task, data: Split, planner: ShotPlanner, variant: str, model: str, concurrency: int) -> dict[str, Any]:
    """Stack the Kaggle model with the variant's Jev probabilities; also score the Kaggle-only control."""
    shot_rows, _ = planner.examples(variant)
    result = run_stacking(task, data, model, concurrency, shot_rows)
    y = data.scored["label"]
    return {
        **result.info,
        "control_metrics": score(task, y, result.control.labels, result.control.positive_scores),
        "metrics": score(task, y, result.stacked.labels, result.stacked.positive_scores),
    }


def write_request_examples(task: Task, data: Split, planner: ShotPlanner, variants: list[str], model: str) -> None:
    """Save the request each variant would send for the first scored row (no API calls)."""
    for variant in variants:
        payload = request_payload(task, data.scored.iloc[0], model, planner.questions(variant))
        suffix = jev_method(variant).key[len("jev"):]  # "", "_3shot", "_cluster", ...
        path = config.RESULTS_DIR / f"{task.name}_request_example{suffix}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[examples] wrote {path.name} ({len(json.dumps(payload))} chars)")
