"""Offline tests: no API key needed. Jev is replaced by a fake that answers from the true label.

They use the small Iris and Titanic datasets (downloaded to data/raw on first run).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from jev_bench import config, experiments, stacking
from jev_bench.cluster_shots import select_cluster_shots, select_matched_random
from jev_bench.datasets import split
from jev_bench.few_shot import few_shot_questions, select_shots
from jev_bench.jev_runner import request_key, request_payload
from jev_bench.pipeline import PipelineConfig, run
from jev_bench.tasks import TASKS


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch):
    """Write results and cache into a temporary directory instead of the repository."""
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return tmp_path


def _oracle_answer(task, label: Any) -> dict[str, Any]:
    """A Jev answer that is always right, in the same shape as the real API's."""
    if task.is_binary:
        return {"type": "noul", "noul": 0.9 if label == 1 else 0.1}
    probabilities = {str(l): float(l == label) for l in task.labels}
    return {"type": "choice", "choice": str(label), "confidence": 1.0, "probabilities": probabilities}


def fake_ask_jev(task, rows: pd.DataFrame, model, concurrency, questions=None) -> list[dict[str, Any]]:
    return [{"key": str(i), "answer": _oracle_answer(task, label), "latency_s": 0.0, "usage": {"input_tokens": 1}}
            for i, label in enumerate(rows["label"])]


def test_split_is_stratified_and_complete():
    data = split("iris")
    assert len(data.train) + len(data.test) == 150
    assert data.scored.equals(data.test)  # no sampling by default
    assert data.test["label"].value_counts().nunique() == 1  # 10 of each species


def test_scored_subset_is_a_stratified_sample():
    data = split("titanic", scored_n=50)
    assert len(data.scored) == 50
    assert abs(data.scored["label"].mean() - data.test["label"].mean()) < 0.05


def test_few_shot_examples_come_from_train_and_sit_in_criteria():
    task, data = TASKS["iris"], split("iris")
    shots = select_shots(task, data.train, 3)
    assert shots.index.isin(data.train.index).all()
    assert shots["label"].value_counts().to_dict() == {label: 3 for label in task.labels}
    criteria = few_shot_questions(task, shots)["label"].criteria
    assert all(len(criteria[label]["examples"]) == 3 for label in task.labels)


def test_cluster_shots_cover_every_label_and_control_matches_counts():
    task, data = TASKS["iris"], split("iris")
    shots, info = select_cluster_shots(task, data.train)
    assert set(shots["label"]) == set(task.labels)
    assert info["k"] >= len(task.labels)
    control = select_matched_random(task, data.train, shots)
    assert control["label"].value_counts().to_dict() == shots["label"].value_counts().to_dict()


def test_request_key_depends_on_examples():
    task, data = TASKS["iris"], split("iris")
    row = data.scored.iloc[0]
    zero_shot = request_key(request_payload(task, row, "jev-latest"))
    few_shot = request_key(request_payload(task, row, "jev-latest", few_shot_questions(task, select_shots(task, data.train, 1))))
    assert zero_shot != few_shot


def test_full_pipeline_with_fake_jev(isolated_outputs, monkeypatch):
    monkeypatch.setattr(experiments, "ask_jev", fake_ask_jev)
    monkeypatch.setattr(stacking, "ask_jev", fake_ask_jev)

    run(PipelineConfig(stages=["baseline", "jev", "stack", "report"], tasks=["iris", "titanic"],
                       shots=["0", "cluster", "cluster-random"]))

    summary = (config.RESULTS_DIR / "summary.md").read_text()
    for heading in ("## 1. Summary", "## 2. Accuracy", "## 3. Macro-F1", "## 4. ROC AUC"):
        assert heading in summary
    for column in ("Jev 0-shot", "Jev cluster-shot", "Stage 2 · Kaggle only", "Kaggle + Jev 0-shot"):
        assert column in summary

    iris = json.loads((config.RESULTS_DIR / "iris.json").read_text())
    assert iris["jev"]["metrics"]["accuracy"] == 1.0  # the fake Jev is always right
    assert iris["stack"]["metrics"]["accuracy"] >= iris["stack"]["control_metrics"]["accuracy"]
