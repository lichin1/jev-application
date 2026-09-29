"""Stack Jev's probabilities with the Kaggle model's scores.

Jev's answer becomes a feature next to the Kaggle model's own score, in a second-stage model::

    stage 1  Kaggle model  -> out-of-fold scores on train (5-fold), scores on the scored test rows
    Jev      same question -> probabilities on a stratified train sample and on the scored test rows
    stage 2  logistic regression on [Kaggle score, Jev probability], fit on the train sample

Out-of-fold scores keep stage 2 honest: each training row's Kaggle score comes from a model that
did not see that row, just as the test scores come from a model that did not see the test rows.
Jev is only asked about a sample of the training rows (``STAGE2_TRAIN_N``): stage 2 has a handful
of inputs and does not need 40,000 IMDB rows. Rows used as few-shot examples are left out of that
sample, so Jev never scores a row it was shown together with its label.

The control fits the same stage 2 on the Kaggle score alone. Re-fitting moves the decision
threshold by itself, so Jev's contribution is the difference between the stacked model and the
control, not between the stacked model and the original Kaggle model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from jev_bench import config
from jev_bench.datasets import Split
from jev_bench.few_shot import few_shot_questions
from jev_bench.jev_runner import Record, ask_jev
from jev_bench.tasks import Task

STAGE2_TRAIN_N = 2000
CV_FOLDS = 5
# Probabilities are clipped away from 0 and 1 before taking logits.
EPS = 1e-4


@dataclass
class Prediction:
    labels: np.ndarray
    positive_scores: np.ndarray | None  # P(positive) for binary tasks, else None


@dataclass
class StackingResult:
    stacked: Prediction  # stage 2 on [Kaggle score, Jev probability]
    control: Prediction  # stage 2 on the Kaggle score alone
    info: dict[str, Any]  # sample size, missing Jev answers, stage-2 weights


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _score_method(task: Task) -> str:
    """How the Kaggle model exposes scores: probabilities, or decision values (LinearSVC)."""
    return "predict_proba" if hasattr(task.make_baseline(), "predict_proba") else "decision_function"


def _as_features(task: Task, raw: np.ndarray, method: str) -> np.ndarray:
    """Stage-1 output as a 2-D feature matrix: one column for binary tasks, one per label otherwise."""
    if method == "predict_proba":
        return _logit(raw[:, 1:2]) if task.is_binary else _logit(raw)
    return raw.reshape(-1, 1) if raw.ndim == 1 else raw


def kaggle_features(task: Task, data: Split) -> tuple[np.ndarray, np.ndarray]:
    """(out-of-fold scores for every training row, scores for the scored test rows)."""
    method = _score_method(task)
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.SEED)
    oof = cross_val_predict(
        task.make_baseline(), task.baseline_features(data.train), data.train["label"], cv=folds, method=method
    )
    model = task.make_baseline().fit(task.baseline_features(data.train), data.train["label"])
    test = getattr(model, method)(task.baseline_features(data.scored))
    return _as_features(task, oof, method), _as_features(task, test, method)


def jev_features(task: Task, records: list[Record]) -> tuple[np.ndarray, int]:
    """Jev probabilities as logits (one column for Noul, one per label for Choice), and how many failed.

    A failed answer gets the uninformative prior, so one missing call does not drop the row.
    """
    rows, missing = [], 0
    for record in records:
        if "error" in record:
            missing += 1
            rows.append([0.5] if task.is_binary else [1 / len(task.labels)] * len(task.labels))
        elif task.is_binary:
            rows.append([record["answer"]["noul"]])
        else:
            rows.append([record["answer"]["probabilities"].get(str(label), 0.0) for label in task.labels])
    return _logit(np.array(rows, dtype=float)), missing


def stage2_sample(train: pd.DataFrame, excluded: pd.Index) -> pd.Index:
    """Training rows used to fit stage 2: a stratified sample, without the few-shot example rows."""
    eligible = train.index.difference(excluded)
    if len(eligible) <= STAGE2_TRAIN_N:
        return eligible
    sample, _ = train_test_split(
        eligible, train_size=STAGE2_TRAIN_N, random_state=config.SEED, stratify=train.loc[eligible, "label"]
    )
    return sample


def _fit_stage2(task: Task, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray) -> tuple[Pipeline, Prediction]:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1] if task.is_binary else None
    return model, Prediction(model.predict(X_test), scores)


def run_stacking(task: Task, data: Split, model: str, concurrency: int, shot_rows: pd.DataFrame | None) -> StackingResult:
    """Stack the Kaggle model with Jev (zero-shot, or with ``shot_rows`` as examples) and its control."""
    questions = task.questions if shot_rows is None else few_shot_questions(task, shot_rows)
    excluded = pd.Index([]) if shot_rows is None else shot_rows.index

    oof, kaggle_test = kaggle_features(task, data)
    sample_index = stage2_sample(data.train, excluded)
    sample = data.train.loc[sample_index]
    kaggle_train = oof[data.train.index.get_indexer(sample_index)]

    jev_train, missing_train = jev_features(task, ask_jev(task, sample, model, concurrency, questions))
    jev_test, missing_test = jev_features(task, ask_jev(task, data.scored, model, concurrency, questions))

    stacked_model, stacked = _fit_stage2(
        task, np.hstack([kaggle_train, jev_train]), sample["label"], np.hstack([kaggle_test, jev_test])
    )
    _, control = _fit_stage2(task, kaggle_train, sample["label"], kaggle_test)

    info: dict[str, Any] = {"stage2_train_rows": len(sample), "jev_missing": missing_train + missing_test}
    if task.is_binary:
        # Coefficients on standardised inputs: how much stage 2 leans on each source.
        coef = stacked_model[-1].coef_[0]
        info["stage2_weight"] = {"kaggle": round(float(coef[0]), 3), "jev": round(float(coef[1]), 3)}
    return StackingResult(stacked=stacked, control=control, info=info)
