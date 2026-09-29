"""The scores every method is compared on.

All methods report the same dictionary, so the report can put them side by side:

* ``accuracy`` and ``macro_f1`` for every task;
* ``f1_positive`` and ``roc_auc`` for binary tasks (AUC needs a score, not just a label).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from jev_bench.tasks import Task


def score(task: Task, y_true: Any, y_pred: Any, y_score: Any | None = None) -> dict[str, float]:
    """Metrics of one method on the scored rows; ``y_score`` is P(positive) for binary tasks."""
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=task.labels)),
    }
    if task.is_binary:
        out["f1_positive"] = float(f1_score(y_true, y_pred, pos_label=1))
        if y_score is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def positive_scores(model: Any, X: Any) -> np.ndarray | None:
    """A fitted scikit-learn model's positive-class scores, for ROC AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None
