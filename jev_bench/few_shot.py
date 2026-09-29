"""Few-shot versions of a task's Jev question.

Labeled examples are drawn from the training split only (never the test rows) and attached to
the criteria of the answer they illustrate, so each outcome is defined by a description plus
concrete cases::

    criteria = {"true":  {"description": "...", "examples": [state, state, ...]},
                "false": {"description": "...", "examples": [...]}}

The instructions stay unchanged, so a few-shot run differs from the zero-shot run only by the
examples. Every test row sees the same examples, which keeps results comparable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd
from typesafe_sdk import Choice, Noul

from jev_bench import config
from jev_bench.tasks import LABEL_QUESTION, Task

# Examples longer than this (as JSON) are skipped when enough shorter ones exist, so a few long
# reviews or articles do not multiply the request size.
MAX_EXAMPLE_CHARS = 700
# Text inside an example is cut to this length. It matters when no short example exists (almost
# every BBC article is longer); the opening of an article or review carries most of its label signal.
MAX_EXAMPLE_TEXT = 600


def example_len(task: Task, row: pd.Series) -> int:
    """Size of a row's state as JSON, the measure of how much it adds to a request."""
    return len(json.dumps(task.to_state(row), ensure_ascii=False))


def is_short(task: Task, rows: pd.DataFrame) -> pd.Series:
    """Which rows are short enough to be preferred as examples."""
    return pd.Series([example_len(task, row) <= MAX_EXAMPLE_CHARS for _, row in rows.iterrows()], index=rows.index)


def sample_per_label(task: Task, train: pd.DataFrame, counts: Mapping[Any, int]) -> pd.DataFrame:
    """``counts[label]`` random training rows per label, preferring short rows when there are enough."""
    picked = []
    for label in task.labels:
        n = counts.get(label, 0)
        if n == 0:
            continue
        rows = train[train["label"] == label]
        short = rows[is_short(task, rows)]
        pool = short if len(short) >= n else rows
        picked.append(pool.sample(n=min(n, len(pool)), random_state=config.SEED))
    return pd.concat(picked)


def select_shots(task: Task, train: pd.DataFrame, k: int) -> pd.DataFrame:
    """``k`` random training examples per label."""
    return sample_per_label(task, train, {label: k for label in task.labels})


def _shorten(value: Any) -> Any:
    """Cut every string inside an example state to ``MAX_EXAMPLE_TEXT`` characters."""
    if isinstance(value, str):
        return value if len(value) <= MAX_EXAMPLE_TEXT else value[:MAX_EXAMPLE_TEXT].rstrip() + " …"
    if isinstance(value, dict):
        return {k: _shorten(v) for k, v in value.items()}
    return value


def _criterion_key(task: Task, label: Any) -> str:
    """The criteria entry that describes ``label``: Noul uses "true"/"false", Choice the label itself."""
    if task.question.type == "noul":
        return "true" if label == 1 else "false"
    return str(label)


def few_shot_questions(task: Task, shots: pd.DataFrame) -> dict[str, Any]:
    """The task's question with ``shots`` attached as examples to each outcome's criteria."""
    question = task.question
    examples: dict[str, list[Any]] = {}
    for _, row in shots.iterrows():
        examples.setdefault(_criterion_key(task, row["label"]), []).append(_shorten(task.to_state(row)))
    criteria = {
        key: {"description": description, "examples": examples.get(key, [])}
        for key, description in dict(question.criteria).items()
    }
    question_type = Noul if question.type == "noul" else Choice
    return {LABEL_QUESTION: question_type(instructions=question.instructions, criteria=criteria)}
