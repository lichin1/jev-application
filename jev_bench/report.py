"""Build ``results/summary.md`` from every task's results file.

The report has one overview table and one table per metric (accuracy, macro-F1, ROC AUC). Each
metric table has one column per method, and marks the best score in each row in bold and underlined.
It is plain Markdown (no HTML), so it renders the same in every Markdown viewer.
"""

from __future__ import annotations

import json
from typing import Any

from jev_bench import config
from jev_bench.methods import KAGGLE, Method, report_methods
from jev_bench.tasks import TASKS

Results = dict[str, dict[str, Any]]  # task name -> contents of results/<task>.json

# Run-detail fields that are too long for one line of the report (they stay in the JSON files).
_LONG_FIELDS = {"metrics", "full_test_metrics", "control_metrics", "shot_train_rows", "clustering"}

_INTRO = [
    "Kaggle is the classic Kaggle solution trained on the 80% train split. Jev 0-shot never sees training labels;",
    "Jev k-shot adds k random labeled training examples per class to each answer's criteria; Jev cluster-shot takes",
    "one example per (label, k-means cluster) cell; Jev matched random is its control with the same per-label counts.",
    "Kaggle + Jev columns stack the Kaggle model's out-of-fold score with Jev's probability in a logistic regression;",
    "Stage 2 · Kaggle only is the same logistic regression without Jev, the control to compare them with.",
    "Examples always come from the train split. Bold and underline mark the best score in each row.",
]


def load_results() -> Results:
    """Every task that has a results file, in report order."""
    paths = {name: config.RESULTS_DIR / f"{name}.json" for name in TASKS}
    return {name: json.loads(path.read_text()) for name, path in paths.items() if path.exists()}


# ---------------------------------------------------------------- Markdown helpers


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _cell(text: str) -> str:
    """Text made safe for one table cell: no line breaks, escaped pipes."""
    return " ".join(text.split()).replace("|", "\\|")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _underline(text: str) -> str:
    """Underline with a combining low line (U+0332) after each character.

    Markdown has no underline syntax and some viewers do not render HTML such as <ins>, so the
    underline is carried by the text itself.
    """
    return "".join(ch + "̲" for ch in text)


def _highlight(value: float) -> str:
    return f"**{_underline(f'{value:.3f}')}**"


# ---------------------------------------------------------------- tables


def _uses_subset(results: Results) -> bool:
    """Whether any task was scored on a sample of its test split rather than all of it."""
    return any(r["pipeline"]["n_scored"] < r["pipeline"]["n_full_test"] for r in results.values() if "pipeline" in r)


def _n_label(results: Results) -> str:
    return "Subset n" if _uses_subset(results) else "Full test n"


def overview_table(results: Results) -> list[str]:
    """Per task: dataset, sizes, problem, the Jev question, and the first state sent to Jev."""
    header = ["Dataset", _n_label(results), "Train", "Problem", "Jev question", "State (first row)", "True label"]
    rows = [_row(header), "| --- | ---: | ---: | --- | --- | --- | --- |"]
    for name, result in results.items():
        p = result.get("pipeline")
        if p is None:
            continue
        # Backticks would end the code span early, so they are dropped from the example state.
        state = _truncate(json.dumps(p["state_example"], ensure_ascii=False).replace("`", ""), 110)
        question = f"**{p['question_type']}**: {p['question']} ({p['answers']})"
        rows.append(
            _row([f"[{name}]({result['kaggle']})", str(p["n_scored"]), str(p["n_train"]), p["problem"], question,
                  f"`{state}`", p["state_example_label"]])
        )
    return rows


def metric_table(results: Results, metric: str, methods: list[Method]) -> list[str]:
    """One metric for every method; tasks without that metric (ROC AUC on multiclass) are skipped."""
    # When every task is scored on its whole test split, a full-test column would repeat "Kaggle".
    show_full_test = _uses_subset(results)
    header = ["Dataset", _n_label(results), KAGGLE.label, *(m.label for m in methods)]
    header += ["Kaggle full test"] if show_full_test else []
    rows = [_row(header), "| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |"]

    for name, result in results.items():
        kaggle = KAGGLE.scores(result).get(metric)
        if "pipeline" not in result or kaggle is None:
            continue
        values = [kaggle, *(m.scores(result).get(metric) for m in methods)]
        # Compared at the printed precision, so scores that print the same are marked the same.
        best = max(round(v, 3) for v in values if v is not None)
        cells = ["—" if v is None else _highlight(v) if round(v, 3) == best else f"{v:.3f}" for v in values]
        if show_full_test:
            full = result["baseline"].get("full_test_metrics", {}).get(metric)
            cells.append("—" if full is None else f"{full:.3f}")
        rows.append(_row([name, str(result["pipeline"]["n_scored"]), *cells]))
    return rows


def run_details(results: Results, methods: list[Method]) -> list[str]:
    """One line per task and method with its cost and settings (latency, tokens, examples, stage-2 weights)."""
    lines = []
    for name, result in results.items():
        for method in methods:
            if method.field != "metrics" or method.key not in result:
                continue  # the stacking control shares its entry with the stacked model
            details = {k: v for k, v in result[method.key].items() if k not in _LONG_FIELDS}
            lines.append(f"- **{name}** {method.label}: `{json.dumps(details, ensure_ascii=False)}`")
    return lines


# ---------------------------------------------------------------- the report


def write_summary() -> None:
    results = load_results()
    methods = report_methods(results)
    if _uses_subset(results):
        scope = ("`Subset n` is the stratified test subset every method is scored on; \"Kaggle full test\" scores "
                 "the same Kaggle model on the whole 20% test split, to show whether the subset is representative.")
    else:
        scope = "Every method is scored on the whole 20% test split (`Full test n` rows), never seen in training."

    sections = [
        ["# Kaggle × Jev benchmark results", "", scope, *_INTRO],
        ["## 1. Summary", "", *overview_table(results)],
        ["## 2. Accuracy", "", *metric_table(results, "accuracy", methods)],
        ["## 3. Macro-F1", "", *metric_table(results, "macro_f1", methods)],
        [
            "## 4. ROC AUC (binary tasks)", "",
            "Uses the Kaggle model's positive-class score and Jev's `noul` probability.", "",
            *metric_table(results, "roc_auc", methods),
        ],
        ["## Run details", "", *run_details(results, methods)],
    ]
    text = "\n\n".join("\n".join(section) for section in sections) + "\n"
    (config.RESULTS_DIR / "summary.md").write_text(text)
    print(f"[report] wrote {config.RESULTS_DIR / 'summary.md'}")
