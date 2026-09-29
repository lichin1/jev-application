"""The methods being compared, and how each is named in results files and in the report.

A method's scores live in ``results/<task>.json`` under ``result[key][field]``. Jev runs are
named by their *shot variant*:

* ``"0"``              zero-shot
* ``"3"`` (any k)      k random training examples per label
* ``"cluster"``        one example per (label, k-means cluster) cell
* ``"cluster-random"`` control for "cluster": the same per-label counts, drawn at random

Stacking runs use the zero-shot or cluster variant as their Jev feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CLUSTER_VARIANTS = ("cluster", "cluster-random")
STACK_VARIANTS = ("0", "cluster")


@dataclass(frozen=True)
class Method:
    key: str  # entry in results/<task>.json
    label: str  # column title in the report
    field: str = "metrics"  # where that entry keeps the scores

    def scores(self, result: dict[str, Any]) -> dict[str, Any]:
        """This method's metrics for one task ({} if it has not been run)."""
        return result.get(self.key, {}).get(self.field, {})


KAGGLE = Method("baseline", "Kaggle")
# The control shares the zero-shot stacking entry, which stores it next to the stacked scores.
STACK_CONTROL = Method("stack", "Stage 2 · Kaggle only", field="control_metrics")


def is_valid_variant(variant: str) -> bool:
    return variant.isdigit() or variant in CLUSTER_VARIANTS


def jev_method(variant: str) -> Method:
    if variant == "cluster":
        return Method("jev_cluster", "Jev cluster-shot")
    if variant == "cluster-random":
        return Method("jev_cluster_random", "Jev matched random")
    k = int(variant)
    return Method("jev", "Jev 0-shot") if k == 0 else Method(f"jev_{k}shot", f"Jev {k}-shot")


def stack_method(variant: str) -> Method:
    if variant == "0":
        return Method("stack", "Kaggle + Jev 0-shot")
    return Method(f"stack_{variant.replace('-', '_')}", f"Kaggle + Jev {variant}")


def report_methods(results: dict[str, dict[str, Any]]) -> list[Method]:
    """Every method (after Kaggle) that appears in any task's results, in report order.

    Order: Jev k-shot by k, the cluster variants, then the stacking control followed by the
    stacked models it is compared with.
    """
    present = {key for result in results.values() for key in result}
    shot_counts = sorted(
        {0 if key == "jev" else int(key[4:-4]) for key in present if key == "jev" or (key.endswith("shot") and key[4:-4].isdigit())}
    )
    methods = [jev_method(str(k)) for k in shot_counts]
    methods += [jev_method(v) for v in CLUSTER_VARIANTS if jev_method(v).key in present]
    stacks = [stack_method(v) for v in STACK_VARIANTS if stack_method(v).key in present]
    if stacks:
        methods += [STACK_CONTROL, *stacks]
    return methods
