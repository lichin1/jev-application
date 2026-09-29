"""Project-wide paths and defaults.

Every module reads its locations and fixed settings from here, so a run can be redirected
(for example to a temporary directory in tests) by changing one place.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Downloaded Kaggle files (not committed).
RAW_DATA_DIR = ROOT / "data" / "raw"
# Jev answers, one JSONL file per task (not committed). Re-running reads from here instead of the API.
CACHE_DIR = ROOT / ".cache" / "jev"
# Per-task results and the Markdown report (committed).
RESULTS_DIR = ROOT / "results"

# One seed for every random step: the train/test split, example sampling, k-means and cross-validation.
SEED = 42
# Share of each dataset held out for testing.
TEST_SIZE = 0.2

DEFAULT_MODEL = "jev-latest"
DEFAULT_CONCURRENCY = 8
