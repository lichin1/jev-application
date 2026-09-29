# Shortcuts for the jev-bench pipeline. Run `make help` for the list.
# Jev targets need TYPESAFE_API_KEY (see .env.example); answers are cached in .cache/jev/.

PYTHON ?= python3
CONCURRENCY ?= 16

.PHONY: help install test baseline jev fewshot cluster stack all report examples clean-cache

help:  ## Show this list
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

install:  ## Install the package and test tools (editable)
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run the offline tests (no API key needed)
	$(PYTHON) -m pytest -q

baseline:  ## Train and score the Kaggle solutions (no API key needed)
	jev-bench baseline

jev:  ## Jev zero-shot on every task
	jev-bench jev --shots 0 --concurrency $(CONCURRENCY)

fewshot:  ## Jev with 3 random examples per label
	jev-bench jev --shots 3 --concurrency $(CONCURRENCY)

cluster:  ## Jev with cluster-selected examples, and its matched random control
	jev-bench jev --shots cluster cluster-random --concurrency $(CONCURRENCY)

stack:  ## Stack the Kaggle model with Jev (0-shot and cluster features)
	jev-bench stack --concurrency $(CONCURRENCY)

all:  ## The full experiment: baseline, every Jev variant, stacking, report
	jev-bench all --concurrency $(CONCURRENCY)

report:  ## Rebuild results/summary.md (no API calls)
	jev-bench report

examples:  ## Save example Jev requests for every variant (no API calls)
	jev-bench examples --shots 0 3 cluster cluster-random

clean-cache:  ## Delete cached Jev answers (the next run calls the API again)
	rm -rf .cache/jev
