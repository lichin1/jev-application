"""Ask Jev one question per row, with bounded concurrency and an on-disk answer cache.

Each request is identified by a hash of its full body (model, state and questions). Answers are
appended to ``.cache/jev/<task>.jsonl`` as they arrive, so:

* re-running the pipeline, or changing only metrics or reports, makes no API calls;
* an interrupted run resumes where it stopped;
* failed calls are not cached, so the next run retries exactly those rows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from jev_bench import config
from jev_bench.tasks import LABEL_QUESTION, Task

# A record is {"key", "answer", "model", "latency_s", "usage"} for a success, {"key", "error"} for a failure.
Record = dict[str, Any]

REQUEST_TIMEOUT_S = 60.0
MAX_RETRIES = 4


def request_payload(task: Task, row: pd.Series, model: str, questions: dict[str, Any] | None = None) -> dict[str, Any]:
    """The request body for one row; ``questions`` defaults to the task's zero-shot question."""
    questions = task.questions if questions is None else questions
    wire_questions = {name: q.model_dump(mode="json") for name, q in questions.items()}
    return {"model": model, "state": task.to_state(row), "questions": wire_questions}


def request_key(payload: dict[str, Any]) -> str:
    """Stable cache key of a request body."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class AnswerCache:
    """Append-only JSONL file of successful Jev answers for one task."""

    def __init__(self, task_name: str) -> None:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path: Path = config.CACHE_DIR / f"{task_name}.jsonl"
        self.records: dict[str, Record] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["key"]] = record

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def __getitem__(self, key: str) -> Record:
        return self.records[key]

    def remember(self, record: Record, file) -> None:
        """Keep a record for this run; write it to disk only if it is a successful answer."""
        self.records[record["key"]] = record
        if "error" not in record:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()  # a crash loses at most the in-flight requests


class _Progress:
    """Prints roughly every 10% of the requests instead of one line per batch."""

    def __init__(self, task_name: str, total: int) -> None:
        self.task_name, self.total, self.done = task_name, total, 0
        self.step = max(1, total // 10)

    def tick(self) -> None:
        self.done += 1
        if self.done % self.step == 0 or self.done == self.total:
            print(f"[jev] {self.task_name}: {self.done}/{self.total}")


async def _request_missing(
    task: Task, payloads: list[dict[str, Any]], keys: list[str], todo: list[int],
    questions: dict[str, Any], model: str, concurrency: int, cache: AnswerCache,
) -> None:
    """Send the uncached requests, ``concurrency`` at a time, storing each answer as it arrives."""
    limit = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    progress = _Progress(task.name, len(todo))
    retry = RetryPolicy(max_retries=MAX_RETRIES)

    async with AsyncTypeSafeClient(model=model, timeout=REQUEST_TIMEOUT_S, retry=retry) as client:
        with cache.path.open("a") as file:

            async def request(i: int) -> None:
                async with limit:
                    start = time.perf_counter()
                    try:
                        response = await client.system_one(payloads[i]["state"], questions)
                        record = {
                            "key": keys[i],
                            "answer": response.answers[LABEL_QUESTION].model_dump(mode="json"),
                            "model": response.model,
                            "latency_s": time.perf_counter() - start,
                            "usage": response.usage.model_dump(mode="json"),
                        }
                    except TypeSafeError as err:
                        record = {"key": keys[i], "error": f"{type(err).__name__}: {err}"}
                async with write_lock:
                    cache.remember(record, file)
                    progress.tick()

            await asyncio.gather(*(request(i) for i in todo))


def ask_jev(
    task: Task, rows: pd.DataFrame, model: str, concurrency: int, questions: dict[str, Any] | None = None
) -> list[Record]:
    """One record per row, in row order; cached answers are reused and only the rest are requested."""
    questions = task.questions if questions is None else questions
    cache = AnswerCache(task.name)
    payloads = [request_payload(task, row, model, questions) for _, row in rows.iterrows()]
    keys = [request_key(p) for p in payloads]
    todo = [i for i, key in enumerate(keys) if key not in cache]
    print(f"[jev] {task.name}: {len(rows) - len(todo)} cached, {len(todo)} to request (model={model})")
    if todo:
        asyncio.run(_request_missing(task, payloads, keys, todo, questions, model, concurrency, cache))
    return [cache[key] for key in keys]
