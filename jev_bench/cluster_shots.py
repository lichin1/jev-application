"""Cluster-stratified few-shot selection.

Random k-per-label sampling mostly draws the typical member of each label. Here the training rows
are first grouped with k-means, then every (label, cluster) cell with enough rows contributes one
representative example. A label that spreads over several clusters therefore gets one example
per sub-pattern, including the minority cells near the label boundary, such as a first-class woman
who died on the Titanic, that random sampling rarely draws.

``select_matched_random`` is the control: the same number of examples per label, drawn at random,
to separate the effect of diversity from the effect of simply showing more examples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

from jev_bench import config
from jev_bench.few_shot import is_short, sample_per_label
from jev_bench.tasks import Task

# Cluster counts tried run from the number of labels up to MAX_K. Fewer clusters than labels
# cannot separate the labels (Iris alone would score best at k=2: setosa vs. the rest).
MAX_K = 8
# A cell needs at least this many rows to count as a pattern rather than an outlier.
MIN_CELL_SIZE = 3
# Silhouette is quadratic in the number of rows, so it is estimated on a sample.
SILHOUETTE_SAMPLE = 3000


def cluster_features(task: Task, train: pd.DataFrame) -> np.ndarray:
    """Numeric vectors for k-means, built from the same inputs the Kaggle model uses."""
    X = task.baseline_features(train)
    if isinstance(X, pd.Series):
        # Text: TF-IDF reduced to 100 dense dimensions and length-normalised, so k-means
        # compares topics rather than document lengths.
        text_pipeline = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50_000, stop_words="english"),
            TruncatedSVD(n_components=100, random_state=config.SEED),
            Normalizer(),
        )
        return text_pipeline.fit_transform(X)
    # Tables: standardised numbers plus one-hot categories (rare categories grouped together).
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False),
    )
    table_pipeline = ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_exclude="number")),
        ]
    )
    return table_pipeline.fit_transform(X)


def _kmeans(k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=4, random_state=config.SEED)


def choose_k(X: np.ndarray, min_k: int) -> tuple[int, dict[int, float]]:
    """The cluster count between ``min_k`` and ``MAX_K`` with the best silhouette score, and all scores."""
    scores = {}
    for k in range(max(2, min_k), max(MAX_K, min_k) + 1):
        clusters = _kmeans(k).fit_predict(X)
        sample = min(SILHOUETTE_SAMPLE, len(X))
        scores[k] = float(silhouette_score(X, clusters, sample_size=sample, random_state=config.SEED))
    return max(scores, key=scores.get), scores


def _representative(X: np.ndarray, cell: np.ndarray, short: np.ndarray) -> int:
    """Position of the row nearest the cell's centroid, among the cell's short rows when it has any."""
    candidates = cell[short[cell]] if short[cell].any() else cell
    centroid = X[cell].mean(axis=0)
    return int(candidates[np.argmin(np.linalg.norm(X[candidates] - centroid, axis=1))])


def select_cluster_shots(task: Task, train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One example per (label, k-means cluster) cell, plus a description of the clustering."""
    X = cluster_features(task, train)
    k, silhouette = choose_k(X, min_k=len(task.labels))
    clusters = _kmeans(k).fit_predict(X)
    labels = train["label"].to_numpy()
    short = is_short(task, train).to_numpy()

    picked, cells = [], []
    for label in task.labels:
        for cluster in range(k):
            cell = np.flatnonzero((labels == label) & (clusters == cluster))
            if len(cell) < MIN_CELL_SIZE:
                continue
            chosen = _representative(X, cell, short)
            picked.append(chosen)
            cluster_size = int((clusters == cluster).sum())
            cells.append(
                {
                    "label": str(label),
                    "cluster": cluster,
                    "cell_size": len(cell),
                    "cluster_size": cluster_size,
                    # Below 0.5 the label is a minority of its cluster: a boundary pattern.
                    "label_share_of_cluster": round(len(cell) / cluster_size, 3),
                    "train_row": int(train.index[chosen]),
                }
            )
    info = {
        "k": k,
        "silhouette": {str(kk): round(v, 4) for kk, v in silhouette.items()},
        "min_cell_size": MIN_CELL_SIZE,
        "cells": cells,
    }
    return train.iloc[picked], info


def select_matched_random(task: Task, train: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Random examples with the same per-label counts as ``reference`` (the cluster selection)."""
    counts = reference["label"].value_counts()
    return sample_per_label(task, train, {label: int(counts.get(label, 0)) for label in task.labels})
