"""Download, load, and split the five Kaggle datasets.

Kaggle itself needs an account and API token, so each dataset is fetched from a public GitHub
mirror of the same Kaggle file and cached under ``data/raw``. Every loader returns a frame with a
``label`` column; the rest of the project never needs to know the original column names.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from jev_bench import config

# Public mirrors of each Kaggle file; the comment names the original Kaggle page.
SOURCES = {
    # kaggle.com/competitions/titanic
    "titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    # kaggle.com/datasets/uciml/sms-spam-collection-dataset
    "sms_spam": "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    # kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    "imdb": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
    # kaggle.com/competitions/learn-ai-bbc (BBC News Classification)
    "bbc_news": "https://raw.githubusercontent.com/mdsohaib/BBC-News-Classification/master/bbc-text.csv",
    # kaggle.com/datasets/uciml/iris
    "iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
}


@dataclass
class Split:
    """The data one task is evaluated on.

    ``scored`` is the set of test rows every method is scored on: the whole ``test`` split by
    default, or a stratified sample of it when a run limits the number of Jev calls.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    scored: pd.DataFrame


def fetch(name: str) -> Path:
    """Local path of the raw file, downloading it on first use."""
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = SOURCES[name]
    path = config.RAW_DATA_DIR / f"{name}{Path(url).suffix}"
    if not path.exists():
        print(f"[data] downloading {name} from {url}")
        partial = path.with_suffix(path.suffix + ".part")  # renamed only once complete
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return path


def _load_titanic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["Survived"].astype(int)
    return df


def _load_sms_spam(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["label_str", "text"])
    df["label"] = (df["label_str"] == "spam").astype(int)
    return df


def _load_imdb(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Reviews contain HTML line breaks; they carry no meaning for classification.
    df["text"] = df["review"].map(lambda text: re.sub(r"<br\s*/?>", " ", text).strip())
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df


def _load_bbc_news(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).rename(columns={"category": "label"})


def _load_iris(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).rename(columns={"species": "label"})


_LOADERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "titanic": _load_titanic,
    "sms_spam": _load_sms_spam,
    "imdb": _load_imdb,
    "bbc_news": _load_bbc_news,
    "iris": _load_iris,
}


def load(name: str) -> pd.DataFrame:
    """The full dataset with a ``label`` column."""
    return _LOADERS[name](fetch(name)).reset_index(drop=True)


def split(name: str, scored_n: int | None = None) -> Split:
    """Stratified train/test split; ``scored_n`` optionally limits the scored test rows to a stratified sample."""
    df = load(name)
    train, test = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=df["label"])
    test = test.reset_index(drop=True)
    if scored_n is None or scored_n >= len(test):
        scored = test
    else:
        scored, _ = train_test_split(test, train_size=scored_n, random_state=config.SEED, stratify=test["label"])
    return Split(train.reset_index(drop=True), test, scored.reset_index(drop=True))
