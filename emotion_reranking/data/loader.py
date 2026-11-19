"""Loading and sampling of the Amazon Reviews corpus."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from emotion_reranking.data.preprocessing import clean_text, tokenize_and_stem

HIGH_RATING = "high_rating (>=4)"
LOW_RATING = "low_rating (<4)"
UNKNOWN_RATING = "unknown"

TEXT_COLUMN_CANDIDATES = ("Review Text", "reviewText", "Text", "review_body", "text", "review", "body")
RATING_COLUMN_CANDIDATES = ("Rating", "overall", "Score", "star_rating", "rating", "label")

_RATED_PATTERN = re.compile(r"Rated (\d)")


def parse_stars(value) -> float:
    """Star rating from values such as ``"Rated 4 out of 5 stars"`` or ``"4"``."""
    match = _RATED_PATTERN.search(str(value))
    if match:
        return int(match.group(1))
    try:
        return float(str(value).strip())
    except ValueError:
        return np.nan


def rating_category(value) -> str:
    """Binary rating class: ``HIGH_RATING`` for 4-5 stars, ``LOW_RATING`` otherwise."""
    try:
        match = _RATED_PATTERN.search(str(value))
        stars = int(match.group(1)) if match else float(str(value).strip())
        return HIGH_RATING if stars >= 4 else LOW_RATING
    except ValueError:
        return UNKNOWN_RATING


def stratified_sample(df: pd.DataFrame, column: str, n_target: int,
                      min_per_class: int = 150, seed: int = 42) -> pd.DataFrame:
    """Sample ``n_target`` rows keeping class proportions, with a minimum per class."""
    groups = []
    for _, group in df.groupby(column):
        share = len(group) / len(df)
        take = min(max(min_per_class, int(round(n_target * share))), len(group))
        groups.append(group.sample(n=take, random_state=seed))
    out = pd.concat(groups).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out.head(n_target) if len(out) > n_target else out


class AmazonReviewsLoader:
    """Loads the Kaggle *Amazon Reviews* CSV and prepares it for indexing.

    The CSV is looked up at ``data_path``; if missing, any CSV in the same
    folder is used; as a last resort the dataset is downloaded with
    ``kagglehub``.

    The resulting frame has the original columns plus ``rating_category``,
    ``stars``, ``clean_text`` and ``bm25_tokens``.
    """

    def __init__(self, data_path: Path, kaggle_dataset: str, n_docs: int,
                 min_per_class: int = 150, seed: int = 42) -> None:
        self.data_path = Path(data_path)
        self.kaggle_dataset = kaggle_dataset
        self.n_docs = n_docs
        self.min_per_class = min_per_class
        self.seed = seed

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._resolve_csv(), engine="python", on_bad_lines="skip")
        return self.prepare(df)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in df.columns), df.columns[0])
        rating_col = next((c for c in RATING_COLUMN_CANDIDATES if c in df.columns), None)

        df = df.copy()
        df["rating_category"] = df[rating_col].apply(rating_category) if rating_col else UNKNOWN_RATING
        df["stars"] = df[rating_col].apply(parse_stars) if rating_col else np.nan

        df = df.dropna(subset=[text_col, "stars"]).reset_index(drop=True)
        df = stratified_sample(df, "stars", min(len(df), self.n_docs),
                               min_per_class=self.min_per_class, seed=self.seed)

        df[text_col] = df[text_col].astype(str).str.replace(r"<br\s*/?>", " ", regex=True)
        df["clean_text"] = df[text_col].apply(clean_text)
        df["bm25_tokens"] = df[text_col].apply(tokenize_and_stem)
        return df

    def _resolve_csv(self) -> Path:
        if self.data_path.is_file():
            return self.data_path
        if self.data_path.parent.is_dir():
            local = sorted(self.data_path.parent.rglob("*.csv"))
            if local:
                return local[0]
        return self._download()

    def _download(self) -> Path:
        import kagglehub

        print(f"[Data] Downloading '{self.kaggle_dataset}' from Kaggle...")
        dataset_dir = Path(kagglehub.dataset_download(self.kaggle_dataset))
        csv_files = sorted(dataset_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {dataset_dir}")
        return csv_files[0]
