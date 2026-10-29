import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import compute_recency_score, prepare_books


class ContentRecommender:
    """TF-IDF over title, author, and publisher; optionally boosted by trustworthy recency."""

    def __init__(self, books: pd.DataFrame, fit_book_ids: set[str] | None = None):
        self.books = prepare_books(books).drop_duplicates("book_id", keep="first").reset_index(drop=True)
        self.index = {str(book_id): i for i, book_id in enumerate(self.books.book_id)}
        text = self.books.metadata_text
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        try:
            if fit_book_ids is None:
                self.matrix = self.vectorizer.fit_transform(text)
            else:
                training_rows = self.books.book_id.astype(str).isin(set(map(str, fit_book_ids)))
                self.vectorizer.fit(text.loc[training_rows])
                self.matrix = self.vectorizer.transform(text)
        except ValueError:
            self.matrix = None
        if "year" in self.books.columns:
            self.recency_scores = compute_recency_score(self.books.year)
        else:
            self.recency_scores = np.zeros(len(self.books), dtype=float)

    def recommend(self, book_ids: list[str] | tuple[str, ...] | str, top_k: int = 10,
                  exclude_ids: list[str] | set[str] | tuple[str, ...] | None = None,
                  recency_weight: float = 0.0,
                  liked_ids: list[str] | set[str] | tuple[str, ...] | None = None,
                  liked_weight: float = 0.5) -> pd.DataFrame:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer from 1 to 100")
        if not isinstance(recency_weight, (int, float)) or not 0 <= recency_weight <= 1:
            raise ValueError("recency_weight must be a float between 0 and 1")
        if not isinstance(liked_weight, (int, float)) or liked_weight < 0:
            raise ValueError("liked_weight must be a non-negative float")
        if isinstance(book_ids, str):
            book_ids = [book_ids]
        ids = [str(v) for v in (book_ids or []) if str(v) in self.index]
        if isinstance(liked_ids, str):
            liked_ids = [liked_ids]
        liked_list = [str(v) for v in (liked_ids or []) if str(v) in self.index]
        if (not ids and not liked_list) or self.matrix is None:
            return self._empty()

        if ids:
            seed_rows = [self.index[value] for value in dict.fromkeys(ids)]
            seed_scores = np.asarray(cosine_similarity(self.matrix[seed_rows], self.matrix).mean(axis=0)).ravel()
        else:
            seed_rows = []
            seed_scores = np.zeros(self.matrix.shape[0], dtype=float)

        if liked_list:
            liked_rows = [self.index[value] for value in dict.fromkeys(liked_list)]
            liked_scores = np.asarray(cosine_similarity(self.matrix[liked_rows], self.matrix).mean(axis=0)).ravel()
        else:
            liked_rows = []
            liked_scores = np.zeros(self.matrix.shape[0], dtype=float)

        if ids and liked_list:
            scores = seed_scores + float(liked_weight) * liked_scores
        elif ids:
            scores = seed_scores
        else:
            scores = liked_scores

        selected = set(seed_rows) | set(liked_rows)
        scores[list(selected)] = -1
        excluded = selected | {self.index[str(value)] for value in (exclude_ids or [])
                               if str(value) in self.index}
        if excluded:
            scores[list(excluded)] = -1
        # Zero-similarity rows are not evidence and would otherwise create a huge tie to sort.
        valid = np.isfinite(scores) & (scores > 0)
        if recency_weight > 0 and np.any(valid):
            # Boost candidate scores gently without promoting zero-evidence items
            scores[valid] = scores[valid] * (1.0 + float(recency_weight) * self.recency_scores[valid])
        positions = np.flatnonzero(valid)
        if len(positions) > top_k:
            positions = positions[np.argpartition(scores[positions], -top_k)[-top_k:]]
        positions = sorted(positions, key=lambda i: (-scores[i], str(self.books.iloc[i].book_id)))
        result = self.books.iloc[positions].copy()
        result["score"] = scores[positions]
        if recency_weight > 0:
            result["recency_score"] = self.recency_scores[positions]
        if liked_rows and len(positions) > 0:
            result["feedback_boost"] = liked_scores[positions]
        return result.reset_index(drop=True)

    def _empty(self) -> pd.DataFrame:
        result = self.books.iloc[0:0].copy()
        result["score"] = pd.Series(dtype=float)
        return result
