import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import explicit_ratings


class ItemCollaborativeRecommender:
    """Item-based cosine on user-mean-centered explicit ratings, with overlap shrinkage."""

    def __init__(self, books: pd.DataFrame, ratings: pd.DataFrame, min_overlap: int = 2):
        self.books = books.reset_index(drop=True).copy()
        self.source_interaction_counts = ratings.groupby("user_id").size().to_dict()
        self.ratings = explicit_ratings(ratings)
        self.explicit_rating_counts = self.ratings.groupby("user_id").size().to_dict()
        self.min_overlap = max(1, int(min_overlap))
        self.book_ids = self.books.book_id.astype(str).tolist()
        self.item_index = {book_id: i for i, book_id in enumerate(self.book_ids)}
        known = self.ratings[self.ratings.book_id.astype(str).isin(self.item_index)].copy()
        self.user_ids = sorted(known.user_id.astype(str).unique())
        self.user_index = {user_id: i for i, user_id in enumerate(self.user_ids)}
        known["_i"] = known.book_id.astype(str).map(self.item_index)
        known["_u"] = known.user_id.astype(str).map(self.user_index)
        self.user_rated: dict[str, set[int]] = {}
        self.user_values: dict[str, dict[int, float]] = {}
        for user, group in known.groupby("user_id", sort=False):
            self.user_rated[str(user)] = set(group._i.astype(int))
            self.user_values[str(user)] = dict(zip(group._i.astype(int), group.rating.astype(float)))
        means = known.groupby("user_id").rating.mean()
        known["_centered"] = known.rating - known.user_id.map(means)
        self.item_user = csr_matrix((known._centered.astype(float), (known._i, known._u)),
                                    shape=(len(self.book_ids), len(self.user_ids)))
        self.binary = csr_matrix((np.ones(len(known)), (known._i, known._u)),
                                 shape=(len(self.book_ids), len(self.user_ids)))

    def recommend(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        if not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer from 1 to 100")
        user_id = str(user_id)
        history = self.user_rated.get(user_id, set())
        if len(history) < 2:
            return self._empty()
        # Compare each rated item to the catalog in sparse form; never allocate a full catalog square matrix.
        numerator = np.zeros(len(self.book_ids), dtype=float)
        denominator = np.zeros(len(self.book_ids), dtype=float)
        target_mean = float(np.mean(list(self.user_values[user_id].values())))
        for item in sorted(history):
            vector = self.item_user.getrow(item)
            similarities = cosine_similarity(vector, self.item_user).ravel()
            overlaps = (self.binary.getrow(item) @ self.binary.T).toarray().ravel()
            similarities[overlaps < self.min_overlap] = 0
            similarities[item] = 0
            rating = self.user_values[user_id][item]
            numerator += similarities * (rating - target_mean)
            denominator += np.abs(similarities)
        scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        seen = set(history)
        scores[list(seen)] = -np.inf
        # Positive evidence only; use finite scores and deterministic tie-breaking.
        valid = np.flatnonzero(np.isfinite(scores) & (scores > 0) & (denominator > 0))
        if len(valid) > top_k:
            valid = valid[np.argpartition(scores[valid], -top_k)[-top_k:]]
        valid = sorted(valid, key=lambda i: (-scores[i], self.book_ids[i]))
        result = self.books.iloc[valid].copy()
        result["score"] = scores[valid]
        return result.reset_index(drop=True)

    def _empty(self) -> pd.DataFrame:
        result = self.books.iloc[:0].copy()
        result["score"] = pd.Series(dtype=float)
        return result
