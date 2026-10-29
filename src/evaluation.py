from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .collaborative import ItemCollaborativeRecommender
from .config import RANDOM_SEED, RATING_THRESHOLD
from .content_based import ContentRecommender
from .catalog_search import exclude_seen_editions
from .hybrid import hybrid_recommend, hybrid_recommend_previous
from .preprocess import explicit_ratings


def _ranking_metrics(recommended: list[str], relevant: set[str], k: int) -> dict[str, float]:
    ids = list(dict.fromkeys(map(str, recommended)))[:k]
    relevant = set(map(str, relevant))
    hit = [int(book_id in relevant) for book_id in ids]

    def precision_at(cutoff: int) -> float:
        return sum(hit[:cutoff]) / cutoff

    def recall_at(cutoff: int) -> float:
        return sum(hit[:cutoff]) / len(relevant) if relevant else 0.0

    discounts = np.log2(np.arange(2, len(hit) + 2, dtype=float))
    dcg = float(np.sum(np.asarray(hit, dtype=float) / discounts)) if hit else 0.0
    ideal_len = min(len(relevant), k)
    idcg = float(np.sum(1 / np.log2(np.arange(2, ideal_len + 2, dtype=float)))) if ideal_len else 0.0
    precision_sum = 0.0
    found = 0
    for rank, is_hit in enumerate(hit, start=1):
        if is_hit:
            found += 1
            precision_sum += found / rank
    average_precision = precision_sum / min(len(relevant), k) if relevant else 0.0
    return {
        "precision_at_5": precision_at(5),
        "precision_at_10": precision_at(10),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "map_at_10": average_precision,
    }


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = RANDOM_SEED
    threshold: int = RATING_THRESHOLD
    max_users: int = 100
    top_k: int = 10
    test_fraction: float = 0.2


def precision_recall_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> tuple[float, float]:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    hits = len(set(map(str, recommended[:k])) & set(map(str, relevant)))
    return hits / k, hits / len(relevant) if relevant else 0.0


def _relevant_work_ids(test_rows: pd.DataFrame, train_rows: pd.DataFrame,
                       books: pd.DataFrame, threshold: int) -> set[str]:
    """Return held-out relevant works not already represented in training history."""
    relevant_rows = test_rows.loc[test_rows.rating.ge(threshold)].copy()
    if relevant_rows.empty:
        return set()
    seen_ids = set(train_rows.book_id.astype(str))
    metadata = [column for column in ("book_id", "title", "author") if column in books]
    relevant_rows = relevant_rows.merge(books[metadata].drop_duplicates("book_id"),
                                        on="book_id", how="left")
    relevant_rows = exclude_seen_editions(relevant_rows, books, seen_ids)
    return set(relevant_rows.book_id.astype(str))


def evaluate_models(books: pd.DataFrame, ratings: pd.DataFrame,
                    config: EvaluationConfig = EvaluationConfig()) -> tuple[pd.DataFrame, dict]:
    """Per-user holdout evaluation; every recommender sees training interactions only."""
    if (not isinstance(config.max_users, int) or config.max_users < 1 or
            not isinstance(config.top_k, int) or config.top_k < 1 or
            not 0 < config.test_fraction < 1 or not 1 <= config.threshold <= 10):
        raise ValueError("Invalid evaluation settings")
    catalog_book_ids = set(books.book_id.astype(str))
    data = explicit_ratings(ratings)
    data = data[data.book_id.astype(str).isin(catalog_book_ids)].reset_index(drop=True)
    eligible = data.groupby("user_id").filter(lambda rows: len(rows) >= 5)
    user_counts = eligible.groupby("user_id").size()
    rng = np.random.default_rng(config.seed)
    eligible_users = np.array(sorted(user_counts.index.astype(str)))
    rng.shuffle(eligible_users)
    eligible_users = eligible_users[:config.max_users]

    test_by_user, heldout_pairs, relevant_interactions = {}, set(), 0
    eligible_set = set(eligible_users)
    for user, group in eligible[eligible.user_id.astype(str).isin(eligible_set)].groupby("user_id", sort=True):
        group = group.sample(frac=1, random_state=config.seed)
        n_test = max(1, int(round(len(group) * config.test_fraction)))
        # Keep at least two train ratings so a profile can be formed.
        n_test = min(n_test, len(group) - 2)
        test = group.iloc[:n_test]
        train = group.iloc[n_test:]
        heldout_pairs.update((str(user), book_id) for book_id in test.book_id.astype(str))
        relevant = _relevant_work_ids(test, train, books, config.threshold)
        if not relevant:
            continue
        test_by_user[str(user)] = (set(train.book_id.astype(str)), relevant)
        relevant_interactions += len(relevant)
    # Keep every non-held-out interaction in training. Only sampled evaluation users have
    # a holdout; other readers provide legitimate collaborative evidence in the train set.
    training_mask = [(str(user), str(book_id)) not in heldout_pairs
                     for user, book_id in zip(data.user_id, data.book_id)]
    train_ratings = data.loc[training_mask].reset_index(drop=True)
    training_pairs = set(zip(train_ratings.user_id.astype(str), train_ratings.book_id.astype(str)))
    training_heldout_overlap = len(training_pairs & heldout_pairs)
    training_book_ids = set(train_ratings.book_id.astype(str))
    # Catalogue candidates are known at serving time; learned vocabulary/IDF comes
    # only from catalogue records represented in explicit training interactions.
    content = ContentRecommender(books, fit_book_ids=training_book_ids)
    collaborative = ItemCollaborativeRecommender(books, train_ratings)
    item_rater_counts = np.asarray(collaborative.binary.getnnz(axis=1)).ravel()
    popularity_order = (train_ratings.groupby("book_id").size()
                       .sort_values(ascending=False, kind="stable").index.astype(str).tolist())
    candidate_k = min(100, max(config.top_k * 5, config.top_k))
    records = []
    methods = ("Most popular (train only)", "Content-based (train-only fit)",
               "Collaborative (item-based)", "Previous hybrid (min-max union)",
               "Revised hybrid (content-preserving)")
    recommended_by_method: dict[str, list[list[str]]] = {name: [] for name in methods}
    for method in methods:
        per_user, diversity_scores = [], []
        for user, (seen, relevant) in test_by_user.items():
            if method == "Most popular (train only)":
                ids = [book_id for book_id in popularity_order if book_id not in seen][:candidate_k]
                recommendations = books.loc[books.book_id.astype(str).isin(ids)].copy()
                recommendations["score"] = 0.0
                recommendations = recommendations.set_index("book_id").loc[ids].reset_index() if ids else recommendations.iloc[:0]
            elif method == "Content-based (train-only fit)":
                # Training books rated >= 8 define this user's content profile.
                likes = train_ratings[(train_ratings.user_id.astype(str) == user) &
                                      (train_ratings.rating >= config.threshold)].book_id.astype(str).tolist()
                recommendations = content.recommend(likes, candidate_k, exclude_ids=seen)
            elif method == "Collaborative (item-based)":
                recommendations = collaborative.recommend(user, candidate_k)
            elif method == "Previous hybrid (min-max union)":
                likes = train_ratings[(train_ratings.user_id.astype(str) == user) &
                                      (train_ratings.rating >= config.threshold)].book_id.astype(str).tolist()
                recommendations = hybrid_recommend_previous(books, content, collaborative, user, likes,
                                                             candidate_k)
            else:
                likes = train_ratings[(train_ratings.user_id.astype(str) == user) &
                                      (train_ratings.rating >= config.threshold)].book_id.astype(str).tolist()
                recommendations = hybrid_recommend(books, content, collaborative, user, likes,
                                                   candidate_k)
            recommendations = exclude_seen_editions(recommendations, books, seen)
            ids = recommendations.book_id.astype(str).tolist()
            ids = [book_id for book_id in ids if book_id not in seen]
            recommended_by_method[method].append(ids[:config.top_k])
            per_user.append(_ranking_metrics(ids, relevant, config.top_k))
            item_rows = [content.index[value] for value in ids[:config.top_k] if value in content.index]
            if len(item_rows) > 1:
                pairwise = cosine_similarity(content.matrix[item_rows], content.matrix[item_rows])
                upper = pairwise[np.triu_indices(len(item_rows), k=1)]
                diversity_scores.append(float(1 - np.mean(upper)))
            else:
                diversity_scores.append(0.0)
        metrics = pd.DataFrame(per_user).mean().to_dict() if per_user else {}
        unique_recommendations = {value for row in recommended_by_method[method] for value in row}
        records.append({"model": method,
                        **metrics,
                        "catalogue_coverage": len(unique_recommendations) / max(1, len(books)),
                        "intra_list_diversity": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
                        "average_list_length": float(np.mean([len(row) for row in recommended_by_method[method]])) if recommended_by_method[method] else 0.0,
                        "empty_recommendation_users": int(sum(not row for row in recommended_by_method[method])),
                        "evaluated_users": len(per_user),
                        "sampled_eligible_users": len(eligible_users),
                        "users_with_relevant_holdout": len(test_by_user),
                        "relevant_heldout_interactions": relevant_interactions,
                        "top_k": config.top_k})
    details = {"dataset": "Book-Crossing explicit ratings; all catalogue books are candidates for sampled users",
               "seed": config.seed, "threshold": config.threshold,
               "strategy": f"Per-user random holdout of {config.test_fraction:.0%}; minimum 5 explicit ratings; at least 2 retained for training",
               "max_users": config.max_users,
               "holdout_fraction": config.test_fraction,
               "minimum_explicit_ratings": 5,
               "minimum_train_ratings": 2,
               "explicit_rating_rows_matched_to_catalogue": int(len(data)),
               "training_explicit_ratings": int(len(train_ratings)),
               "users_with_training_ratings": int(train_ratings.user_id.nunique()),
               "books_with_training_ratings": int(train_ratings.book_id.nunique()),
               "item_user_matrix_density": float(len(train_ratings) /
                   max(1, len(books) * train_ratings.user_id.nunique())),
               "collaborative_minimum_shared_raters": collaborative.min_overlap,
               "catalogue_books_with_zero_training_ratings": int(np.sum(item_rater_counts == 0)),
               "catalogue_books_with_one_training_rater": int(np.sum(item_rater_counts == 1)),
               "catalogue_books_with_two_or_more_training_raters": int(np.sum(item_rater_counts >= 2)),
               "median_raters_per_rated_book": float(np.median(item_rater_counts[item_rater_counts > 0])) if np.any(item_rater_counts > 0) else 0.0,
               "top_k": config.top_k,
               "sampled_eligible_users": int(len(eligible_users)),
               "users_with_relevant_holdout": int(len(test_by_user)),
               "users_without_relevant_holdout": int(len(eligible_users) - len(test_by_user)),
               "all_heldout_explicit_interactions": int(len(heldout_pairs)),
               "training_heldout_pair_overlap": int(training_heldout_overlap),
               "metric_averaging": "Macro-average per-user ranking metrics over sampled users with at least one relevant held-out book. Precision@K divides by K, so short lists count empty slots as misses. Catalog coverage is unique recommended items divided by all catalogue items; intra-list diversity is 1 minus mean pairwise TF-IDF cosine similarity within each top-K list.",
               "excluded_from_metric_average": "Sampled users with no held-out rating at or above the relevance threshold; zero-valued interactions",
               "excluded_from_training": "All held-out explicit interactions for sampled users",
               "candidate_and_seen_item_rule": "Every catalogue book is a candidate; exclude each sampled user's training-rated ISBNs and same title-author work before truncating to top K. Users and held-out items are restricted to explicit interactions whose book IDs match the catalogue after checksum-validated ISBN normalization.",
               "split_limit": "Book-Crossing has no usable timestamps in this project; use a deterministic seeded per-user random holdout.",
               "relevant_heldout_interactions": relevant_interactions}
    details["feature_fit_scope"] = (
        "TF-IDF vocabulary and IDF were fitted only on catalogue items with explicit training ratings; "
        "the fitted transform was applied to the shared full-catalogue candidate universe."
    )
    details["model_comparison"] = (
        "All five model variants use the same seeded per-user holdout, relevant users, relevance threshold, "
        "catalogue candidates, seen-item/work exclusions, metrics, and training interactions."
    )
    details["relevance_definition"] = (
        f"A held-out explicit rating of at least {config.threshold} on the 1–10 Book-Crossing scale, "
        "where the same title-author work was not already represented by an ISBN in that reader's "
        "training history; duplicate held-out editions of a work count once. Zero-valued implicit "
        "interactions are excluded."
    )
    details["metric_definitions"] = (
        "Precision@K is hits divided by K (short lists count empty slots as misses); Recall@K is hits divided "
        "by that user's relevant held-out items; NDCG@10 uses binary relevance and ideal DCG; MAP@10 is "
        "mean precision at relevant ranks divided by min(relevant count, 10). Metrics are macro-averaged "
        "over sampled users with at least one relevant held-out item. Coverage is unique recommended ISBNs "
        "over catalogue size; diversity is 1 minus mean pairwise TF-IDF cosine within each list."
    )
    details["catalogue_book_count"] = int(books.book_id.astype(str).nunique())
    details["unique_test_relevant_items"] = int(len({book for _user, _seen_relevant in test_by_user.items()
                                                       for book in _seen_relevant[1]}))
    return pd.DataFrame(records), details
