"""Application-level fallback policy for unreliable rating-overlap suggestions."""

import pandas as pd


def reader_rating_fallback(content_model, collaborative_model, catalog: pd.DataFrame,
                           popular_catalog: pd.DataFrame, user_id: str,
                           seed_ids: list[str] | tuple[str, ...], top_k: int = 50,
                           recency_weight: float = 0.0,
                           liked_ids: list[str] | set[str] | None = None,
                           exclude_ids: list[str] | set[str] | None = None) -> pd.DataFrame:
    """Use real selected-book, liked feedback, or rating-popularity evidence when sparse CF is unreliable."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    user_id = str(user_id).strip()
    rated_indexes = collaborative_model.user_rated.get(user_id, set())
    rated_ids = set(collaborative_model.books.iloc[list(rated_indexes)].book_id.astype(str)) if rated_indexes else set()
    seeds = list(dict.fromkeys(map(str, seed_ids)))
    excluded = rated_ids | set(seeds) | set(map(str, exclude_ids or [])) | set(map(str, liked_ids or []))

    if seeds or liked_ids:
        try:
            result = content_model.recommend(
                seeds, top_k, exclude_ids=excluded, recency_weight=recency_weight, liked_ids=liked_ids
            )
        except TypeError:
            result = content_model.recommend(seeds, top_k, exclude_ids=excluded)
        if seeds:
            explanation = (
                "Reader-overlap patterns are too sparse to rely on, so this fallback matches the "
                "title, author, and publisher details of your selected books."
            )
        else:
            explanation = (
                "Reader-overlap patterns are too sparse to rely on, so this fallback matches the "
                "title, author, and publisher details of books you liked."
            )
        result["explanation"] = explanation
        result["recommendation_source"] = "selected-book content fallback"
        return result

    ranking_columns = [column for column in ("rating_count", "avg_rating", "title")
                       if column in popular_catalog.columns]
    ascending = [False, False, True][:len(ranking_columns)]
    ranked = (popular_catalog.sort_values(ranking_columns, ascending=ascending, kind="stable")
              if ranking_columns else popular_catalog)
    candidates = ranked[~ranked.book_id.astype(str).isin(excluded)].head(top_k).copy()
    candidates["score"] = range(len(candidates), 0, -1)
    candidates["explanation"] = (
        "Reader-overlap patterns are too sparse to rely on, so this catalogue-wide fallback "
        "uses explicit rating count and average rating."
    )
    candidates["recommendation_source"] = "catalogue rating fallback"
    return candidates.reset_index(drop=True)
