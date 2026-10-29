import pandas as pd
import numpy as np

from .collaborative import ItemCollaborativeRecommender
from .content_based import ContentRecommender
from .preprocess import parse_publication_year


def normalize_scores(values: pd.Series) -> pd.Series:
    """Convert arbitrary model scores to a stable ordinal score in [0, 1]."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.empty:
        return pd.Series(dtype=float, index=values.index)
    finite = numeric.dropna()
    if finite.empty or float(finite.max()) == float(finite.min()):
        return pd.Series(0.0, index=values.index)
    ranks = numeric.rank(method="min", ascending=False)
    if len(ranks) == 1:
        return pd.Series(1.0, index=values.index)
    return 1.0 - (ranks - 1.0) / (len(ranks) - 1.0)


def hybrid_recommend_previous(books: pd.DataFrame, content_model: ContentRecommender,
                              collaborative_model: ItemCollaborativeRecommender,
                              user_id: str | None = None, favorite_ids: list[str] | None = None,
                              top_k: int = 10, alpha: float = 0.5) -> pd.DataFrame:
    """Historical v5 union + per-source min-max blend, kept for fair comparison only."""
    candidate_k = min(100, max(10, top_k * 5))
    content = content_model.recommend(favorite_ids or [], candidate_k)
    collab = collaborative_model.recommend(user_id, candidate_k) if user_id else collaborative_model._empty()
    parts = []
    for frame, name in ((content, "content_score"), (collab, "collaborative_score")):
        if not frame.empty:
            part = frame[["book_id", "score"]].rename(columns={"score": name}).copy()
            values = pd.to_numeric(part[name], errors="coerce")
            if values.notna().any() and values.max() != values.min():
                part[name] = (values - values.min()) / (values.max() - values.min())
            else:
                part[name] = 0.0
            parts.append(part)
    if not parts:
        return books.iloc[:0].assign(score=pd.Series(dtype=float), explanation=pd.Series(dtype=str))
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="book_id", how="outer")
    has_content = "content_score" in merged
    has_collab = "collaborative_score" in merged
    if has_content and has_collab:
        content_weight = alpha if alpha > 0 else 0.0
        collab_weight = (1 - alpha) if alpha < 1 else 0.0
        content_present = merged.content_score.notna() & (content_weight > 0)
        collab_present = merged.collaborative_score.notna() & (collab_weight > 0)
        available = (content_present.astype(float) * content_weight +
                     collab_present.astype(float) * collab_weight)
        weighted = (merged.content_score.fillna(0) * content_weight +
                    merged.collaborative_score.fillna(0) * collab_weight)
        merged["score"] = weighted.div(available.replace(0, np.nan))
        merged["explanation"] = np.select(
            [content_present & collab_present, content_present],
            ["Catalogue similarity and overlapping rating evidence both contributed.",
             "Metadata is similar to your selected books."],
            default="Rating patterns on similar books support this recommendation.")
        merged = merged[merged.score.notna()]
    elif has_content:
        merged["score"] = merged.content_score.fillna(0)
        merged["explanation"] = "Content-based fallback: metadata is similar to your selected books."
    else:
        merged["score"] = merged.collaborative_score.fillna(0)
        merged["explanation"] = "Reader-rating fallback: rating patterns on similar books support this recommendation."
    exclude = set(map(str, favorite_ids or []))
    if user_id:
        rated = collaborative_model.user_rated.get(str(user_id), set())
        exclude.update(collaborative_model.books.iloc[list(rated)].book_id.astype(str))
    merged = merged[~merged.book_id.astype(str).isin(exclude)]
    merged = merged.sort_values(["score", "book_id"], ascending=[False, True], kind="stable").drop_duplicates("book_id").head(top_k)
    return books.merge(merged, on="book_id", how="inner").sort_values(
        ["score", "book_id"], ascending=[False, True], kind="stable").reset_index(drop=True)


def hybrid_recommend(books: pd.DataFrame, content_model: ContentRecommender,
                     collaborative_model: ItemCollaborativeRecommender,
                     user_id: str | None = None, favorite_ids: list[str] | None = None,
                     top_k: int = 10, alpha: float = 0.5,
                     recency_weight: float = 0.0,
                     liked_ids: list[str] | set[str] | None = None,
                     exclude_ids: list[str] | set[str] | None = None) -> pd.DataFrame:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer from 1 to 100")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(recency_weight, (int, float)) or not 0 <= recency_weight <= 1:
        raise ValueError("recency_weight must be a float between 0 and 1")
    candidate_k = min(100, max(10, top_k * 5))
    all_exclusions = set(map(str, exclude_ids or []))
    if liked_ids:
        all_exclusions.update(map(str, liked_ids))
    try:
        content = content_model.recommend(
            favorite_ids or [],
            candidate_k,
            exclude_ids=all_exclusions,
            recency_weight=recency_weight,
            liked_ids=liked_ids,
        )
    except TypeError:
        content = content_model.recommend(favorite_ids or [], candidate_k)
    collab = collaborative_model.recommend(user_id, candidate_k) if user_id else collaborative_model._empty()
    if content.empty and collab.empty and not user_id:
        return books.iloc[:0].assign(score=pd.Series(dtype=float), explanation=pd.Series(dtype=str))
    has_content = not content.empty and alpha > 0
    has_collab = not collab.empty and alpha < 1
    if has_content:
        merged = content[["book_id", "score"]].copy()
        if "feedback_boost" in content.columns:
            merged["feedback_boost"] = content["feedback_boost"]
        merged["content_score"] = normalize_scores(merged.score)
        merged = merged.drop(columns="score")
        if has_collab:
            collab_rank = collab[["book_id", "score"]].copy()
            collab_rank["collaborative_score"] = normalize_scores(collab_rank.score)
            collab_rank = collab_rank.drop(columns="score")
            merged = merged.merge(collab_rank, on="book_id", how="left")
            both = merged.collaborative_score.notna()
            # Content defines the candidate pool whenever taste picks exist. A missing
            # collaborative score therefore cannot lower a content candidate's rank.
            merged["score"] = merged.content_score
            merged.loc[both, "score"] = (
                alpha * merged.loc[both, "content_score"] +
                (1 - alpha) * merged.loc[both, "collaborative_score"]
            )
            year_col = books[["book_id", "year"]].drop_duplicates("book_id") if "year" in books.columns else pd.DataFrame(columns=["book_id", "year"])
            if not year_col.empty:
                merged = merged.merge(year_col, on="book_id", how="left")
                parsed_years = parse_publication_year(merged["year"])
                is_recent = parsed_years.ge(2015)
                has_fb = merged.get("feedback_boost", pd.Series(0, index=merged.index)).fillna(0).gt(0)
                merged["explanation"] = np.where(
                    both,
                    np.where(has_fb, "Catalogue similarity, reader ratings, and your liked feedback contributed.",
                             "Catalogue similarity and overlapping reader ratings both contributed."),
                    np.where(
                        is_recent,
                        "Recent publication matching your reading taste; no reader ratings in the historical Book-Crossing data.",
                        np.where(has_fb, "Boosted by positive feedback on your liked books.",
                                 "Catalogue metadata provides the match.")
                    )
                )
                merged = merged.drop(columns=["year"])
            else:
                has_fb = merged.get("feedback_boost", pd.Series(0, index=merged.index)).fillna(0).gt(0)
                merged["explanation"] = np.where(
                    both,
                    np.where(has_fb, "Catalogue similarity, reader ratings, and your liked feedback contributed.",
                             "Catalogue similarity and overlapping reader ratings both contributed."),
                    np.where(has_fb, "Boosted by positive feedback on your liked books.",
                             "Catalogue metadata provides the match.")
                )
        else:
            merged["score"] = merged.content_score
            year_col = books[["book_id", "year"]].drop_duplicates("book_id") if "year" in books.columns else pd.DataFrame(columns=["book_id", "year"])
            if not year_col.empty:
                merged = merged.merge(year_col, on="book_id", how="left")
                parsed_years = parse_publication_year(merged["year"])
                is_recent = parsed_years.ge(2015)
                has_fb = merged.get("feedback_boost", pd.Series(0, index=merged.index)).fillna(0).gt(0)
                merged["explanation"] = np.where(
                    is_recent,
                    "Recent publication matching your reading taste; no reader ratings in the historical Book-Crossing data.",
                    np.where(has_fb, "Boosted by positive feedback on your liked books.",
                             "Catalogue metadata provides the match.")
                )
                merged = merged.drop(columns=["year"])
            else:
                has_fb = merged.get("feedback_boost", pd.Series(0, index=merged.index)).fillna(0).gt(0)
                merged["explanation"] = np.where(
                    has_fb,
                    "Boosted by positive feedback on your liked books.",
                    "Catalogue metadata provides the match."
                )
        if "feedback_boost" in merged.columns:
            merged = merged.drop(columns=["feedback_boost"])
    elif user_id:
        # Offline evaluation found the sparse item-neighbour model unreliable. When
        # there is no usable metadata profile, fall back to real training rating
        # volume/mean rather than pretending noisy collaborative scores are strong.
        rated = collaborative_model.ratings[
            collaborative_model.ratings.book_id.astype(str).isin(collaborative_model.item_index)
        ]
        popularity = (rated.groupby("book_id").rating.agg(rating_count="count", avg_rating="mean")
                      .reset_index())
        popularity = popularity.merge(books[["book_id", "title"]], on="book_id", how="left")
        popularity = popularity.sort_values(["rating_count", "avg_rating", "title"],
                                            ascending=[False, False, True], kind="stable")
        exclude = set(map(str, favorite_ids or []))
        if liked_ids:
            exclude.update(map(str, liked_ids))
        if exclude_ids:
            exclude.update(map(str, exclude_ids))
        exclude.update(collaborative_model.books.iloc[
            list(collaborative_model.user_rated.get(str(user_id), set()))
        ].book_id.astype(str))
        merged = popularity[~popularity.book_id.astype(str).isin(exclude)].head(top_k).copy()
        merged["score"] = list(range(len(merged), 0, -1))
        merged["explanation"] = (
            "Rating overlap is too sparse to trust for this profile; this fallback uses "
            "training rating count and average rating."
        )
        merged["recommendation_source"] = "training rating fallback"
        merged = merged.sort_values(["score", "book_id"], ascending=[False, True], kind="stable")
        return books.merge(merged.drop(columns="title"), on="book_id", how="inner").reset_index(drop=True)
    else:
        return books.iloc[:0].assign(score=pd.Series(dtype=float), explanation=pd.Series(dtype=str))
    exclude = set(map(str, favorite_ids or []))
    if liked_ids:
        exclude.update(map(str, liked_ids))
    if exclude_ids:
        exclude.update(map(str, exclude_ids))
    if user_id:
        exclude.update(collaborative_model.books.iloc[list(collaborative_model.user_rated.get(str(user_id), set()))].book_id.astype(str))
    merged = merged[~merged.book_id.astype(str).isin(exclude)]
    merged = merged.sort_values(["score", "book_id"], ascending=[False, True], kind="stable").drop_duplicates("book_id").head(top_k)
    return (books.merge(merged, on="book_id", how="inner")
            .sort_values(["score", "book_id"], ascending=[False, True], kind="stable")
            .reset_index(drop=True))
