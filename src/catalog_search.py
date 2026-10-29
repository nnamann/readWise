import numpy as np
import pandas as pd


from .preprocess import parse_publication_year


def search_catalog(catalog: pd.DataFrame, title_query: str = "", author_query: str = "",
                   publisher_query: str = "", year_query: str = "",
                   sort_by: str = "Most rated") -> pd.DataFrame:
    """Filter catalogue records and apply deterministic relevance-aware sorting."""
    if sort_by not in {"Most rated", "Highest average", "Title A–Z", "Newest first"}:
        raise ValueError("Unknown catalogue sort order")
    result = catalog.copy()
    for column, query in (("title", title_query), ("author", author_query),
                          ("publisher", publisher_query), ("year", year_query)):
        values = result[column] if column in result else pd.Series("", index=result.index)
        if str(query).strip():
            matched = values.astype(str).str.contains(str(query).strip(), case=False, regex=False, na=False)
            if column == "title":
                authors = result["author"] if "author" in result else pd.Series("", index=result.index)
                matched |= authors.astype(str).str.contains(str(query).strip(), case=False, regex=False, na=False)
            result = result.loc[matched]
    if result.empty:
        return result.reset_index(drop=True)
    if str(title_query).strip():
        titles = result.title.fillna("").astype(str).str.casefold()
        term = str(title_query).strip().casefold()
        result = result.assign(_search_rank=np.select(
            [titles.eq(term), titles.str.startswith(term), titles.str.contains(term, regex=False)],
            [0, 1, 2], default=3))
    search_rank = ["_search_rank"] if "_search_rank" in result else []
    if sort_by == "Highest average":
        for column in ("avg_rating", "rating_count"):
            if column not in result:
                result[column] = np.nan if column == "avg_rating" else 0
        result = result.sort_values(search_rank + ["avg_rating", "rating_count", "title"],
                                    ascending=([True] if search_rank else []) + [False, False, True],
                                    na_position="last", kind="stable")
    elif sort_by == "Title A–Z":
        result = result.sort_values(search_rank + ["title"], ascending=True, kind="stable")
    elif sort_by == "Newest first":
        result["_clean_year"] = parse_publication_year(result["year"]) if "year" in result else np.nan
        if "rating_count" not in result:
            result["rating_count"] = 0
        result = result.sort_values(
            search_rank + ["_clean_year", "rating_count", "title"],
            ascending=([True] if search_rank else []) + [False, False, True],
            na_position="last",
            kind="stable",
        ).drop(columns=["_clean_year"])
    else:
        if "rating_count" not in result:
            result["rating_count"] = 0
        result = result.sort_values(search_rank + ["rating_count", "title"],
                                    ascending=([True] if search_rank else []) + [False, True],
                                    kind="stable")
    return result.drop(columns=["_search_rank"], errors="ignore").reset_index(drop=True)


def exclude_seen_editions(candidates: pd.DataFrame, catalog: pd.DataFrame,
                          seen_ids: set[str] | list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Hide selected/read ISBN editions and duplicate catalogue records for the same title-author work."""
    if candidates.empty:
        return candidates.copy().reset_index(drop=True)
    seen_ids = set(map(str, seen_ids))
    seen = catalog.loc[catalog.book_id.astype(str).isin(seen_ids)]

    def work_keys(frame: pd.DataFrame) -> pd.Series:
        title = frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
        author = frame.get("author", pd.Series("", index=frame.index)).fillna("").astype(str)
        title = title.str.casefold()
        # Strip only clear format labels; parenthetical subtitles can distinguish different works.
        title = title.str.replace(
            r"\s*\((?:paperback|hardcover|mass market paperback|large print|library binding|audio(?: book)?|ebook|e-book|reprint|second edition|third edition)\)",
            "", regex=True)
        title = title.str.replace(r"\s*(?:[:,-]\s*(?:a\s+)?novel|\(\s*a\s+novel\s*\))$", "", regex=True)
        title = title.str.replace(r"\s+", " ", regex=True).str.strip(" :,-")
        author = author.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
        return title + "::" + author

    seen_work_keys = work_keys(seen)
    seen_keys = set(seen_work_keys.loc[seen_work_keys.str.split("::").str[1].ne("")])
    candidate_keys = work_keys(candidates)
    same_ids = candidates.book_id.astype(str).isin(seen_ids)
    same_work = candidate_keys.isin(seen_keys)
    result = candidates.loc[~(same_ids | same_work)].copy()
    candidate_keys = candidate_keys.loc[result.index]
    known_author = candidate_keys.str.split("::").str[1].ne("")
    # Recommendation frames are already ranked; keep the highest-ranked edition per work.
    duplicate_work = known_author & candidate_keys.duplicated(keep="first")
    return result.loc[~duplicate_work].reset_index(drop=True)


def filter_feedback_recommendations(candidates: pd.DataFrame, catalog: pd.DataFrame,
                                    feedback_map: dict[str, str]) -> pd.DataFrame:
    """Hide books the reader passed on or has already read; keep liked books visible."""
    hidden_ids = {
        str(book_id) for book_id, feedback_type in feedback_map.items()
        if feedback_type in {"dislike", "read"}
    }
    return exclude_seen_editions(candidates, catalog, hidden_ids)
