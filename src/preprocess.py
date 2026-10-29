import re

import numpy as np
import pandas as pd


def prepare_books(books: pd.DataFrame) -> pd.DataFrame:
    result = books.copy()
    for col in ("title", "author", "publisher"):
        if col not in result:
            result[col] = ""
        result[col] = result[col].fillna("").astype(str).str.strip()
    result["metadata_text"] = (result.title + " " + result.author + " " + result.publisher).map(
        lambda value: re.sub(r"\s+", " ", value).strip()
    )
    return result


def parse_publication_year(years: pd.Series | str | int | float | None) -> pd.Series | float:
    """Safely and honestly extract publication years as numeric values in [1800, 2026].
    
    Returns np.nan for missing, zero, invalid, or out-of-range years.
    Does not invent publication years.
    """
    if isinstance(years, pd.Series):
        s = years.fillna("").astype(str).str.strip()
        # Extract 4-digit years
        extracted = s.str.extract(r"(\b\d{4}\b)", expand=False)
        numeric = pd.to_numeric(extracted, errors="coerce")
        invalid = numeric.isna() | numeric.lt(1800) | numeric.gt(2026) | s.isin({"0", "nan", "none", "null", ""})
        numeric[invalid] = np.nan
        return numeric
    # Scalar handling
    val_str = str(years or "").strip()
    match = re.search(r"\b(\d{4})\b", val_str)
    if not match or val_str in {"0", "nan", "none", "null", ""}:
        return float("nan")
    val = float(match.group(1))
    return val if 1800 <= val <= 2026 else float("nan")


def compute_recency_score(years: pd.Series, base_year: int = 1970, current_year: int = 2026) -> np.ndarray:
    """Compute a normalized recency signal in [0.0, 1.0].
    
    Books published before base_year receive 0.0.
    Books published between base_year and current_year scale smoothly from 0.0 to 1.0.
    Missing or uncertain years receive 0.0 without penalizing relevance.
    """
    numeric = parse_publication_year(years)
    scores = np.zeros(len(numeric), dtype=float)
    valid = numeric.notna() & numeric.ge(base_year) & numeric.le(current_year)
    span = max(1, current_year - base_year)
    scores[valid] = (numeric[valid] - base_year) / span
    return scores


def explicit_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Return genuine nonzero ratings; zero-valued BX interactions are implicit only."""
    result = ratings.loc[ratings.rating.gt(0)].copy()
    return result.drop_duplicates(["user_id", "book_id"], keep="last").reset_index(drop=True)
