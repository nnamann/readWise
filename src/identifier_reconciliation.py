"""Conservative ISBN normalisation for linking rating rows to catalogue editions."""
import re

import pandas as pd


def canonical_isbn(value: object) -> str:
    """Remove presentation punctuation while preserving leading zeroes and ISBN X."""
    return re.sub(r"[^0-9Xx]", "", str(value)).upper()


def is_valid_isbn(value: object) -> bool:
    value = canonical_isbn(value)
    if len(value) == 13 and value.isdigit():
        return sum(int(char) * (1 if i % 2 == 0 else 3)
                   for i, char in enumerate(value)) % 10 == 0
    if len(value) == 10 and value[:9].isdigit() and (value[-1].isdigit() or value[-1] == "X"):
        return sum((10 - i) * (10 if char == "X" else int(char))
                   for i, char in enumerate(value)) % 11 == 0
    return False


def reconcile_rating_ids(books: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repair only checksum-valid, uniquely matching formatted ISBNs; return a row audit."""
    book_ids = books.book_id.astype(str).str.strip()
    canonical_to_ids: dict[str, set[str]] = {}
    for book_id in book_ids:
        if is_valid_isbn(book_id):
            canonical_to_ids.setdefault(canonical_isbn(book_id), set()).add(book_id)
    ratings = ratings.copy()
    original_ids = ratings.book_id.astype(str).str.strip()
    known_ids = set(book_ids)
    id_resolution: dict[str, tuple[str, str]] = {}
    for raw_id in original_ids.drop_duplicates():
        if raw_id in known_ids:
            id_resolution[raw_id] = (raw_id, "matched_exact")
        elif is_valid_isbn(raw_id) and len(canonical_to_ids.get(canonical_isbn(raw_id), set())) == 1:
            id_resolution[raw_id] = (next(iter(canonical_to_ids[canonical_isbn(raw_id)])), "repaired_format")
        elif is_valid_isbn(raw_id):
            id_resolution[raw_id] = (raw_id, "valid_isbn_missing_book")
        else:
            id_resolution[raw_id] = (raw_id, "invalid_or_non_isbn")
    repaired = original_ids.map({key: value[0] for key, value in id_resolution.items()})
    statuses = original_ids.map({key: value[1] for key, value in id_resolution.items()})
    ratings["book_id"] = repaired
    audit = pd.DataFrame({"raw_book_id": original_ids, "resolved_book_id": repaired, "status": statuses})
    return ratings, audit
