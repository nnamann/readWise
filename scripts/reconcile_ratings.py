"""Create a row-level audit of raw ratings and conservative ISBN-to-book linking."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, RAW_DIR
from src.data_loader import _find_file, _normalize_columns, _read_table, load_curated_dataset
from src.identifier_reconciliation import reconcile_rating_ids


def reconcile(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    books, _ = load_curated_dataset(raw_dir)
    raw = _normalize_columns(_read_table(_find_file(("BX-Book-Ratings.csv", "Ratings.csv"), raw_dir)))

    def select(*choices: str) -> str:
        return next(name for name in choices if name in raw.columns)

    user_col = select("user-id", "user_id")
    isbn_col = select("isbn", "book-id", "book_id")
    rating_col = select("book-rating", "rating")
    rows = pd.DataFrame({
        "parsed_row_number": range(1, len(raw) + 1),
        "raw_user_id": raw[user_col].astype(str).str.strip(),
        "raw_book_id": raw[isbn_col].astype(str).str.strip(),
        "raw_rating": raw[rating_col].astype(str).str.strip(),
    })
    numeric = pd.to_numeric(rows.raw_rating, errors="coerce")
    loader_valid = (numeric.notna() & rows.raw_user_id.ne("") & rows.raw_book_id.ne("") &
                    numeric.between(0, 10) & numeric.mod(1).eq(0))
    mapped, identifier_audit = reconcile_rating_ids(books, rows.rename(columns={"raw_book_id": "book_id"})[["book_id"]])
    rows["resolved_book_id"] = mapped.book_id.astype(str).values
    rows["identifier_status"] = identifier_audit.status.values
    rows["loader_valid"] = loader_valid
    rows["rating"] = numeric
    known_ids = set(books.book_id.astype(str))
    keys = pd.DataFrame({"user": rows.raw_user_id, "book": rows.resolved_book_id})
    last_occurrence = pd.Series(False, index=rows.index)
    last_occurrence.loc[loader_valid] = ~keys.loc[loader_valid].duplicated(keep="last")
    model_status = pd.Series("included_explicit_model", index=rows.index, dtype=object)
    model_status.loc[~loader_valid] = "excluded_invalid_rating_or_identifier"
    valid_unresolved = loader_valid & ~rows.resolved_book_id.isin(known_ids)
    model_status.loc[valid_unresolved] = "unresolved_no_catalogue_record"
    model_status.loc[loader_valid & ~last_occurrence] = "superseded_duplicate_user_book"
    model_status.loc[loader_valid & last_occurrence & numeric.eq(0)] = "zero_implicit_not_explicit_rating"
    rows["model_status"] = model_status
    unresolved_explicit = rows.model_status.eq("unresolved_no_catalogue_record") & rows.rating.gt(0)
    recovered_explicit = rows.identifier_status.eq("repaired_format") & rows.model_status.eq("included_explicit_model")
    summary = {
        "raw_rating_rows": int(len(rows)),
        "valid_rows_after_loader_rules": int(loader_valid.sum()),
        "identifier_status_counts": {str(k): int(v) for k, v in rows.identifier_status.value_counts().items()},
        "model_status_counts": {str(k): int(v) for k, v in rows.model_status.value_counts().items()},
        "explicit_unresolved_rows_after_format_repairs_and_dedup": int(unresolved_explicit.sum()),
        "explicit_unresolved_unique_ids": int(rows.loc[unresolved_explicit, "raw_book_id"].nunique()),
        "explicit_unresolved_valid_isbn_rows": int((unresolved_explicit & rows.identifier_status.eq("valid_isbn_missing_book")).sum()),
        "explicit_unresolved_invalid_or_non_isbn_rows": int((unresolved_explicit & rows.identifier_status.eq("invalid_or_non_isbn")).sum()),
        "recovered_explicit_rows_used_by_model": int(recovered_explicit.sum()),
        "recovered_explicit_unique_ids": int(rows.loc[recovered_explicit, "raw_book_id"].nunique()),
        "raw_identifier_format_repaired_rows": int(rows.identifier_status.eq("repaired_format").sum()),
        "identifier_normalization": "Strip punctuation and uppercase X only for checksum-valid ISBN-10/13 values. Apply a repair only when the normalized identifier maps to one unique catalogue ISBN. Original IDs remain in raw_book_id.",
    }
    return rows.drop(columns="rating"), summary


def main() -> None:
    audit, summary = reconcile()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_DIR / "rating_reconciliation_rows.csv", index=False)
    (OUTPUT_DIR / "rating_reconciliation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote row-level rating audit: {OUTPUT_DIR / 'rating_reconciliation_rows.csv'}")


if __name__ == "__main__":
    main()
