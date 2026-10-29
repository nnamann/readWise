from pathlib import Path
import pathlib

import pandas as pd

from .config import RAW_DIR
from .config import DATA_DIR
from .covers import clean_text_encoding
from .identifier_reconciliation import reconcile_rating_ids


class DatasetError(ValueError):
    """Raised when source files are missing or structurally invalid."""


def _find_file(names: tuple[str, ...], data_dir: Path) -> Path:
    for directory in (data_dir, data_dir / "raw"):
        for name in names:
            path = directory / name
            if path.is_file():
                return path
    raise DatasetError(f"Missing {', '.join(names)} in {data_dir} or {data_dir / 'raw'}.")


def _is_needed_column(col_name: str) -> bool:
    clean = str(col_name).strip().strip('"').casefold().replace("_", "-")
    # Exclude columns never used by the recommender models, UI, or evaluation
    if clean in {"image-url-l", "location", "age"}:
        return False
    return True


def _read_table(path: Path, encoding: str = "latin-1") -> pd.DataFrame:
    # sep=None handles comma and semicolon formats with proper CSV quoting.
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding=encoding, dtype=str,
                           usecols=lambda c: _is_needed_column(c),
                           keep_default_na=False, on_bad_lines="skip")
    except (UnicodeDecodeError, UnicodeError):
        alt_encoding = "latin-1" if str(encoding).lower().startswith("utf") else "utf-8"
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=alt_encoding, dtype=str,
                               usecols=lambda c: _is_needed_column(c),
                               keep_default_na=False, on_bad_lines="skip")
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise DatasetError(f"Could not read {path.name}: {exc}") from exc
    except (OSError, pd.errors.ParserError) as exc:
        raise DatasetError(f"Could not read {path.name}: {exc}") from exc


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(c).strip().strip('"').casefold().replace("_", "-") for c in result.columns]
    return result


def load_dataset(data_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize book metadata and ratings without coercing IDs to numbers."""
    books_path = _find_file(("BX-Books.csv", "Books.csv"), data_dir)
    ratings_path = _find_file(("BX-Book-Ratings.csv", "Ratings.csv"), data_dir)
    books = _normalize_columns(_read_table(books_path))
    ratings = _normalize_columns(_read_table(ratings_path))

    def choose(frame: pd.DataFrame, *candidates: str) -> str:
        for name in candidates:
            if name.casefold() in frame.columns:
                return name.casefold()
        raise DatasetError(f"Required column missing: {' or '.join(candidates)}")

    isbn = choose(books, "isbn", "book-id")
    title = choose(books, "book-title", "title")
    author = next((x for x in ("book-author", "author") if x in books.columns), None)
    publisher = next((x for x in ("publisher",) if x in books.columns), None)
    year = next((x for x in ("year-of-publication", "year") if x in books.columns), None)
    cover = next((x for x in ("image-url-m", "image-url-s", "cover") if x in books.columns), None)
    books = books.rename(columns={isbn: "book_id", title: "title", author: "author"})
    books["book_id"] = books.book_id.astype(str).str.strip()
    books["title"] = books.title.astype(str).str.strip().map(clean_text_encoding)
    books["author"] = books["author"].astype(str).str.strip().map(clean_text_encoding) if "author" in books.columns else ""
    books["publisher"] = books[publisher].astype(str).str.strip().map(clean_text_encoding) if publisher else ""
    books["year"] = books[year].astype(str).str.strip() if year else ""
    books["cover_url"] = books[cover].astype(str).str.strip() if cover else ""
    books = books[books.book_id.ne("") & books.title.ne("") & books.title.str.casefold().ne("nan")]
    books = books.drop_duplicates("book_id", keep="first").reset_index(drop=True)
    if books.empty:
        raise DatasetError("The book catalogue contains no usable records.")

    uid = choose(ratings, "user-id", "user_id")
    rid = choose(ratings, "isbn", "book-id", "book_id")
    value = choose(ratings, "book-rating", "rating")
    ratings = ratings.rename(columns={uid: "user_id", rid: "book_id", value: "rating"})
    ratings["user_id"] = ratings.user_id.astype(str).str.strip()
    ratings["book_id"] = ratings.book_id.astype(str).str.strip()
    ratings["rating"] = pd.to_numeric(ratings.rating, errors="coerce")
    ratings = ratings.dropna(subset=["rating"])
    ratings = ratings[ratings.user_id.ne("") & ratings.book_id.ne("")]
    ratings = ratings[ratings.rating.between(0, 10) & ratings.rating.mod(1).eq(0)]
    ratings, _ = reconcile_rating_ids(books, ratings)
    # A user's last occurrence is retained deterministically; zeros remain recorded but are not explicit ratings.
    ratings = ratings.drop_duplicates(["user_id", "book_id"], keep="last").reset_index(drop=True)
    return books, ratings


def load_curated_dataset(data_dir: Path = RAW_DIR,
                         additions_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Book-Crossing plus verified metadata-only records; ratings stay raw and unchanged."""
    books, ratings = load_dataset(data_dir)
    path = additions_path or (DATA_DIR / "curated" / "additional_books.csv")
    if not path.is_file():
        return books, ratings
    additions = _normalize_columns(_read_table(path, encoding="utf-8"))
    required = {"book-id", "title", "author", "publisher", "year", "language", "source-name", "source-url"}
    missing = required - set(additions.columns)
    if missing:
        raise DatasetError(f"Curated additions missing columns: {', '.join(sorted(missing))}.")
    additions = additions.rename(columns={"book-id": "book_id", "source-name": "source_name",
                                          "source-url": "source_url"}).fillna("")
    additions["book_id"] = additions.book_id.astype(str).str.strip()
    additions["title"] = additions.title.astype(str).str.strip().map(clean_text_encoding)
    additions["author"] = additions.author.astype(str).str.strip().map(clean_text_encoding)
    if "publisher" in additions.columns:
        additions["publisher"] = additions.publisher.astype(str).str.strip().map(clean_text_encoding)
    if additions.book_id.eq("").any() or additions.title.eq("").any():
        raise DatasetError("Curated additions require non-empty identifiers and titles.")
    if additions.source_url.astype(str).str.strip().eq("").any():
        raise DatasetError("Every curated addition must include its bibliographic source URL.")
    if additions.source_name.astype(str).str.strip().eq("").any():
        raise DatasetError("Every curated addition must name its bibliographic source.")
    def valid_isbn(value: str) -> bool:
        value = str(value).replace("-", "").strip().upper()
        if len(value) == 13 and value.isdigit():
            return sum(int(char) * (1 if index % 2 == 0 else 3)
                       for index, char in enumerate(value)) % 10 == 0
        if len(value) == 10 and value[:9].isdigit() and (value[-1].isdigit() or value[-1] == "X"):
            return sum((10 - index) * (10 if char == "X" else int(char))
                       for index, char in enumerate(value)) % 11 == 0
        return False
    if not additions.book_id.map(valid_isbn).all():
        raise DatasetError("Every supplemental book identifier must be a valid ISBN-10 or ISBN-13.")
    if additions.book_id.duplicated().any():
        raise DatasetError("Curated additions contain duplicate identifiers.")
    collisions = set(additions.book_id) & set(books.book_id.astype(str))
    if collisions:
        raise DatasetError(f"Curated identifiers already exist in the source catalogue: {', '.join(sorted(collisions)[:5])}.")
    # Match title + author conservatively; never collapse editions on title alone.
    def work_key(frame):
        title_key = frame.title.astype(str).str.casefold().str.replace(r"[^\w]+", " ", regex=True).str.strip()
        author_key = frame.author.astype(str).str.casefold().str.replace(r"[^\w]+", " ", regex=True).str.strip()
        return title_key + "::" + author_key
    duplicate_works = set(work_key(books)) & set(work_key(additions))
    if duplicate_works:
        raise DatasetError("Curated additions duplicate an existing normalized title-author record.")
    for column in ("publisher", "year", "cover_url"):
        if column not in books.columns:
            books[column] = ""
        if column not in additions.columns:
            additions[column] = ""
    keep_columns = list(dict.fromkeys([*books.columns, *additions.columns]))
    books = pd.concat([books.reindex(columns=keep_columns), additions.reindex(columns=keep_columns)], ignore_index=True)
    # Load recent Open Library books and merge if the filtered JSONL file exists
    recent_path = pathlib.Path("data/books_2020_2024.jsonl")
    if recent_path.is_file():
        from .load_recent_books import load_recent_books
        recent_df = load_recent_books(str(recent_path))
        recent_df = recent_df.reindex(columns=keep_columns, fill_value="")
        books = pd.concat([books, recent_df], ignore_index=True)
    return books, ratings
