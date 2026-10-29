import json
import pathlib
import pandas as pd


def load_recent_books(
    jsonl_path: str = "data/books_2020_2024.jsonl",
) -> pd.DataFrame:
    """Load the filtered Open Library recent‑books JSON‑Lines file.

    The output DataFrame contains the columns expected by the rest of
    the ReadWise pipeline:
        - ``book_id``   – a stable identifier derived from the Open Library work key.
        - ``title``
        - ``author``   – first author name or a comma‑separated list.
        - ``year``     – integer publication year (or ``None`` if missing).
        - ``cover_url`` – full URL to a cover image (or empty string).
    """
    path = pathlib.Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Recent books file not found: {jsonl_path}")

    records = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            rec = json.loads(line)
            # Build a stable book_id from the Open Library key
            book_id = rec.get("key", "").replace("/works/", "OL") + "W"
            title = rec.get("title", "").strip()
            # authors is a list of dicts; join names
            authors = rec.get("authors", [])
            if isinstance(authors, list) and authors:
                author_names = [a.get("name", "").strip() for a in authors if a.get("name")]
                author = ", ".join(author_names)
            else:
                author = ""
            year = rec.get("first_publish_year")
            cover_url = rec.get("cover_url", "")
            records.append(
                {
                    "book_id": book_id,
                    "title": title,
                    "author": author,
                    "year": int(year) if year else None,
                    "cover_url": cover_url,
                }
            )
    df = pd.DataFrame.from_records(records)
    # Drop rows without a title – they are not useful for recommendations
    df = df[df.title.astype(bool)].reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_recent_books()
    print(f"Loaded {len(df)} recent books")
    # Optional quick export to CSV for inspection
    df.to_csv("data/recent_books.csv", index=False)
