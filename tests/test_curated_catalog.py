import pandas as pd
import pytest

from src.catalog_search import search_catalog
from src.content_based import ContentRecommender
from src.collaborative import ItemCollaborativeRecommender
from src.data_loader import DatasetError, load_curated_dataset
from scripts.build_catalog import build_report


def _write_raw(path):
    (path / "Books.csv").write_text(
        "book_id,title,author,publisher,year\n00001,Known Book,Ada,Press,2001\n",
        encoding="utf-8",
    )
    (path / "Ratings.csv").write_text(
        "user_id,book_id,rating\n0007,00001,8\n0008,00001,0\n", encoding="utf-8"
    )


def _write_additions(path, rows=None):
    frame = pd.DataFrame(rows or [{
        "book_id": "9780099282273", "title": "New Book", "author": "Bea",
        "publisher": "Verified Press", "year": "2020", "language": "English",
        "cover_url": "", "source_name": "Library", "source_url": "https://example.org/new",
        "curation_reason": "coverage",
    }])
    frame.to_csv(path, index=False)


def test_curated_load_is_reproducible_and_preserves_ratings_and_isbn_strings(tmp_path):
    _write_raw(tmp_path)
    additions = tmp_path / "additional.csv"
    _write_additions(additions)
    books_a, ratings_a = load_curated_dataset(tmp_path, additions)
    books_b, ratings_b = load_curated_dataset(tmp_path, additions)
    pd.testing.assert_frame_equal(books_a, books_b)
    pd.testing.assert_frame_equal(ratings_a, ratings_b)
    assert books_a.book_id.tolist() == ["00001", "9780099282273"]
    assert ratings_a.book_id.tolist() == ["00001", "00001"]
    assert books_a.loc[1, "book_id"] == "9780099282273"


def test_additions_are_searchable_and_content_model_compatible(tmp_path):
    _write_raw(tmp_path)
    additions = tmp_path / "additional.csv"
    _write_additions(additions)
    books, _ = load_curated_dataset(tmp_path, additions)
    assert search_catalog(books, author_query="Bea").title.tolist() == ["New Book"]
    assert search_catalog(books, title_query="New Book").title.tolist() == ["New Book"]
    model = ContentRecommender(books)
    assert "9780099282273" in model.index
    collaborative = ItemCollaborativeRecommender(books, pd.DataFrame({
        "user_id": ["u1"], "book_id": ["00001"], "rating": [8]
    }))
    assert "9780099282273" not in collaborative.recommend("u1").book_id.tolist()


def test_duplicate_ids_and_existing_title_author_raise(tmp_path):
    _write_raw(tmp_path)
    additions = tmp_path / "additional.csv"
    row = {"book_id": "9780099282273", "title": "New Book", "author": "Bea",
           "publisher": "P", "year": "2020", "language": "English", "source_name": "Library",
           "source_url": "https://example.org"}
    _write_additions(additions, [row, {**row, "title": "Second title"}])
    with pytest.raises(DatasetError, match="duplicate identifiers"):
        load_curated_dataset(tmp_path, additions)
    _write_additions(additions, [{**row, "title": "Known Book", "author": "Ada"}])
    with pytest.raises(DatasetError, match="title-author"):
        load_curated_dataset(tmp_path, additions)


def test_missing_source_url_is_rejected(tmp_path):
    _write_raw(tmp_path)
    additions = tmp_path / "additional.csv"
    row = {"book_id": "9780099282273", "title": "New Book", "author": "Bea",
           "publisher": "P", "year": "2020", "language": "English", "source_name": "Library",
           "source_url": ""}
    _write_additions(additions, [row])
    with pytest.raises(DatasetError, match="source URL"):
        load_curated_dataset(tmp_path, additions)


def test_invalid_supplemental_isbn_is_rejected(tmp_path):
    _write_raw(tmp_path)
    additions = tmp_path / "additional.csv"
    _write_additions(additions, [{
        "book_id": "9780000000001", "title": "New Book", "author": "Bea",
        "publisher": "P", "year": "2020", "language": "English",
        "source_name": "Library", "source_url": "https://example.org/new",
    }])
    with pytest.raises(DatasetError, match="valid ISBN"):
        load_curated_dataset(tmp_path, additions)


def test_report_marks_ratings_unchanged_and_additions_unrated(tmp_path):
    _write_raw(tmp_path)
    additions = tmp_path / "additional.csv"
    _write_additions(additions)
    report = build_report(tmp_path, additions, tmp_path / "no-evaluation")
    assert report["curated"]["rating_data_identical_to_source"] is True
    assert report["curated"]["books_without_explicit_rating_evidence"] == 1
    assert report["curated"]["verified_metadata_only_additions"] == 1
    assert report["evaluation"]["historical_outputs_preserved"] is True
    assert report["evaluation"]["new_metrics_computed"] is False
