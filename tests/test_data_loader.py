import pandas as pd
import pytest

from src.data_loader import DatasetError, load_dataset


def test_semicolon_latin1_quotes_and_isbn(tmp_path):
    (tmp_path / "BX-Books.csv").write_bytes('"ISBN";"Book-Title";"Book-Author";"Publisher"\n"00001";"Café; stories";"Ana";"Éditions"\n"00002";"";"No title";"X"\n'.encode("latin-1"))
    (tmp_path / "BX-Book-Ratings.csv").write_text('"User-ID";"ISBN";"Book-Rating"\n"01";"00001";"0"\n"02";"00001";"9"\n"03";"00001";"9.5"\n', encoding="latin-1")
    books, ratings = load_dataset(tmp_path)
    assert books.book_id.tolist() == ["00001"]
    assert books.title.iloc[0] == "Café; stories"
    assert ratings.rating.tolist() == [0, 9]
    assert ratings.user_id.tolist() == ["01", "02"]


def test_missing_schema_raises(tmp_path):
    (tmp_path / "Books.csv").write_text("title,author\nA,B\n")
    (tmp_path / "Ratings.csv").write_text("user_id,book_id,rating\nu,1,8\n")
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)


def test_author_is_optional_and_empty_catalogue_is_rejected(tmp_path):
    (tmp_path / "Books.csv").write_text("book_id,title,publisher\n001,The Book,Press\n", encoding="utf-8")
    (tmp_path / "Ratings.csv").write_text("user_id,book_id,rating\nu,001,8\n", encoding="utf-8")
    books, _ = load_dataset(tmp_path)
    assert books.loc[0, "author"] == ""

    (tmp_path / "Books.csv").write_text("book_id,title,publisher\n001,,Press\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no usable records"):
        load_dataset(tmp_path)


def test_rating_isbn_format_repair_preserves_original_book_identifier(tmp_path):
    (tmp_path / "Books.csv").write_text(
        "book_id,title,author\n014025448X,Edition,Writer\n", encoding="utf-8"
    )
    (tmp_path / "Ratings.csv").write_text(
        "user_id,book_id,rating\nu1,01402.5448x,9\n", encoding="utf-8"
    )
    books, ratings = load_dataset(tmp_path)
    assert books.book_id.tolist() == ["014025448X"]
    assert ratings.book_id.tolist() == ["014025448X"]
    assert ratings.rating.tolist() == [9]
