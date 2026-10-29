"""Focused tests for recency signals, contemporary catalog additions, and optimized loading."""

import numpy as np
import pandas as pd
import pytest

from src.catalog_search import search_catalog
from src.collaborative import ItemCollaborativeRecommender
from src.content_based import ContentRecommender
from src.data_loader import _is_needed_column, load_curated_dataset
from src.hybrid import hybrid_recommend
from src.preprocess import compute_recency_score, parse_publication_year


def test_parse_publication_year_honest_validation():
    # Valid years
    assert parse_publication_year("2024") == 2024.0
    assert parse_publication_year(1995) == 1995.0
    assert parse_publication_year("Published in 2021") == 2021.0

    # Invalid, zero, or uncertain years return NaN
    assert np.isnan(parse_publication_year("0"))
    assert np.isnan(parse_publication_year(""))
    assert np.isnan(parse_publication_year("nan"))
    assert np.isnan(parse_publication_year("NULL"))
    assert np.isnan(parse_publication_year("1388"))  # ancient / typo
    assert np.isnan(parse_publication_year("2050"))  # future typo

    # Series parsing
    s = pd.Series(["2023", "0", "1984", "unknown", "2026", "2999"])
    parsed = parse_publication_year(s)
    assert parsed.iloc[0] == 2023.0
    assert np.isnan(parsed.iloc[1])
    assert parsed.iloc[2] == 1984.0
    assert np.isnan(parsed.iloc[3])
    assert parsed.iloc[4] == 2026.0
    assert np.isnan(parsed.iloc[5])


def test_compute_recency_score_scaling():
    years = pd.Series(["2024", "2020", "2000", "1970", "1950", "0", "invalid"])
    scores = compute_recency_score(years)
    assert scores[0] > scores[1] > scores[2] > scores[3]
    assert scores[3] == 0.0  # base year 1970
    assert scores[4] == 0.0  # pre-1970
    assert scores[5] == 0.0  # missing
    assert scores[6] == 0.0  # invalid
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_content_recency_boost_surfaces_recent_book_without_overwhelming_relevance():
    books = pd.DataFrame({
        "book_id": ["seed", "old_match", "new_match", "irrelevant_new"],
        "title": ["Quantum Computing Basics", "Quantum Physics Guide", "Quantum Computing in 2024", "Baking Artisan Bread"],
        "author": ["Physicist Alan", "Physicist Alan", "Physicist Alan", "Chef Baker"],
        "publisher": ["Tech Publishing", "Tech Publishing", "Tech Publishing", "Culinary House"],
        "year": ["2010", "1990", "2024", "2024"],
    })
    model = ContentRecommender(books)

    # Without recency boost (pure TF-IDF cosine)
    res_raw = model.recommend("seed", top_k=3, recency_weight=0.0)
    assert "seed" not in res_raw.book_id.tolist()
    assert "irrelevant_new" not in res_raw.book_id.tolist()  # Zero match is excluded

    # With recency boost
    res_boosted = model.recommend("seed", top_k=3, recency_weight=0.25)
    assert res_boosted.iloc[0].book_id == "new_match"  # recent title boosted above old match
    assert "irrelevant_new" not in res_boosted.book_id.tolist()  # Irrelevant item NEVER surfaces
    assert "recency_score" in res_boosted.columns


def test_search_catalog_newest_first_sort():
    books = pd.DataFrame({
        "book_id": ["1", "2", "3", "4"],
        "title": ["Old Classic", "Modern Bestseller", "Ancient Tale", "No Year Book"],
        "author": ["Author A", "Author B", "Author C", "Author D"],
        "year": ["1980", "2023", "1950", ""],
        "rating_count": [100, 10, 5, 50],
    })
    sorted_df = search_catalog(books, sort_by="Newest first")
    assert sorted_df.book_id.tolist() == ["2", "1", "3", "4"]


def test_hybrid_recommends_recent_unrated_book_with_transparent_explanation():
    books = pd.DataFrame({
        "book_id": ["seed", "recent_book", "rated_old"],
        "title": ["Artificial Intelligence", "Artificial Intelligence: Modern Principles", "Artificial Intelligence 1990"],
        "author": ["Dr. Smith", "Dr. Smith", "Dr. Smith"],
        "publisher": ["AI Press", "AI Press", "AI Press"],
        "year": ["2015", "2023", "1990"],
    })
    ratings = pd.DataFrame({
        "user_id": ["u1", "u1"],
        "book_id": ["seed", "rated_old"],
        "rating": [9, 8],
    })
    content = ContentRecommender(books)
    collab = ItemCollaborativeRecommender(books, ratings)

    res = hybrid_recommend(books, content, collab, user_id="u1", favorite_ids=["seed"],
                           top_k=2, alpha=0.5, recency_weight=0.2)
    assert "recent_book" in res.book_id.tolist()
    recent_row = res[res.book_id == "recent_book"].iloc[0]
    assert "Recent publication" in recent_row.explanation
    assert "no reader ratings in the historical Book-Crossing data" in recent_row.explanation


def test_column_filter_skips_unused_columns():
    assert _is_needed_column("ISBN")
    assert _is_needed_column("Book-Title")
    assert _is_needed_column("Year-Of-Publication")
    assert _is_needed_column("Image-URL-M")
    # Unused columns
    assert not _is_needed_column("Image-URL-L")
    assert not _is_needed_column("Location")
    assert not _is_needed_column("Age")


def test_curated_contemporary_dataset_has_valid_recent_metadata():
    books, ratings = load_curated_dataset()
    curated_recent = books.loc[
        books.get("source_name", pd.Series("", index=books.index)).eq("Open Library") &
        parse_publication_year(books.year).ge(2015)
    ]
    assert len(curated_recent) >= 20  # rich collection of contemporary additions
    assert "The Midnight Library" in curated_recent.title.values
    assert "Project Hail Mary" in curated_recent.title.values
    assert "Klara and the Sun" in curated_recent.title.values
    assert "Yellowface" in curated_recent.title.values
    # Curated recent books have zero fabricated ratings in the ratings table
    assert not set(curated_recent.book_id) & set(ratings.book_id)
