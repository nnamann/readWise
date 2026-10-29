import pandas as pd

from src.catalog_search import exclude_seen_editions, search_catalog


def test_search_matches_partial_title_author_and_publisher_case_insensitively():
    books = pd.DataFrame({
        "book_id": ["1", "2", "3"],
        "title": ["The Quiet River", "Quietly Brilliant", "Garden Notes"],
        "author": ["A. Reader", "B. Writer", "A. Reader"],
        "publisher": ["North Press", "South Press", "North Press"],
        "year": ["2001", "1998", "2001"],
        "rating_count": [2, 10, 7],
    })
    assert search_catalog(books, title_query="QUIET").book_id.tolist() == ["2", "1"]
    assert search_catalog(books, title_query="Writer").book_id.tolist() == ["2"]
    assert search_catalog(books, author_query="reader").book_id.tolist() == ["3", "1"]
    assert search_catalog(books, publisher_query="north", sort_by="Title A–Z").book_id.tolist() == ["3", "1"]
    assert search_catalog(books, year_query="2001").book_id.tolist() == ["3", "1"]
    assert search_catalog(books, title_query="not a real book").empty


def test_title_relevance_precedes_popularity_but_sort_controls_ties():
    books = pd.DataFrame({"book_id": ["1", "2", "3"],
                          "title": ["A Story Within", "Story of a River", "Story"],
                          "author": ["A", "B", "C"],
                          "rating_count": [100, 5, 1]})
    assert search_catalog(books, title_query="Story").book_id.tolist() == ["3", "2", "1"]


def test_missing_optional_metadata_does_not_break_search():
    books = pd.DataFrame({"book_id": ["1"], "title": ["One Book"]})
    assert search_catalog(books, author_query="any author").empty
    assert search_catalog(books).book_id.tolist() == ["1"]


def test_excluding_seen_books_also_hides_duplicate_editions_of_the_same_work():
    catalog = pd.DataFrame({
        "book_id": ["seen", "edition", "other"],
        "title": ["The Quiet River (River Stories (Paperback))",
                  " the   quiet river (River Stories (Hardcover)) ", "Another River"],
        "author": ["A. Reader", "a. reader", "B. Writer"],
    })
    candidates = catalog.iloc[[1, 2]].copy()
    assert exclude_seen_editions(candidates, catalog, {"seen"}).book_id.tolist() == ["other"]


def test_distinct_companion_titles_are_not_mistaken_for_duplicate_editions():
    catalog = pd.DataFrame({
        "book_id": ["seen", "companion", "other"],
        "title": ["The Lovely Bones: A Novel",
                  "The Lovely Bones (Reader's Companion)", "Lucky"],
        "author": ["Alice Sebold", "Alice Sebold", "Alice Sebold"],
    })
    candidates = catalog.iloc[[1, 2]].copy()
    assert exclude_seen_editions(candidates, catalog, {"seen"}).book_id.tolist() == ["companion", "other"]


def test_format_labels_are_removed_when_comparing_editions():
    catalog = pd.DataFrame({
        "book_id": ["seen", "edition"],
        "title": ["The Quiet River (Paperback)", "The Quiet River (Hardcover)"],
        "author": ["A. Reader", "A. Reader"],
    })
    assert exclude_seen_editions(catalog.iloc[[1]], catalog, {"seen"}).empty


def test_recommendation_list_collapses_duplicate_editions_even_without_seen_books():
    candidates = pd.DataFrame({
        "book_id": ["top", "edition", "different"],
        "title": ["The Same Novel: A Novel", "The Same Novel (Paperback)", "Another Story"],
        "author": ["A. Writer", "A. Writer", "B. Writer"],
        "score": [0.9, 0.8, 0.7],
    })
    actual = exclude_seen_editions(candidates, candidates, set())
    assert actual.book_id.tolist() == ["top", "different"]
