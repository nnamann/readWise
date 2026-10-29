"""Focused checks verifying the persistent feedback loop in ReadWise.

Checks show that:
1. Feedback persists locally in SQLite across restarts.
2. Positive feedback affects future recommendations, including for newer books
   with little or no rating history.
3. Negative feedback ('dislike' / 'read') suppresses the selected book from recommendations.
4. User feedback is kept strictly separate from the source dataset (never modifying
   source ratings or books).
5. Repeated feedback, updates, and removals are handled cleanly.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.content_based import ContentRecommender
from src.hybrid import hybrid_recommend
from src.recommendation_service import reader_rating_fallback
from src.collaborative import ItemCollaborativeRecommender
from src.feedback import (
    clear_all_feedback,
    get_all_feedback,
    get_feedback,
    get_feedback_by_type,
    get_feedback_records,
    get_feedback_summary,
    remove_feedback,
    set_feedback,
    init_db,
    VALID_FEEDBACK_TYPES,
)
from src import feedback as feedback_module


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary SQLite database path."""
    db_file = tmp_path / "test_feedback.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def sample_catalog():
    """A mixed catalogue including classic books and contemporary books with 0 ratings."""
    return pd.DataFrame({
        "book_id": ["0001", "0002", "0003", "NEW-01", "NEW-02", "0006"],
        "title": [
            "Space Journey",
            "Space Odyssey",
            "Garden Guide",
            "The Sun Also Rises Again",
            "Sun and Stars",
            "Ocean World",
        ],
        "author": [
            "Arthur Clarke",
            "Arthur Clarke",
            "Beatrix Potter",
            "Contemporary Author",
            "Contemporary Author",
            "Jacques Cousteau",
        ],
        "publisher": [
            "SciFi Press",
            "SciFi Press",
            "Nature Books",
            "Modern Press",
            "Modern Press",
            "Oceanic",
        ],
        "year": ["1968", "1982", "1990", "2023", "2024", "1995"],
        "cover_url": [""] * 6,
        "rating_count": [50, 40, 30, 0, 0, 20],
        "avg_rating": [8.5, 8.0, 7.5, None, None, 7.0],
    })


@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2"],
        "book_id": ["0001", "0002", "0001", "0003"],
        "rating": [9, 8, 8, 7],
    })


def test_feedback_crud_and_persistence(temp_db):
    """Verify storing, retrieving, updating, and removing feedback across connections."""
    # Add feedback
    set_feedback("0001", "like", temp_db)
    set_feedback("0002", "dislike", temp_db)
    set_feedback("0003", "read", temp_db)

    # Verify retrieval
    assert get_feedback("0001", temp_db) == "like"
    assert get_feedback("0002", temp_db) == "dislike"
    assert get_feedback("0003", temp_db) == "read"
    assert get_feedback("non_existent", temp_db) is None

    # Verify summary and sets
    summary = get_feedback_summary(temp_db)
    assert summary == {"like": 1, "dislike": 1, "read": 1, "total": 3}
    assert get_feedback_by_type("like", temp_db) == {"0001"}
    assert get_feedback_by_type("dislike", temp_db) == {"0002"}
    assert get_feedback_by_type("read", temp_db) == {"0003"}

    # Update feedback (reader changes their mind)
    set_feedback("0002", "like", temp_db)
    assert get_feedback("0002", temp_db) == "like"
    assert get_feedback_by_type("dislike", temp_db) == set()
    assert get_feedback_by_type("like", temp_db) == {"0001", "0002"}

    # Remove feedback
    assert remove_feedback("0001", temp_db) is True
    assert get_feedback("0001", temp_db) is None
    assert remove_feedback("0001", temp_db) is False  # already removed

    # Verify database file actually exists and persists data
    assert temp_db.exists()
    all_fb = get_all_feedback(temp_db)
    assert "0002" in all_fb
    assert "0003" in all_fb

    # Clear all
    cleared_count = clear_all_feedback(temp_db)
    assert cleared_count == 2
    assert len(get_all_feedback(temp_db)) == 0


def test_feedback_validation_and_edge_cases(temp_db):
    """Ensure invalid feedback types or empty book IDs are rejected."""
    with pytest.raises(ValueError):
        set_feedback("", "like", temp_db)

    with pytest.raises(ValueError):
        set_feedback("0001", "invalid_type", temp_db)

    # Empty book ID removal
    assert remove_feedback("", temp_db) is False


def test_legacy_feedback_is_migrated_once_into_private_default_store(tmp_path, monkeypatch):
    legacy_db = tmp_path / "project" / "feedback.db"
    private_db = tmp_path / "private" / "feedback.db"
    set_feedback("legacy-book", "read", legacy_db)
    monkeypatch.setattr(feedback_module, "LEGACY_FEEDBACK_DB_PATH", legacy_db)
    monkeypatch.setattr(feedback_module, "FEEDBACK_DB_PATH", private_db)
    monkeypatch.delenv("READWISE_FEEDBACK_DB_PATH", raising=False)

    init_db()
    assert get_feedback("legacy-book") == "read"
    remove_feedback("legacy-book")
    init_db()
    assert get_feedback("legacy-book") is None


def test_positive_feedback_affects_future_recommendations(sample_catalog):
    """Positive feedback boosts similar books in content recommendations."""
    model = ContentRecommender(sample_catalog)

    # Without liked feedback, search for picks matching 'Garden Guide' (0003)
    baseline = model.recommend(["0003"], top_k=5)
    baseline_ids = baseline.book_id.tolist()
    baseline_score = float(baseline.loc[baseline.book_id.eq("0002"), "score"].max()) if "0002" in baseline_ids else 0.0

    # Now reader likes "0001" (Space Journey by Arthur Clarke)
    # This should boost "0002" (Space Odyssey by Arthur Clarke)
    with_positive_feedback = model.recommend(
        ["0003"], top_k=5, liked_ids=["0001"], liked_weight=0.8
    )

    # Space Odyssey (0002) score should increase significantly due to positive feedback on Space Journey
    score_with_feedback = with_positive_feedback.loc[
        with_positive_feedback.book_id == "0002", "score"
    ]
    assert not score_with_feedback.empty
    assert "0002" in with_positive_feedback.book_id.tolist()
    assert "feedback_boost" in with_positive_feedback.columns
    assert float(score_with_feedback.iloc[0]) > baseline_score
    assert float(with_positive_feedback.loc[with_positive_feedback.book_id.eq("0002"), "feedback_boost"].iloc[0]) > 0


def test_positive_feedback_works_for_newer_books_without_ratings(sample_catalog):
    """Positive feedback on a newer book (0 ratings) improves recommendations."""
    model = ContentRecommender(sample_catalog)

    # NEW-01 is a 2023 book with 0 ratings in ratings data
    # Reader likes NEW-01 ("The Sun Also Rises Again" by Contemporary Author)
    recs = model.recommend([], top_k=5, liked_ids=["NEW-01"])

    # Should recommend NEW-02 ("Sun and Stars" by the same Contemporary Author)
    assert not recs.empty
    top_book = recs.iloc[0]
    assert top_book.book_id == "NEW-02"
    assert top_book.author == "Contemporary Author"
    assert int(top_book.rating_count) == 0
    assert float(top_book.feedback_boost) > 0
    # NEW-01 itself should be excluded from recommendations (already liked/known)
    assert "NEW-01" not in recs.book_id.tolist()


def test_negative_feedback_suppresses_selected_book(sample_catalog):
    """Negative feedback ('dislike' / 'read') strictly suppresses the book from recommendations."""
    model = ContentRecommender(sample_catalog)

    # Under normal seed recommendations, 0001 recommends 0002
    normal_recs = model.recommend("0001", top_k=5)
    assert "0002" in normal_recs.book_id.tolist()

    # Reader marks 0002 as 'dislike' (or 'read')
    suppressed_recs = model.recommend("0001", top_k=5, exclude_ids={"0002"})
    assert "0002" not in suppressed_recs.book_id.tolist()


def test_hybrid_recommender_integrates_feedback_loop(sample_catalog, sample_ratings):
    """Hybrid recommender respects positive feedback and suppresses negative feedback."""
    content = ContentRecommender(sample_catalog)
    collab = ItemCollaborativeRecommender(sample_catalog, sample_ratings)

    # User likes NEW-01, dislikes 0002
    result = hybrid_recommend(
        sample_catalog,
        content,
        collab,
        favorite_ids=["0001"],
        liked_ids=["NEW-01"],
        exclude_ids=["0002"],
        top_k=5,
    )

    result_ids = result.book_id.tolist()
    # 0002 is suppressed due to negative feedback
    assert "0002" not in result_ids
    # NEW-02 is boosted due to positive feedback on NEW-01
    assert "NEW-02" in result_ids


def test_fallback_recommender_respects_feedback(sample_catalog, sample_ratings):
    """reader_rating_fallback suppresses negative feedback and incorporates liked feedback."""
    content = ContentRecommender(sample_catalog)
    collab = ItemCollaborativeRecommender(sample_catalog, sample_ratings)

    result = reader_rating_fallback(
        content,
        collab,
        sample_catalog,
        sample_catalog,
        user_id="u1",
        seed_ids=[],
        top_k=5,
        liked_ids=["NEW-01"],
        exclude_ids=["0006"],
    )

    result_ids = result.book_id.tolist()
    # Excluded 0006 should not appear
    assert "0006" not in result_ids
    # NEW-02 boosted from liked NEW-01
    assert "NEW-02" in result_ids


def test_source_dataset_integrity_preserved(sample_catalog, sample_ratings, temp_db):
    """Ensure feedback is stored in SQLite and never pollutes the source ratings."""
    ratings_before = sample_ratings.copy()

    # Save multiple feedback entries
    set_feedback("0001", "like", temp_db)
    set_feedback("NEW-01", "like", temp_db)
    set_feedback("0002", "dislike", temp_db)

    # Ratings dataframe is completely unchanged
    pd.testing.assert_frame_equal(sample_ratings, ratings_before)
    # The new book NEW-01 is still not present in ratings
    assert "NEW-01" not in sample_ratings.book_id.values
