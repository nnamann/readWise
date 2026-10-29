import pandas as pd

from src.collaborative import ItemCollaborativeRecommender
from src.content_based import ContentRecommender
from src.recommendation_service import reader_rating_fallback


def test_reader_fallback_uses_seed_content_and_excludes_profile_history(books, ratings):
    content = ContentRecommender(books)
    collaborative = ItemCollaborativeRecommender(books, ratings)
    popular = books.assign(rating_count=[10, 9, 8, 7, 6, 5], avg_rating=[9, 8, 7, 8, 9, 10])
    result = reader_rating_fallback(content, collaborative, books, popular, "u1", ["0001"], 5)
    assert "selected-book content fallback" == result.recommendation_source.iloc[0]
    assert result.book_id.is_unique
    assert not set(result.book_id) & {"0001", "0002", "0003"}
    assert result.explanation.str.contains("too sparse").all()


def test_reader_fallback_uses_real_rating_rank_for_new_profile(books, ratings):
    content = ContentRecommender(books)
    collaborative = ItemCollaborativeRecommender(books, ratings)
    popular = books.assign(rating_count=[1, 20, 5, 8, 3, 10], avg_rating=[9, 8, 7, 8, 9, 10])
    result = reader_rating_fallback(content, collaborative, books, popular, "u1", [], 5)
    assert result.book_id.tolist()[0] == "0006"  # highest-count remaining catalogue record
    assert not set(result.book_id) & {"0001", "0002", "0003"}
    assert result.recommendation_source.eq("catalogue rating fallback").all()


def test_sparse_reader_history_excludes_its_only_known_book(books, ratings):
    sparse_ratings = pd.concat([
        ratings,
        pd.DataFrame({"user_id": ["sparse"], "book_id": ["0005"], "rating": [9]}),
    ], ignore_index=True)
    content = ContentRecommender(books)
    collaborative = ItemCollaborativeRecommender(books, sparse_ratings)
    popular = books.assign(rating_count=[10, 9, 8, 7, 20, 5], avg_rating=[9, 8, 7, 8, 9, 10])
    result = reader_rating_fallback(content, collaborative, books, popular, "sparse", [], 5)
    assert "0005" not in result.book_id.tolist()
    assert result.recommendation_source.eq("catalogue rating fallback").all()
