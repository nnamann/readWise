import numpy as np
import pandas as pd
import pytest

from src.collaborative import ItemCollaborativeRecommender
from src.content_based import ContentRecommender
from src.hybrid import hybrid_recommend, hybrid_recommend_previous, normalize_scores


def test_content_ranking_and_validation(books):
    model = ContentRecommender(books)
    result = model.recommend("0001", 3)
    assert "0001" not in result.book_id.tolist()
    assert result.iloc[0].book_id == "0002"
    assert np.isfinite(result.score).all()
    assert model.recommend("missing").empty
    with pytest.raises(ValueError):
        model.recommend("0001", 0)


def test_content_exclusions_apply_before_top_k_and_empty_metadata_has_no_signal(books):
    model = ContentRecommender(books)
    result = model.recommend("0001", 5, exclude_ids={"0002"})
    assert "0002" not in result.book_id.tolist()
    assert "0001" not in result.book_id.tolist()
    blank = books.assign(title="", author="", publisher="")
    assert ContentRecommender(blank).recommend("0001").empty


def test_content_training_fit_scope_and_duplicate_isbn_handling(books):
    books.loc[0, "title"] = "Training stories"
    books.loc[1, "title"] = "Quasarglyph futures"
    duplicated = pd.concat([books, books.iloc[[1]]], ignore_index=True)
    model = ContentRecommender(duplicated, fit_book_ids={"0001"})
    assert "quasarglyph" not in model.vectorizer.vocabulary_
    assert model.matrix.shape[0] == books.book_id.nunique()
    result = model.recommend("0001", top_k=5)
    assert result.book_id.is_unique


def test_collaborative_known_user_excludes_seen_and_unknown_empty(books, ratings):
    model = ItemCollaborativeRecommender(books, ratings)
    result = model.recommend("u1")
    assert not set(result.book_id) & {"0001", "0002", "0003"}
    assert np.isfinite(result.score).all()
    assert model.recommend("not-a-user").empty


def test_zero_and_sparse_history(books, ratings):
    more = pd.concat([ratings, pd.DataFrame({"user_id": ["only-zero"], "book_id": ["0006"], "rating": [0]})])
    model = ItemCollaborativeRecommender(books, more)
    assert model.recommend("only-zero").empty
    assert model.explicit_rating_counts["u1"] == 3
    assert model.recommend("u1", 100).shape[0] <= 3


def test_hybrid_normalization_and_previously_rated(books, ratings):
    content = ContentRecommender(books)
    collaborative = ItemCollaborativeRecommender(books, ratings)
    result = hybrid_recommend(books, content, collaborative, "u1", ["0001"], top_k=5)
    assert not set(result.book_id) & {"0001", "0002", "0003"}
    assert result.book_id.is_unique
    assert np.isfinite(result.score).all()
    assert normalize_scores(pd.Series([3.0, 3.0])).tolist() == [0.0, 0.0]
    assert normalize_scores(pd.Series([np.nan, 2.0])).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        hybrid_recommend(books, content, collaborative, alpha=1.1)
    with pytest.raises(ValueError):
        hybrid_recommend(books, content, collaborative, top_k=0)


def test_hybrid_renormalizes_when_candidate_has_one_source(books, ratings):
    from src.hybrid import hybrid_recommend

    class StubContent:
        def recommend(self, *_args):
            return pd.DataFrame({"book_id": ["0004", "0005"], "score": [0.4, 0.2]})

    class StubCollaborative:
        user_rated = {"u1": {0, 1, 2}}

        def recommend(self, *_args):
            return pd.DataFrame({"book_id": ["0005"], "score": [0.7]})

        def _empty(self):
            return pd.DataFrame(columns=["book_id", "score"])

    collab_stub = StubCollaborative()
    collab_stub.books = books
    result = hybrid_recommend(books, StubContent(), collab_stub, "u1",
                              favorite_ids=["0001"], top_k=5)
    only_content = result[result.book_id == "0004"].iloc[0]
    assert only_content.score == 1.0  # available content score is renormalized
    assert only_content.explanation == "Catalogue metadata provides the match."


def test_hybrid_weight_one_preserves_content_ranking(books):
    ids = ["0001", "0002"]

    class StubContent:
        def recommend(self, *_args):
            return pd.DataFrame({"book_id": ids, "score": [1.0, 0.0]})

    class StubCollaborative:
        user_rated = {}

        def recommend(self, *_args):
            return pd.DataFrame({"book_id": ids, "score": [0.0, 1.0]})

        def _empty(self):
            return pd.DataFrame(columns=["book_id", "score"])

    collaborative = StubCollaborative()
    collaborative.books = books
    content_first = hybrid_recommend(books, StubContent(), collaborative, user_id="u1", top_k=2, alpha=1.0)
    assert content_first.book_id.iloc[0] == "0001"


def test_zero_content_weight_uses_rating_popularity_fallback(books, ratings):
    content = ContentRecommender(books)
    collaborative = ItemCollaborativeRecommender(books, ratings)
    result = hybrid_recommend(books, content, collaborative, user_id="u1", top_k=5, alpha=0.0)
    assert result.recommendation_source.eq("training rating fallback").all()
    assert not set(result.book_id) & {"0001", "0002", "0003"}


def test_revised_and_previous_hybrid_are_deterministic_and_exclude_seen(books, ratings):
    content = ContentRecommender(books)
    collaborative = ItemCollaborativeRecommender(books, ratings)
    revised1 = hybrid_recommend(books, content, collaborative, "u1", ["0001"], 5, 0.5)
    revised2 = hybrid_recommend(books, content, collaborative, "u1", ["0001"], 5, 0.5)
    previous = hybrid_recommend_previous(books, content, collaborative, "u1", ["0001"], 5, 0.5)
    pd.testing.assert_frame_equal(revised1, revised2)
    for result in (revised1, previous):
        assert result.book_id.is_unique
        assert not set(result.book_id) & {"0001", "0002", "0003"}


def test_revised_hybrid_uses_rating_popularity_fallback_without_content_profile(books, ratings):
    content = ContentRecommender(books)
    collaborative = ItemCollaborativeRecommender(books, ratings)
    result = hybrid_recommend(books, content, collaborative, "u1", ["not-in-catalogue"], 5, 0.5)
    assert not result.empty
    assert result.recommendation_source.eq("training rating fallback").all()
    assert result.explanation.str.contains("too sparse to trust").all()
    assert not set(result.book_id) & {"0001", "0002", "0003"}
