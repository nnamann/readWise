import pandas as pd
import pytest

from src.evaluation import (EvaluationConfig, _ranking_metrics, _relevant_work_ids,
                            evaluate_models, precision_recall_at_k)


def test_precision_recall():
    assert precision_recall_at_k(["a", "b", "c"], {"b", "c", "d"}, 3) == (2 / 3, 2 / 3)
    assert precision_recall_at_k([], set(), 10) == (0, 0)


def test_evaluation_is_reproducible_and_counts_users(books, ratings):
    extra = pd.DataFrame({"user_id": [f"u{i}" for i in range(5, 10)] * 6,
                          "book_id": [f"000{i}" for i in range(1, 7) for _ in range(5)],
                          "rating": [8, 9, 10, 8, 9, 10] * 5})
    data = pd.concat([ratings, extra], ignore_index=True)
    cfg = EvaluationConfig(seed=4, threshold=8, max_users=20)
    first, detail1 = evaluate_models(books, data, cfg)
    second, detail2 = evaluate_models(books, data, cfg)
    pd.testing.assert_frame_equal(first, second)
    assert detail1 == detail2
    assert first.model.tolist() == [
        "Most popular (train only)", "Content-based (train-only fit)",
        "Collaborative (item-based)", "Previous hybrid (min-max union)",
        "Revised hybrid (content-preserving)",
    ]
    assert (first.evaluated_users >= 0).all()
    assert detail1["sampled_eligible_users"] >= detail1["users_with_relevant_holdout"]
    assert first.users_with_relevant_holdout.nunique() == 1
    assert detail1["training_heldout_pair_overlap"] == 0
    assert first.catalogue_coverage.between(0, 1).all()
    assert first.intra_list_diversity.between(0, 1).all()
    assert first.empty_recommendation_users.between(0, first.evaluated_users).all()
    assert detail1["training_heldout_pair_overlap"] == 0
    assert "explicit training ratings" in detail1["feature_fit_scope"]
    assert {"precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10", "ndcg_at_10", "map_at_10"}.issubset(first.columns)


def test_ranking_metrics_on_known_order():
    result = _ranking_metrics(["a", "b", "c"], {"b", "c", "d"}, 10)
    assert result["precision_at_5"] == 2 / 5
    assert result["precision_at_10"] == 2 / 10
    assert result["recall_at_5"] == 2 / 3
    assert result["recall_at_10"] == 2 / 3
    assert 0 < result["ndcg_at_10"] < 1
    assert 0 < result["map_at_10"] < 1


def test_short_recommendation_precision_keeps_k_as_denominator():
    precision, recall = precision_recall_at_k(["hit"], {"hit", "miss"}, k=10)
    assert precision == 0.1
    assert recall == 0.5


def test_metric_rejects_non_integer_k():
    with pytest.raises(ValueError, match="positive integer"):
        precision_recall_at_k([], set(), k=1.5)


def test_relevant_heldout_edition_is_not_a_new_work_when_training_has_same_work():
    books = pd.DataFrame({
        "book_id": ["seen", "heldout-edition", "other"],
        "title": ["A Shared Story", "A Shared Story (Paperback)", "A Different Story"],
        "author": ["One Author", "One Author", "Another Author"],
    })
    train = pd.DataFrame({"user_id": ["reader"], "book_id": ["seen"], "rating": [8]})
    test = pd.DataFrame({"user_id": ["reader", "reader"],
                         "book_id": ["heldout-edition", "other"], "rating": [10, 9]})
    assert _relevant_work_ids(test, train, books, threshold=8) == {"other"}
