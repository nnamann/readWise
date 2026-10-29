import pandas as pd

from src.identifier_reconciliation import canonical_isbn, is_valid_isbn, reconcile_rating_ids


def test_canonical_isbn_keeps_leading_zero_and_uppercases_x():
    assert canonical_isbn("01402.5448x") == "014025448X"
    assert canonical_isbn("0000000000000") == "0000000000000"
    assert is_valid_isbn("01402.5448x")
    assert not is_valid_isbn("01402.5448Y")


def test_only_unique_checksum_valid_format_variants_are_repaired():
    books = pd.DataFrame({"book_id": ["014025448X", "00001"], "title": ["Edition", "Other"]})
    ratings = pd.DataFrame({"book_id": ["01402.5448X", "00001", "9780000000001"],
                            "user_id": ["u1", "u2", "u3"], "rating": [9, 8, 7]})
    repaired, audit = reconcile_rating_ids(books, ratings)
    assert repaired.book_id.tolist() == ["014025448X", "00001", "9780000000001"]
    assert audit.status.tolist() == ["repaired_format", "matched_exact", "invalid_or_non_isbn"]
