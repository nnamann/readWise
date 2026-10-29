import pandas as pd
import pytest


@pytest.fixture
def books():
    return pd.DataFrame({"book_id": ["0001", "0002", "0003", "0004", "0005", "0006"],
                         "title": ["Space Journey", "Space Odyssey", "Garden Guide", "Garden Design", "Ocean Life", "Ocean World"],
                         "author": ["A Writer", "A Writer", "B Author", "B Author", "C Person", "C Person"],
                         "publisher": ["Press"] * 6, "cover_url": [""] * 6})


@pytest.fixture
def ratings():
    return pd.DataFrame({"user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3", "u4", "u4", "u4"],
                         "book_id": ["0001", "0002", "0003", "0001", "0002", "0004", "0001", "0002", "0005", "0003", "0004", "0006"],
                         "rating": [9, 8, 7, 8, 9, 9, 7, 8, 9, 8, 9, 8]})
