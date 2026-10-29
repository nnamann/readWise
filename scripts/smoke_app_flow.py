"""Exercise the For You app through Streamlit's in-process AppTest harness."""
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_curated_dataset


ROOT = Path(__file__).resolve().parents[1]
books, ratings = load_curated_dataset()
author_sizes = books.groupby("author").size()
author_sizes = author_sizes[author_sizes.index.astype(str).str.strip() != ""]
seed_author = author_sizes.sort_values(ascending=False, kind="stable").index[0]
seed_id = str(books.loc[books.author.eq(seed_author), "book_id"].iloc[0])
catalog_ids = set(books.book_id.astype(str))
eligible = ratings.loc[ratings.rating.gt(0) & ratings.book_id.astype(str).isin(catalog_ids)]
user_counts = eligible.groupby("user_id").size()
reader_id = str(user_counts[user_counts.ge(2)].index[0])

app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
assert not app.exception, [exception.message for exception in app.exception]
app.button(key="nav_for_you").click().run()
assert not app.exception, [exception.message for exception in app.exception]
assert app.session_state["page_route"] == "For you"
assert not app.session_state.get("for_you_result")

# New-reader state should render as empty and can generate a content-based list.
app.session_state["seed_ids"] = [seed_id]
app.session_state["favorite_ids"] = [seed_id]
app.run()
app.button(key="build_list").click().run()
assert not app.exception, [exception.message for exception in app.exception]
content_result = app.session_state["for_you_result"]
assert not content_result.empty
assert seed_id not in content_result.book_id.astype(str).tolist()
assert {"title", "author", "cover_url", "explanation"}.issubset(content_result.columns)

# A real reader ID exercises the sparse-rating fallback with a selected-book profile.
app.session_state["reader_id"] = reader_id
app.session_state["for_you_method_v2"] = "Based on reader ratings"
app.run()
app.button(key="build_list").click().run()
assert not app.exception, [exception.message for exception in app.exception]
rating_result = app.session_state["for_you_result"]
assert not rating_result.empty
assert seed_id not in rating_result.book_id.astype(str).tolist()
assert rating_result.explanation.str.contains("too sparse", case=False).all()
assert {"title", "author", "cover_url", "explanation"}.issubset(rating_result.columns)

print("For You AppTest passed: empty state, content recommendation, reader ID, fallback, metadata, explanations, and seen-book exclusion.")
