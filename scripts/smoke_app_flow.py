"""Exercise the For You app through Streamlit's in-process AppTest harness."""
from pathlib import Path
import sys
import tempfile
import os

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_curated_dataset
from src import feedback as feedback_store


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

# Keep the UI regression isolated from a real reader's local feedback database.
feedback_tmp = tempfile.TemporaryDirectory(prefix="readwise-feedback-smoke-")
test_feedback_db_path = Path(feedback_tmp.name) / "feedback.db"
original_feedback_db_env = os.environ.get("READWISE_FEEDBACK_DB_PATH")
os.environ["READWISE_FEEDBACK_DB_PATH"] = str(test_feedback_db_path)

explicit = ratings.loc[ratings.rating.gt(0)]
rating_stats = explicit.groupby("book_id").rating.agg(avg_rating="mean", rating_count="count").reset_index()
popular = books.merge(rating_stats, on="book_id", how="left")
popular["rating_count"] = popular.rating_count.fillna(0).astype(int)
featured_id = str(popular.sort_values(
    ["rating_count", "avg_rating", "title"], ascending=[False, False, True], kind="stable"
).iloc[0].book_id)

app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
assert not app.exception, [exception.message for exception in app.exception]

# Exercise the actual feedback controls, then start a fresh app session to prove
# the saved marker is read back from SQLite after an app restart.
app.button(key="open_featured_book").click().run()
assert not app.exception, [exception.message for exception in app.exception]
app.button(key=f"detail_fb_like_detail_{featured_id}").click().run()
assert not app.exception, [exception.message for exception in app.exception]
assert feedback_store.get_feedback(featured_id) == "like"

restarted_app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
assert not restarted_app.exception, [exception.message for exception in restarted_app.exception]
restarted_app.session_state["page_route"] = "Browse"
restarted_app.session_state["selected_book_id"] = featured_id
restarted_app.run()
assert not restarted_app.exception, [exception.message for exception in restarted_app.exception]
assert any("Your saved feedback" in element.value for element in restarted_app.info)
restarted_app.button(key=f"detail_fb_dislike_detail_{featured_id}").click().run()
assert not restarted_app.exception, [exception.message for exception in restarted_app.exception]
assert feedback_store.get_feedback(featured_id) == "dislike"
restarted_app.button(key=f"detail_fb_rm_detail_{featured_id}").click().run()
assert not restarted_app.exception, [exception.message for exception in restarted_app.exception]
assert feedback_store.get_feedback(featured_id) is None
assert feedback_store.get_feedback_records() == []

# Static Discovery shelves are also suggestion surfaces: prior likes, passes,
# and read markers must disappear there, including duplicate editions.
feedback_store.set_feedback("0385504209", "read")
feedback_store.set_feedback("0446672211", "dislike")
feedback_store.set_feedback("059035342X", "like")
discovery_app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
assert not discovery_app.exception, [exception.message for exception in discovery_app.exception]
visible_action_keys = [button.key for button in discovery_app.button]
assert not any("0446672211" in key for key in visible_action_keys)
assert any("059035342X" in key for key in visible_action_keys)
assert any(element.value == "Already read" for element in discovery_app.header)
read_book_cards = [key for key in visible_action_keys
                   if key.startswith("open_") and key.endswith("_0385504209")]
assert len(read_book_cards) == 1
feedback_store.clear_all_feedback()
assert feedback_store.get_feedback_records() == []

if original_feedback_db_env is None:
    os.environ.pop("READWISE_FEEDBACK_DB_PATH", None)
else:
    os.environ["READWISE_FEEDBACK_DB_PATH"] = original_feedback_db_env
feedback_tmp.cleanup()

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

print("AppTest passed: feedback like/change/remove persists across a fresh session; For You empty state, content recommendations, reader ID fallback, metadata, explanations, and seen-book exclusion.")
