from html import escape
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.collaborative import ItemCollaborativeRecommender
from src.config import DATA_DIR, FEEDBACK_DB_PATH, OUTPUT_DIR, RAW_DIR
from src.content_based import ContentRecommender
from src.covers import clean_text_encoding, render_cover_html
from src.catalog_search import exclude_seen_editions, filter_feedback_recommendations, search_catalog
from src.data_loader import DatasetError, load_curated_dataset
from src.feedback import (
    FEEDBACK_LABELS,
    VALID_FEEDBACK_TYPES,
    clear_all_feedback,
    get_all_feedback,
    get_feedback,
    get_feedback_by_type,
    get_feedback_records,
    get_feedback_summary,
    remove_feedback,
    set_feedback,
)
from src.hybrid import hybrid_recommend
from src.preprocess import parse_publication_year
from src.recommendation_service import reader_rating_fallback


st.set_page_config(page_title="ReadWise — Find your next read", page_icon="📚", layout="wide")
st.markdown("""
<style>
:root { --paper:#f6f2eb; --ink:#25232b; --accent:#5947bd; --accent-soft:#eeeafd; --line:#e4ded5; --card:#fffefa; --muted:#706d77; --coral:#c66b55; --teal:#547c79; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(246,242,235,.96); }
[data-testid="stToolbar"], #MainMenu, footer { visibility:hidden; }
.block-container { max-width:1440px; padding-top:1.5rem; padding-bottom:4rem; }
h1,h2,h3 { font-family:Georgia,serif !important; color:var(--ink); letter-spacing:-.035em; }
h1 { font-size:3.7rem !important; line-height:1.02 !important; }
h2 { font-size:2.25rem !important; }
h3 { font-size:1.35rem !important; }
p, label, .stMarkdown { color:var(--ink); }
.brand-mark { font-family:Georgia,serif; color:var(--ink); font-size:1.8rem; font-weight:700; letter-spacing:-.06em; }
.brand-mark span { color:var(--accent); }
.brand-note { color:var(--muted); font-size:.77rem; margin-top:.1rem; }
.topline { border-top:1px solid var(--line); margin:.8rem 0 1rem; }
.section-kicker { color:var(--accent); letter-spacing:.15em; font-size:.68rem; font-weight:750; text-transform:uppercase; }
.small-note { color:var(--muted); font-size:.92rem; }
.book-meta { color:var(--muted); font-size:.85rem; min-height:2rem; }
.cover-fallback { width:100%; max-width:196px; height:264px; margin:0 auto .75rem; padding:1rem .85rem; background:#eeeafd; border-left:6px solid var(--accent); box-shadow:0 4px 10px rgba(42,34,67,.06); display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; overflow:hidden; box-sizing:border-box; border-radius:4px; }
.cover-fallback .fallback-title { font:650 1rem/1.3 Georgia,serif; color:var(--ink); display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:4; overflow:hidden; word-break:break-word; margin-bottom:.35rem; }
.cover-fallback .fallback-author { font:italic .8rem/1.25 system-ui,-apple-system,sans-serif; color:var(--muted); display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:2; overflow:hidden; word-break:break-word; }
.detail-cover { max-width:260px; height:360px; }
.detail-cover .fallback-title { font-size:1.25rem; -webkit-line-clamp:5; }
.detail-cover .fallback-author { font-size:.9rem; }
.cover-frame { width:100%; max-width:196px; height:264px; margin:0 auto .75rem; display:flex; align-items:center; justify-content:center; }
.cover-frame img { max-width:100%; max-height:100%; width:100%; height:100%; object-fit:contain; transition:transform .2s ease; }
.shelf-empty { border-left:3px solid var(--accent); background:var(--card); padding:1.5rem; color:var(--muted); }
.hero-copy { max-width:660px; padding:1rem 0 1.5rem; }
.hero-copy h1 { max-width:660px; margin:.55rem 0 1rem; }
.hero-copy p { color:var(--muted); max-width:560px; font-size:1.08rem; line-height:1.65; }
  .st-key-hero_side { background:var(--accent-soft); padding:1.5rem 1.2rem; align-self:flex-start; }
.section-head { display:flex; align-items:baseline; justify-content:space-between; gap:1rem; }
.author-name { font:600 1.25rem Georgia,serif; color:var(--ink); margin-bottom:.25rem; min-height:3.1rem; }
.author-sub { color:var(--muted); font-size:.87rem; margin-bottom:.65rem; }
.book-title { color:var(--ink); font-size:.96rem; font-weight:650; line-height:1.35; min-height:2.7em; display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:2; overflow:hidden; margin:.15rem 0 .35rem; }
.book-rating { color:#625b79; font-size:.8rem; min-height:1.25rem; margin:.25rem 0 .4rem; }
[data-testid="stMetric"] { background:transparent; border:0; padding:.5rem 0; }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--card); border:1px solid var(--line) !important; }
.stTextInput input, [data-baseweb="select"] > div, [data-baseweb="input"] > div { background:var(--card) !important; color:var(--ink) !important; border-color:var(--line) !important; border-radius:5px !important; }
.stTextInput input::placeholder { color:#858691 !important; }
.stRadio label, [data-testid="stWidgetLabel"] p { color:var(--ink) !important; }
div.stButton > button { border-radius:5px; border:1px solid transparent; color:var(--ink); background:transparent; min-height:2.4rem; transition:background .16s ease,color .16s ease,border-color .16s ease; }
div.stButton > button:hover { color:var(--accent); background:var(--accent-soft); border-color:transparent; }
[data-testid="stBaseButton-primary"] { color:white !important; background:var(--accent) !important; border-color:var(--accent) !important; }
[data-testid="stBaseButton-primary"]:hover { background:#493ba8 !important; }
button[kind="primary"] *, [data-testid="stBaseButton-primary"] * { color:white !important; }
[data-testid="stBaseButton-primaryFormSubmit"] { color:white !important; background:var(--accent) !important; border-color:var(--accent) !important; }
[data-testid="stBaseButton-primaryFormSubmit"]:hover { background:#493ba8 !important; }
[data-testid="stBaseButton-primaryFormSubmit"] * { color:white !important; }
.st-key-desktop_nav [data-testid="stBaseButton-secondary"] { border:0; border-radius:0; border-bottom:2px solid transparent; color:#62636e; }
.st-key-desktop_nav [data-testid="stBaseButton-secondary"]:hover { color:var(--accent); background:transparent; border-bottom-color:#c8c3ed; }
.st-key-desktop_nav [data-testid="stBaseButton-primary"] { color:var(--accent) !important; background:transparent !important; border:0; border-bottom:2px solid var(--accent) !important; border-radius:0; }
.st-key-desktop_nav [data-testid="stBaseButton-primary"] * { color:var(--accent) !important; }
.st-key-clear_browse_filters button { background:#fff !important; border:1px solid var(--line) !important; }
[data-testid="stExpander"] { background:var(--card); border-color:var(--line); }
hr { border-color:var(--line); }
[class*="st-key-book_tile_"] { height:100%; padding:.85rem .85rem 1rem; background:var(--card); border:1px solid var(--line); border-radius:10px; box-shadow:0 2px 8px rgba(42,34,67,.045); transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease; }
[class*="st-key-book_tile_"]:hover { transform:translateY(-3px); border-color:#cbc4e8; box-shadow:0 9px 20px rgba(42,34,67,.09); }
[class*="st-key-book_tile_"]:hover .cover-frame img { transform:scale(1.025); }
[class*="st-key-book_tile_"] [data-testid="stBaseButton-primary"] { min-height:2.35rem; }
[class*="st-key-book_tile_"] [data-testid="stBaseButton-secondary"] { min-height:2.35rem; color:var(--accent) !important; background:#f1effb !important; border:1px solid #e0dcf3 !important; font-weight:600; }
[class*="st-key-author_tile_"] { height:100%; padding:1.15rem; background:var(--card); border:1px solid var(--line); border-radius:9px; box-shadow:0 2px 8px rgba(42,34,67,.04); }
[class*="st-key-author_dir_tile_"] { height:100%; padding:1rem; background:var(--card); border:1px solid var(--line); border-radius:9px; }
.st-key-hero_side { border:1px solid #ded8f0; border-radius:12px; box-shadow:0 12px 30px rgba(42,34,67,.075); }
.st-key-quiet_section { margin:1.4rem 0; padding:1.35rem 1.1rem .5rem; background:#f0eee9; border:1px solid #e6e0d6; border-radius:12px; }
.st-key-author_section { margin:1.5rem 0; padding:1.35rem 1.1rem; background:#eeedf7; border:1px solid #e0dded; border-radius:12px; }
button:focus-visible, input:focus-visible, [role="combobox"]:focus-visible { outline:3px solid #8277df !important; outline-offset:2px; }
/* ── Inline feedback icon row ─────────────────────────────── */
.fb-row { display:flex; gap:.35rem; margin-top:.45rem; align-items:center; }
.fb-btn {
  flex:1; display:flex; align-items:center; justify-content:center;
  height:2rem; border-radius:6px; border:1.5px solid var(--line);
  background:var(--card); cursor:pointer; transition:background .14s, border-color .14s, transform .12s;
  padding:0;
}
.fb-btn:hover { background:var(--accent-soft); border-color:#c8c3ed; transform:scale(1.07); }
.fb-btn.fb-active-like  { background:#eaf4ea; border-color:#5c9e5c; }
.fb-btn.fb-active-dislike { background:#fdf0ed; border-color:#c66b55; }
.fb-btn.fb-active-read  { background:#eeeafd; border-color:var(--accent); }
.fb-btn svg { width:15px; height:15px; display:block; pointer-events:none; }
.fb-hint { font-size:.68rem; color:var(--muted); margin-top:.2rem; min-height:.9rem; }
/* invisible Streamlit buttons layered over the SVG icon row */
[class*="st-key-fb_wrap_"] {
  position:relative;
}
[class*="st-key-fb_wrap_"] .fb-row {
  position:relative; z-index:1; pointer-events:none;
}
[class*="st-key-fb_wrap_"] [data-testid="stHorizontalBlock"] {
  position:absolute; top:0; left:0; width:100%; height:2rem;
  opacity:0; z-index:2; margin:0 !important;
}
[class*="st-key-fb_wrap_"] [data-testid="stHorizontalBlock"] button {
  height:2rem; min-height:2rem; border:none !important; padding:0 !important;
  background:transparent !important; cursor:pointer !important;
}
@media (max-width: 850px) {
  .block-container { padding:3.5rem .8rem 2rem; }
  .topline { margin:.35rem 0 .5rem; }
  h1 { font-size:2.65rem !important; }
  h2 { font-size:1.8rem !important; }
  .hero-copy { padding:.25rem 0 1rem; }
  .hero-copy p { font-size:1rem; }
  .st-key-hero_side { padding:.8rem; }
  .st-key-hero_side .cover-frame { height:220px !important; max-width:145px !important; }
  .stForm [data-testid="stHorizontalBlock"] { display:grid !important; grid-template-columns:minmax(0,1fr) 92px; gap:.55rem !important; }
  .stForm [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width:auto !important; min-width:0 !important; flex:unset !important; }
  .st-key-hero_for_you { display:none !important; }
  .brand-note { display:none; }
  .st-key-top_header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2),
  .st-key-top_header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) { display:none !important; }
  .st-key-top_header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child { width:100% !important; min-width:100% !important; flex:1 1 100% !important; }
  .st-key-desktop_nav { display:none !important; }
  .st-key-mobile_nav { display:block !important; }
  [class*="st-key-book_grid_"] [data-testid="stHorizontalBlock"] {
    display:grid !important; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:.8rem !important;
  }
  [class*="st-key-book_grid_"] [data-testid="stHorizontalBlock"] > div {
    width:auto !important; min-width:0 !important; flex:unset !important;
  }
  [class*="st-key-book_tile_"] { padding:.65rem .55rem .8rem; }
  .st-key-quiet_section, .st-key-author_section { padding:.85rem .65rem .4rem; }
}
@media (min-width: 851px) and (max-width: 1100px) {
  .st-key-top_header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child { width:150px !important; min-width:150px !important; flex:0 0 150px !important; }
  .st-key-top_header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) { display:none !important; }
  .st-key-desktop_nav button { font-size:.78rem !important; padding-left:.15rem !important; padding-right:.15rem !important; }
}
.st-key-mobile_nav { display:none; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior:auto !important; animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Opening the Book-Crossing catalogue…")
def get_data(additions_path, additions_mtime_ns: int):
    # Include the supplement's modification time in the cache key so edits invalidate data and models.
    return load_curated_dataset(RAW_DIR, additions_path)


@st.cache_resource(show_spinner="Building the recommendation models…")
def get_models(books: pd.DataFrame, ratings: pd.DataFrame):
    return ContentRecommender(books), ItemCollaborativeRecommender(books, ratings)


@st.cache_data
def get_rating_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    explicit = ratings.loc[ratings.rating.gt(0)]
    return explicit.groupby("book_id").rating.agg(avg_rating="mean", rating_count="count").reset_index()


@st.cache_data(show_spinner="Preparing the book catalogue…")
def get_catalog(books: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    result = books.merge(get_rating_stats(ratings), on="book_id", how="left")
    result["rating_count"] = result.rating_count.fillna(0).astype(int)
    return result


@st.cache_data
def get_popular_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    return catalog.sort_values(["rating_count", "avg_rating", "title"],
                               ascending=[False, False, True], kind="stable").reset_index(drop=True)


@st.cache_data
def get_quieter_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    return (catalog[catalog.rating_count.between(10, 80)]
            .sort_values(["avg_rating", "rating_count", "title"],
                         ascending=[False, True, True], kind="stable").head(8))


@st.cache_data
def get_recent_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    clean_years = parse_publication_year(catalog.year)
    recent = catalog[clean_years.ge(2015)].copy()
    recent["_clean_year"] = clean_years.loc[recent.index]
    return (recent.sort_values(["_clean_year", "title"], ascending=[False, True], kind="stable")
            .drop(columns=["_clean_year"]).head(8))


@st.cache_data
def get_author_catalog_counts(catalog: pd.DataFrame) -> pd.DataFrame:
    authors = catalog.author.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    author_keys = authors.str.casefold()
    invalid = {"", "nan", "none", "unknown", "unknown author", "n/a", "na", "null", "not available"}
    valid = ~author_keys.isin(invalid) & ~author_keys.str.startswith("not applicable")
    source = catalog.loc[valid, ["book_id"]].copy()
    source["author"] = authors.loc[valid]
    source["_author_key"] = source.author.str.casefold()
    return (source.groupby("_author_key", as_index=False)
            .agg(author=("author", "first"), catalogue_records=("book_id", "nunique"))
            .sort_values(["catalogue_records", "author"], ascending=[False, True], kind="stable")
            .reset_index(drop=True))


def book_label(row) -> str:
    year = str(row.get("year", "")).strip()
    suffix = f" · {year}" if year and year not in {"0", "nan", "NULL"} else ""
    title = clean_text_encoding(str(row.title))
    author = clean_text_encoding(str(row.author or "Author not listed"))
    return f"{title} — {author}{suffix}"


def display_author_name(value: str) -> str:
    name = clean_text_encoding(str(value).strip())
    return name.title() if name.isupper() and len(name) > 1 else name


def go_to(page: str):
    routes = {"Explore": "Browse", "For You": "For you", "Research": "Results"}
    st.session_state["page_route"] = routes.get(page, page)


def reset_browse_page():
    st.session_state["browse_page"] = 0


def clear_browse_search():
    for key in ("browse_query", "browse_author", "browse_publisher", "browse_year"):
        st.session_state[key] = ""
    st.session_state["browse_page"] = 0


def browse_author(author: str):
    st.session_state["browse_query"] = ""
    st.session_state["browse_publisher"] = ""
    st.session_state["browse_year"] = ""
    st.session_state["browse_author"] = author
    st.session_state["browse_page"] = 0
    st.session_state["page_route"] = "Browse"


def toggle_author_directory():
    st.session_state["author_directory_open"] = not st.session_state.get("author_directory_open", False)
    st.session_state["author_directory_page"] = 0


def reset_author_directory_page():
    st.session_state["author_directory_page"] = 0


def submit_discover_search():
    st.session_state["browse_query"] = str(st.session_state.get("discover_query", "")).strip()
    st.session_state["browse_author"] = ""
    st.session_state["browse_publisher"] = ""
    st.session_state["browse_year"] = ""
    st.session_state["browse_page"] = 0
    st.session_state["page_route"] = "Browse"


def open_book(book_id: str):
    if str(st.session_state.get("selected_book_id", "")) != str(book_id):
        st.session_state["detail_similar"] = None
    st.session_state["selected_book_id"] = book_id
    st.session_state["page_route"] = "Browse"


def render_cover(row, large: bool = False, small: bool = False):
    html = render_cover_html(row, large=large, small=small)
    st.markdown(html, unsafe_allow_html=True)


def add_to_shelf(book_id: str):
    favorites = list(st.session_state.get("favorite_ids", []))
    if book_id in favorites:
        favorites.remove(book_id)
        st.session_state["seed_ids"] = [value for value in st.session_state.get("seed_ids", []) if str(value) != str(book_id)]
    else:
        favorites.append(book_id)
    st.session_state["favorite_ids"] = favorites


def reading_list_signature(seed_ids: list[str], user_id: str, method: str, alpha: float,
                           recency_weight: float = 0.0, feedback_items: tuple = ()) -> tuple:
    return tuple(sorted(map(str, seed_ids))), str(user_id).strip(), method, float(alpha), float(recency_weight), feedback_items


def explain_user_id(model: ItemCollaborativeRecommender, user_id: str) -> tuple[str, int]:
    user_id = str(user_id).strip()
    rated = len(model.user_rated.get(user_id, set()))
    if not user_id:
        return "", 0
    if user_id not in model.source_interaction_counts:
        return "That ID is not in the Book-Crossing ratings data. Check it or leave the field blank.", 0
    if rated < 2:
        explicit_count = model.explicit_rating_counts.get(user_id, 0)
        return f"This ID has {explicit_count:,} explicit ratings in the source, but only {rated} match books in the catalogue. At least two usable ratings are needed for reader-pattern recommendations.", rated
    return f"Found {rated:,} usable explicit ratings for this reader.", rated


_SVG_LIKE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="%s" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/>'
    '<path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>'
    "</svg>"
)
_SVG_DISLIKE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="%s" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/>'
    '<path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>'
    "</svg>"
)
_SVG_READ = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="%s" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
    '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'
    "</svg>"
)

def _fb_icon_row(book_id: str, current_fb: str | None, key_suffix: str) -> None:
    """Render three inline SVG icon buttons (like / dislike / read) directly in the card."""
    like_cls    = "fb-btn fb-active-like"    if current_fb == "like"    else "fb-btn"
    dislike_cls = "fb-btn fb-active-dislike" if current_fb == "dislike" else "fb-btn"
    read_cls    = "fb-btn fb-active-read"    if current_fb == "read"    else "fb-btn"

    like_color    = "#3a7d3a" if current_fb == "like"    else "#9390a0"
    dislike_color = "#b05038" if current_fb == "dislike" else "#9390a0"
    read_color    = "#5947bd" if current_fb == "read"    else "#9390a0"

    if current_fb == "like":
        hint = "Liked · boosts similar books"
    elif current_fb == "dislike":
        hint = "Passed · won't be recommended"
    elif current_fb == "read":
        hint = "Read · won't be recommended"
    else:
        hint = ""

    wrap_key = f"fb_wrap_{key_suffix}_{book_id}"
    with st.container(key=wrap_key):
        st.markdown(
            f"<div class='fb-row'>"
            f"<div class='{like_cls}' title='Like this book'>{_SVG_LIKE % like_color}</div>"
            f"<div class='{dislike_cls}' title='Not interested'>{_SVG_DISLIKE % dislike_color}</div>"
            f"<div class='{read_cls}' title='Already read'>{_SVG_READ % read_color}</div>"
            f"</div>"
            f"<div class='fb-hint'>{escape(hint)}</div>",
            unsafe_allow_html=True,
        )
        # Invisible Streamlit buttons layered over the icon row (click targets)
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            if st.button("·", key=f"fb_like_{key_suffix}_{book_id}", help="Like this book", width="stretch"):
                if current_fb == "like":
                    remove_feedback(book_id)
                else:
                    set_feedback(book_id, "like")
                st.rerun()
        with ic2:
            if st.button("·", key=f"fb_dislike_{key_suffix}_{book_id}", help="Not interested", width="stretch"):
                if current_fb == "dislike":
                    remove_feedback(book_id)
                else:
                    set_feedback(book_id, "dislike")
                st.rerun()
        with ic3:
            if st.button("·", key=f"fb_read_{key_suffix}_{book_id}", help="Mark as already read", width="stretch"):
                if current_fb == "read":
                    remove_feedback(book_id)
                else:
                    set_feedback(book_id, "read")
                st.rerun()


def render_feedback_control(book_id: str, title: str, key_suffix: str, compact: bool = False):
    current_fb = get_feedback(book_id)
    if compact:
        _fb_icon_row(book_id, current_fb, key_suffix)
    else:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown("##### Reader Feedback")
        if current_fb:
            badge_icon = "👍" if current_fb == "like" else ("👎" if current_fb == "dislike" else "📖")
            st.info(f"{badge_icon} Your saved feedback: **{FEEDBACK_LABELS.get(current_fb, current_fb)}**")
        else:
            st.caption("Feedback helps ReadWise tailor future recommendations to your taste.")
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            if st.button("👍 Like this book", key=f"detail_fb_like_{key_suffix}_{book_id}",
                         type="primary" if current_fb == "like" else "secondary", width="stretch"):
                set_feedback(book_id, "like")
                st.rerun()
        with fcol2:
            if st.button("👎 Not interested", key=f"detail_fb_dislike_{key_suffix}_{book_id}",
                         type="primary" if current_fb == "dislike" else "secondary", width="stretch"):
                set_feedback(book_id, "dislike")
                st.rerun()
        with fcol3:
            if st.button("📖 Mark as already read", key=f"detail_fb_read_{key_suffix}_{book_id}",
                         type="primary" if current_fb == "read" else "secondary", width="stretch"):
                set_feedback(book_id, "read")
                st.rerun()
        if current_fb:
            if st.button("✕ Remove my feedback for this book", key=f"detail_fb_rm_{key_suffix}_{book_id}"):
                remove_feedback(book_id)
                st.rerun()
        st.caption("• **Like**: Boosts similar books in your recommendations, including newer releases with no Book-Crossing rating history.\n• **Not interested / Already read**: Avoids recommending this book in the future.")


def render_book_grid(frame: pd.DataFrame, catalog: pd.DataFrame, columns: int = 4,
                     controls: bool = True, deduplicate: bool = False,
                     max_items: int | None = 16):
    if frame.empty:
        st.markdown("<div class='shelf-empty'>No books matched this view. Try a broader search or another title.</div>", unsafe_allow_html=True)
        return
    frame = frame.copy()
    if deduplicate:
        title_key = frame.title.fillna("").astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
        author_key = frame.author.fillna("").astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
        frame = frame.assign(_display_key=title_key + "::" + author_key).drop_duplicates("_display_key", keep="first")
    if max_items is not None:
        frame = frame.head(max_items)
    frame = frame.reset_index(drop=True)
    favorites = set(st.session_state.get("favorite_ids", []))
    grid_number = st.session_state.get("_book_grid_number", 0)
    st.session_state["_book_grid_number"] = grid_number + 1
    with st.container(key=f"book_grid_{grid_number}"):
        for start in range(0, len(frame), columns):
            slots = st.columns(columns, gap="medium")
            for slot, (_, book) in zip(slots, frame.iloc[start:start + columns].iterrows()):
                book_id = str(book.book_id)
                with slot:
                    with st.container(key=f"book_tile_{grid_number}_{book_id}"):
                        render_cover(book)
                        st.markdown(f"<div class='book-title' title='{escape(str(book.title), quote=True)}'>{escape(str(book.title))}</div>", unsafe_allow_html=True)
                        year = str(book.get("year", "")).strip()
                        if year in {"0", "nan", "NULL"}:
                            year = ""
                        author_meta = escape(str(book.author or "Author not listed"))
                        if year:
                            author_meta += f" · {escape(year)}"
                        st.markdown(f"<div class='book-meta'>{author_meta}</div>", unsafe_allow_html=True)
                        avg = book.get("avg_rating")
                        votes = int(book.get("rating_count", 0) or 0)
                        if pd.notna(avg) and votes:
                            rating_label = f"★ {float(avg):.1f} · {votes:,} ratings"
                        else:
                            rating_label = ""
                        st.markdown(f"<div class='book-rating'>{escape(rating_label)}</div>", unsafe_allow_html=True)
                        if book.get("explanation"):
                            st.markdown(f"<div class='book-meta'>{escape(str(book.explanation))}</div>", unsafe_allow_html=True)
                        if controls:
                            st.button("Open book", key=f"open_{grid_number}_{book_id}", type="primary", width="stretch",
                                      on_click=open_book, args=(book_id,))
                            if st.button("In your shelf ✓" if book_id in favorites else "＋ Add to shelf",
                                         key=f"shelf_{grid_number}_{book_id}", width="stretch"):
                                add_to_shelf(book_id)
                                st.rerun()
                            render_feedback_control(book_id, book.title, key_suffix=f"grid_{grid_number}", compact=True)


def show_book_details(book_id: str, catalog: pd.DataFrame, content_model):
    row = catalog[catalog.book_id.astype(str).eq(str(book_id))].iloc[0]
    left, right = st.columns([1, 2.5], gap="large")
    with left:
        render_cover(row, large=True)
    with right:
        st.markdown("<div class='section-kicker'>BOOK NOTES</div>", unsafe_allow_html=True)
        st.subheader(str(row.title))
        st.markdown(f"**{escape(str(row.author or 'Author not listed'))}**")
        metadata = [str(x) for x in [row.get("year", ""), row.get("publisher", "")]
                    if str(x).strip() and str(x) not in {"0", "nan", "NULL"}]
        st.caption(" · ".join(metadata) or "Publication details are not available in this dataset.")
        count, average = int(row.get("rating_count", 0)), row.get("avg_rating")
        st.markdown(f"**{average:.1f} / 10** average from {count:,} explicit ratings" if pd.notna(average) and count else "No explicit reader ratings in this dataset.")
        st.markdown("This catalogue record includes title, author, publisher and reader feedback where available. Book-Crossing does not provide a synopsis or reliable genre labels for every title.")
        first, second = st.columns(2)
        with first:
            if st.button("Remove from shelf" if book_id in st.session_state.get("favorite_ids", []) else "Add to my shelf",
                         key=f"detail_shelf_{book_id}", type="primary", width="stretch"):
                add_to_shelf(book_id)
                st.rerun()
        with second:
            if st.button("Find similar books", key=f"similar_{book_id}", width="stretch"):
                related = content_model.recommend(book_id, 40)
                st.session_state["detail_similar"] = exclude_seen_editions(
                    related, catalog, {str(book_id)}).head(8)
        render_feedback_control(book_id, row.title, key_suffix="detail", compact=False)
        if st.session_state.get("detail_similar") is not None:
            st.markdown("#### Similar in the catalogue")
            render_book_grid(st.session_state["detail_similar"], catalog, columns=4, controls=False)


try:
    additions_path = DATA_DIR / "curated" / "additional_books.csv"
    books, ratings = get_data(additions_path, additions_path.stat().st_mtime_ns)
except DatasetError:
    st.error("The Book-Crossing files are missing or invalid. Place BX-Books.csv and BX-Book-Ratings.csv in data/raw.")
    st.stop()

catalog = get_catalog(books, ratings)
content_model, collaborative_model = get_models(books, ratings)
explicit_count = int(ratings.rating.gt(0).sum())
popular = get_popular_catalog(catalog)
feedback_map = get_all_feedback()
liked_ids = {bid for bid, ftype in feedback_map.items() if ftype == "like"}
disliked_ids = {bid for bid, ftype in feedback_map.items() if ftype == "dislike"}
read_ids = {bid for bid, ftype in feedback_map.items() if ftype == "read"}
suppressed_ids = disliked_ids | read_ids
discover_popular = filter_feedback_recommendations(popular, catalog, feedback_map)

st.session_state.setdefault("favorite_ids", [])
st.session_state.setdefault("seed_ids", [])
st.session_state.setdefault("page_route", "Discover")
current_page = st.session_state["page_route"]
nav_names = ["Discover", "Explore", "My Shelf", "For You", "How it works", "Research"]
route_for_nav = {"Discover": "Discover", "Explore": "Browse", "My Shelf": "My Shelf",
                 "For You": "For you", "How it works": "How it works", "Research": "Results"}
nav_active = next((label for label, route in route_for_nav.items() if route == current_page), "Discover")
with st.container(key="top_header"):
    brand, nav_area, tagline = st.columns([1.1, 5.4, .8], vertical_alignment="center")
    with brand:
        st.markdown("<div class='brand-mark'>read<span>wise</span></div><div class='brand-note'>A shelf of better next reads</div>", unsafe_allow_html=True)
    with nav_area:
        with st.container(key="desktop_nav"):
            nav_columns = st.columns(len(nav_names), gap="small")
            for column, nav_name in zip(nav_columns, nav_names):
                key = "nav_" + nav_name.casefold().replace(" ", "_")
                with column:
                    st.button(nav_name, key=key, type="primary" if nav_name == nav_active else "secondary",
                              on_click=go_to, args=(nav_name,), width="stretch")
    with tagline:
        st.markdown("<div class='brand-note' style='text-align:right'>Books first. Better suggestions.</div>", unsafe_allow_html=True)
st.markdown("<div class='topline'></div>", unsafe_allow_html=True)
st.session_state["mobile_route"] = nav_active
with st.container(key="mobile_nav"):
    st.selectbox("Choose a page", nav_names, key="mobile_route", on_change=lambda: go_to(st.session_state["mobile_route"]),
                 label_visibility="collapsed")
st.markdown("<div class='topline'></div>", unsafe_allow_html=True)
page = st.session_state["page_route"]
# Grid keys must repeat in the same order on Streamlit reruns so card actions stay attached.
st.session_state["_book_grid_number"] = 0


if page == "Discover":
    left, right = st.columns([1.12, .88], gap="large", vertical_alignment="top")
    with left:
        st.markdown("<div class='hero-copy'><div class='section-kicker'>A GOOD PLACE TO GET LOST</div><h1>Find a story that stays with you.</h1><p>Browse the Book-Crossing collection and build a shelf around what you love.</p></div>", unsafe_allow_html=True)
        with st.form("discover_search", border=False):
            search_col, action_col = st.columns([4, 1.1], vertical_alignment="bottom")
            with search_col:
                st.text_input("Search books or authors", placeholder="Try a title or author name", key="discover_query")
            with action_col:
                st.form_submit_button("Search", type="primary", width="stretch", on_click=submit_discover_search)
        st.button("Find my next read", key="hero_for_you", type="primary", on_click=go_to,
                  args=("For You",), width="stretch")
    with right:
        with st.container(key="hero_side"):
            st.markdown("<div class='section-kicker'>A PLACE TO START · MOST RATED</div>", unsafe_allow_html=True)
            # Stable selection: the catalogue is sorted by explicit rating count, average, then title.
            if discover_popular.empty:
                st.info("You’ve already marked every title here. Clear or change feedback on My Shelf to see more suggestions.")
            else:
                hero_book = discover_popular.iloc[0]
                render_cover(hero_book, large=True)
                st.markdown(f"**{escape(str(hero_book.title))}**")
                hero_author = escape(str(hero_book.author or "Author not listed"))
                hero_votes = int(hero_book.get("rating_count", 0) or 0)
                hero_average = hero_book.get("avg_rating")
                if pd.notna(hero_average) and hero_votes:
                    st.caption(f"{hero_author} · ★ {float(hero_average):.1f} from {hero_votes:,} ratings")
                else:
                    st.caption(f"{hero_author} · a reader favourite in the catalogue")
                st.button("Open this book", key="open_featured_book", on_click=open_book,
                          args=(str(hero_book.book_id),), width="stretch")
    st.markdown("<div class='section-kicker'>READER FAVOURITES</div>", unsafe_allow_html=True)
    st.header("Popular with readers")
    st.markdown("<div class='small-note'>Sorted by explicit rating count, with observed average rating as a tie-breaker.</div>", unsafe_allow_html=True)
    render_book_grid(discover_popular.head(8), catalog, controls=True, columns=4, deduplicate=True)
    st.button("See the full collection →", key="browse_more", on_click=go_to, args=("Explore",))
    quiet = filter_feedback_recommendations(get_quieter_catalog(catalog), catalog, feedback_map)
    if not quiet.empty:
        with st.container(key="quiet_section"):
            st.markdown("<div class='section-kicker'>A QUIETER CORNER</div>", unsafe_allow_html=True)
            st.header("Well-rated, less often seen")
            st.markdown("These titles have 10–80 explicit ratings and are ordered by their observed average rating.")
            render_book_grid(quiet, catalog, controls=True, columns=4, deduplicate=True)
    recent_shelf = filter_feedback_recommendations(get_recent_catalog(catalog), catalog, feedback_map)
    if not recent_shelf.empty:
        with st.container(key="recent_section"):
            st.markdown("<div class='section-kicker'>RECENT ADDITIONS & CONTEMPORARY READS</div>", unsafe_allow_html=True)
            st.header("Contemporary books (2015–2024)")
            st.markdown("Verified modern titles sourced from Open Library under CC0 public domain terms, preserving genuine publication dates.")
            render_book_grid(recent_shelf, catalog, controls=True, columns=4, deduplicate=True)
    authors = get_author_catalog_counts(catalog)
    if not authors.empty:
        with st.container(key="author_section"):
            st.markdown("<div class='section-kicker'>FOLLOW A NAME ON THE COVER</div>", unsafe_allow_html=True)
            st.header("Explore an author")
            st.markdown("Choose a familiar name, or search the full author directory.")
            featured_authors = authors.head(8)
            for start in range(0, len(featured_authors), 4):
                author_columns = st.columns(4, gap="medium")
                for (index, author_row), column in zip(featured_authors.iloc[start:start + 4].iterrows(), author_columns):
                    with column:
                        with st.container(key=f"author_tile_{index}"):
                            st.markdown(f"<div class='author-name'>{escape(display_author_name(author_row.author))}</div>", unsafe_allow_html=True)
                            st.caption(f"{int(author_row.catalogue_records):,} catalogue books")
                            st.button("Explore books →", key=f"author_pick_{index}", on_click=browse_author,
                                      args=(str(author_row.author),), width="stretch")
            directory_open = st.session_state.get("author_directory_open", False)
            st.button("Close author directory" if directory_open else "Explore all authors →",
                      key="toggle_author_directory", type="primary", on_click=toggle_author_directory)
            if directory_open:
                st.markdown("#### Author directory")
                author_search = st.text_input("Search authors", placeholder="Enter an author name",
                                               key="author_directory_search", on_change=reset_author_directory_page)
                author_matches = authors
                if author_search.strip():
                    author_matches = authors[authors.author.str.contains(author_search.strip(), case=False, regex=False)]
                st.caption(f"{len(author_matches):,} authors matched")
                author_page_size = 12
                author_page_count = max(1, (len(author_matches) + author_page_size - 1) // author_page_size)
                st.session_state["author_directory_page"] = min(
                    st.session_state.get("author_directory_page", 0), author_page_count - 1)
                author_page = st.session_state["author_directory_page"]
                page_start = author_page * author_page_size
                directory_page = author_matches.iloc[page_start:page_start + author_page_size]
                if directory_page.empty:
                    st.info("No authors matched. Try a shorter name or another spelling.")
                else:
                    for row_start in range(0, len(directory_page), 4):
                        author_columns = st.columns(4, gap="medium")
                        for (index, author_row), column in zip(directory_page.iloc[row_start:row_start + 4].iterrows(), author_columns):
                            with column:
                                with st.container(key=f"author_dir_tile_{author_page}_{index}"):
                                    st.markdown(f"<div class='author-name'>{escape(display_author_name(author_row.author))}</div>", unsafe_allow_html=True)
                                    st.caption(f"{int(author_row.catalogue_records):,} catalogue books")
                                    st.button("Explore books →", key=f"author_dir_pick_{author_page}_{index}",
                                              on_click=browse_author, args=(str(author_row.author),), width="stretch")
                previous, page_info, next_page = st.columns([1, 2, 1])
                with previous:
                    if st.button("← Previous", key="author_dir_previous", disabled=author_page == 0, width="stretch"):
                        st.session_state["author_directory_page"] -= 1
                        st.rerun()
                with page_info:
                    st.caption(f"Page {author_page + 1} of {author_page_count}")
                with next_page:
                    if st.button("Next →", key="author_dir_next", disabled=author_page >= author_page_count - 1, width="stretch"):
                        st.session_state["author_directory_page"] += 1
                        st.rerun()
    active_likes = get_feedback_by_type("like")
    active_dislikes = get_feedback_by_type("dislike")
    active_reads = get_feedback_by_type("read")
    all_suppressed = active_dislikes | active_reads
    active_seeds = list(map(str, st.session_state.get("seed_ids", [])))
    if active_seeds or active_likes:
        seen_all = set(active_seeds) | all_suppressed | active_likes
        similar = content_model.recommend(
            active_seeds, 40, exclude_ids=all_suppressed | active_likes, liked_ids=list(active_likes)
        )
        similar = exclude_seen_editions(similar, catalog, seen_all)
        if not similar.empty:
            if active_seeds:
                seed_titles = catalog.loc[catalog.book_id.astype(str).isin(active_seeds), "title"].dropna().astype(str).tolist()
                shelf_heading = f"More like {seed_titles[0]}" if len(active_seeds) == 1 and seed_titles else "More like your picks"
            else:
                shelf_heading = "More like books you’ve liked"
            st.markdown("<div class='section-kicker'>A LITTLE MORE LIKE YOUR PICKS</div>", unsafe_allow_html=True)
            st.header(shelf_heading)
            render_book_grid(similar.head(8), catalog, controls=True, columns=4, deduplicate=True)

    st.divider()
    st.markdown("<div class='section-kicker'>YOUR READING HISTORY</div>", unsafe_allow_html=True)
    st.header("Already read")
    if not active_reads:
        st.caption("Books you mark as already read will be collected here.")
    else:
        read_books = catalog.loc[
            catalog.book_id.astype(str).isin(active_reads), ["title", "author", "book_id"]
        ].copy()
        known_read_ids = set(read_books.book_id.astype(str))
        missing_rows = pd.DataFrame([
            {"title": "Book details unavailable", "author": f"Book ID {book_id}",
             "book_id": book_id, "cover_url": ""}
            for book_id in sorted(active_reads - known_read_ids)
        ])
        read_list = pd.concat([read_books, missing_rows], ignore_index=True)
        read_list["book_id"] = read_list.book_id.astype(str)
        read_list = read_list.sort_values(["title", "author", "book_id"], kind="stable")
        st.caption(f"{len(read_list):,} book(s) marked as already read")
        render_book_grid(read_list, catalog, columns=4, controls=True, max_items=None)


elif page == "Browse":
    author_focus = str(st.session_state.get("browse_author", "")).strip()
    st.markdown("<div class='section-kicker'>THE BOOK COLLECTION</div>", unsafe_allow_html=True)
    st.title(f"Books by {author_focus}" if author_focus else "Explore the collection")
    st.markdown("<div class='small-note'>Find a title, follow an author, or narrow the catalogue by publication details.</div>", unsafe_allow_html=True)
    search, author, clear = st.columns([1.35, 1, .62])
    query = search.text_input("Search titles and authors", placeholder="A title or author name", key="browse_query", on_change=reset_browse_page)
    author_query = author.text_input("Author", placeholder="Search an author", key="browse_author", on_change=reset_browse_page)
    with clear:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        st.button("Clear", key="clear_browse_filters", on_click=clear_browse_search, width="stretch")
    with st.expander("Publication filters", expanded=bool(st.session_state.get("browse_publisher") or st.session_state.get("browse_year"))):
        publisher, year = st.columns([2, 1])
        publisher_query = publisher.text_input("Publisher", placeholder="Filter by publisher", key="browse_publisher", on_change=reset_browse_page)
        year_query = year.text_input("Publication year", placeholder="e.g. 2001", key="browse_year", on_change=reset_browse_page)
    sort, _ = st.columns([1.2, 3.8])
    with sort:
        order = st.selectbox("Sort books", ["Most rated", "Highest average", "Newest first", "Title A–Z"], key="browse_sort")
    filtered = search_catalog(catalog, query, author_query, publisher_query, year_query, order)
    if st.session_state.get("selected_book_id"):
        detail_id = st.session_state["selected_book_id"]
        if catalog.book_id.astype(str).eq(str(detail_id)).any():
            with st.container(border=True):
                show_book_details(detail_id, catalog, content_model)
            if st.button("Close book details", key="close_details"):
                st.session_state.pop("selected_book_id", None)
                st.session_state["detail_similar"] = None
                st.rerun()
            st.divider()
    page_size = 12
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    st.session_state["browse_page"] = min(st.session_state.get("browse_page", 0), total_pages - 1)
    if author_focus:
        st.caption(f"{len(filtered):,} matching catalogue titles by {author_focus} · sorted by {order.lower()}")
    else:
        st.caption(f"{len(filtered):,} matching titles · page {st.session_state['browse_page'] + 1} of {total_pages} · {order}")
    previous, next_col, _ = st.columns([1, 1, 4])
    with previous:
        if st.button("← Previous", disabled=st.session_state["browse_page"] == 0, key="previous_page"):
            st.session_state["browse_page"] -= 1
            st.rerun()
    with next_col:
        if st.button("Next →", disabled=st.session_state["browse_page"] >= total_pages - 1, key="next_page"):
            st.session_state["browse_page"] += 1
            st.rerun()
    start = st.session_state["browse_page"] * page_size
    render_book_grid(filtered.iloc[start:start + page_size], catalog, columns=4)


elif page == "My Shelf":
    st.markdown("<div class='section-kicker'>YOUR PERSONAL COLLECTION</div>", unsafe_allow_html=True)
    st.title("My Shelf")
    tab_shelf, tab_feedback = st.tabs(["📚 Saved to Shelf", "💬 Saved Feedback & Preferences"])
    with tab_shelf:
        shelf = catalog[catalog.book_id.astype(str).isin(st.session_state.get("favorite_ids", []))]
        st.markdown("Books you save stay here for this session. Your shelf is private to this browser session and does not require an account.")
        if shelf.empty:
            st.markdown("<div class='shelf-empty'>Your shelf is waiting for its first book. Save titles from Discover, Explore, or For You and they’ll gather here.</div>", unsafe_allow_html=True)
            st.button("Browse the collection →", key="empty_shelf_browse", on_click=go_to, args=("Explore",))
        else:
            st.caption(f"{len(shelf):,} saved catalogue titles")
            if st.button("Recommend from shelf", key="shelf_to_recs", type="primary", width="content"):
                st.session_state["seed_ids"] = list(dict.fromkeys(map(str, shelf.book_id.tolist())))[:5]
                st.session_state["page_route"] = "For you"
                st.rerun()
            st.caption("Uses up to five saved books as starting picks.")
            render_book_grid(shelf, catalog, columns=4, controls=True)
    with tab_feedback:
        st.markdown("### Saved Reader Feedback")
        st.markdown(
            f"ReadWise stores your feedback locally in a SQLite database (`{FEEDBACK_DB_PATH}`). "
            "Your feedback persists across app restarts, is never uploaded or shared, and is kept strictly "
            "separate from the original Book-Crossing ratings dataset."
        )
        st.markdown(
            "**How feedback affects recommendations:**\n"
            "- **Like this book (👍)**: Learns your taste and boosts related titles across authors and publishers, including recent books with no Book-Crossing rating history.\n"
            "- **Not interested (👎)**: Avoids recommending this title and alternative editions again.\n"
            "- **Mark as already read (📖)**: Suppresses this title so your recommendations stay fresh."
        )
        fb_summary = get_feedback_summary()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total feedback", fb_summary["total"])
        m2.metric("👍 Liked", fb_summary["like"])
        m3.metric("👎 Not interested", fb_summary["dislike"])
        m4.metric("📖 Already read", fb_summary["read"])

        fb_records = get_feedback_records()
        if not fb_records:
            st.markdown("<div class='shelf-empty'>No feedback recorded yet. You can like, pass on, or mark books as read from Discover, Explore, or For You.</div>", unsafe_allow_html=True)
        else:
            c_clear, _ = st.columns([1.5, 4])
            with c_clear:
                if st.button("🗑 Clear all saved feedback", key="clear_all_fb_btn", width="stretch"):
                    clear_all_feedback()
                    st.rerun()
            st.markdown("#### Review and edit your feedback")
            for idx, item in enumerate(fb_records):
                f_book_id = item["book_id"]
                f_type = item["feedback_type"]
                book_matches = catalog[catalog.book_id.astype(str).eq(f_book_id)]
                if not book_matches.empty:
                    b_row = book_matches.iloc[0]
                    b_title = str(b_row.title)
                    b_author = str(b_row.author or "Author not listed")
                    b_year = str(b_row.get("year", "")).strip()
                    if b_year in {"0", "nan", "NULL"}:
                        b_year = ""
                else:
                    b_title = f"Book record #{f_book_id}"
                    b_author = "Unknown author"
                    b_year = ""

                with st.container(border=True):
                    col_info, col_status, col_actions = st.columns([3, 1.5, 2], vertical_alignment="center")
                    with col_info:
                        meta = b_author + (f" · {b_year}" if b_year else "")
                        st.markdown(f"**{escape(b_title)}**")
                        st.caption(f"{escape(meta)} (ID: `{escape(f_book_id)}`)")
                    with col_status:
                        icon = "👍" if f_type == "like" else ("👎" if f_type == "dislike" else "📖")
                        st.markdown(f"**{icon} {FEEDBACK_LABELS.get(f_type, f_type)}**")
                    with col_actions:
                        act1, act2 = st.columns(2)
                        with act1:
                            new_type = st.selectbox(
                                "Change",
                                ["like", "dislike", "read"],
                                index=["like", "dislike", "read"].index(f_type),
                                key=f"sel_fb_{idx}_{f_book_id}",
                                label_visibility="collapsed",
                                format_func=lambda x: {"like": "👍 Like", "dislike": "👎 Pass", "read": "📖 Read"}[x]
                            )
                            if new_type != f_type:
                                set_feedback(f_book_id, new_type)
                                st.rerun()
                        with act2:
                            if st.button("Remove", key=f"del_fb_{idx}_{f_book_id}", width="stretch"):
                                remove_feedback(f_book_id)
                                st.rerun()


elif page == "For you":
    st.markdown("<div class='section-kicker'>MADE AROUND YOUR TASTE</div>", unsafe_allow_html=True)
    st.title("More books you’ll love.")
    st.markdown("Choose books you’ve enjoyed. ReadWise will use them as clues for your next reading list.")

    if feedback_map:
        fb_sum = get_feedback_summary()
        st.info(
            f"💡 **Feedback loop active:** {fb_sum['like']} liked book(s) boosting recommendations · "
            f"{fb_sum['dislike'] + fb_sum['read']} book(s) suppressed (not interested or read). "
            f"Saved locally in SQLite (`{FEEDBACK_DB_PATH}`)."
        )
        with st.expander("Review or clear your saved feedback", expanded=False):
            st.caption("Feedback is stored locally on this machine and never written to original Book-Crossing ratings.")
            c_clr, _ = st.columns([1.5, 4])
            with c_clr:
                if st.button("🗑 Clear all feedback", key="for_you_clear_fb", width="stretch"):
                    clear_all_feedback()
                    st.rerun()
            for idx, (f_id, f_t) in enumerate(list(feedback_map.items())[:10]):
                b_match = catalog[catalog.book_id.astype(str).eq(f_id)]
                b_name = str(b_match.iloc[0].title) if not b_match.empty else f"Book #{f_id}"
                ico = "👍" if f_t == "like" else ("👎" if f_t == "dislike" else "📖")
                st.caption(f"{ico} **{escape(b_name)}** — {FEEDBACK_LABELS.get(f_t, f_t)}")

    st.markdown("#### Your favourite books")
    st.caption(f"Pick up to five from the real catalogue. ({len(st.session_state['seed_ids'])}/5 selected)")
    fav_search = st.text_input("Search for a book", placeholder="Type a title or author", key="favorite_search")
    options = catalog.sort_values(["rating_count", "title"], ascending=[False, True], kind="stable")
    if fav_search.strip():
        mask = (options.title.str.contains(fav_search.strip(), case=False, regex=False, na=False) |
                options.author.str.contains(fav_search.strip(), case=False, regex=False, na=False))
        options = options[mask]
    options = options.head(100)
    if options.empty:
        st.info("No books matched that search. Try another title or author.")
    else:
        label_to_id = {}
        label_counts = {}
        for _, row in options.iterrows():
            label = book_label(row)
            label_counts[label] = label_counts.get(label, 0) + 1
            if label in label_to_id:
                label = f"{label} · edition {label_counts[book_label(row)]}"
            label_to_id[label] = str(row.book_id)
        selected_label = st.selectbox("Choose a book to add", options=list(label_to_id), index=None,
                                      placeholder="Search results appear here", key="favorite_candidate")
        if selected_label:
            candidate_id = label_to_id[selected_label]
            candidate_row = catalog.loc[catalog.book_id.astype(str).eq(candidate_id)].iloc[0]
            preview, action = st.columns([1, 3], vertical_alignment="center")
            with preview:
                render_cover(candidate_row)
            with action:
                st.markdown(f"**{escape(str(candidate_row.title))}**")
                st.caption(f"{escape(str(candidate_row.author or 'Author not listed'))} · choose this book to shape your list")
                if candidate_id in st.session_state["seed_ids"]:
                    st.success("Already in your picks")
                elif len(st.session_state["seed_ids"]) >= 5:
                    st.info("You can choose up to five books. Remove one below to add another.")
                elif st.button("Add to my picks", type="primary", key="add_seed_book"):
                    st.session_state["seed_ids"].append(candidate_id)
                    shelf = list(st.session_state.get("favorite_ids", []))
                    if candidate_id not in shelf:
                        shelf.append(candidate_id)
                    st.session_state["favorite_ids"] = shelf
                    st.rerun()
    seeds = list(st.session_state["seed_ids"])
    if seeds:
        st.markdown("##### Your picks")
        seed_columns = st.columns(min(5, len(seeds)))
        for seed_id, seed_column in zip(seeds, seed_columns):
            row = catalog.loc[catalog.book_id.astype(str).eq(str(seed_id))].iloc[0]
            with seed_column:
                render_cover(row, small=True)
                st.markdown(f"**{escape(str(row.title))}**")
                st.caption(str(row.author or "Author not listed"))
                if st.button("Remove", key=f"remove_seed_{seed_id}"):
                    st.session_state["seed_ids"] = [value for value in seeds if value != seed_id]
                    st.session_state["favorite_ids"] = [value for value in st.session_state.get("favorite_ids", []) if value != seed_id]
                    st.rerun()
    else:
        st.markdown("<div class='shelf-empty'>Your picks will appear here. Search for a book you enjoyed to get started.</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("##### Ready when you are")
    st.caption("Your picks work on their own. Open preferences to add an existing reader profile or choose a different recommendation approach.")
    user_id = str(st.session_state.get("reader_id", "")).strip()
    user_status, user_rating_count = explain_user_id(collaborative_model, user_id)
    methods = {
        "Based on my favourite books": "Content-based",
        "Based on reader ratings": "Collaborative",
        "A blend of both": "Hybrid",
    }
    with st.expander("Advanced preferences", expanded=False):
        st.caption("Optional: add a Book-Crossing reader profile for genuine rating signals, or choose another recommendation strategy.")
        user_id = st.text_input("Book-Crossing reader ID", placeholder="Optional", key="reader_id").strip()
        user_status, user_rating_count = explain_user_id(collaborative_model, user_id)
        if user_id and user_status.startswith("That ID"):
            st.warning(user_status)
        elif user_id and user_rating_count < 2:
            st.info(user_status)
        elif user_id:
            st.success(user_status)
        method_label = st.radio("Recommendation approach", list(methods), key="for_you_method_v2")
        if method_label == "Based on my favourite books":
            st.caption("Finds titles with similar title, author, or publisher details.")
        elif method_label == "Based on reader ratings":
            st.caption("Uses your Book-Crossing history when possible. Rating overlap is sparse, so ReadWise falls back to your selected books or highly rated catalogue titles when needed.")
        else:
            st.caption("Combines catalogue details and reader-rating evidence when both are available.")
            metadata_percent = st.slider("Influence of book details", 0, 100, 50, 5,
                                         format="%d%%", key="metadata_weight")
            if metadata_percent == 0:
                st.caption("With no book-detail signal, ReadWise uses a catalogue rating fallback because reader overlap is sparse.")
            else:
                st.caption(f"Book details influence: {metadata_percent}%. Available reader-rating evidence can rerank matching books; missing rating evidence does not lower a content match.")
        recency_percent = st.slider("Influence of publication recency", 0, 50, 15, 5,
                                    format="%d%%", key="recency_weight_pct")
        if recency_percent > 0:
            st.caption(f"Recency influence: {recency_percent}%. Surfaces contemporary titles without overwhelming catalogue relevance or rating evidence.")
    method_label = st.session_state.get("for_you_method_v2", "Based on my favourite books")
    method = methods[method_label]
    alpha = float(st.session_state.get("metadata_weight", 50)) / 100
    recency_weight = float(st.session_state.get("recency_weight_pct", 15)) / 100
    feedback_tuple = tuple(sorted(feedback_map.items()))
    current_signature = reading_list_signature(seeds, user_id, method, alpha, recency_weight, feedback_tuple)
    if st.button("Find My Next Read", type="primary", key="build_list", width="stretch"):
        error = None
        if method in {"Content-based", "Hybrid"} and not seeds and not user_id and not liked_ids:
            error = "Choose at least one book first (or like a book in the catalogue), or enter a reader ID for rating-based suggestions."
        if method == "Collaborative" and not user_id:
            error = "Enter a valid Book-Crossing reader ID to use reader-pattern recommendations."
        if user_id and user_status.startswith("That ID"):
            error = user_status
        if method == "Content-based" and not seeds and not liked_ids:
            error = "Choose at least one book or like a book so ReadWise can find similar catalogue records."
        if error:
            st.error(error)
        else:
            with st.spinner("Looking for books that fit your reading taste…"):
                try:
                    known_rated = collaborative_model.user_rated.get(user_id, set()) if user_id else set()
                    known_rated_ids = set(collaborative_model.books.iloc[list(known_rated)].book_id.astype(str)) if known_rated else set()
                    all_excluded = known_rated_ids | suppressed_ids | set(map(str, liked_ids))
                    seen_ids = set(map(str, seeds)) | known_rated_ids | suppressed_ids | set(map(str, liked_ids))
                    if method == "Content-based":
                        result = content_model.recommend(seeds, 50, exclude_ids=all_excluded, recency_weight=recency_weight, liked_ids=list(liked_ids))
                        if not result.empty:
                            clean_years = parse_publication_year(result["year"]) if "year" in result else pd.Series(dtype=float)
                            is_recent = clean_years.ge(2015)
                            has_fb = result.get("feedback_boost", pd.Series(0, index=result.index)).fillna(0).gt(0)
                            result["explanation"] = np.where(
                                has_fb,
                                "Boosted by positive feedback on your liked books.",
                                np.where(
                                    is_recent,
                                    "Recent publication matching your reading taste; catalogue metadata provides the match.",
                                    "Recommended because its title, author, or publisher metadata resembles your selected books."
                                )
                            )
                    elif method == "Collaborative":
                        result = reader_rating_fallback(content_model, collaborative_model, catalog,
                                                        popular, user_id, seeds, top_k=50, recency_weight=recency_weight,
                                                        liked_ids=list(liked_ids), exclude_ids=all_excluded)
                    else:
                        result = hybrid_recommend(books, content_model, collaborative_model,
                                                  user_id or None, seeds, 50, alpha, recency_weight=recency_weight,
                                                  liked_ids=list(liked_ids), exclude_ids=all_excluded)
                    result = exclude_seen_editions(result, catalog, seen_ids)
                    result = result.head(10).reset_index(drop=True)
                except Exception:
                    st.error("ReadWise couldn’t build that list from the available data. Your picks are saved; try another approach or book.")
                else:
                    st.session_state["for_you_result"] = result
                    st.session_state["for_you_signature"] = current_signature
                    st.session_state["for_you_method_used"] = method_label
    st.divider()
    st.markdown("#### 4 · Your next reads")
    if "for_you_result" not in st.session_state:
        st.markdown("<div class='shelf-empty'>Your recommendations will appear here after you choose a few books and ask ReadWise to find your next read.</div>", unsafe_allow_html=True)
    elif st.session_state.get("for_you_signature") != current_signature:
        st.info("Your picks or settings changed. Build your reading list again to refresh these results.")
    elif st.session_state["for_you_result"].empty:
        st.info("There isn’t enough matching evidence to build a list with these inputs. Try another book, or choose the blended approach with a valid reader ID.")
    else:
        st.caption(f"Built with: {st.session_state.get('for_you_method_used', 'your selected approach')}. Reasons describe the catalogue or rating evidence used for each match.")
        render_book_grid(st.session_state["for_you_result"], catalog, columns=4, controls=True, deduplicate=True)
    shelf = catalog[catalog.book_id.astype(str).isin(st.session_state.get("favorite_ids", []))]
    if not shelf.empty:
        st.divider()
        st.markdown("#### Saved to your shelf")
        render_book_grid(shelf, catalog, columns=5, controls=False)


elif page == "How it works":
    st.markdown("<div class='section-kicker'>A CLEAR LOOK UNDER THE COVER</div>", unsafe_allow_html=True)
    st.title("How ReadWise finds a book")
    st.markdown("A short path from your taste to a useful, explainable reading list.")
    steps = [
        ("01 · Choose", "Pick books you love, use an optional Book-Crossing reader profile, or rely on your saved likes."),
        ("02 · Gather", "Build a candidate list from available catalogue details and explicit ratings."),
        ("03 · Rank", "Order candidates with your chosen approach: similar book details, reader behaviour, or both."),
        ("04 · Feedback", "Your local feedback (likes, passes, and read markings) stored in SQLite continuously refines suggestions."),
        ("05 · Explain", "Show the evidence behind each suggestion and let you open its catalogue record."),
        ("06 · Evaluate", "Measure ranking quality with books held out from the training interactions."),
    ]
    for heading, description in steps:
        number, text_col = st.columns([1, 5], vertical_alignment="center")
        with number:
            st.markdown(f"<div class='section-kicker'>{heading}</div>", unsafe_allow_html=True)
        with text_col:
            st.markdown(description)
        st.divider()
    st.info("Book-Crossing provides titles, authors, publishers and ratings, but no dependable genre or synopsis for every book. ReadWise does not invent those details.")
    with st.expander("Technical details"):
        st.markdown("**Book details:** TF-IDF and cosine similarity compare title, author and publisher text. This is catalogue similarity, not plot or mood understanding.")
        st.markdown("**Reader behaviour:** Item-based cosine similarity uses nonzero ratings after centering each reader around their own mean. It requires enough overlap between books.")
        st.markdown("**Blend:** Candidate scores are normalized within each source and combined using the selected book-detail weight. If one source has no candidates, the available source is used.")
        st.markdown(f"**Feedback loop:** Reader feedback is stored locally in SQLite (`{FEEDBACK_DB_PATH}`) and kept strictly separate from the source Book-Crossing data. Positive feedback boosts similar titles using content TF-IDF vectors (allowing even brand-new books with zero rating history to shape recommendations). Negative and already-read feedback suppresses titles and matching editions from future lists.")
        st.markdown("**Offline evaluation:** Ratings are split per user before fitting the rating-based model. See Results for the actual cohort, relevance rule, metrics and limitations.")
    st.caption("The source is the old, sparse Book-Crossing collection. Recommendations can be empty when the available evidence is too thin.")


elif page == "Results":
    st.markdown("<div class='section-kicker'>THE EXPERIMENT</div>", unsafe_allow_html=True)
    st.title("How the models performed")
    st.markdown("This page reports a reproducible offline ranking experiment on real Book-Crossing ratings. It is separate from the everyday book-discovery experience.")
    result_path = OUTPUT_DIR / "recommendation_evaluation_v7_results.csv"
    detail_path = OUTPUT_DIR / "recommendation_evaluation_v7_details.json"
    if not result_path.exists() or not detail_path.exists():
        result_path = OUTPUT_DIR / "recommendation_evaluation_v5_results.csv"
        detail_path = OUTPUT_DIR / "recommendation_evaluation_v5_details.json"
    if result_path.exists() and detail_path.exists():
        metrics = pd.read_csv(result_path)
        details = json.loads(detail_path.read_text(encoding="utf-8"))
        sampled_users = int(details.get("sampled_eligible_users", metrics.get("sampled_eligible_users", pd.Series([metrics.evaluated_users.max()])).max()))
        relevant_users = int(details.get("users_with_relevant_holdout", metrics.get("users_with_relevant_holdout", pd.Series([metrics.evaluated_users.max()])).max()))
        relevant_items = int(details["relevant_heldout_interactions"])
        k = int(metrics.top_k.max())
        a, b, c, d = st.columns(4)
        a.metric("Sampled eligible readers", f"{sampled_users:,}")
        b.metric("Readers with a relevant held-out book", f"{relevant_users:,}")
        c.metric("Held-out interactions", f"{int(details.get('all_heldout_explicit_interactions', relevant_items)):,}")
        d.metric("Relevant held-out ratings", f"{relevant_items:,}")
        st.caption(f"{details['dataset']} · Seed {details['seed']} · {details['strategy']}. Relevance means a held-out explicit rating of {details['threshold']}–10. Zero-valued interactions are excluded.")
        if details.get("feature_fit_scope"):
            st.caption(details["feature_fit_scope"])
        st.subheader(f"Ranking quality at K = {k}")
        chart_col_a, chart_col_b = st.columns(2)
        for target, metric_key, title, color in [
            (chart_col_a, "precision_at_10", "Precision@10", "#5a4bc4"),
            (chart_col_b, "recall_at_10", "Recall@10", "#d26b52"),
        ]:
            with target:
                st.markdown(f"**{title}**")
                fig = px.bar(metrics, x="model", y=metric_key, color_discrete_sequence=[color], text_auto=".4f")
                fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#20212a",
                                  showlegend=False, yaxis_title="Mean score", xaxis_title="",
                                  margin=dict(l=10, r=10, t=15, b=10), height=320)
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if "empty_recommendation_users" not in metrics:
            metrics["empty_recommendation_users"] = 0
        table = metrics[["model", "precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10",
                         "ndcg_at_10", "map_at_10", "catalogue_coverage", "intra_list_diversity",
                         "empty_recommendation_users"]].rename(
            columns={"model": "Approach", "precision_at_5": "Precision@5",
                     "precision_at_10": "Precision@10", "recall_at_5": "Recall@5",
                     "recall_at_10": "Recall@10", "ndcg_at_10": "NDCG@10",
                     "map_at_10": "MAP@10", "catalogue_coverage": "Catalogue coverage",
                     "intra_list_diversity": "List diversity",
                     "empty_recommendation_users": "Empty lists"})
        st.dataframe(table.style.format({column: "{:.4f}" for column in table.columns if column != "Approach"}),
                     hide_index=True, width="stretch")
        best_precision = metrics.loc[metrics.precision_at_10.idxmax(), "model"]
        best_recall = metrics.loc[metrics.recall_at_10.idxmax(), "model"]
        st.subheader("What these results mean")
        st.markdown(f"Across the {relevant_users} readers who had at least one relevant book in the held-out portion, **{best_precision}** had the highest mean Precision@10 and **{best_recall}** had the highest mean Recall@10 in this run. Precision divides relevant suggestions by K even when a model returns a short list, so unfilled places count as misses. Recall is averaged per reader, then those reader-level values are averaged.")
        st.caption(f"Average list diversity is 1 minus mean pairwise TF-IDF cosine similarity. Catalogue coverage is the fraction of the full catalogue appearing in top-10 lists. The item-user matrix density was {details['item_user_matrix_density']:.6%}.")
        st.markdown("These are low offline scores on an old, sparse dataset. They show that the current metadata and rating signals have limited ability to recover held-out favourites; they are not evidence of production-quality personalization.")
    else:
        st.warning("Real-data evaluation has not run yet. Run .venv\\Scripts\\python.exe scripts\\evaluate_models.py to create measured results.")
    with st.expander("Evaluation protocol"):
        min_ratings = int(details.get("minimum_explicit_ratings", 5))
        min_train = int(details.get("minimum_train_ratings", 2))
        holdout = float(details.get("holdout_fraction", 0.2))
        st.markdown(f"Readers with at least {min_ratings} explicit ratings matched to catalogue books are eligible; {sampled_users} were sampled using seed {details['seed']}. For every sampled reader, {holdout:.0%} of their matched explicit ratings are held out before model fitting. At least {min_train} ratings remain for training. A relevant book has a held-out rating of {details['threshold']}/10 or higher. Readers without a relevant held-out book are excluded from the per-reader metric average, but all held-out pairs stay out of training (measured overlap: {details['training_heldout_pair_overlap']}). All models share the same catalogue and candidate rules; the popularity baseline is counted using training ratings only. Precision divides hits by K (empty slots count as misses); recall, NDCG@10 and MAP@10 are macro-averaged over readers with at least one relevant held-out book.")
        st.caption(f"{details.get('excluded_from_metric_average', details.get('excluded', ''))} · {details.get('candidate_and_seen_item_rule', '')}")
    with st.expander("Limitations and next steps"):
        st.markdown("Book-Crossing is old and sparse, and its readers are not a representative sample. Title/author/publisher text is a narrow signal; rating overlap is limited. This bounded run supports a project comparison, not a broad claim about real-world reader satisfaction.")
        st.markdown("Future work could add richer, licensed book descriptions, confidence-aware score blending, a larger user sample, and a reader study. None of those signals are currently present in these results.")
