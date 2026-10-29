"""Tests for cover handling, URL normalization, fallback behavior, and text decoding."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.covers import (
    clean_text_encoding,
    is_valid_url,
    normalize_cover_url,
    is_usable_cover_url,
    resolve_cover_url,
    render_cover_fallback_html,
    render_cover_html,
)


def test_clean_text_encoding_mojibake_repairs():
    # Double-mojibake pattern from Book-Crossing
    assert clean_text_encoding("Das groÃ?Â?e BÃ?Â¶se- MÃ?Â¤dchen- Lesebuch.") == "Das große Böse- Mädchen- Lesebuch."
    assert clean_text_encoding("Der KÃ?Â¶nig in Gelb.") == "Der König in Gelb."
    assert clean_text_encoding("Der illustrierte Mann. ErzÃ?Â¤hlungen.") == "Der illustrierte Mann. Erzählungen."
    assert clean_text_encoding("LÃ?Â¶wenzahnwein. Roman.") == "Löwenzahnwein. Roman."

    # Single mojibake from UTF-8 decoded as Latin-1 / CP1252
    assert clean_text_encoding("groÃŸe") == "große"
    assert clean_text_encoding("Schwarzer Tee mit drei StÃ¼ck Zucker") == "Schwarzer Tee mit drei Stück Zucker"
    assert clean_text_encoding("Antoine de Saint-ExupÃ©ry") == "Antoine de Saint-Exupéry"
    assert clean_text_encoding("Isabel-Clara SimÃ³") == "Isabel-Clara Simó"

    # Already-correct text must remain intact
    assert clean_text_encoding("Café; stories") == "Café; stories"
    assert clean_text_encoding("James") == "James"
    assert clean_text_encoding("Tomorrow, and Tomorrow, and Tomorrow") == "Tomorrow, and Tomorrow, and Tomorrow"
    assert clean_text_encoding("Ãncora") == "Ãncora"
    assert clean_text_encoding("") == ""
    assert clean_text_encoding(None) == ""


def test_is_valid_url_and_normalization():
    assert is_valid_url("https://covers.openlibrary.org/b/isbn/9780385537858-M.jpg")
    assert is_valid_url("http://images.amazon.com/images/P/0380000059.01.MZZZZZZZ.jpg")
    assert not is_valid_url("")
    assert not is_valid_url("not_a_url")
    assert not is_valid_url("ftp://example.com/cover.jpg")
    assert not is_valid_url(None)

    # Normalization forces HTTPS
    assert normalize_cover_url("http://images.amazon.com/test.jpg") == "https://images.amazon.com/test.jpg"
    assert normalize_cover_url("https://images.amazon.com/test.jpg") == "https://images.amazon.com/test.jpg"
    assert normalize_cover_url("") == ""
    assert normalize_cover_url("invalid") == ""


def test_is_usable_cover_url_rejects_spacers_and_errors():
    # 1. 43-byte 1x1 transparent GIF (spacer) should be rejected
    mock_spacer = MagicMock()
    mock_spacer.status_code = 200
    mock_spacer.headers = {"Content-Type": "image/gif", "Content-Length": "43"}
    mock_spacer.iter_content.return_value = [b"GIF89a" + b"\x00" * 37]

    with patch("src.covers.requests.get") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_spacer
        assert not is_usable_cover_url("https://example.com/spacer.gif")

    # 2. HTTP 404 / 403 should be rejected
    mock_err = MagicMock()
    mock_err.status_code = 404
    mock_err.headers = {"Content-Type": "text/html"}
    with patch("src.covers.requests.get") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_err
        assert not is_usable_cover_url("https://example.com/missing.jpg")

    # 3. HTML response with 200 should be rejected
    mock_html = MagicMock()
    mock_html.status_code = 200
    mock_html.headers = {"Content-Type": "text/html"}
    with patch("src.covers.requests.get") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_html
        assert not is_usable_cover_url("https://example.com/page.html")

    # 4. Valid image response should be accepted
    mock_ok = MagicMock()
    mock_ok.status_code = 200
    mock_ok.headers = {"Content-Type": "image/jpeg", "Content-Length": "15420"}
    mock_ok.iter_content.return_value = [b"\xff\xd8\xff" + b"\x00" * 500]
    with patch("src.covers.requests.get") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_ok
        assert is_usable_cover_url("https://example.com/real_cover.jpg")


def test_resolve_cover_url_with_fallbacks():
    # Mock is_usable_cover_url for predictable routing
    def fake_usable(url, timeout=2.0):
        if "amazon" in url and "0380000059" in url:
            return False  # CLOUT spacer on Amazon
        if "openlibrary" in url and "0380000059" in url:
            return True   # CLOUT valid on Open Library
        if "openlibrary" in url and "9780385537858" in url:
            return True   # James valid on Open Library
        if "9780063076235" in url:
            return False  # Yellowface missing cover everywhere
        return "valid_cover" in url

    with patch("src.covers.is_usable_cover_url", side_effect=fake_usable):
        # CLOUT on Amazon should fall back to Open Library
        res_clout = resolve_cover_url("http://images.amazon.com/images/P/0380000059.01.MZZZZZZZ.jpg", "0380000059")
        assert "covers.openlibrary.org" in res_clout
        assert "0380000059" in res_clout

        # James with empty cover should resolve to Open Library
        res_james = resolve_cover_url("", "9780385537858")
        assert "covers.openlibrary.org" in res_james

        # Yellowface with missing cover should return empty string
        res_yf = resolve_cover_url("", "9780063076235")
        assert res_yf == ""


def test_render_cover_fallback_html():
    html = render_cover_fallback_html("The Great Gatsby", "F. Scott Fitzgerald", width=170, height=264)
    assert "class='cover-fallback" in html
    assert "max-width:170px" in html
    assert "height:264px" in html
    assert "display:flex;" in html
    assert "The Great Gatsby" in html
    assert "F. Scott Fitzgerald" in html

    # Hidden variant for embed behind img
    hidden_html = render_cover_fallback_html("Hidden Book", "Author", is_hidden=True)
    assert "display:none;" in hidden_html


def test_render_cover_html_never_leaves_blank_area():
    # 1. Book with usable cover
    with patch("src.covers.resolve_cover_url", return_value="https://example.com/cover.jpg"):
        row = {"book_id": "123", "title": "Real Book", "author": "Real Author", "cover_url": "https://example.com/cover.jpg"}
        html = render_cover_html(row)
        assert "<div class='cover-frame'" in html
        assert "<img src='https://example.com/cover.jpg'" in html
        assert "onload='if(this.naturalWidth<=1" in html
        assert "onerror='this.style.display=" in html
        assert "cover-fallback" in html
        assert "display:none;" in html

    # 2. Book with NO usable cover: renders fallback directly with title and author
    with patch("src.covers.resolve_cover_url", return_value=""):
        row = {"book_id": "9780063076235", "title": "Yellowface", "author": "R. F. Kuang", "cover_url": ""}
        html = render_cover_html(row)
        assert "<img" not in html
        assert "cover-fallback" in html
        assert "Yellowface" in html
        assert "R. F. Kuang" in html
        assert "display:flex;" in html

    # 3. Handles pandas Series and NaN values gracefully
    with patch("src.covers.resolve_cover_url", return_value=""):
        series = pd.Series({"book_id": "999", "title": "Untitled Book", "author": float("nan"), "cover_url": float("nan")})
        html = render_cover_html(series)
        assert "Untitled Book" in html
        assert "fallback-author" not in html
