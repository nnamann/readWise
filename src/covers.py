"""Cover image validation, normalization, resolution, and fallback rendering for ReadWise.

Handles missing covers, broken links, 1x1 transparent spacer GIFs, unsupported URL formats,
timeouts, and non-image responses while preserving genuine provider covers and showing
consistent, aligned fallbacks with title and author.
"""

from __future__ import annotations

import functools
from html import escape
import re
import urllib.parse
import urllib.request
import urllib.error

import requests


def clean_text_encoding(text: str) -> str:
    """Repair corrupted mojibake encoding while leaving already-correct text untouched.

    Handles:
    1. Double-mojibake artifacts in legacy Book-Crossing records (e.g. ``Ã?Â(.)`` -> original character).
    2. Single mojibake where UTF-8 bytes were decoded as Latin-1 or CP1252 (e.g. ``groÃŸe`` -> ``große``).
    3. Preserves pure Latin-1 (e.g. ``Café``) and standard ASCII/UTF-8 strings.
    """
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    # Fast path: if no mojibake indicators, return immediately
    if "Ã" not in text and "â€" not in text and "Â" not in text and "â\x80" not in text:
        return text

    # 1. Repair double-mojibake pattern: Ã?Â(.) or Ã?Â?
    def _repl_double(match: re.Match) -> str:
        ch = match.group(1)
        code = 0x9F if ch == "?" else ord(ch)
        try:
            return bytes([0xC3, code]).decode("utf-8")
        except Exception:
            return match.group(0)

    repaired = re.sub(r"Ã\?Â(.)", _repl_double, text)

    # 2. Repair single mojibake: UTF-8 decoded as CP1252 or Latin-1
    if any(m in repaired for m in ("Ã", "Â", "â€", "â\x80")):
        for enc in ("cp1252", "latin-1"):
            try:
                candidate = repaired.encode(enc).decode("utf-8")
                repaired = candidate
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue

    return repaired


def is_valid_url(url: str) -> bool:
    """Validate whether url is a well-formed HTTP/HTTPS URL string."""
    if not isinstance(url, str):
        return False
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        return bool(parsed.scheme in ("http", "https") and parsed.netloc)
    except Exception:
        return False


def normalize_cover_url(url: str) -> str:
    """Normalize cover URL to HTTPS, stripping whitespace and rejecting malformed URLs."""
    if not is_valid_url(url):
        return ""
    url = url.strip()
    if url.startswith("http://"):
        url = "https://" + url[7:]
    return url


@functools.lru_cache(maxsize=4096)
def is_usable_cover_url(url: str, timeout: float = 2.0) -> bool:
    """Check whether a remote cover URL responds with a genuine image.

    Rejects:
    - Non-200 HTTP statuses (e.g. 403, 404, 500)
    - Non-image content types (e.g. HTML error pages)
    - 1x1 transparent placeholder GIFs (e.g. Amazon / Open Library 43-byte spacers)
    - Timeouts and connection failures
    """
    if not is_valid_url(url):
        return False

    norm_url = normalize_cover_url(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 ReadWise/1.0"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    try:
        # Stream first chunk to inspect headers and avoid downloading unnecessary data
        with requests.get(norm_url, headers=headers, timeout=timeout, stream=True) as resp:
            if resp.status_code != 200:
                return False

            content_type = resp.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                return False

            # Check content length header if present
            cl_header = resp.headers.get("Content-Length")
            if cl_header and cl_header.isdigit():
                cl_int = int(cl_header)
                if cl_int <= 100:  # Amazon and OL return 43-byte 1x1 GIFs for missing covers
                    return False

            # Read first chunk to verify actual byte payload size
            chunk = next(iter(resp.iter_content(chunk_size=512)), b"")
            if len(chunk) <= 100 and not resp.headers.get("Content-Length"):
                return False

            return True
    except (requests.RequestException, OSError, TimeoutError):
        return False


def get_open_library_cover_url(isbn: str) -> str:
    """Return Open Library standard cover URL for a valid ISBN with default=false."""
    clean_isbn = str(isbn).replace("-", "").strip().upper()
    if not clean_isbn:
        return ""
    return f"https://covers.openlibrary.org/b/isbn/{clean_isbn}-M.jpg?default=false"


def resolve_cover_url(cover_url: str, book_id: str = "") -> str:
    """Resolve a usable cover URL for a book, falling back to Open Library where possible.

    If the provided cover URL is empty, broken, or a 1x1 placeholder, checks if
    the book_id is a valid ISBN and has an Open Library cover.
    """
    norm_url = normalize_cover_url(cover_url)

    # 1. If an existing URL is provided, check if it's usable
    if norm_url:
        if is_usable_cover_url(norm_url):
            return norm_url

    # 2. Fallback to Open Library if book_id is an ISBN
    clean_id = str(book_id).replace("-", "").strip().upper()
    if len(clean_id) in (10, 13) and (clean_id[:9].isdigit() or clean_id[:12].isdigit()):
        ol_url = get_open_library_cover_url(clean_id)
        if is_usable_cover_url(ol_url):
            return ol_url

    return ""


def render_cover_fallback_html(title: str, author: str = "", width: int = 170,
                               height: int = 264, extra_class: str = "",
                               is_hidden: bool = False) -> str:
    """Generate consistent, polished fallback cover HTML showing title and author."""
    clean_title = clean_text_encoding(title)
    clean_author = clean_text_encoding(author)
    display_title = escape(str(clean_title)[:80], quote=True)
    display_author = escape(str(clean_author)[:60], quote=True) if clean_author else ""

    hidden_style = "display:none;" if is_hidden else "display:flex;"
    author_markup = f"<div class='fallback-author'>{display_author}</div>" if display_author else ""

    return (
        f"<div class='cover-fallback{extra_class}' style='{hidden_style}max-width:{width}px;height:{height}px;'>"
        f"<div class='fallback-title'>{display_title}</div>"
        f"{author_markup}"
        f"</div>"
    )


def render_cover_html(row: dict | object, large: bool = False, small: bool = False) -> str:
    """Generate robust HTML markup for a book cover, guaranteeing no blank image areas."""
    get_fn = getattr(row, "get", None) or (lambda k, d="": getattr(row, k, d))
    raw_url = str(get_fn("cover_url", "") or "").strip()
    book_id = str(get_fn("book_id", "") or "").strip()
    raw_title = str(get_fn("title", "ReadWise") or "ReadWise")
    raw_author = str(get_fn("author", "") or "")

    if raw_url.lower() in ("nan", "none", "null"):
        raw_url = ""
    if raw_author.lower() in ("nan", "none", "null"):
        raw_author = ""

    title = clean_text_encoding(raw_title)
    author = clean_text_encoding(raw_author)

    width = 115 if small else (205 if large else 170)
    height = 155 if small else (245 if large else 264)
    extra_class = " detail-cover" if large else ""

    # Resolve usable URL (checks URL validity, 1x1 GIFs, and Open Library fallback)
    usable_url = resolve_cover_url(raw_url, book_id)

    if usable_url:
        escaped_url = escape(usable_url, quote=True)
        escaped_title = escape(title[:58], quote=True)
        fallback_html = render_cover_fallback_html(title, author, width=width, height=height,
                                                   extra_class=extra_class, is_hidden=True)
        # Two-layer defense:
        # onload checks naturalWidth/Height to catch 1x1 spacer GIFs from CDNs
        # onerror catches HTTP errors, network drops, or broken image decoding
        return (
            f"<div class='cover-frame' style='max-width:{width}px;height:{height}px;'>"
            f"<img src='{escaped_url}' alt='Cover: {escaped_title}' loading='lazy' "
            f"onload='if(this.naturalWidth<=1||this.naturalHeight<=1){{this.style.display=\"none\";this.nextElementSibling.style.display=\"flex\";}}' "
            f"onerror='this.style.display=\"none\";this.nextElementSibling.style.display=\"flex\";'>"
            f"{fallback_html}"
            f"</div>"
        )

    # When no usable cover exists, show polished fallback directly
    return render_cover_fallback_html(title, author, width=width, height=height,
                                      extra_class=extra_class, is_hidden=False)
