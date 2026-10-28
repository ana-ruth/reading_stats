"""Look up genres and moods for Goodreads books.

Sources, in order:
  1. Hardcover (https://hardcover.app) - community genres AND moods. Needs a free
     API token, read from HARDCOVER_TOKEN in .streamlit/secrets.toml or the environment.
  2. Open Library (https://openlibrary.org) - subjects mapped to genres. No key needed.

Results are cached in book_metadata_cache.json next to this file, so each book is
only looked up once. Network failures are never cached, so they are retried next time.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_PATH = Path(__file__).with_name("book_metadata_cache.json")
HARDCOVER_URL = "https://api.hardcover.app/v1/graphql"
OPENLIBRARY_URL = "https://openlibrary.org"
HTTP_HEADERS = {"User-Agent": "reading-stats-streamlit-app"}
HARDCOVER_BATCH = 25
OPENLIBRARY_BATCH = 40
TAGS_PER_BOOK = 3

# Tags too broad to be useful as a genre
GENERIC_TAGS = {
    "fiction", "nonfiction", "non-fiction", "general", "novel", "novels", "literature",
    "adult", "adult fiction", "audiobook", "audiobooks", "books", "ebook", "book",
    "accessible book", "protected daisy", "in library", "large type books",
}

# Maps raw subjects/tags onto one consistent genre list. First match wins, so more
# specific keywords ("science fiction") come before broader ones ("science").
GENRE_KEYWORDS = [
    ("science fiction", "Science Fiction"), ("sci-fi", "Science Fiction"),
    ("fiction, historical", "Historical Fiction"), ("historical fiction", "Historical Fiction"),
    ("fantasy", "Fantasy"),
    ("romance", "Romance"), ("love stories", "Romance"),
    ("mystery", "Mystery"), ("detective", "Mystery"),
    ("thriller", "Thriller"), ("suspense", "Thriller"),
    ("horror", "Horror"), ("ghost stories", "Horror"),
    ("dystopia", "Dystopian"),
    ("young adult", "Young Adult"),
    ("juvenile fiction", "Children's"), ("children", "Children's"),
    ("true crime", "True Crime"),
    ("autobiography", "Biography & Memoir"), ("biography", "Biography & Memoir"),
    ("memoir", "Biography & Memoir"),
    ("self-help", "Self-Help"), ("self help", "Self-Help"), ("personal development", "Self-Help"),
    ("poetry", "Poetry"),
    ("graphic novel", "Graphic Novels & Comics"), ("comic", "Graphic Novels & Comics"),
    ("manga", "Graphic Novels & Comics"),
    ("humor", "Humor"), ("humour", "Humor"),
    ("philosophy", "Philosophy"), ("psychology", "Psychology"),
    ("business", "Business & Economics"), ("economics", "Business & Economics"),
    ("history", "History"),
    ("science", "Science"),
    ("classic", "Classics"),
    ("literary fiction", "Literary Fiction"), ("domestic fiction", "Literary Fiction"),
    ("contemporary", "Contemporary"),
]


class HardcoverAuthError(Exception):
    """The Hardcover token was rejected."""


# ---------- helpers ----------

def clean_isbn(raw):
    """Goodreads writes ISBNs as ="9780141439518"; return digits only, or None."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    isbn = re.sub(r"[^0-9Xx]", "", str(raw)).upper()
    return isbn if len(isbn) in (10, 13) else None


def search_title(title):
    """Drop the series part Goodreads adds: 'Dune (Dune, #1)' -> 'Dune'."""
    return re.sub(r"\s*\([^)]*#[^)]*\)\s*$", "", str(title)).strip()


def book_key(row):
    for col in ("ISBN13", "ISBN"):
        isbn = clean_isbn(row.get(col))
        if isbn:
            return f"isbn:{isbn}"
    return f"title:{search_title(row.get('Title', '')).lower()}|{str(row.get('Author', '')).lower()}"


def to_genre(tag):
    """Map one raw subject/tag to a canonical genre, or None if it is too generic."""
    text = str(tag).strip().lower()
    if not text or text in GENERIC_TAGS:
        return None
    for keyword, genre in GENRE_KEYWORDS:
        if keyword in text:
            return genre
    return None


def normalize_genres(tags, keep_unmapped=False):
    genres = []
    for tag in tags:
        genre = to_genre(tag)
        if genre is None and keep_unmapped:
            text = str(tag).strip()
            if text and text.lower() not in GENERIC_TAGS:
                genre = text.title()
        if genre and genre not in genres:
            genres.append(genre)
        if len(genres) == TAGS_PER_BOOK:
            break
    return genres


def tag_names(cached_tags, category):
    """Pull tag names for one category ('Genre', 'Mood') from Hardcover's cached_tags."""
    if isinstance(cached_tags, str):
        try:
            cached_tags = json.loads(cached_tags)
        except ValueError:
            return []
    entries = (cached_tags or {}).get(category) or []
    names = []
    for entry in entries:
        name = entry.get("tag") if isinstance(entry, dict) else entry
        if name:
            names.append(str(name))
    return names


def load_cache():
    try:
        return json.loads(CACHE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_cache(cache):
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True))
    except OSError:
        pass  # read-only host (e.g. some cloud deployments): just skip the disk cache


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------- Hardcover ----------

HARDCOVER_QUERY = """
query BooksByIsbn($isbns: [String!]) {
  editions(where: {_or: [{isbn_13: {_in: $isbns}}, {isbn_10: {_in: $isbns}}]}) {
    isbn_13
    isbn_10
    book { cached_tags }
  }
}
"""


def hardcover_lookup(session, token, isbns):
    """Return {isbn: {"genres": [...], "moods": [...]}} for the ISBNs Hardcover knows."""
    auth = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    resp = session.post(
        HARDCOVER_URL,
        json={"query": HARDCOVER_QUERY, "variables": {"isbns": isbns}},
        headers={**HTTP_HEADERS, "authorization": auth, "content-type": "application/json"},
        timeout=20,
    )
    if resp.status_code in (401, 403):
        raise HardcoverAuthError(resp.text[:200])
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        message = str(payload["errors"])[:300]
        if "jwt" in message.lower() or "auth" in message.lower():
            raise HardcoverAuthError(message)
        raise RuntimeError(message)

    wanted = set(isbns)
    found = {}
    for edition in payload.get("data", {}).get("editions", []):
        tags = (edition.get("book") or {}).get("cached_tags")
        result = {
            "genres": normalize_genres(tag_names(tags, "Genre"), keep_unmapped=True),
            "moods": [m.title() for m in tag_names(tags, "Mood")[:TAGS_PER_BOOK]],
        }
        for isbn in (edition.get("isbn_13"), edition.get("isbn_10")):
            if isbn in wanted:
                # Several editions can match; keep the one with the most tags
                current = found.get(isbn)
                if current is None or len(result["moods"]) + len(result["genres"]) > len(current["moods"]) + len(current["genres"]):
                    found[isbn] = result
    return found


# ---------- Open Library ----------

def openlibrary_isbn_lookup(session, isbns):
    """Return {isbn: [subjects]} using Open Library's batch books API."""
    resp = session.get(
        f"{OPENLIBRARY_URL}/api/books",
        params={"bibkeys": ",".join(f"ISBN:{i}" for i in isbns), "format": "json", "jscmd": "data"},
        headers=HTTP_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    out = {}
    for isbn in isbns:
        subjects = (data.get(f"ISBN:{isbn}") or {}).get("subjects") or []
        out[isbn] = [s.get("name", "") if isinstance(s, dict) else str(s) for s in subjects]
    return out


def openlibrary_title_lookup(session, title, author):
    """Return subjects for the best title/author match on Open Library."""
    resp = session.get(
        f"{OPENLIBRARY_URL}/search.json",
        params={"title": title, "author": author, "fields": "subject", "limit": 1},
        headers=HTTP_HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    docs = resp.json().get("docs") or []
    return (docs[0].get("subject") or []) if docs else []


# ---------- main entry point ----------

def enrich_books(books, hardcover_token=None, progress=None):
    """Add 'Genres' and 'Moods' (lists) for every row of a Goodreads dataframe.

    Returns (dataframe with Genres/Moods columns aligned to books.index, list of warnings).
    """
    progress = progress or (lambda fraction, text: None)
    warnings = []
    cache = load_cache()
    session = requests.Session()

    keys = books.apply(book_key, axis=1)
    rows_by_key = {}
    for idx, key in keys.items():
        rows_by_key.setdefault(key, books.loc[idx])

    isbn_keys = [k for k in rows_by_key if k.startswith("isbn:")]
    title_keys = [k for k in rows_by_key if k.startswith("title:")]

    # Work out what still needs looking up
    hc_todo = []
    if hardcover_token:
        hc_todo = [k[5:] for k in isbn_keys if not cache.get(k, {}).get("hc_checked")]
    ol_isbn_todo = [k for k in isbn_keys if k not in cache]
    ol_title_todo = [k for k in title_keys if k not in cache]

    total_steps = (
        len(list(chunks(hc_todo, HARDCOVER_BATCH)))
        + len(list(chunks(ol_isbn_todo, OPENLIBRARY_BATCH)))
        + len(ol_title_todo)
    )
    done = 0

    def step(text):
        nonlocal done
        done += 1
        progress(min(done / max(total_steps, 1), 1.0), text)

    # 1. Hardcover: genres + moods by ISBN
    hc_ok = bool(hardcover_token)
    for batch in chunks(hc_todo, HARDCOVER_BATCH):
        if not hc_ok:
            break
        try:
            found = hardcover_lookup(session, hardcover_token, batch)
        except HardcoverAuthError:
            warnings.append("Hardcover rejected the API token, so moods were skipped. Check HARDCOVER_TOKEN.")
            hc_ok = False
            break
        except (requests.RequestException, RuntimeError, ValueError) as e:
            warnings.append(f"Some Hardcover lookups failed and will be retried next time ({type(e).__name__}).")
            step("Looking up genres and moods on Hardcover…")
            continue
        for isbn in batch:
            key = f"isbn:{isbn}"
            entry = cache.get(key, {})
            hit = found.get(isbn)
            if hit and (hit["genres"] or hit["moods"]):
                entry.update(genres=hit["genres"], moods=hit["moods"], source="hardcover")
            entry["hc_checked"] = True
            cache[key] = entry
        save_cache(cache)
        step("Looking up genres and moods on Hardcover…")
        time.sleep(1)  # Hardcover allows about 60 requests a minute

    # Books Hardcover had no genres for still need Open Library
    ol_isbn_todo = [k for k in isbn_keys if not cache.get(k, {}).get("genres") and not cache.get(k, {}).get("ol_checked")]

    # 2. Open Library: genres by ISBN, in batches
    for batch_keys in chunks(ol_isbn_todo, OPENLIBRARY_BATCH):
        isbns = [k[5:] for k in batch_keys]
        try:
            subjects = openlibrary_isbn_lookup(session, isbns)
        except (requests.RequestException, ValueError) as e:
            warnings.append(f"Some Open Library lookups failed and will be retried next time ({type(e).__name__}).")
            step("Looking up genres on Open Library…")
            continue
        for key, isbn in zip(batch_keys, isbns):
            entry = cache.get(key, {})
            genres = normalize_genres(subjects.get(isbn, []))
            if genres:
                entry.update(genres=genres, source=entry.get("source") or "openlibrary")
            entry["ol_checked"] = True
            cache[key] = entry
        save_cache(cache)
        step("Looking up genres on Open Library…")

    # 3. Open Library: title/author search for books without an ISBN,
    #    and for ISBN books that neither source had genres for
    title_search = ol_title_todo + [
        k for k in isbn_keys
        if not cache.get(k, {}).get("genres") and cache.get(k, {}).get("ol_checked")
        and not cache.get(k, {}).get("title_checked")
    ]
    total_steps += len(title_search) - len(ol_title_todo)
    for n, key in enumerate(title_search, 1):
        row = rows_by_key[key]
        try:
            subjects = openlibrary_title_lookup(session, search_title(row.get("Title", "")), str(row.get("Author", "")))
        except (requests.RequestException, ValueError) as e:
            if not any("Open Library" in w for w in warnings):
                warnings.append(f"Some Open Library lookups failed and will be retried next time ({type(e).__name__}).")
            step("Searching Open Library by title…")
            continue
        entry = cache.get(key, {})
        genres = normalize_genres(subjects)
        if genres:
            entry.update(genres=genres, source=entry.get("source") or "openlibrary")
        entry.update(ol_checked=True, title_checked=True)
        cache[key] = entry
        if n % 10 == 0:
            save_cache(cache)
        step(f"Searching Open Library by title… ({n}/{len(title_search)})")
        time.sleep(0.2)
    save_cache(cache)

    result = pd.DataFrame(index=books.index)
    result["Genres"] = keys.map(lambda k: list(cache.get(k, {}).get("genres", [])))
    result["Moods"] = keys.map(lambda k: list(cache.get(k, {}).get("moods", [])))
    return result, warnings