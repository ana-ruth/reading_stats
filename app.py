import hashlib
import html
import math
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from book_metadata import enrich_books

# One hue for single-series charts; everything else follows the Streamlit theme
BAR_COLOR = "#2a78d6"
# Hover labels list at most this many books per bar
HOVER_LIMIT = 20

EXPECTED_COLUMNS = [
    "Book Id", "Title", "Author", "Author l-f", "Additional Authors", "ISBN",
    "ISBN13", "My Rating", "Publisher", "Binding", "Number of Pages",
    "Year Published", "Original Publication Year", "Date Read", "Date Added",
    "Bookshelves", "Bookshelves with positions", "Exclusive Shelf", "My Review",
    "spoiler", "Private Notes", "Read Count", "Owned Copies",
]


def load_read_books(df):
    """Keep only books on the 'read' shelf and clean the columns used for stats."""
    read = df[df["Exclusive Shelf"].astype(str).str.strip().str.lower() == "read"].copy()

    # Goodreads exports ISBNs as ="0123456789"
    for col in ["ISBN", "ISBN13"]:
        read[col] = read[col].astype(str).str.strip('="')

    read["My Rating"] = pd.to_numeric(read["My Rating"], errors="coerce").fillna(0).astype(int)
    read["Number of Pages"] = pd.to_numeric(read["Number of Pages"], errors="coerce")
    read["Original Publication Year"] = pd.to_numeric(read["Original Publication Year"], errors="coerce")
    read["Date Read"] = pd.to_datetime(read["Date Read"], errors="coerce")
    read["Date Added"] = pd.to_datetime(read["Date Added"], errors="coerce")
    # Books without a read date fall back to when they were added
    read["Finished"] = read["Date Read"].fillna(read["Date Added"])
    return read


Chart = namedtuple("Chart", "fig summaries books")


def book_list(labels):
    """Hover text listing the books behind a bar, one per line."""
    lines = [html.escape(l) for l in labels[:HOVER_LIMIT]]
    if len(labels) > HOVER_LIMIT:
        lines.append(f"<i>…and {len(labels) - HOVER_LIMIT} more (click the bar to see all)</i>")
    return "<br>".join(lines)


def books_by(books, key, categories, sort_by="Title", ascending=True):
    """The books behind each category, in display order; empty for categories without books."""
    books = books.sort_values(sort_by, ascending=ascending)
    groups = dict(list(books.groupby(key, sort=False)))
    return [groups.get(c, books.iloc[:0]) for c in categories]


def hover_text(summaries, groups, detail=None):
    """Bold summary line above each bar's book list; `detail(books)` is appended to each title."""
    texts = []
    for summary, books in zip(summaries, groups):
        labels = books["Title"].map(lambda t: t if len(t) <= 55 else t[:52] + "…")
        if detail is not None and not books.empty:
            labels = labels + detail(books)
        texts.append(f"<b>{html.escape(summary)}</b><br><br>{book_list(list(labels))}")
    return texts


def count_axis(max_count):
    """Integer ticks for axes that count books."""
    return dict(tick0=0, dtick=max(1, math.ceil(max_count / 6)), rangemode="tozero")


def bar_chart(categories, values, summaries, groups, title, axis_title, detail=None,
              text=None, horizontal=False, value_axis=None):
    """Single-series bar chart whose hover shows a summary line plus the books."""
    categories = [str(c) for c in categories]
    fig = go.Figure(go.Bar(
        x=values if horizontal else categories,
        y=categories if horizontal else values,
        orientation="h" if horizontal else "v",
        marker_color=BAR_COLOR,
        text=text if text is not None else values,
        textposition="outside",
        cliponaxis=False,
        customdata=hover_text(summaries, groups, detail),
        hovertemplate="%{customdata}<extra></extra>",
    ))
    value_axis = dict(title=axis_title, **(value_axis or {}))
    category_axis = dict(type="category", showgrid=False)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        xaxis=value_axis if horizontal else category_axis,
        yaxis=category_axis if horizontal else value_axis,
        height=max(300, 32 * len(categories) + 120) if horizontal else 400,
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(align="left"),
    )
    return Chart(fig, summaries, groups)


def plot_top_authors(read, n):
    counts = read["Author"].value_counts().head(n).sort_values()
    summaries = [f"{a}: {c} book{'s' * (c != 1)}" for a, c in counts.items()]
    return bar_chart(counts.index, counts.values, summaries, books_by(read, "Author", counts.index),
                     f"Top {len(counts)} most read authors", "Books read", horizontal=True,
                     value_axis=count_axis(counts.max()))


def plot_rating_distribution(read):
    rated = read[read["My Rating"] > 0]
    counts = rated["My Rating"].value_counts().reindex(range(1, 6), fill_value=0)
    summaries = [f"{r} ★: {c} books" for r, c in counts.items()]
    return bar_chart(counts.index, counts.values, summaries, books_by(rated, "My Rating", counts.index),
                     "Rating distribution", "Books", value_axis=count_axis(counts.max()))


def plot_books_per_year(read):
    finished = read[read["Finished"].notna()].assign(Year=lambda d: d["Finished"].dt.year)
    per_year = finished["Year"].value_counts().sort_index()
    if per_year.empty:
        return None
    years = range(per_year.index.min(), per_year.index.max() + 1)
    per_year = per_year.reindex(years, fill_value=0)
    summaries = [f"{y}: {c} books" for y, c in per_year.items()]
    chart = bar_chart(per_year.index, per_year.values, summaries,
                      books_by(finished, "Year", per_year.index, sort_by="Finished"),
                      "Books read per year", "Books",
                      text=per_year.values if len(per_year) <= 15 else [""] * len(per_year),
                      value_axis=count_axis(per_year.max()))
    chart.fig.update_xaxes(tickangle=-45 if len(per_year) > 8 else 0)
    return chart


def plot_page_distribution(read):
    books = read[read["Number of Pages"] > 0]
    if books.empty:
        return None
    pages = books["Number of Pages"]
    counts, edges = np.histogram(pages, bins=20)
    books = books.assign(Bin=pd.cut(pages, edges, include_lowest=True, labels=False))
    groups = books_by(books, "Bin", range(len(counts)), sort_by="Number of Pages")
    summaries = [f"{lo:.0f}–{hi:.0f} pages: {c} books" for lo, hi, c in zip(edges[:-1], edges[1:], counts)]
    fig = go.Figure(go.Bar(
        x=(edges[:-1] + edges[1:]) / 2,
        y=counts,
        width=np.diff(edges) * 0.92,
        marker_color=BAR_COLOR,
        customdata=hover_text(summaries, groups,
                              lambda b: " (" + b["Number of Pages"].astype(int).astype(str) + " p.)"),
        hovertemplate="%{customdata}<extra></extra>",
    ))
    median = pages.median()
    fig.add_vline(x=median, line_dash="dash", line_width=1.5,
                  annotation_text=f"median {median:.0f}", annotation_position="top right")
    fig.update_layout(
        title=dict(text="Book length", x=0, xanchor="left"),
        xaxis=dict(title="Pages", showgrid=False),
        yaxis=dict(title="Books", **count_axis(counts.max())),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(align="left"),
    )
    return Chart(fig, summaries, groups)


def plot_publication_decades(read):
    books = read[read["Original Publication Year"].notna()]
    if books.empty:
        return None
    books = books.assign(Decade=(books["Original Publication Year"] // 10 * 10).astype(int))
    decades = books["Decade"].value_counts().sort_index()
    summaries = [f"{d}s: {c} books" for d, c in decades.items()]
    chart = bar_chart([f"{d}s" for d in decades.index], decades.values, summaries,
                      books_by(books, "Decade", decades.index, sort_by="Original Publication Year"),
                      "Original publication decade", "Books",
                      detail=lambda b: " (" + b["Original Publication Year"].astype(int).astype(str) + ")",
                      value_axis=count_axis(decades.max()))
    chart.fig.update_xaxes(tickangle=-45 if len(decades) > 8 else 0)
    return chart


def rating_chart(rated, key, title):
    """Highest average ratings per `key`; `rated` has one row per (book, key)."""
    stats = rated.groupby(key)["My Rating"].agg(["mean", "count"]).sort_values("mean").tail(10)
    if stats.empty:
        return None
    summaries = [f"{k}: {m:.2f} ★ avg over {c} books"
                 for k, m, c in zip(stats.index, stats["mean"], stats["count"])]
    return bar_chart(stats.index, stats["mean"], summaries,
                     books_by(rated, key, stats.index, sort_by="My Rating", ascending=False),
                     title, "Average rating (books rated)",
                     detail=lambda b: "  ★" + b["My Rating"].astype(str),
                     text=[f"{m:.2f} ({c})" for m, c in zip(stats["mean"], stats["count"])],
                     horizontal=True, value_axis=dict(range=[0, 5.6]))


def plot_avg_rating_by_author(read, min_books):
    rated = read[read["My Rating"] > 0]
    counts = rated["Author"].value_counts()
    rated = rated[rated["Author"].isin(counts[counts >= min_books].index)]
    return rating_chart(rated, "Author", f"Highest rated authors (≥ {min_books} rated books)")


def explode_tags(read, column):
    """One row per (book, tag) with the tag in 'Tag'; books without tags are dropped."""
    exploded = read.assign(Tag=read[column]).explode("Tag")
    return exploded[exploded["Tag"].notna() & (exploded["Tag"] != "")]


def plot_top_tags(read, column, title, n=12):
    tags = explode_tags(read, column)
    if tags.empty:
        return None
    tagged_books = read[column].map(len).gt(0).sum()
    counts = tags["Tag"].value_counts().head(n).sort_values()
    summaries = [f"{t}: {c} books ({c / tagged_books:.0%})" for t, c in counts.items()]
    return bar_chart(counts.index, counts.values, summaries, books_by(tags, "Tag", counts.index),
                     title, f"Books (share of {tagged_books} tagged books)",
                     text=[f"{c}  ({c / tagged_books:.0%})" for c in counts.values],
                     horizontal=True, value_axis=dict(count_axis(counts.max()), range=[0, counts.max() * 1.25]))


def plot_avg_rating_by_tag(read, column, label, min_books):
    tags = explode_tags(read[read["My Rating"] > 0], column)
    if tags.empty:
        return None
    counts = tags["Tag"].value_counts()
    tags = tags[tags["Tag"].isin(counts[counts >= min_books].index)]
    return rating_chart(tags, "Tag", f"Highest rated {label} (≥ {min_books} rated books)")


def book_table(books):
    """Books as a readable table, with genre and mood lists joined."""
    table = books[["Title", "Author", "My Rating", "Genres", "Moods", "Number of Pages",
                   "Original Publication Year", "Date Read"]].copy()
    for col in ["Genres", "Moods"]:
        table[col] = table[col].map(", ".join)
    return table


def get_hardcover_token():
    """Optional free token from hardcover.app/account/api, for mood data."""
    try:
        token = st.secrets.get("HARDCOVER_TOKEN")
    except Exception:  # no secrets.toml
        token = None
    return (token or os.environ.get("HARDCOVER_TOKEN") or "").strip() or None


def get_book_metadata(read, file_bytes):
    """Look up genres/moods once per uploaded file; slider changes reuse the result."""
    file_id = hashlib.md5(file_bytes).hexdigest()
    cached = st.session_state.get("book_metadata")
    if cached and cached[0] == file_id:
        return cached[1], cached[2]
    bar = st.progress(0.0, text="Looking up genres and moods…")
    meta, warnings = enrich_books(
        read, get_hardcover_token(), progress=lambda f, text: bar.progress(f, text=text)
    )
    bar.empty()
    st.session_state["book_metadata"] = (file_id, meta, warnings)
    return meta, warnings


def show(chart, empty_message, key):
    """Draw a chart; clicking a bar lists all of its books underneath."""
    if chart is None:
        st.info(empty_message)
        return
    event = st.plotly_chart(chart.fig, key=key, on_select="rerun", selection_mode="points")
    for point in event.selection.points:
        i = point["point_index"]
        if i < len(chart.books):  # the selection can outlive a chart redrawn with fewer bars
            st.markdown(f"**{chart.summaries[i]}**")
            st.dataframe(book_table(chart.books[i]), hide_index=True)


st.set_page_config(page_title="Reading Stats", layout="wide")
st.title("📚 Reading Stats")
st.write("Upload your Goodreads library export (CSV) to see statistics about the books you've read.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Awaiting CSV file upload. Please select a file above.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"An error occurred while reading the file: {e}")
    st.stop()

missing = [c for c in ["Title", "Author", "My Rating", "Exclusive Shelf", "Number of Pages",
                       "Original Publication Year", "Date Read", "Date Added", "ISBN", "ISBN13"]
           if c not in df.columns]
if missing:
    st.error(f"The file is missing required columns: {', '.join(missing)}")
    st.stop()

extra = set(EXPECTED_COLUMNS) - set(df.columns)
if extra:
    st.caption(f"Note: some expected columns are absent and will be ignored: {', '.join(sorted(extra))}")

read = load_read_books(df)
if read.empty:
    st.warning("No books on the 'read' shelf were found in this file.")
    st.stop()

meta, lookup_warnings = get_book_metadata(read, uploaded_file.getvalue())
read = read.join(meta)
for warning in lookup_warnings:
    st.warning(warning)

# Optional year filter on when the book was finished
years = sorted(read["Finished"].dropna().dt.year.astype(int).unique())
if len(years) > 1:
    start, end = st.sidebar.select_slider(
        "Years read", options=years, value=(years[0], years[-1])
    )
    in_range = read["Finished"].dt.year.between(start, end) | read["Finished"].isna()
    read = read[in_range]

# Headline numbers
rated = read[read["My Rating"] > 0]
pages = read["Number of Pages"].dropna()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Books read", f"{len(read):,}")
c2.metric("Pages read", f"{pages.sum():,.0f}")
c3.metric("Average rating", f"{rated['My Rating'].mean():.2f} ★" if not rated.empty else "—")
c4.metric("Different authors", f"{read['Author'].nunique():,}")

st.divider()
st.caption("Hover over a bar to see its books, or click it to list them all below the chart"
           " (double-click the chart to clear).")

left, right = st.columns(2)
with left:
    top_n = st.slider("Number of authors to show", 5, 25, 10)
    show(plot_top_authors(read, top_n), "No authors to show.", "top_authors")
with right:
    show(plot_rating_distribution(read), "No rated books.", "ratings")
    unrated = (read["My Rating"] == 0).sum()
    if unrated:
        st.caption(f"{unrated} read book(s) have no rating and are not included.")

left, right = st.columns(2)
with left:
    show(plot_books_per_year(read), "No read dates available.", "per_year")
with right:
    show(plot_page_distribution(read), "No page counts available.", "pages")

left, right = st.columns(2)
with left:
    show(plot_publication_decades(read), "No publication years available.", "decades")
with right:
    min_books = st.slider("Minimum rated books per author", 1, 5, 2)
    show(plot_avg_rating_by_author(read, min_books), "No authors meet this minimum.", "author_ratings")

st.divider()

# Genres and moods (looked up from Hardcover / Open Library)
st.header("Genres & moods")
with_genres = read["Genres"].map(len).gt(0).sum()
with_moods = read["Moods"].map(len).gt(0).sum()
coverage = f"Genres found for {with_genres} of {len(read)} books, moods for {with_moods}."
if get_hardcover_token() is None:
    coverage += (" Moods come from Hardcover: add a free HARDCOVER_TOKEN to"
                 " .streamlit/secrets.toml to include them.")
st.caption(coverage)

min_tagged = st.slider("Minimum rated books per genre or mood", 1, 10, 3)

left, right = st.columns(2)
with left:
    show(plot_top_tags(read, "Genres", "Most read genres"), "No genre data found for these books.", "genres")
with right:
    show(plot_avg_rating_by_tag(read, "Genres", "genres", min_tagged), "No genres meet this minimum.", "genre_ratings")

if with_moods:
    left, right = st.columns(2)
    with left:
        show(plot_top_tags(read, "Moods", "Most common moods"), "No mood data found for these books.", "moods")
    with right:
        show(plot_avg_rating_by_tag(read, "Moods", "moods", min_tagged), "No moods meet this minimum.", "mood_ratings")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Longest books")
    st.dataframe(
        read.nlargest(5, "Number of Pages")[["Title", "Author", "Number of Pages"]],
        hide_index=True,
    )
with right:
    st.subheader("Shortest books")
    st.dataframe(
        read[read["Number of Pages"] > 0].nsmallest(5, "Number of Pages")[["Title", "Author", "Number of Pages"]],
        hide_index=True,
    )

with st.expander("All read books"):
    st.dataframe(book_table(read), hide_index=True)