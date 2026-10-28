import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chart styling: one hue for single-series charts, quiet ink for text and axes
BAR_COLOR = "#2a78d6"
MUTED_COLOR = "#b5b4ae"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "grid.color": GRID_COLOR,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
)

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


def bar_labels(ax, bars):
    """Put the count at the end of each bar in secondary ink."""
    ax.bar_label(bars, padding=3, color=TEXT_SECONDARY, fontsize=9)


def plot_top_authors(read, n):
    counts = read["Author"].value_counts().head(n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(counts))))
    bars = ax.barh(counts.index, counts.values, color=BAR_COLOR, height=0.7)
    bar_labels(ax, bars)
    ax.set_title(f"Top {len(counts)} most read authors")
    ax.set_xlabel("Books read")
    ax.grid(axis="y", visible=False)
    ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig


def plot_rating_distribution(read):
    rated = read[read["My Rating"] > 0]
    counts = rated["My Rating"].value_counts().reindex(range(1, 6), fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(r) for r in counts.index], counts.values, color=BAR_COLOR, width=0.7)
    bar_labels(ax, bars)
    ax.set_title("Rating distribution")
    ax.set_xlabel("My rating (stars)")
    ax.set_ylabel("Books")
    ax.grid(axis="x", visible=False)
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig


def plot_books_per_year(read):
    per_year = read["Finished"].dropna().dt.year.value_counts().sort_index()
    if per_year.empty:
        return None
    years = range(per_year.index.min(), per_year.index.max() + 1)
    per_year = per_year.reindex(years, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([str(y) for y in per_year.index], per_year.values, color=BAR_COLOR, width=0.7)
    if len(per_year) <= 15:
        bar_labels(ax, bars)
    ax.set_title("Books read per year")
    ax.set_ylabel("Books")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=45 if len(per_year) > 8 else 0)
    fig.tight_layout()
    return fig


def plot_page_distribution(read):
    pages = read["Number of Pages"].dropna()
    pages = pages[pages > 0]
    if pages.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(pages, bins=20, color=BAR_COLOR, edgecolor="white", linewidth=1, alpha=1, ax=ax)
    ax.axvline(pages.median(), color=TEXT_PRIMARY, linewidth=1.5, linestyle="--")
    ax.annotate(
        f"median {pages.median():.0f}", xy=(pages.median(), ax.get_ylim()[1] * 0.95),
        xytext=(4, 0), textcoords="offset points", color=TEXT_PRIMARY, fontsize=9,
    )
    ax.set_title("Book length")
    ax.set_xlabel("Pages")
    ax.set_ylabel("Books")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_publication_decades(read):
    years = read["Original Publication Year"].dropna()
    if years.empty:
        return None
    decades = (years // 10 * 10).astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([f"{d}s" for d in decades.index], decades.values, color=BAR_COLOR, width=0.7)
    bar_labels(ax, bars)
    ax.set_title("Original publication decade")
    ax.set_ylabel("Books")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=45 if len(decades) > 8 else 0)
    fig.tight_layout()
    return fig


def plot_avg_rating_by_author(read, min_books):
    rated = read[read["My Rating"] > 0]
    stats = rated.groupby("Author")["My Rating"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_books].sort_values("mean").tail(10)
    if stats.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(stats))))
    bars = ax.barh(stats.index, stats["mean"], color=BAR_COLOR, height=0.7)
    ax.bar_label(
        bars, labels=[f"{m:.2f} ({c})" for m, c in zip(stats["mean"], stats["count"])],
        padding=3, color=TEXT_SECONDARY, fontsize=9,
    )
    ax.set_xlim(0, 5.6)
    ax.set_title(f"Highest rated authors (≥ {min_books} rated books)")
    ax.set_xlabel("Average rating (books rated)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def show(fig, empty_message):
    if fig is None:
        st.info(empty_message)
    else:
        st.pyplot(fig)
        plt.close(fig)


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

left, right = st.columns(2)
with left:
    top_n = st.slider("Number of authors to show", 5, 25, 10)
    show(plot_top_authors(read, top_n), "No authors to show.")
with right:
    show(plot_rating_distribution(read), "No rated books.")
    unrated = (read["My Rating"] == 0).sum()
    if unrated:
        st.caption(f"{unrated} read book(s) have no rating and are not included.")

left, right = st.columns(2)
with left:
    show(plot_books_per_year(read), "No read dates available.")
with right:
    show(plot_page_distribution(read), "No page counts available.")

left, right = st.columns(2)
with left:
    show(plot_publication_decades(read), "No publication years available.")
with right:
    min_books = st.slider("Minimum rated books per author", 1, 5, 2)
    show(plot_avg_rating_by_author(read, min_books), "No authors meet this minimum.")

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
    st.dataframe(
        read[["Title", "Author", "My Rating", "Number of Pages", "Original Publication Year", "Date Read"]],
        hide_index=True,
    )
