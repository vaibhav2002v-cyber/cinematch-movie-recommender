import ast
import html
import os

import pandas as pd
import requests
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===========================================================================
# PASTE YOUR TMDB API KEY BETWEEN THE QUOTES BELOW (keep the quotes)
# ===========================================================================
MY_TMDB_KEY = "5f0900ca8f46067335c05f1822f77bb5"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #3a0d1a 0%, transparent 60%),
                radial-gradient(900px 500px at 100% 0%, #0f2a4a 0%, transparent 55%),
                #0b0c10;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1300px; }

/* Hero */
.hero { text-align: center; padding: 2.2rem 1rem 1.2rem; }
.hero h1 {
    font-size: 3.4rem; font-weight: 800; margin: 0; letter-spacing: -1px;
    background: linear-gradient(90deg, #ff4d6d, #ff9a3c, #ffd166);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #aab2c0; font-size: 1.1rem; margin-top: .4rem; }

/* Stat chips */
.chips { display: flex; justify-content: center; gap: .6rem; flex-wrap: wrap; margin: 1rem 0 1.6rem; }
.chip {
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1);
    color: #dfe4ee; padding: .35rem .9rem; border-radius: 999px; font-size: .85rem;
    backdrop-filter: blur(6px);
}

/* Selected movie banner */
.picked {
    display: flex; gap: 1.4rem; align-items: center;
    background: linear-gradient(135deg, rgba(255,77,109,.14), rgba(255,154,60,.08));
    border: 1px solid rgba(255,255,255,.1); border-radius: 20px;
    padding: 1.2rem; margin: 1rem 0 2rem;
}
.picked img, .picked .ph {
    width: 130px; height: 195px; border-radius: 14px; object-fit: cover; flex-shrink: 0;
    box-shadow: 0 10px 30px rgba(0,0,0,.5);
}
.picked h2 { color: #fff; margin: 0 0 .3rem; font-weight: 700; }
.picked .meta { color: #ffb86b; font-size: .9rem; margin-bottom: .6rem; }
.picked .ov { color: #b9c0cd; font-size: .92rem; line-height: 1.55; }

/* Section title */
.sec { color: #fff; font-size: 1.5rem; font-weight: 700; margin: .5rem 0 1rem; }
.sec span { color: #ff4d6d; }

/* Movie card */
.card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px; overflow: hidden; height: 100%;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    position: relative;
}
.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 18px 40px rgba(255,77,109,.25);
    border-color: rgba(255,77,109,.6);
}
.card .poster { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.ph {
    display: flex; align-items: center; justify-content: center; text-align: center;
    padding: 1rem; color: #fff; font-weight: 700; font-size: 1.05rem;
    background: linear-gradient(160deg, #3b1c32, #1b2a49);
}
.card .body { padding: .8rem .9rem 1rem; }
.card .title { color: #fff; font-weight: 600; font-size: .98rem; line-height: 1.25; margin-bottom: .35rem; }
.card .sub { color: #8f98a8; font-size: .78rem; margin-bottom: .5rem; }
.badge {
    position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,.72);
    color: #ffd166; font-weight: 700; font-size: .8rem; padding: .25rem .6rem;
    border-radius: 999px; backdrop-filter: blur(6px);
}
.match {
    position: absolute; top: 10px; right: 10px; background: linear-gradient(90deg,#ff4d6d,#ff9a3c);
    color: #fff; font-weight: 700; font-size: .75rem; padding: .25rem .6rem; border-radius: 999px;
}
.tag {
    display: inline-block; background: rgba(255,255,255,.08); color: #cfd6e4;
    padding: .12rem .55rem; border-radius: 999px; font-size: .7rem; margin: 0 .25rem .25rem 0;
}

/* Widgets */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.06) !important; border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,.14) !important;
}
.stButton > button {
    background: linear-gradient(90deg, #ff4d6d, #ff9a3c); color: #fff; border: 0;
    border-radius: 14px; padding: .7rem 1.4rem; font-weight: 700; font-size: 1rem; width: 100%;
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover { transform: scale(1.02); box-shadow: 0 8px 24px rgba(255,77,109,.4); color:#fff; border:0; }
section[data-testid="stSidebar"] { background: #0e1016; border-right: 1px solid rgba(255,255,255,.06); }

.footer { text-align: center; color: #6b7384; font-size: .8rem; margin: 3rem 0 1rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data + model (same logic as your notebook, cached so it runs only once)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ps = PorterStemmer()


def names(obj, limit=None, job=None):
    """Convert the stringified list of dicts into a list of names."""
    out = []
    for item in ast.literal_eval(obj):
        if job is not None:
            if item.get("job") == job:
                out.append(item["name"])
                break
            continue
        out.append(item["name"])
        if limit and len(out) == limit:
            break
    return out


def stem(text):
    return " ".join(ps.stem(w) for w in text.split())


def squash(items):
    return [i.replace(" ", "") for i in items]


@st.cache_resource(show_spinner="Loading movies and building the model...")
def build_model():
    movies = pd.read_csv(os.path.join(BASE_DIR, "tmdb_5000_movies.csv"))
    credits = pd.read_csv(os.path.join(BASE_DIR, "tmdb_5000_credits.csv.gz"))
    df = movies.merge(credits, on="title")

    df = df[
        ["id", "title", "overview", "genres", "keywords", "cast", "crew",
         "vote_average", "release_date"]
    ].dropna()
    df = df.drop_duplicates(subset="title").reset_index(drop=True)

    df["genres"] = df["genres"].apply(names)
    df["keywords"] = df["keywords"].apply(names)
    df["cast"] = df["cast"].apply(lambda x: names(x, limit=3))
    df["crew"] = df["crew"].apply(lambda x: names(x, job="Director"))
    df["year"] = df["release_date"].str[:4]

    tags = (
        df["overview"].apply(lambda x: x.split())
        + df["genres"].apply(squash)
        + df["keywords"].apply(squash)
        + df["cast"].apply(squash)
        + df["crew"].apply(squash)
    )
    df["tags"] = tags.apply(lambda x: " ".join(x).lower()).apply(stem)

    cv = CountVectorizer(stop_words="english", max_features=5000)
    vectors = cv.fit_transform(df["tags"])
    similarity = cosine_similarity(vectors).astype("float32")
    return df, similarity


def recommend(df, similarity, title, n):
    idx = df.index[df["title"] == title][0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1 : n + 1]
    return [(df.iloc[i], float(s)) for i, s in scores]


def fetch_poster(movie_id, api_key):
    if not api_key:
        return None
    cache = st.session_state.setdefault("poster_cache", {})
    if movie_id in cache:
        return cache[movie_id]
    for _ in range(3):  # try up to 3 times
        try:
            r = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}",
                params={"api_key": api_key},
                timeout=8,
            )
            if r.status_code == 200:
                path = r.json().get("poster_path")
                url = f"https://image.tmdb.org/t/p/w500{path}" if path else None
                cache[movie_id] = url  # only remember real answers
                return url
        except Exception:
            pass
    return None


def poster_html(url, title, cls):
    if url:
        return f'<img class="{cls}" src="{url}" alt="{html.escape(title)}">'
    return f'<div class="{cls} ph">{html.escape(title)}</div>'


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
try:
    df, similarity = build_model()
except FileNotFoundError:
    st.error(
        "Could not find **tmdb_5000_movies.csv** and **tmdb_5000_credits.csv.gz**. "
        "Put both files in the same folder as `movie_recommend.py` and restart."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    n_recs = st.slider("Number of recommendations", 3, 12, 6)

    st.markdown("### 🖼️ Posters (optional)")
    default_key = MY_TMDB_KEY
    try:
        default_key = MY_TMDB_KEY or st.secrets.get("TMDB_API_KEY", "")
    except Exception:
        pass
    api_key = st.text_input(
        "TMDB API key",
        value=default_key,
        type="password",
        help="Free key from themoviedb.org. Without it, stylish placeholders are shown.",
    )
    st.markdown("---")
    st.caption("Recommendations use content-based filtering: overview, genres, "
               "keywords, top 3 cast and director, compared with cosine similarity.")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🎬 CineMatch</h1>
        <p>Pick a movie you love. We'll find your next favorite.</p>
    </div>
    <div class="chips">
        <div class="chip">🎞️ {len(df):,} movies</div>
        <div class="chip">🧠 Content-based AI</div>
        <div class="chip">⚡ Instant results</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, mid, right = st.columns([1, 2.2, 1])
with mid:
    selected = st.selectbox(
        "Choose a movie",
        df["title"].values,
        index=None,
        placeholder="Type to search, e.g. Avatar, Spider-Man 3, The Dark Knight...",
        label_visibility="collapsed",
    )
    go = st.button("✨ Recommend movies")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if go and not selected:
    with mid:
        st.warning("Please choose a movie first.")

if go and selected:
    row = df[df["title"] == selected].iloc[0]
    purl = fetch_poster(int(row["id"]), api_key)

    genres = " • ".join(row["genres"][:3])
    director = row["crew"][0] if row["crew"] else "Unknown"
    st.markdown(
        f"""
        <div class="picked">
            {poster_html(purl, row['title'], 'poster')}
            <div>
                <h2>{html.escape(row['title'])}</h2>
                <div class="meta">⭐ {row['vote_average']:.1f} &nbsp;|&nbsp; {row['year']}
                &nbsp;|&nbsp; {html.escape(genres)} &nbsp;|&nbsp; 🎥 {html.escape(director)}</div>
                <div class="ov">{html.escape(row['overview'][:320])}{'...' if len(row['overview']) > 320 else ''}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec">Because you liked this, <span>try these</span></div>',
                unsafe_allow_html=True)

    recs = recommend(df, similarity, selected, n_recs)
    per_row = 3 if n_recs <= 6 else 4
    for start in range(0, len(recs), per_row):
        cols = st.columns(per_row)
        for col, (m, score) in zip(cols, recs[start : start + per_row]):
            url = fetch_poster(int(m["id"]), api_key)
            tags = "".join(f'<span class="tag">{html.escape(g)}</span>' for g in m["genres"][:3])
            director = m["crew"][0] if m["crew"] else "Unknown"
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="badge">⭐ {m['vote_average']:.1f}</div>
                        <div class="match">{score * 100:.0f}% match</div>
                        {poster_html(url, m['title'], 'poster')}
                        <div class="body">
                            <div class="title">{html.escape(m['title'])}</div>
                            <div class="sub">{m['year']} • 🎥 {html.escape(director)}</div>
                            {tags}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")

st.markdown('<div class="footer">Built with Streamlit • Data from TMDB 5000</div>',
            unsafe_allow_html=True)