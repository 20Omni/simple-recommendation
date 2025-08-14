import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import os
from math import ceil

# ========= CONFIG =========
USER_DATA_FILE = "user_data.json"

# ========= USER DATA STORAGE =========
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f)

def signup_user(username, genres):
    data = load_user_data()
    if username in data:
        return False
    data[username] = {"genres": genres, "watched": []}
    save_user_data(data)
    return True

def load_user(username):
    return load_user_data().get(username)

def update_watched(username, watched_list):
    data = load_user_data()
    if username in data:
        data[username]['watched'] = watched_list
        save_user_data(data)

# ========= LOAD MODEL =========
@st.cache_resource
def load_model():
    df = joblib.load("movies_df.pkl")
    cosine_sim = joblib.load("cosine_similarity.pkl")
    indices = joblib.load("title_indices.pkl")
    return df, cosine_sim, indices

df, cosine_sim, indices = load_model()

# ========= RECOMMENDER =========
def recommend_for_user(preferred_genres, watched_titles, top_n=10):
    scores = np.zeros(len(df))

    # Adaptive weight logic
    if len(watched_titles) >= 3:
        genre_weight = 0.3
        watch_weight = 4.0
    elif len(watched_titles) > 0:
        genre_weight = 0.5
        watch_weight = 3.5
    else:
        genre_weight = 2.0
        watch_weight = 0.0

    # Boost for signup genres
    for genre in preferred_genres:
        mask = df['Genre'].str.contains(genre, case=False, na=False)
        scores[mask] += genre_weight

    # Boost for watched movies similarity
    for title in watched_titles:
        if title in indices:
            idx = indices[title]
            if isinstance(idx, (pd.Series, list, np.ndarray)):
                sim_vec = cosine_sim[idx].mean(axis=0)
            else:
                sim_vec = cosine_sim[idx]
            scores += watch_weight * sim_vec

    # Exclude watched
    watched_idx = []
    for t in watched_titles:
        if t in indices:
            idx_val = indices[t]
            if isinstance(idx_val, (pd.Series, list, np.ndarray)):
                watched_idx.extend(idx_val.tolist())
            else:
                watched_idx.append(idx_val)
    scores[watched_idx] = -1

    rec_idx = np.argsort(scores)[::-1]
    rec_df = df.iloc[rec_idx]
    rec_df = rec_df[~rec_df['Series_Title'].isin(watched_titles)]

    # Keep 2–3 from signup genres to maintain variety
    signup_df = rec_df[rec_df['Genre'].str.contains('|'.join(preferred_genres), case=False)]
    mixed_df = pd.concat([signup_df.head(3), rec_df]).drop_duplicates().head(top_n)

    return mixed_df[['Series_Title', 'Genre', 'IMDB_Rating']]

# ========= MOVIE CARD =========
def movie_card(row, watched_list, username, section, reason=None, show_button=True, dark_mode=False):
    bg_color = "#1e1e1e" if dark_mode else "#fefefe"
    text_color = "#f5f5f5" if dark_mode else "#000"
    genre_color = "#bbb" if dark_mode else "#555"

    card_css = f"""
    <style>
    .movie-card {{
        border: 1px solid #444 if {dark_mode} else #ddd;
        border-radius: 10px;
        padding: 16px;
        background-color: {bg_color};
        color: {text_color};
        box-shadow: 0 2px 5px rgba(0,0,0,0.07);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .movie-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 18px rgba(0,0,0,0.25);
    }}
    .movie-title {{
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 4px;
    }}
    .movie-genres {{
        font-size: 0.9rem;
        color: {genre_color};
        margin-bottom: 8px;
        font-style: italic;
    }}
    .movie-rating {{
        font-size: 1.3rem;
        color: #f39c12;
        margin-bottom: 8px;
    }}
    .reason-text {{
        font-size: 0.85rem;
        color: #4aa3df;
        margin-bottom: 10px;
    }}
    </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)

    with st.container():
        st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="movie-title">{row["Series_Title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="movie-genres">{row["Genre"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="movie-rating">⭐</div>', unsafe_allow_html=True)

        if reason:
            st.markdown(f'<div class="reason-text">💡 {reason}</div>', unsafe_allow_html=True)

        key = f"watched_btn_{section}_{row.name}"
        if show_button:
            if row['Series_Title'] not in watched_list:
                if st.button("Watched", key=key):
                    watched_list.append(row['Series_Title'])
                    update_watched(username, watched_list)
                    st.success(f"Added '{row['Series_Title']}' ✅")
                    st.rerun()
            else:
                st.button("Watched", key=key, disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ========= SESSION STATE INIT =========
if "auth" not in st.session_state:
    st.session_state.auth = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "genres" not in st.session_state:
    st.session_state.genres = []
if "watched" not in st.session_state:
    st.session_state.watched = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ========= LOGIN/SIGNUP =========
def login_signup_page():
    st.title("🎬 Movie Recommender Dashboard")
    option = st.radio("Select Option", ["Login", "Signup"], horizontal=True)

    if option == "Signup":
        username = st.text_input("Choose Username")
        password = st.text_input("Choose Password", type="password")
        genres = st.multiselect("Select Favourite Genres",
                                 sorted(set(g for glist in df['Genre'].str.split(', ') for g in glist)))
        if st.button("Signup"):
            if username and password and genres:
                if signup_user(username, genres):
                    st.session_state.auth = True
                    st.session_state.username = username
                    st.session_state.genres = genres
                    st.session_state.watched = []
                    st.success(f"Account created for {username}! 🎉")
                    st.rerun()
                else:
                    st.error("Username already exists!")
            else:
                st.error("Fill all fields and pick at least one genre.")
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user_data = load_user(username)
            if user_data:
                st.session_state.auth = True
                st.session_state.username = username
                st.session_state.genres = user_data.get("genres", [])
                st.session_state.watched = user_data.get("watched", [])
                st.success(f"Welcome back {username}! 👋")
                st.rerun()
            else:
                st.error("User not found. Please sign up.")

# ========= MULTI-COLUMN CARD RENDERER =========
def render_cards(dataframe, watched_list, username, section, show_button=True, reason_map=None):
    cols_per_row = 3
    total_rows = ceil(len(dataframe) / cols_per_row)
    dark = st.session_state.dark_mode
    for r in range(total_rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < len(dataframe):
                row = dataframe.iloc[idx]
                reason = reason_map.get(row['Series_Title']) if reason_map else None
                with cols[c]:
                    movie_card(row, watched_list, username, section, reason, show_button, dark)

# ========= DASHBOARD =========
def dashboard():
    st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode,
                         key="dark_mode", help="Toggle dark mode")

    st.markdown(f"### 👋 Welcome, **{st.session_state.username}**")
    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.auth = False
        st.session_state.username = ""
        st.session_state.genres = []
        st.session_state.watched = []
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])

    # Top Rated
    with tab1:
        st.subheader("Top IMDb Rated Movies (Mixed Genres)")
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False)
        genre_set = set()
        for g_list in df['Genre'].str.split(', '):
            genre_set.update(g_list)
        mixed_movies = []
        for g in genre_set:
            genre_subset = top_movies[top_movies['Genre'].str.contains(g, case=False, na=False)]
            mixed_movies.append(genre_subset.head(3))
        mixed_df = pd.concat(mixed_movies).drop_duplicates(subset="Series_Title")
        mixed_df = mixed_df[~mixed_df['Series_Title'].isin(st.session_state.watched)]
        mixed_df = mixed_df.sort_values(by="IMDB_Rating", ascending=False).head(50)
        render_cards(mixed_df, st.session_state.watched, st.session_state.username, section="top")

    # Watching
    with tab2:
        st.subheader("Your Watched Movies")
        if st.session_state.watched:
            watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
            render_cards(watched_df, st.session_state.watched, st.session_state.username,
                         section="your", show_button=False)
        else:
            st.info("No watched movies yet.")

    # Recommendations
    with tab3:
        st.subheader("Recommended for You")
        if not st.session_state.genres and not st.session_state.watched:
            st.warning("Add some genres or watched movies first.")
        else:
            recs = recommend_for_user(st.session_state.genres,
                                      st.session_state.watched,
                                      top_n=10)
            reason_map = {}
            for idx, row in recs.iterrows():
                watched_reasons = []
                for watched in st.session_state.watched:
                    if watched in indices:
                        w_idx = indices[watched]
                        if isinstance(w_idx, (pd.Series, list, np.ndarray)):
                            w_idx = w_idx.iloc[0] if hasattr(w_idx, "iloc") else w_idx[0]
                        if cosine_sim[w_idx][idx] > 0.1:
                            watched_reasons.append(watched)
                watched_reasons = watched_reasons[:3]
                genre_matches = [g for g in st.session_state.genres if g.lower() in row["Genre"].lower()][:3]
                reasons_list = []
                if watched_reasons:
                    reasons_list.append("You watched " + ", ".join(watched_reasons))
                if genre_matches:
                    reasons_list.append("You selected genre(s) " + ", ".join(genre_matches))
                reason_map[row['Series_Title']] = " and ".join(reasons_list) if reasons_list else None
            render_cards(recs, st.session_state.watched, st.session_state.username,
                         section="rec", reason_map=reason_map)

# ========= ENTRY =========
if not st.session_state.auth:
    login_signup_page()
else:
    dashboard()
