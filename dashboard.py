import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import os
from math import ceil

USER_DATA_FILE = "user_data.json"

# ===== User Data Storage =====
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f)

def signup_user(username):
    data = load_user_data()
    if username in data:
        return False
    data[username] = {"genres": [], "watched": []}
    save_user_data(data)
    return True

def load_user(username):
    return load_user_data().get(username)

def update_user_genres(username, genres):
    data = load_user_data()
    if username in data:
        data[username]['genres'] = genres
        save_user_data(data)

def update_watched(username, watched_list):
    data = load_user_data()
    if username in data:
        data[username]['watched'] = watched_list
        save_user_data(data)

# ===== Load Model/Data =====
@st.cache_resource
def load_model():
    df = joblib.load("movies_df.pkl")
    cosine_sim = joblib.load("cosine_similarity.pkl")
    indices = joblib.load("title_indices.pkl")
    return df, cosine_sim, indices

df, cosine_sim, indices = load_model()

# ===== Recommendation Logic =====
def recommend_for_user(preferred_genres, watched_titles, top_n=10):
    scores = np.zeros(len(df))
    if len(watched_titles) >= 3:
        genre_weight = 0.3
        watch_weight = 4.0
    elif watched_titles:
        genre_weight = 0.5
        watch_weight = 3.5
    else:
        genre_weight = 2.0
        watch_weight = 0.0

    for genre in preferred_genres:
        mask = df['Genre'].str.contains(genre, case=False, na=False)
        scores[mask] += genre_weight

    for title in watched_titles:
        if title in indices:
            idx = indices[title]
            sim_vec = cosine_sim[idx].mean(axis=0) if isinstance(idx, (pd.Series, list, np.ndarray)) else cosine_sim[idx]
            scores += watch_weight * sim_vec

    watched_idx = []
    for t in watched_titles:
        if t in indices:
            idx_val = indices[t]
            watched_idx.extend(idx_val if isinstance(idx_val, (pd.Series, list, np.ndarray)) else [idx_val])
    scores[watched_idx] = -1

    rec_df = df.iloc[np.argsort(scores)[::-1]]
    rec_df = rec_df[~rec_df['Series_Title'].isin(watched_titles)]
    signup_df = rec_df[rec_df['Genre'].str.contains('|'.join(preferred_genres), case=False)]
    mixed_df = pd.concat([signup_df.head(3), rec_df]).drop_duplicates().head(top_n)
    return mixed_df[['Series_Title', 'Genre', 'IMDB_Rating']]

# ===== Emoji Mapping =====
genre_emojis = {
    "action": "🎬", "comedy": "😂", "drama": "🎭", "romance": "❤️",
    "thriller": "🔪", "horror": "👻", "sci-fi": "👽", "science fiction": "👽",
    "adventure": "🧭", "fantasy": "🦄", "animation": "🐭", "documentary": "🎥",
    "crime": "🕵️", "mystery": "🕵️", "war": "⚔️", "musical": "🎶", "music": "🎶"
}
def get_dominant_genre_with_emoji(genre_string, signup_genres=None):
    genres_list = [g.strip() for g in genre_string.split(",")]
    if signup_genres:
        for sg in signup_genres:
            for g in genres_list:
                if sg.lower() in g.lower():
                    return genre_emojis.get(g.lower(), "🎞️"), genre_string
    for g in genres_list:
        if g.lower() in genre_emojis:
            return genre_emojis[g.lower()], genre_string
    return "🎞️", genre_string

# ===== Card Renderer =====
def movie_card(row, watched_list, username, section, reason=None, show_button=True, signup_genres=None):
    dark_mode = st.session_state.dark_mode
    bg_color = "#23272e" if dark_mode else "#fdfdfe"
    text_color = "#f5f5f5" if dark_mode else "#222"
    border_color = "#3d434d" if dark_mode else "#e2e3e6"
    genre_color = "#b2b2b2" if dark_mode else "#5A5A5A"
    rating_color = "#fcb900"
    shadow = "0 2px 7px rgba(0,0,0,0.10)"
    shadow_hover = "0 8px 22px rgba(0,0,0,0.17)"
    emoji, genre_text = get_dominant_genre_with_emoji(row["Genre"], signup_genres)

    st.markdown(f"""
    <style>
    .movie-card {{
        border: 1.5px solid {border_color}; border-radius: 16px;
        padding: 16px; margin-bottom: 22px;
        background: {bg_color}; color: {text_color};
        box-shadow: {shadow};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .movie-card:hover {{
        transform: translateY(-6px); box-shadow: {shadow_hover};
    }}
    .movie-title {{ font-size: 1.15rem; font-weight: 700; margin-bottom: 4px; }}
    .movie-genres {{ font-size: 0.9rem; color: {genre_color}; margin-bottom: 6px; }}
    .movie-rating {{ font-size: 1.2rem; color: {rating_color}; margin-bottom: 6px; }}
    .reason-text {{ font-size: 0.9rem; color: #399ed7; margin-bottom: 8px; }}
    </style>
    """, unsafe_allow_html=True)

    html = f'<div class="movie-card">'
    html += f'<div class="movie-title">{row["Series_Title"]}</div>'
    html += f'<div class="movie-genres"><span style="font-style: normal;">{emoji}</span> <span style="font-style: italic;">{genre_text}</span></div>'
    html += f'<div class="movie-rating">⭐ {row["IMDB_Rating"]:.1f}/10</div>'
    if reason:
        html += f'<div class="reason-text">💡 {reason}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    if show_button:
        key = f"watched_btn_{section}_{row.name}"
        if row['Series_Title'] not in watched_list:
            if st.button("Watched", key=key):
                watched_list.append(row['Series_Title'])
                update_watched(username, watched_list)
                st.rerun()
        else:
            st.button("Watched", key=key, disabled=True)

def render_cards(dataframe, watched_list, username, section, show_button=True, reason_map=None, signup_genres=None):
    cols_per_row = 3
    total_rows = ceil(len(dataframe) / cols_per_row)
    for r in range(total_rows):
        cols = st.columns(cols_per_row, gap="large")
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < len(dataframe):
                row = dataframe.iloc[idx]
                reason = reason_map.get(row['Series_Title']) if reason_map else None
                with cols[c]:
                    movie_card(row, watched_list, username, section, reason, show_button, signup_genres)

# ===== Pages =====
def login_signup_page():
    st.title("Movie Recommender – Login / Signup")
    option = st.radio("Select option", ["Login", "Signup"], horizontal=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if option == "Signup":
        if st.button("Signup"):
            if username and password:
                if signup_user(username):
                    st.session_state.username = username
                    st.session_state.page = "genre_select"
                    st.rerun()
                else:
                    st.error("Username already exists")
            else:
                st.error("Fill both fields")
    else:
        if st.button("Login"):
            user = load_user(username)
            if user:
                st.session_state.username = username
                st.session_state.watched = user.get("watched", [])
                st.session_state.genres = user.get("genres", [])
                st.session_state.page = "dashboard" if st.session_state.genres else "genre_select"
                st.rerun()
            else:
                st.error("User not found")

def genre_selection_page():
    st.title(f"Welcome, {st.session_state.username}!")
    st.subheader("Select your favourite genres")
    all_genres = sorted(set(g for glist in df['Genre'].str.split(', ') for g in glist))
    if "temp_selected_genres" not in st.session_state:
        st.session_state.temp_selected_genres = st.session_state.genres or []
    def toggle(genre):
        if genre in st.session_state.temp_selected_genres:
            st.session_state.temp_selected_genres.remove(genre)
        else:
            st.session_state.temp_selected_genres.append(genre)
    cols = st.columns(4)
    for i, genre in enumerate(all_genres):
        col = cols[i % 4]
        label = f"✅ {genre}" if genre in st.session_state.temp_selected_genres else genre
        if col.button(label, key=f"btn_{genre}"):
            toggle(genre)
    if st.button("Next"):
        if st.session_state.temp_selected_genres:
            update_user_genres(st.session_state.username, st.session_state.temp_selected_genres)
            st.session_state.genres = st.session_state.temp_selected_genres
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Select at least one genre")

def dashboard_page():
    st.sidebar.checkbox("🌙 Dark Mode", key="dark_mode")
    st.write(f"### Welcome, {st.session_state.username}")
    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])

    with tab1:
        search = st.text_input("🔍 Search in Top Rated", key="search_top").lower()
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False)
        genre_set = set()
        for genres in df['Genre'].str.split(', '):
            genre_set.update(genres)
        mixed_movies = [top_movies[top_movies['Genre'].str.contains(g, case=False)].head(3) for g in genre_set]
        mixed_df = pd.concat(mixed_movies).drop_duplicates(subset="Series_Title")
        mixed_df = mixed_df[~mixed_df['Series_Title'].isin(st.session_state.watched)]
        if search:
            mixed_df = mixed_df[mixed_df["Series_Title"].str.lower().str.contains(search) |
                                mixed_df["Genre"].str.lower().str.contains(search)]
        render_cards(mixed_df.head(50), st.session_state.watched, st.session_state.username, "top", show_button=True, signup_genres=st.session_state.genres)

    with tab2:
        search = st.text_input("🔍 Search in Your Watched", key="search_watched").lower()
        watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
        if search:
            watched_df = watched_df[watched_df["Series_Title"].str.lower().str.contains(search) |
                                    watched_df["Genre"].str.lower().str.contains(search)]
        render_cards(watched_df, st.session_state.watched, st.session_state.username, "your", show_button=False, signup_genres=st.session_state.genres)

    with tab3:
        search = st.text_input("🔍 Search in Recommendations", key="search_rec").lower()
        recs = recommend_for_user(st.session_state.genres, st.session_state.watched, top_n=10)
        if search:
            recs = recs[recs["Series_Title"].str.lower().str.contains(search) |
                        recs["Genre"].str.lower().str.contains(search)]
        reason_map = {}
        for idx, row in recs.iterrows():
            reasons = []
            watched_reasons = [w for w in st.session_state.watched if w in indices and cosine_sim[indices[w]][idx] > 0.1]
            if watched_reasons:
                reasons.append("You watched " + ", ".join(watched_reasons[:3]))
            genre_matches = [g for g in st.session_state.genres if g.lower() in row["Genre"].lower()]
            if genre_matches:
                reasons.append("You selected genre(s) " + ", ".join(genre_matches[:3]))
            reason_map[row['Series_Title']] = " and ".join(reasons) if reasons else None
        render_cards(recs, st.session_state.watched, st.session_state.username, "rec", show_button=True, reason_map=reason_map, signup_genres=st.session_state.genres)

# ===== Routing =====
if "page" not in st.session_state:
    st.session_state.page = "login_signup"
if "genres" not in st.session_state:
    st.session_state.genres = []
if "watched" not in st.session_state:
    st.session_state.watched = []

if st.session_state.page == "login_signup":
    login_signup_page()
elif st.session_state.page == "genre_select":
    genre_selection_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
