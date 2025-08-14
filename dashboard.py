import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import os

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
    df = joblib.load("movies_df.pkl")  # Must contain Poster_Link col
    cosine_sim = joblib.load("cosine_similarity.pkl")
    indices = joblib.load("title_indices.pkl")
    return df, cosine_sim, indices

df, cosine_sim, indices = load_model()

# ========= RECOMMENDER =========
def recommend_for_user(preferred_genres, watched_titles, top_n=10):
    scores = np.zeros(len(df))
    genre_weight = 0.5 if watched_titles else 2.0
    watch_weight = 3.0 if watched_titles else 0.0

    for genre in preferred_genres:
        mask = df['Genre'].str.contains(genre, case=False, na=False)
        scores[mask] += genre_weight

    for title in watched_titles:
        if title in indices:
            idx = indices[title]
            if isinstance(idx, (pd.Series, list, np.ndarray)):
                sim_vec = cosine_sim[idx].mean(axis=0)
            else:
                sim_vec = cosine_sim[idx]
            scores += watch_weight * sim_vec

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
    # Exclude watched
    rec_df = rec_df[~rec_df['Series_Title'].isin(watched_titles)]
    return rec_df.head(top_n)[['Series_Title', 'Genre', 'IMDB_Rating', 'Poster_Link']] \
           if 'Poster_Link' in df.columns else rec_df.head(top_n)

# ========= MOVIE CARD =========
def movie_card(row, watched_list, username, section, reason=None):
    col_img, col_info = st.columns([1, 3])

    if 'Poster_Link' in row and pd.notna(row['Poster_Link']):
        col_img.image(row['Poster_Link'], width=100)
    else:
        col_img.image("https://via.placeholder.com/100x150.png?text=No+Image", width=100)

    col_info.markdown(f"**{row['Series_Title']}** ({row['IMDB_Rating']})")
    col_info.write(f"Genres: {row['Genre']}")
    if reason:
        col_info.caption(f"💡 {reason}")

    key = f"watched_btn_{section}_{row.name}"
    if row['Series_Title'] in watched_list:
        col_info.button("Watched", key=key, disabled=True)
    else:
        if col_info.button("Mark as Watched", key=key):
            watched_list.append(row['Series_Title'])
            update_watched(username, watched_list)
            st.success(f"Added '{row['Series_Title']}' ✅")
            st.rerun()

# ========= SESSION STATE =========
if "auth" not in st.session_state:
    st.session_state.auth = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "genres" not in st.session_state:
    st.session_state.genres = []
if "watched" not in st.session_state:
    st.session_state.watched = []

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

# ========= DASHBOARD =========
def dashboard():
    st.markdown(f"### 👋 Welcome, **{st.session_state.username}**")
    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.auth = False
        st.session_state.username = ""
        st.session_state.genres = []
        st.session_state.watched = []
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])

    # --- Top Rated ---
    with tab1:
        st.subheader("Top IMDb Rated Movies")
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False).head(50)
        top_movies = top_movies[~top_movies['Series_Title'].isin(st.session_state.watched)]
        for _, row in top_movies.iterrows():
            movie_card(row, st.session_state.watched, st.session_state.username, section="top")

    # --- Your Watching ---
    with tab2:
        st.subheader("Your Watched Movies")
        if st.session_state.watched:
            watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
            for _, row in watched_df.iterrows():
                movie_card(row, st.session_state.watched, st.session_state.username, section="your")
        else:
            st.info("No watched movies yet.")

    # --- Recommendations ---
    with tab3:
        st.subheader("Recommended for You")
        if not st.session_state.genres and not st.session_state.watched:
            st.warning("Add some genres or watched movies first.")
        else:
            recs = recommend_for_user(st.session_state.genres,
                                      st.session_state.watched,
                                      top_n=10)

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
                    reasons_list.append("You selected genre(s) " + ", ".join(genre_matches) + " at signup")

                reason_text = " and ".join(reasons_list) if reasons_list else None

                movie_card(row, st.session_state.watched, st.session_state.username, section="rec", reason=reason_text)

# ========= APP ROUTER =========
if not st.session_state.auth:
    login_signup_page()
else:
    dashboard()
