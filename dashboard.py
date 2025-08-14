import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import os

# ========= CONFIG =========
USER_DATA_FILE = "user_data.json"

# ========= USER DATA STORAGE HELPERS =========
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    else:
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
    data = load_user_data()
    return data.get(username)

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
    if watched_titles:
        genre_weight = 0.5
        watch_weight = 3.0
    else:
        genre_weight = 2.0
        watch_weight = 0.0

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

    rec_idx = np.argsort(scores)[::-1][:top_n]
    return df.iloc[rec_idx][['Series_Title', 'Genre', 'IMDB_Rating']]

# ========= SESSION STATE =========
if "auth" not in st.session_state:
    st.session_state.auth = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "genres" not in st.session_state:
    st.session_state.genres = []
if "watched" not in st.session_state:
    st.session_state.watched = []

# ========= LOGIN/SIGNUP PAGE =========
def login_signup_page():
    st.title("🎬 Movie Recommender Dashboard")
    option = st.radio("Select Option", ["Login", "Signup"], horizontal=True)

    if option == "Signup":
        username = st.text_input("Choose Username")
        password = st.text_input("Choose Password", type="password")  # placeholder
        genres = st.multiselect(
            "Select Favourite Genres",
            sorted(set(g for glist in df['Genre'].str.split(', ') for g in glist))
        )
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

    else:  # Login
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")  # placeholder
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

    # Top Rated
    with tab1:
        st.subheader("Top IMDb Rated Movies")
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False).head(20)
        for i, row in top_movies.iterrows():
            col1, col2, col3 = st.columns([5, 3, 2])
            col1.write(f"**{row['Series_Title']}** ({row['IMDB_Rating']})")
            col2.write(row['Genre'])
            if col3.button("Watched", key=f"topwatched_{i}"):
                if row['Series_Title'] not in st.session_state.watched:
                    st.session_state.watched.append(row['Series_Title'])
                    update_watched(st.session_state.username, st.session_state.watched)
                    st.success(f"Added '{row['Series_Title']}' ✅")
                    st.rerun()

    # Your Watching
    with tab2:
        if st.session_state.watched:
            st.write("🎬 Your Watched Movies:")
            for title in st.session_state.watched:
                st.write(f"- {title}")
        else:
            st.info("No watched movies yet.")

    # Recommendations
    with tab3:
        st.subheader("Recommended for You")
        if not st.session_state.genres and not st.session_state.watched:
            st.warning("Add some genres or watched movies first.")
        else:
            recs = recommend_for_user(st.session_state.genres, st.session_state.watched, top_n=10)

            for idx, row in recs.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### 🎬 {row['Series_Title']} ({row['IMDB_Rating']})")
                    st.write(f"**Genres:** {row['Genre']}")

                    # Build reason strings
                    reasons = []
                    for watched in st.session_state.watched:
                        if watched in indices:
                            w_idx = indices[watched]
                            if isinstance(w_idx, (pd.Series, list, np.ndarray)):
                                w_idx = w_idx.iloc[0] if hasattr(w_idx, "iloc") else w_idx[0]
                            if cosine_sim[w_idx][idx] > 0.1:
                                reasons.append(f"You watched **{watched}**")
                    for g in st.session_state.genres:
                        if g.lower() in row["Genre"].lower():
                            reasons.append(f"You selected genre **{g}** at signup")

                    if reasons:
                        st.caption("💡 " + " and ".join(list(dict.fromkeys(reasons))) + " → so we recommended this.")

                with col2:
                    if st.button("Watched", key=f"recwatched_{idx}"):
                        if row['Series_Title'] not in st.session_state.watched:
                            st.session_state.watched.append(row['Series_Title'])
                            update_watched(st.session_state.username, st.session_state.watched)
                            st.success(f"Added '{row['Series_Title']}' ✅")
                            st.rerun()

# ========= APP ROUTE =========
if not st.session_state.auth:
    login_signup_page()
else:
    dashboard()
