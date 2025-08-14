import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===== Load pre-trained model artifacts =====
@st.cache_resource
def load_model():
    df = joblib.load("movies_df.pkl")
    cosine_sim = joblib.load("cosine_similarity.pkl")
    indices = joblib.load("title_indices.pkl")
    return df, cosine_sim, indices

df, cosine_sim, indices = load_model()

# ===== Recommendation function =====
def recommend_for_user(preferred_genres, watched_titles, top_n=10):
    scores = np.zeros(len(df))

    # Boost for preferred genres
    for genre in preferred_genres:
        mask = df['Genre'].str.contains(genre, case=False, na=False)
        scores[mask] += 1.0

    # Boost for similarity to watched movies (handle duplicates)
    for title in watched_titles:
        if title in indices:
            idx = indices[title]
            if isinstance(idx, (pd.Series, list, np.ndarray)):
                sim_vec = cosine_sim[idx].mean(axis=0)
            else:
                sim_vec = cosine_sim[idx]
            scores += sim_vec

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

    recommended_idx = np.argsort(scores)[::-1][:top_n]
    return df.iloc[recommended_idx][['Series_Title', 'Genre', 'IMDB_Rating']]

# ===== Session state =====
if "auth" not in st.session_state:
    st.session_state.auth = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "genres" not in st.session_state:
    st.session_state.genres = []
if "watched" not in st.session_state:
    st.session_state.watched = []

# ===== Login/Signup page =====
def login_signup_page():
    st.title("🎬 Movie Recommender Dashboard")
    st.subheader("Login or Signup to continue")

    option = st.radio("Select Option", ["Login", "Signup"], horizontal=True)

    if option == "Signup":
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        genres = st.multiselect(
            "Select Your Favourite Genres",
            sorted(set(g for glist in df['Genre'].str.split(', ') for g in glist))
        )

        if st.button("Signup"):
            if username and password and genres:
                st.session_state.auth = True
                st.session_state.username = username
                st.session_state.genres = genres
                st.session_state.watched = []
                st.success(f"Welcome {username}! You have signed up successfully 🎉")
                st.experimental_rerun()
            else:
                st.error("Please fill all fields and select at least one genre.")

    else:  # Login
        username = st.text_input("Enter Username")
        password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if username:
                st.session_state.auth = True
                st.session_state.username = username
                if not st.session_state.genres:
                    st.session_state.genres = []
                if not st.session_state.watched:
                    st.session_state.watched = []
                st.experimental_rerun()
            else:
                st.error("Please enter username.")

# ===== Dashboard =====
def dashboard():
    st.markdown(f"### 👋 Welcome, **{st.session_state.username}**")
    st.markdown("---")

    # Logout button
    if st.button("🚪 Logout"):
        st.session_state.auth = False
        st.session_state.username = ""
        st.session_state.genres = []
        st.session_state.watched = []
        st.experimental_rerun()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])

    with tab1:
        st.subheader("Top IMDb Rated Movies")
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False)
        for i, row in top_movies.iterrows():
            col1, col2, col3 = st.columns([5, 3, 2])
            col1.write(f"**{row['Series_Title']}** ({row['IMDB_Rating']})")
            col2.write(row['Genre'])
            # FIX: unique key using row index
            if col3.button("Watched", key=f"watched_{i}"):
                if row['Series_Title'] not in st.session_state.watched:
                    st.session_state.watched.append(row['Series_Title'])
                    st.success(f"Added '{row['Series_Title']}' to your watching list ✅")

    with tab2:
        st.subheader("Your Watching List")
        if st.session_state.watched:
            for title in st.session_state.watched:
                st.write(f"🎬 {title}")
        else:
            st.info("You haven't added any movies yet.")

    with tab3:
        st.subheader("Recommended for You")
        if not st.session_state.genres:
            st.warning("No genres selected. Please sign up again with genres to get recommendations!")
        else:
            recs = recommend_for_user(st.session_state.genres, st.session_state.watched, top_n=10)
            st.table(recs)

# ===== App run =====
if not st.session_state.auth:
    login_signup_page()
else:
    dashboard()
