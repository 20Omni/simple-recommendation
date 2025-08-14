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
    df = joblib.load("movies_df.pkl")  # Assume Poster_Link exists but we will ignore posters
    cosine_sim = joblib.load("cosine_similarity.pkl")
    indices = joblib.load("title_indices.pkl")
    return df, cosine_sim, indices

df, cosine_sim, indices = load_model()

# ========= RECOMMENDER =========
def recommend_for_user(preferred_genres, watched_titles, top_n=10):
    scores = np.zeros(len(df))

    if len(watched_titles) >= 3:
        genre_weight = 0.3
        watch_weight = 4.0
    elif len(watched_titles) > 0:
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
    rec_df = rec_df[~rec_df['Series_Title'].isin(watched_titles)]

    signup_df = rec_df[rec_df['Genre'].str.contains('|'.join(preferred_genres), case=False)]
    mixed_df = pd.concat([signup_df.head(3), rec_df]).drop_duplicates().head(top_n)

    return mixed_df[['Series_Title', 'Genre', 'IMDB_Rating']]

# ========= STAR RATING UX =========
def star_rating(rating):
    filled_stars = int(round(rating / 2))  # IMDb 1-10 scale to 5 stars approx
    stars = "⭐" * filled_stars + "✩" * (5 - filled_stars)
    return stars

# ========= NICE MOVIE CARD NO POSTER =========
def movie_card_no_poster(row, watched_list, username, section, reason=None, show_button=True):
    # Boxed card style using st.markdown + container + css
    movie_title = row['Series_Title']
    genre = row['Genre']
    rating = row['IMDB_Rating']
    stars = star_rating(rating)

    card_style = """
    <style>
    .movie-card {
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background: #f9f9f9;
        box-shadow: 2px 2px 6px rgb(200 200 200 / 0.5);
    }
    </style>
    """

    st.markdown(card_style, unsafe_allow_html=True)
    with st.container():
        st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(f"### {movie_title}")
            st.markdown(f"**Genres:** {genre}")
            st.markdown(f"**Rating:** {stars} ({rating:.1f}/10)")
            if reason:
                st.caption(f"💡 {reason}")
        with col2:
            key = f"watched_btn_{section}_{row.name}"
            if movie_title in watched_list:
                st.button("Watched", key=key, disabled=True)
            elif show_button:
                if st.button("Mark as Watched", key=key):
                    watched_list.append(movie_title)
                    update_watched(username, watched_list)
                    st.success(f"Added '{movie_title}' ✅")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

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

    # Top Rated tab
    with tab1:
        st.subheader("Top IMDb Rated Movies (Mixed Genres)")
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False)

        # Include variety: sample 3 movies from each genre found
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

        for _, row in mixed_df.iterrows():
            movie_card_no_poster(row, st.session_state.watched, st.session_state.username, "top")

    # Your Watching tab
    with tab2:
        st.subheader("Your Watched Movies")
        if st.session_state.watched:
            watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
            for _, row in watched_df.iterrows():
                # Minimal card, no button, no poster
                movie_title = row['Series_Title']
                genre = row['Genre']
                rating = row['IMDB_Rating']
                stars = star_rating(rating)
                card_style = """
                <style>
                .movie-card-minimal {
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 10px;
                    margin-bottom: 10px;
                    background: #fafafa;
                }
                </style>
                """
                st.markdown(card_style, unsafe_allow_html=True)
                st.markdown(f'<div class="movie-card-minimal">', unsafe_allow_html=True)
                st.markdown(f"### {movie_title}")
                st.markdown(f"Genres: {genre}")
                st.markdown(f"Rating: {stars} ({rating:.1f}/10)")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No watched movies yet.")

    # Recommendations tab
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

                movie_card_no_poster(row, st.session_state.watched, st.session_state.username, "rec", reason=reason_text)

# ========= APP ROUTER =========
if not st.session_state.auth:
    login_signup_page()
else:
    dashboard()
