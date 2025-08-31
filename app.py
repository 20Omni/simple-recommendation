
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json, os, textwrap
from math import ceil
from datetime import datetime
from streamlit_searchbox import st_searchbox

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
    if username in data: return False
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

# ===== Theme & Animations =====
def apply_theme(theme_choice):
    if theme_choice == "🎬 Classic":
        bg = "linear-gradient(135deg, #2c3e50, #4ca1af)"
    elif theme_choice == "🌙 Dark":
        bg = "linear-gradient(135deg, #000000, #1a1a1a)"
    else:
        bg = "linear-gradient(135deg, #ff4b1f, #1fddff)"

    st.markdown(f"""
        <style>
        body {{
            background: {bg};
        }}
        @keyframes fadeUp {{
            from {{opacity:0; transform: translateY(30px);}}
            to {{opacity:1; transform: translateY(0);}}
        }}
        .card-animate {{
            animation: fadeUp 1s ease forwards;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.2); }}
            100% {{ transform: scale(1); }}
        }}
        .pulse {{ animation: pulse 1.5s infinite; }}
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-8px); }}
        }}
        .bounce {{ animation: bounce 1s infinite; }}
        </style>
    """, unsafe_allow_html=True)

# ===== Time-based Greeting =====
def get_greeting(username):
    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good Morning"
        bg = "linear-gradient(90deg,#ffb347,#ffcc33)"
    elif hour < 18:
        greeting = "Good Afternoon"
        bg = "linear-gradient(90deg,#ff944b,#ff4b4b)"
    else:
        greeting = "Good Evening"
        bg = "linear-gradient(90deg,#2c3e50,#4ca1af)"
    return f"{greeting}, {username} 🍿 ready for a movie night?", bg

# ===== Recommendation Logic =====
def recommend_for_user(preferred_genres, watched_titles, top_n=10):
    scores = np.zeros(len(df))
    if len(watched_titles) >= 3: genre_weight, watch_weight = 0.3, 4.0
    elif watched_titles: genre_weight, watch_weight = 0.5, 3.5
    else: genre_weight, watch_weight = 2.0, 0.0

    for genre in preferred_genres:
        scores[df['Genre'].str.contains(genre, case=False, na=False)] += genre_weight

    for title in watched_titles:
        if title in indices:
            idx = indices[title]
            sim_vec = cosine_sim[idx].mean(axis=0) if isinstance(idx,(pd.Series,list,np.ndarray)) else cosine_sim[idx]
            scores += watch_weight * sim_vec

    watched_idx = []
    for t in watched_titles:
        if t in indices:
            idx_val = indices[t]
            watched_idx.extend(idx_val if isinstance(idx_val,(pd.Series,list,np.ndarray)) else [idx_val])

    scores[watched_idx] = -1
    rec_df = df.iloc[np.argsort(scores)[::-1]]
    rec_df = rec_df[~rec_df['Series_Title'].isin(watched_titles)]

    if preferred_genres:
        signup_df = rec_df[rec_df['Genre'].str.contains('|'.join(preferred_genres), case=False, na=False)]
    else:
        signup_df = rec_df.head(0)

    rec_df = pd.concat([signup_df.head(3), rec_df]).drop_duplicates()
    return rec_df.head(top_n)[['Series_Title','Genre','IMDB_Rating','Certificate','Released_Year']]

# ===== Genre Emojis =====
genre_emojis = {
    "action":"🎬","comedy":"😂","drama":"🎭","romance":"❤️","thriller":"🔪","horror":"👻",
    "sci-fi":"👽","science fiction":"👽","adventure":"🧭","fantasy":"🦄","animation":"🐭",
    "documentary":"🎥","crime":"🕵️","mystery":"🕵️","war":"⚔️","musical":"🎶","music":"🎶"
}

def get_dominant_genre_with_emoji(genre_string, signup_genres=None):
    genres_list = [g.strip() for g in str(genre_string).split(",")]
    if signup_genres:
        for sg in signup_genres:
            for g in genres_list:
                if sg.lower() in g.lower():
                    return genre_emojis.get(g.lower(),"🎞️"), genre_string
    for g in genres_list:
        if g.lower() in genre_emojis:
            return genre_emojis[g.lower()], genre_string
    return "🎞️", genre_string

# ===== Login/Signup Page =====
def login_signup_page():
    st.title("Movie Recommender – Login / Signup")
    opt = st.radio("Select option", ["Login", "Signup"], horizontal=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if opt == "Signup":
        if st.button("Signup", type="primary") and username and password:
            if len(password) < 5:
                st.error("Password must be at least 5 characters long.")
            else:
                if signup_user(username):
                    st.session_state.username = username
                    st.session_state.watched, st.session_state.genres, st.session_state.temp_selected_genres = [], [], []
                    st.session_state.page = "genre_select"
                    st.rerun()
                else:
                    st.error("Username already exists")
    else:
        if st.button("Login", type="primary") and username and password:
            user = load_user(username)
            if user:
                st.session_state.username = username
                st.session_state.watched = user.get("watched", [])
                st.session_state.genres = user.get("genres", [])
                st.session_state.page = "dashboard" if st.session_state.genres else "genre_select"
                st.rerun()
            else:
                st.error("User not found")

# ===== Genre Selection Page =====
def genre_selection_page():
    st.title(f"Welcome, {st.session_state.username}!")
    st.subheader("Select Your Favourite Genres")
    all_genres = sorted(set(g for glist in df['Genre'].str.split(', ') for g in glist))
    if "temp_selected_genres" not in st.session_state:
        st.session_state.temp_selected_genres = []
    cols = st.columns(4)
    for idx, genre in enumerate(all_genres):
        emoji = genre_emojis.get(genre.lower(), "🎞️")
        selected = genre in st.session_state.temp_selected_genres
        btn_label = f"✅ {emoji} {genre}" if selected else f"{emoji} {genre}"
        with cols[idx % 4]:
            if st.button(btn_label, key=f"genre_{genre}_btn"):
                if selected:
                    st.session_state.temp_selected_genres.remove(genre)
                else:
                    st.session_state.temp_selected_genres.append(genre)
                st.rerun()

    if st.button("Next ➡️", type="primary"):
        if st.session_state.temp_selected_genres:
            update_user_genres(st.session_state.username, st.session_state.temp_selected_genres)
            st.session_state.genres = st.session_state.temp_selected_genres.copy()
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Please select at least one genre to continue.")

# ===== Dashboard Page =====
def dashboard_page():
    theme_choice = st.sidebar.selectbox("🎨 Choose Theme", ["🎬 Classic", "🌙 Dark", "🌈 Vibrant"])
    apply_theme(theme_choice)

    if st.sidebar.button("🚪 Logout"):
        st.session_state.page = "login_signup"
        st.session_state.username = ""
        st.session_state.genres = []
        st.session_state.watched = []
        st.session_state.temp_selected_genres = []
        st.rerun()

    greeting, bg_color = get_greeting(st.session_state.username)

    st.markdown(f"""
        <div style="background:{bg_color};
                    padding:20px;border-radius:12px;text-align:center;color:white;">
            <h1 style="margin:0;">{greeting}</h1>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.genres:
        chips_html = " ".join([
            f"<span style='background:#ffffff20;color:white;padding:6px 12px;border-radius:16px;margin:4px;display:inline-block;'>{g}</span>"
            for g in st.session_state.genres
        ])
        st.markdown(f"""
            <div style='text-align:center;margin-top:10px;'>
                {chips_html}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center;margin-top:20px;">
            <span class="pulse">⭐ IMDb</span> &nbsp;&nbsp;
            <span class="bounce">🍿</span>
        </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])

    with tab1:
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False).head(20)
        st.dataframe(top_movies[["Series_Title","Genre","IMDB_Rating"]])

    with tab2:
        watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
        if watched_df.empty:
            st.info("You haven’t watched anything yet!")
        else:
            st.dataframe(watched_df[["Series_Title","Genre","IMDB_Rating"]])

    with tab3:
        recs = recommend_for_user(st.session_state.genres, st.session_state.watched, 10)
        st.dataframe(recs)

# ===== Routing =====
if "page" not in st.session_state:
    st.session_state.page = "login_signup"
if "genres" not in st.session_state:
    st.session_state.genres = []
if "watched" not in st.session_state:
    st.session_state.watched = []
if "temp_selected_genres" not in st.session_state:
    st.session_state.temp_selected_genres = []

if st.session_state.page == "login_signup":
    login_signup_page()
elif st.session_state.page == "genre_select":
    genre_selection_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
