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

# ===== Emoji Mapping =====
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

# ===== Base CSS for cards/buttons =====
st.markdown("""
<style>
.movie-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer;
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    padding-bottom: 32px;
    position: relative;
    margin-bottom: 20px;
    opacity: 1;
}
.movie-card:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    position: relative;
    z-index: 10;
}
div[data-testid="stButton"] > button[kind="primary"]{
    background:#ff4b4b;
    color:#fff;
    border-radius:999px;
    font-weight:700;
    box-shadow:0 2px 6px rgba(0,0,0,0.2);
}
div[data-testid="stButton"] > button[kind="primary"]:hover{
    background:#e64444;
}
</style>
""", unsafe_allow_html=True)

# ===== Premium Theme function =====
THEMES = {
    "🎭 Cinematic Curtain": {
        "bg": "url('https://i.ibb.co/2nF0D9n/red-curtain.jpg')",
        "overlay": "rgba(0,0,0,0.65)",
        "chip": "rgba(0,0,0,0.3)",
        "card_border": "gold"
    },
    "🌌 Galaxy Night": {
        "bg": "linear-gradient(135deg,#0f0c29,#302b63,#24243e)",
        "overlay": "rgba(0,0,0,0.45)",
        "chip": "rgba(255,255,255,0.2)",
        "card_border": "#00ffff"
    },
    "💡 Neon Glow": {
        "bg": "radial-gradient(circle at 10% 20%, #ff2d95 0%, transparent 10%), radial-gradient(circle at 90% 80%, #04e5ff 0%, transparent 10%), #000",
        "overlay": "rgba(0,0,0,0.5)",
        "chip": "rgba(255,255,255,0.15)",
        "card_border": "#ff2d95"
    },
    "🕶️ Minimal Dark": {
        "bg": "linear-gradient(180deg,#000000,#1a1a1a)",
        "overlay": "rgba(0,0,0,0.5)",
        "chip": "rgba(255,255,255,0.1)",
        "card_border": "#444"
    }
}

def apply_theme(theme_choice):
    theme = THEMES[theme_choice]
    st.session_state["_chip_bg"] = theme["chip"]
    st.session_state["_card_border"] = theme["card_border"]
    st.markdown(f"""
    <style>
    .stApp {{
        background: {theme['bg']} !important;
        color: #f0f0f0;
    }}
    .stApp::before {{
        content: '';
        position: fixed;inset:0;
        background:{theme['overlay']};
    }}
    .genre-chip-custom {{
        background: {theme['chip']};
        padding:6px 12px;border-radius:16px;margin:6px;
        display:inline-block;font-weight:700;
        backdrop-filter: blur(4px);
    }}
    </style>
    """, unsafe_allow_html=True)

# ===== Time-based greeting =====
def time_greeting(username):
    h = datetime.now().hour
    if h < 12:
        prefix = "Good Morning"
    elif h < 18:
        prefix = "Good Afternoon"
    else:
        prefix = "Good Evening"
    return f"{prefix}, {username} 🍿 ready for a movie night?"

# ===== Card Renderer =====
def movie_card(row, watched_list, username, section, reason=None, show_button=True, signup_genres=None):
    border_color = st.session_state.get("_card_border", "#e2e3e6")
    emoji, genre_text = get_dominant_genre_with_emoji(row["Genre"], signup_genres)
    html = f"""
    <div class="movie-card" style="border:2px solid {border_color};border-radius:12px;padding:12px;">
      <b>{row['Series_Title']} ({row['Released_Year']})</b><br>
      {emoji} <i>{genre_text}</i><br>
      <span style='color:gold;'>⭐ {row['IMDB_Rating']:.1f}/10</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ===== Render Cards Grid =====
def render_cards(dataframe, watched_list, username, section, show_button=True, reason_map=None, signup_genres=None):
    cols_per_row = 3
    for r in range(ceil(len(dataframe) / cols_per_row)):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r*cols_per_row + c
            if idx < len(dataframe):
                row = dataframe.iloc[idx]
                movie_card(row, watched_list, username, section, signup_genres=signup_genres)

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
    cols_per_row = 4
    cols = st.columns(cols_per_row)
    for idx, genre in enumerate(all_genres):
        emoji = genre_emojis.get(genre.lower(), "🎞️")
        selected = genre in st.session_state.temp_selected_genres
        btn_label = f"✅ {emoji} {genre}" if selected else f"{emoji} {genre}"
        with cols[idx % cols_per_row]:
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
    theme_choice = st.sidebar.radio("🎨 Theme", list(THEMES.keys()))
    apply_theme(theme_choice)

    if st.sidebar.button("🚪 Logout"):
        st.session_state.page = "login_signup"
        st.session_state.username = ""
        st.session_state.genres = []
        st.session_state.watched = []
        st.session_state.temp_selected_genres = []
        st.rerun()

    greeting = time_greeting(st.session_state.username or "Movie Buff")
    st.markdown(f"""
    <div style="backdrop-filter:blur(10px);background:rgba(0,0,0,0.5);
                padding:22px;border-radius:20px;text-align:center;color:white;
                box-shadow:0 6px 18px rgba(0,0,0,0.3);">
        <h1 style="margin:0;font-weight:900;">{greeting}</h1>
        <div style="margin-top:8px;font-size:22px;">
            <span class="bounce">🍿</span> <span class="pulse">⭐</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.genres:
        chips_html = " ".join([f"<span class='genre-chip-custom'>{g}</span>" for g in st.session_state.genres])
        st.markdown(f"<div style='text-align:center;margin-top:12px;'>{chips_html}</div>", unsafe_allow_html=True)

    

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


    with tab1:
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False)
        mixed_df = pd.concat([
            top_movies[top_movies['Genre'].str.contains(g, case=False, na=False)].head(3)
            for g in set(g for lst in df['Genre'].str.split(', ') for g in lst)
        ]).drop_duplicates("Series_Title")
        mixed_df = mixed_df[~mixed_df['Series_Title'].isin(st.session_state.watched)].head(50)
        selected_title = st_searchbox(search_top_movies, placeholder="Search top movies...", key="top_searchbox")
        if selected_title:
            mixed_df = mixed_df[mixed_df['Series_Title'] == selected_title]
        render_cards(mixed_df, st.session_state.watched, st.session_state.username, "top", True, signup_genres=st.session_state.genres)

    with tab2:
        watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
        if watched_df.empty:
            st.info("You haven’t watched anything yet!")
        else:
            selected_title = st_searchbox(search_watched_movies, placeholder="Search watched movies...", key="watched_searchbox")
            if selected_title:
                watched_df = watched_df[watched_df['Series_Title'] == selected_title]
            render_cards(watched_df, st.session_state.watched, st.session_state.username, "your", False, signup_genres=st.session_state.genres)

    with tab3:
        recs = recommend_for_user(st.session_state.genres, st.session_state.watched, 10)
        reason_map = {}
        for idx, row in recs.iterrows():
            reasons = []
            watched_reasons = [
                w for w in st.session_state.watched
                if w in indices and cosine_sim[indices[w]][idx] > 0.1
            ]
            if watched_reasons:
                reasons.append("You watched " + ", ".join(watched_reasons[:3]))
            genre_matches = [g for g in st.session_state.genres if g.lower() in row["Genre"].lower()][:3]
            if genre_matches:
                reasons.append("You selected genre(s) " + ", ".join(genre_matches))
            reason_map[row['Series_Title']] = " and ".join(reasons) if reasons else None
        selected_title = st_searchbox(search_recommended_movies, placeholder="Search recommended movies...", key="rec_searchbox")
        if selected_title:
            recs = recs[recs['Series_Title'] == selected_title]
        render_cards(recs, st.session_state.watched, st.session_state.username, "rec", True, reason_map, signup_genres=st.session_state.genres)

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
