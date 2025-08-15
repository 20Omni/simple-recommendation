import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json, os
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
    if username in data: return False
    data[username] = {"genres": [], "watched": []}
    save_user_data(data)
    return True

def load_user(username): return load_user_data().get(username)

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
    signup_df = rec_df[rec_df['Genre'].str.contains('|'.join(preferred_genres), case=False)]
    return pd.concat([signup_df.head(3), rec_df]).drop_duplicates().head(top_n)[['Series_Title','Genre','IMDB_Rating']]

# ===== Emoji Mapping =====
genre_emojis = {
    "action":"🎬","comedy":"😂","drama":"🎭","romance":"❤️","thriller":"🔪","horror":"👻",
    "sci-fi":"👽","science fiction":"👽","adventure":"🧭","fantasy":"🦄","animation":"🐭",
    "documentary":"🎥","crime":"🕵️","mystery":"🕵️","war":"⚔️","musical":"🎶","music":"🎶"
}
def get_dominant_genre_with_emoji(genre_string, signup_genres=None):
    genres_list = [g.strip() for g in genre_string.split(",")]
    if signup_genres:
        for sg in signup_genres:
            for g in genres_list:
                if sg.lower() in g.lower():
                    return genre_emojis.get(g.lower(),"🎞️"), genre_string
    for g in genres_list:
        if g.lower() in genre_emojis:
            return genre_emojis[g.lower()], genre_string
    return "🎞️", genre_string

# ===== Card Renderer =====
def movie_card(row, watched_list, username, section, reason=None, show_button=True, signup_genres=None):
    dark = st.session_state.dark_mode
    bg_color = "#23272e" if dark else "#fdfdfe"
    text_color = "#f5f5f5" if dark else "#222"
    border_color = "#3d434d" if dark else "#e2e3e6"
    genre_color = "#b2b2b2" if dark else "#5A5A5A"
    rating_color = "#fcb900"

    emoji, genre_text = get_dominant_genre_with_emoji(row["Genre"], signup_genres)

    html = f'''
    <div style="border:1.5px solid {border_color};border-radius:10px;padding:12px;
    margin-bottom:16px;background:{bg_color};color:{text_color};
    box-shadow:0 2px 6px rgba(0,0,0,0.08);">
        <div style="font-weight:700;font-size:1.1rem;">{row["Series_Title"]}</div>
        <div style="color:{genre_color};margin-bottom:5px;">{emoji} <span style="font-style: italic;">{genre_text}</span></div>
        <div style="color:{rating_color};margin-bottom:8px;">⭐ {row["IMDB_Rating"]:.1f}/10</div>
        {f'<div style="color:#399ed7;margin-bottom:8px;">💡 {reason}</div>' if reason else ""}
    '''
    st.markdown(html, unsafe_allow_html=True)

    if show_button:
        key = f"watched_{section}_{row.name}"
        st.markdown("""
            <style>
            .stButton button {
                background-color: #4CAF50; 
                color: white; 
                padding: 6px 14px; 
                font-size: 14px; 
                border: none; 
                border-radius: 6px; 
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: #45a049;
            }
            .stButton button:disabled {
                background-color: #888;
                cursor: default;
            }
            </style>
        """, unsafe_allow_html=True)

        if row['Series_Title'] not in watched_list:
            if st.button("✅ Watched", key=key, help="Mark this as watched"):
                watched_list.append(row['Series_Title'])
                update_watched(username, watched_list)
                st.rerun()
        else:
            st.button("✅ Watched", key=key, disabled=True)

# ===== Render Grid =====
def render_cards(dataframe, watched_list, username, section, show_button=True, reason_map=None, signup_genres=None):
    cols_per_row = 3
    for r in range(ceil(len(dataframe) / cols_per_row)):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r*cols_per_row + c
            if idx < len(dataframe):
                row = dataframe.iloc[idx]
                reason = reason_map.get(row['Series_Title']) if reason_map else None
                with cols[c]:
                    movie_card(row, watched_list, username, section, reason, show_button, signup_genres)

# ===== Login/Signup =====
def login_signup_page():
    st.title("Movie Recommender – Login / Signup")
    opt = st.radio("Select option", ["Login", "Signup"], horizontal=True)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if opt == "Signup":
        if st.button("Signup") and username and password:
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
    else:  # Login
        if st.button("Login") and username and password:
            user = load_user(username)
            if user:
                st.session_state.username = username
                st.session_state.watched = user.get("watched", [])
                st.session_state.genres = user.get("genres", [])
                if not st.session_state.genres:
                    st.session_state.temp_selected_genres = []
                st.session_state.page = "dashboard" if st.session_state.genres else "genre_select"
                st.rerun()
            else:
                st.error("User not found")

# ===== Genre Selection =====
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
            if st.button(btn_label, key=f"genre_{genre}"):
                if selected:
                    st.session_state.temp_selected_genres.remove(genre)
                else:
                    st.session_state.temp_selected_genres.append(genre)
                st.rerun()

    if st.button("Next ➡️"):
        if st.session_state.temp_selected_genres:
            update_user_genres(st.session_state.username, st.session_state.temp_selected_genres)
            st.session_state.genres = st.session_state.temp_selected_genres.copy()
            st.session_state.scroll_to_top = True
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Please select at least one genre to continue.")

# ===== Dashboard with Live Filtering + Suggestions =====
def dashboard_page():
    if st.session_state.get("scroll_to_top", False):
        st.markdown("<script>window.scrollTo({top: 0, behavior: 'instant'});</script>", unsafe_allow_html=True)
        st.session_state.scroll_to_top = False

    st.sidebar.checkbox("🌙 Dark Mode", key="dark_mode")
    st.write(f"### Welcome, {st.session_state.username}")
    
    if st.button("🚪 Logout"):
        st.session_state.page, st.session_state.username = "login_signup", ""
        st.session_state.genres, st.session_state.watched, st.session_state.temp_selected_genres = [], [], []
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])

    # Tab 1 - Top Rated
    with tab1:
        search_query = st.text_input("🔍 Search top movies...", key="top_q").strip().lower()
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False)
        mixed_df = pd.concat([top_movies[top_movies['Genre'].str.contains(g, case=False)].head(3)
                              for g in set(g for lst in df['Genre'].str.split(', ') for g in lst)]
                             ).drop_duplicates("Series_Title")
        mixed_df = mixed_df[~mixed_df['Series_Title'].isin(st.session_state.watched)].head(50)

        if search_query:
            suggestions = mixed_df[mixed_df["Series_Title"].str.lower().str.contains(search_query)]["Series_Title"].head(5).tolist()
            for s in suggestions:
                st.markdown(f"- {s}")
            mixed_df = mixed_df[mixed_df["Series_Title"].str.lower().str.contains(search_query) |
                                mixed_df["Genre"].str.lower().str.contains(search_query)]
        render_cards(mixed_df, st.session_state.watched, st.session_state.username, "top", True, signup_genres=st.session_state.genres)

    # Tab 2 - Watching
    with tab2:
        search_query = st.text_input("🔍 Search your watched movies...", key="watch_q").strip().lower()
        watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
        if search_query:
            suggestions = watched_df[watched_df["Series_Title"].str.lower().str.contains(search_query)]["Series_Title"].head(5).tolist()
            for s in suggestions:
                st.markdown(f"- {s}")
            watched_df = watched_df[watched_df["Series_Title"].str.lower().str.contains(search_query) |
                                    watched_df["Genre"].str.lower().str.contains(search_query)]
        render_cards(watched_df, st.session_state.watched, st.session_state.username, "your", False, signup_genres=st.session_state.genres)

    # Tab 3 - Recommendations
    with tab3:
        search_query = st.text_input("🔍 Search recommended...", key="rec_q").strip().lower()
        recs = recommend_for_user(st.session_state.genres, st.session_state.watched, 10)

        reason_map = {}
        for idx, row in recs.iterrows():
            reasons = []
            watched_reasons = [w for w in st.session_state.watched if w in indices and cosine_sim[indices[w]][idx] > 0.1]
            if watched_reasons:
                reasons.append("You watched " + ", ".join(watched_reasons[:3]))
            genre_matches = [g for g in st.session_state.genres if g.lower() in row["Genre"].lower()][:3]
            if genre_matches:
                reasons.append("You selected genre(s) " + ", ".join(genre_matches))
            reason_map[row['Series_Title']] = " and ".join(reasons) if reasons else None

        if search_query:
            suggestions = recs[recs["Series_Title"].str.lower().str.contains(search_query)]["Series_Title"].head(5).tolist()
            for s in suggestions:
                st.markdown(f"- {s}")
            recs = recs[recs["Series_Title"].str.lower().str.contains(search_query) |
                        recs["Genre"].str.lower().str.contains(search_query)]
        render_cards(recs, st.session_state.watched, st.session_state.username, "rec", True, reason_map, signup_genres=st.session_state.genres)

# ===== Routing =====
if "page" not in st.session_state: st.session_state.page = "login_signup"
if "genres" not in st.session_state: st.session_state.genres = []
if "watched" not in st.session_state: st.session_state.watched = []
if "temp_selected_genres" not in st.session_state: st.session_state.temp_selected_genres = []

if st.session_state.page == "login_signup": login_signup_page()
elif st.session_state.page == "genre_select": genre_selection_page()
elif st.session_state.page == "dashboard": dashboard_page()
