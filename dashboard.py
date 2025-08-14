

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json, os
from math import ceil
from streamlit_extras.st_searchbox import st_searchbox


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
    save_user_data(data); return True
def load_user(username): return load_user_data().get(username)
def update_user_genres(username, genres):
    data = load_user_data()
    if username in data: data[username]['genres'] = genres; save_user_data(data)
def update_watched(username, watched_list):
    data = load_user_data()
    if username in data: data[username]['watched'] = watched_list; save_user_data(data)

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
        if g.lower() in genre_emojis: return genre_emojis[g.lower()], genre_string
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
    st.markdown(f"""
    <style>.movie-card{{border:1.5px solid {border_color};border-radius:16px;padding:16px;
    margin-bottom:22px;background:{bg_color};color:{text_color};
    box-shadow:0 2px 7px rgba(0,0,0,0.10);transition:transform 0.2s ease,box-shadow 0.2s ease;}}
    .movie-card:hover{{transform:translateY(-6px);box-shadow:0 8px 22px rgba(0,0,0,0.17);}}
    .movie-title{{font-size:1.15rem;font-weight:700;margin-bottom:4px;}}
    .movie-genres{{font-size:0.9rem;color:{genre_color};margin-bottom:6px;}}
    .movie-rating{{font-size:1.2rem;color:{rating_color};margin-bottom:6px;}}
    .reason-text{{font-size:0.9rem;color:#399ed7;margin-bottom:8px;}}</style>
    """, unsafe_allow_html=True)
    html = f'<div class="movie-card"><div class="movie-title">{row["Series_Title"]}</div>'
    html += f'<div class="movie-genres"><span style="font-style: normal;">{emoji}</span> <span style="font-style: italic;">{genre_text}</span></div>'
    html += f'<div class="movie-rating">⭐ {row["IMDB_Rating"]:.1f}/10</div>'
    if reason: html += f'<div class="reason-text">💡 {reason}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    if show_button:
        key = f"watched_btn_{section}_{row.name}"
        if row['Series_Title'] not in watched_list:
            if st.button("Watched", key=key):
                watched_list.append(row['Series_Title']); update_watched(username, watched_list); st.rerun()
        else: st.button("Watched", key=key, disabled=True)

# ===== Render Grid =====
def render_cards(dataframe, watched_list, username, section, show_button=True, reason_map=None, signup_genres=None):
    cols_per_row = 3
    for r in range(ceil(len(dataframe) / cols_per_row)):
        cols = st.columns(cols_per_row, gap="large")
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < len(dataframe):
                row = dataframe.iloc[idx]
                reason = reason_map.get(row['Series_Title']) if reason_map else None
                with cols[c]: movie_card(row, watched_list, username, section, reason, show_button, signup_genres)

# ===== True Autocomplete Search =====
def search_and_render(df_tab, search_key, watched_list, username, section, show_button=True, reason_map=None, signup_genres=None):
    def search_func(q):
        if not q: return []
        q = q.lower()
        all_ops = df_tab["Series_Title"].tolist() + [g for lst in df_tab["Genre"].str.split(", ") for g in lst]
        return [v for v in all_ops if q in v.lower()]
    suggestion = st_searchbox(search_func, key=search_key, placeholder="Search by movie title or genre...")
    if suggestion:
        results = df_tab[
            df_tab["Series_Title"].str.lower().str.contains(suggestion.lower()) |
            df_tab["Genre"].str.lower().str.contains(suggestion.lower())
        ]
        if results.empty: st.warning("No results found")
        else: render_cards(results, watched_list, username, section, show_button, reason_map, signup_genres)
    else:
        render_cards(df_tab, watched_list, username, section, show_button, reason_map, signup_genres)

# ===== Pages =====
def login_signup_page():
    st.title("Movie Recommender – Login / Signup")
    opt = st.radio("Select option", ["Login", "Signup"], horizontal=True)
    username = st.text_input("Username"); password = st.text_input("Password", type="password")
    if opt=="Signup":
        if st.button("Signup") and username and password:
            if signup_user(username):
                st.session_state.username = username
                st.session_state.watched, st.session_state.genres, st.session_state.temp_selected_genres = [], [], []
                st.session_state.page = "genre_select"; st.rerun()
            else: st.error("Username exists")
    else:
        if st.button("Login") and username and password:
            user = load_user(username)
            if user:
                st.session_state.username = username
                st.session_state.watched = user.get("watched", [])
                st.session_state.genres = user.get("genres", [])
                if not st.session_state.genres: st.session_state.temp_selected_genres = []
                st.session_state.page = "dashboard" if st.session_state.genres else "genre_select"; st.rerun()
            else: st.error("User not found")

def genre_selection_page():
    st.title(f"Welcome, {st.session_state.username}!")
    st.subheader("Select Your Favourite Genres")
    all_genres = sorted(set(g for glist in df['Genre'].str.split(', ') for g in glist))
    if "temp_selected_genres" not in st.session_state:
        st.session_state.temp_selected_genres = []
    st.markdown("""
    <style>
    .genre-grid{display:flex;flex-wrap:wrap;gap:18px;margin-bottom:20px;}
    .genre-card{display:flex;flex-direction:column;align-items:center;justify-content:center;
        width:110px;height:110px;border-radius:16px;background:#f5f6fa;color:#292640;
        border:2.2px solid #dadbe7;font-size:1.01rem;cursor:pointer;transition:0.12s;
        box-shadow:0 2px 9px rgba(80,85,110,0.08);}
    .genre-card.selected{background:#1e4fff;color:#fff;border:2.6px solid #274ccf;
        box-shadow:0 2px 20px #5368ee25;}
    .genre-emoji{font-size:1.7em;margin-bottom:4px;}
    .genre-label{font-style:italic;font-size:1.06em;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="genre-grid">', unsafe_allow_html=True)
    for genre in all_genres:
        emoji = genre_emojis.get(genre.lower(),"🎞️")
        selected = genre in st.session_state.temp_selected_genres
        sel_class = "genre-card selected" if selected else "genre-card"
        if st.button(f"{emoji}\n{genre}", key=f"btn_{genre}"):
            if selected: st.session_state.temp_selected_genres.remove(genre)
            else: st.session_state.temp_selected_genres.append(genre)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("Next ➡️"):
        if st.session_state.temp_selected_genres:
            update_user_genres(st.session_state.username, st.session_state.temp_selected_genres)
            st.session_state.genres = st.session_state.temp_selected_genres.copy()
            st.session_state.page = "dashboard"; st.rerun()
        else: st.error("Please select at least 1 genre")

def dashboard_page():
    # Scroll-to-top JS injection to avoid page loading mid-tabs
    st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
    st.sidebar.checkbox("🌙 Dark Mode", key="dark_mode")
    st.write(f"### Welcome, {st.session_state.username}")
    if st.button("🚪 Logout"):
        st.session_state.page, st.session_state.username = "login_signup", ""
        st.session_state.genres, st.session_state.watched, st.session_state.temp_selected_genres = [], [], []
        st.rerun()
    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])
    with tab1:
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False)
        genre_set = set(g for lst in df['Genre'].str.split(', ') for g in lst)
        mixed_df = pd.concat([top_movies[top_movies['Genre'].str.contains(g, case=False)].head(3) for g in genre_set])\
                    .drop_duplicates(subset="Series_Title")
        mixed_df = mixed_df[~mixed_df['Series_Title'].isin(st.session_state.watched)].head(50)
        search_and_render(mixed_df, "search_top", st.session_state.watched, st.session_state.username, "top", True, signup_genres=st.session_state.genres)
    with tab2:
        watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
        search_and_render(watched_df, "search_watched", st.session_state.watched, st.session_state.username, "your", False, signup_genres=st.session_state.genres)
    with tab3:
        recs = recommend_for_user(st.session_state.genres, st.session_state.watched, top_n=10)
        reason_map = {}
        for idx, row in recs.iterrows():
            reasons = []
            watched_reasons = [w for w in st.session_state.watched if w in indices and cosine_sim[indices[w]][idx]>0.1]
            if watched_reasons: reasons.append("You watched " + ", ".join(watched_reasons[:3]))
            genre_matches = [g for g in st.session_state.genres if g.lower() in row["Genre"].lower()][:3]
            if genre_matches: reasons.append("You selected genre(s) " + ", ".join(genre_matches))
            reason_map[row['Series_Title']] = " and ".join(reasons) if reasons else None
        search_and_render(recs, "search_rec", st.session_state.watched, st.session_state.username, "rec", True, reason_map, signup_genres=st.session_state.genres)

# ===== Routing =====
if "page" not in st.session_state: st.session_state.page = "login_signup"
if "genres" not in st.session_state: st.session_state.genres = []
if "watched" not in st.session_state: st.session_state.watched = []
if "temp_selected_genres" not in st.session_state: st.session_state.temp_selected_genres = []
if st.session_state.page == "login_signup": login_signup_page()
elif st.session_state.page == "genre_select": genre_selection_page()
elif st.session_state.page == "dashboard": dashboard_page()
