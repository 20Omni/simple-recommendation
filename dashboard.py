import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import numpy as np
from math import ceil

# ===== Session defaults =====
for k, v in {
    "page": "login_signup",
    "genres": [],
    "watched": [],
    "temp_selected_genres": [],
    "username": "",
    "dark_mode": False,
    "_scroll_once": False,
    "_user_data": {}  # store all user data here
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===== User Data Functions =====
def signup_user(username):
    if username in st.session_state._user_data:
        return False
    st.session_state._user_data[username] = {"genres": [], "watched": []}
    return True

def load_user(username):
    return st.session_state._user_data.get(username)

def update_user_genres(username, genres):
    if username in st.session_state._user_data:
        st.session_state._user_data[username]["genres"] = genres

def update_watched(username, watched_list):
    if username in st.session_state._user_data:
        st.session_state._user_data[username]["watched"] = watched_list

# ===== Load Data =====
@st.cache_resource
def load_model():
    df = joblib.load("movies_df.pkl")
    cosine_sim = joblib.load("cosine_similarity.pkl")
    indices = joblib.load("title_indices.pkl")
    return df, cosine_sim, indices

df, cosine_sim, indices = load_model()

# ===== Recommendation =====
def recommend_for_user(preferred_genres, watched_titles, top_n=10):
    scores = np.zeros(len(df))
    if len(watched_titles) >= 3:
        genre_weight, watch_weight = 0.3, 4.0
    elif watched_titles:
        genre_weight, watch_weight = 0.5, 3.5
    else:
        genre_weight, watch_weight = 2.0, 0.0

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
    if watched_idx:
        scores[np.array(watched_idx)] = -1

    rec_df = df.iloc[np.argsort(scores)[::-1]]
    rec_df = rec_df[~rec_df['Series_Title'].isin(watched_titles)]
    if preferred_genres:
        preferred_mask = rec_df['Genre'].str.contains('|'.join(map(repr, preferred_genres)).replace("'", ""), case=False, na=False)
        signup_df = rec_df[preferred_mask]
    else:
        signup_df = rec_df

    result = pd.concat([signup_df.head(3), rec_df]).drop_duplicates('Series_Title').head(top_n)
    return result[['Series_Title', 'Genre', 'IMDB_Rating']]

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
                    return genre_emojis.get(g.lower(), "🎞️"), genre_string
    for g in genres_list:
        if g.lower() in genre_emojis:
            return genre_emojis[g.lower()], genre_string
    return "🎞️", genre_string

# ===== Movie Card =====
def movie_card(row, watched_list, username, section, reason=None, show_button=True, signup_genres=None):
    dark = st.session_state.dark_mode
    bg = "#23272e" if dark else "#fdfdfe"
    text = "#f5f5f5" if dark else "#222"
    border = "#3d434d" if dark else "#e2e3e6"
    genre_color = "#b2b2b2" if dark else "#5A5A5A"
    rating_c = "#fcb900"
    emoji, genre_text = get_dominant_genre_with_emoji(row["Genre"], signup_genres)
    html = f'''
    <div style="border:1.5px solid {border};border-radius:16px;padding:16px;margin-bottom:22px;background:{bg};color:{text};">
        <div style="font-size:1.15rem;font-weight:700;margin-bottom:4px;">{row["Series_Title"]}</div>
        <div style="font-size:0.9rem;color:{genre_color};margin-bottom:6px;">
            <span>{emoji}</span> <span style="font-style: italic;">{genre_text}</span>
        </div>
        <div style="font-size:1.2rem;color:{rating_c};margin-bottom:6px;">⭐ {float(row["IMDB_Rating"]):.1f}/10</div>
        {f'<div style="font-size:0.9rem;color:#399ed7;margin-bottom:8px;">💡 {reason}</div>' if reason else ""}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)
    if show_button:
        key = f"watched_btn_{section}_{row.name}"
        if row['Series_Title'] not in watched_list:
            if st.button("Watched", key=key):
                watched_list.append(row['Series_Title'])
                update_watched(username, watched_list)
                st.experimental_rerun()
        else:
            st.button("Watched", key=key, disabled=True)

# ===== Render Cards =====
def render_cards(dfdata, watched_list, username, section, show_button=True, reason_map=None, signup_genres=None):
    if dfdata is None or len(dfdata) == 0:
        st.warning("No results found")
        return
    cols_per_row = 3
    for r in range(ceil(len(dfdata) / cols_per_row)):
        cols = st.columns(cols_per_row, gap="large")
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < len(dfdata):
                row = dfdata.iloc[idx]
                reason = reason_map.get(row['Series_Title']) if reason_map else None
                with cols[c]:
                    movie_card(row, watched_list, username, section, reason, show_button, signup_genres)

# ===== Search and Render (Fixed) =====
def search_and_render(df_tab, search_key, watched_list, username, section,
                      show_button=True, reason_map=None, signup_genres=None):
    input_key = f"{search_key}_q"
    if input_key not in st.session_state:
        st.session_state[input_key] = ""

    query = st.text_input("🔍 Search", key=input_key, placeholder="Type to search...").strip().lower()

    filtered_df = df_tab[
        df_tab["Series_Title"].str.lower().str.contains(query, na=False) |
        df_tab["Genre"].str.lower().str.contains(query, na=False)
    ] if query else df_tab.copy()

    # Show suggestions
    if query:
        suggestions = filtered_df["Series_Title"].head(5).tolist()
        if suggestions:
            st.markdown("**Suggestions:**")
            cols = st.columns(len(suggestions))
            for i, title in enumerate(suggestions):
                if cols[i].button(title, key=f"sugg_{search_key}_{i}"):
                    st.session_state[input_key] = title
        elif filtered_df.empty:
            st.warning("No results found")
            return

    # Render cards
    if filtered_df.empty:
        st.warning("No results found")
    else:
        render_cards(filtered_df, watched_list, username, section, show_button, reason_map, signup_genres)

# ===== Pages =====
def login_signup_page():
    st.title("Movie Recommender – Login / Signup")
    opt = st.radio("Select option", ["Login", "Signup"], horizontal=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if opt == "Signup":
        if st.button("Signup") and username and password:
            if signup_user(username):
                st.session_state.username = username
                st.session_state.watched = []
                st.session_state.genres = []
                st.session_state.temp_selected_genres = []
                st.session_state.page = "genre_select"
                st.experimental_rerun()
            else:
                st.error("Username exists")
    else:
        if st.button("Login") and username and password:
            user = load_user(username)
            if user:
                st.session_state.username = username
                st.session_state.watched = user.get("watched", [])
                st.session_state.genres = user.get("genres", [])
                st.session_state.page = "dashboard" if st.session_state.genres else "genre_select"
                st.session_state._scroll_once = True
                st.rerun()
            else:
                st.error("User not found")

def genre_selection_page():
    st.title(f"Welcome, {st.session_state.username}!")
    st.subheader("Select Your Favourite Genres")
    all_genres = sorted(set(g for glist in df['Genre'].str.split(', ') for g in glist))
    for genre in all_genres:
        emoji = genre_emojis.get(genre.lower(), "🎞️")
        label = f"{'✅ ' if genre in st.session_state.temp_selected_genres else ''}{emoji} {genre}"
        if st.button(label, key=f"btn_{genre}"):
            if genre in st.session_state.temp_selected_genres:
                st.session_state.temp_selected_genres.remove(genre)
            else:
                st.session_state.temp_selected_genres.append(genre)
            st.experimental_rerun()
    if st.button("Next ➡️"):
        if st.session_state.temp_selected_genres:
            update_user_genres(st.session_state.username, st.session_state.temp_selected_genres)
            st.session_state.genres = st.session_state.temp_selected_genres.copy()
            st.session_state.page = "dashboard"
            st.session_state._scroll_once = True
            st.rerun()
        else:
            st.error("Please select at least 1 genre")

def dashboard_page():
    if st.session_state._scroll_once:
        # Scroll to top reliably
        components.html("<script>window.scrollTo(0,0)</script>", height=0)
        st.session_state._scroll_once = False

    st.sidebar.checkbox("🌙 Dark Mode", key="dark_mode")
    st.write(f"### Welcome, {st.session_state.username}")

    if st.button("🚪 Logout"):
        st.session_state.page = "login_signup"
        st.session_state.username = ""
        st.session_state.genres = []
        st.session_state.watched = []
        st.session_state.temp_selected_genres = []
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])

    with tab1:
        top_movies = df.sort_values(by="IMDB_Rating", ascending=False)
        genre_set = set(g for lst in df['Genre'].str.split(', ') for g in lst)
        mixed_df = pd.concat(
            [top_movies[top_movies['Genre'].str.contains(g, case=False, na=False)].head(3) for g in genre_set]
        ).drop_duplicates("Series_Title")
        mixed_df = mixed_df[~mixed_df['Series_Title'].isin(st.session_state.watched)].head(50)
        search_and_render(mixed_df, "search_top", st.session_state.watched, st.session_state.username, "top", True, signup_genres=st.session_state.genres)

    with tab2:
        watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
        search_and_render(watched_df, "search_watched", st.session_state.watched, st.session_state.username, "your", False, signup_genres=st.session_state.genres)

    with tab3:
        recs = recommend_for_user(st.session_state.genres, st.session_state.watched, 10)
        reason_map = {}
        for idx, row in recs.iterrows():
            reasons = []
            watched_reasons = [w for w in st.session_state.watched if w in indices]
            if watched_reasons:
                reasons.append("You watched " + ", ".join(watched_reasons[:3]))
            genre_matches = [g for g in st.session_state.genres if g.lower() in str(row["Genre"]).lower()][:3]
            if genre_matches:
                reasons.append("You selected genre(s) " + ", ".join(genre_matches))
            reason_map[row['Series_Title']] = " and ".join(reasons) if reasons else None
        search_and_render(recs, "search_rec", st.session_state.watched, st.session_state.username, "rec", True, reason_map, signup_genres=st.session_state.genres)

# ===== Routing =====
if st.session_state.page == "login_signup":
    login_signup_page()
elif st.session_state.page == "genre_select":
    genre_selection_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
