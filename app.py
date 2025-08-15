# ===========================
# CHANGES MADE:
# 1. Card overflow fix: used min-height and word wrap
# 2. Improved "reason" formatting in Recommendations tab
# ===========================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json, os, textwrap
from math import ceil
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
    return pd.concat([signup_df.head(3), rec_df]).drop_duplicates().head(top_n)[['Series_Title','Genre','IMDB_Rating','Certificate','Released_Year']]

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

# ===== Inject CSS =====
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
}
.movie-card:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    position: relative;
    z-index: 10;
}
</style>
""", unsafe_allow_html=True)

# ===== Card Renderer =====
def movie_card(row, watched_list, username, section, reason=None, show_button=True, signup_genres=None):
    dark = st.session_state.get('dark_mode', False)
    bg_color = "#23272e" if dark else "#fdfdfe"
    text_color = "#f5f5f5" if dark else "#222"
    border_color = "#3d434d" if dark else "#e2e3e6"
    genre_color = "#b2b2b2" if dark else "#5A5A5A"
    rating_color = "#fcb900"
    
    emoji, genre_text = get_dominant_genre_with_emoji(row["Genre"], signup_genres)
    
    cert_value = row["Certificate"] if pd.notna(row["Certificate"]) and str(row["Certificate"]).strip() else "UA"
    cert_value = cert_value.strip()
    cert_colors = {"U": "#27ae60", "UA": "#f39c12", "A": "#c0392b"}
    cert_color = cert_colors.get(cert_value.upper(), "#7f8c8d")

    html = textwrap.dedent(f"""<div class="movie-card" style="border:1.5px solid {border_color};
border-radius:10px;padding:12px;background:{bg_color};color:{text_color};
box-shadow:0 2px 6px rgba(0,0,0,0.08);min-height:180px;height:auto;
display:flex;flex-direction:column;justify-content:space-between;
overflow-wrap:break-word;word-break:break-word;white-space:normal;">

<div>
  <div style="font-weight:700;font-size:1.1rem;line-height:1.3;">
    {row["Series_Title"]} ({row["Released_Year"]})
  </div>
  <div style="display:inline-block;background:{cert_color};color:#fff;padding:4px 10px;border-radius:6px;
              font-size:0.85rem;font-weight:bold;min-width:38px;text-align:center;margin-top:6px;">
    {cert_value}
  </div>
</div>

<div>
  <div style="color:{genre_color};margin-top:6px;">
    {emoji} <span style="font-style: italic;">{genre_text}</span>
  </div>
  <div style="color:{rating_color};margin-top:6px;">
    ⭐ {row["IMDB_Rating"]:.1f}/10
  </div>
  {f'<div style="color:#399ed7;margin-top:6px;overflow-wrap:break-word;word-break:break-word;white-space:normal;">{reason}</div>' if reason else ''}
</div>

</div>""")

    st.markdown(html, unsafe_allow_html=True)

    if show_button:
        key = f"watched_{section}_{row.name}"
        if row['Series_Title'] not in watched_list:
            if st.button("✅ Watched", key=key):
                watched_list.append(row['Series_Title'])
                update_watched(username, watched_list)
                st.rerun()
        else:
            st.button("✅ Watched", key=key, disabled=True)

# ===== Render Cards Grid =====
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

# ===== Dashboard Page (recommendations formatting updated) =====
def dashboard_page():
    # ... unchanged for tabs 1 & 2 ...

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
                formatted_watched = "💡 You watched:<br>" + "<br>".join([f"• {w}" for w in watched_reasons[:3]])
                reasons.append(formatted_watched)
            genre_matches = [g for g in st.session_state.genres if g.lower() in row["Genre"].lower()][:3]
            if genre_matches:
                formatted_genres = "💡 You selected genre(s): " + ", ".join(genre_matches)
                reasons.append(formatted_genres)
            if reasons:
                reasons.append("📌 So we recommend this!")
            reason_map[row['Series_Title']] = "<br><br>".join(reasons) if reasons else None
        selected_title = st_searchbox(search_recommended_movies, placeholder="Search recommended movies...", key="rec_searchbox")
        if selected_title:
            recs = recs[recs['Series_Title'] == selected_title]
        render_cards(recs, st.session_state.watched, st.session_state.username, "rec", True, reason_map, signup_genres=st.session_state.genres)
