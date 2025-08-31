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

# ===== Base CSS (keeps your original card/button styles) =====
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

# ===== Premium Theme function (uses .stApp + overlay + fallbacks) =====
def apply_premium_theme(choice: str):
    """
    Three premium themes:
     - 🎭 Cinematic Curtain (red theater curtain image + overlay)
     - 🌌 Galaxy Night (galaxy image + violet overlay)
     - 💡 Neon Glow (dark with neon radial gradients)
    If images fail to load the CSS also includes gradient fallbacks.
    """
    if choice == "🎭 Cinematic Curtain":
        # curtain image hosted on free image host (fallback gradient included)
        image_url = "https://images.unsplash.com/photo-1513104890138-7c749659a591?auto=format&fit=crop&w=2000&q=80"
        overlay = "linear-gradient(180deg, rgba(0,0,0,0.45), rgba(0,0,0,0.6))"
        chip_bg = "rgba(0,0,0,0.25)"
    elif choice == "🌌 Galaxy Night":
        image_url = "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?auto=format&fit=crop&w=2000&q=80"
        overlay = "linear-gradient(180deg, rgba(5,5,25,0.45), rgba(10,10,40,0.6))"
        chip_bg = "rgba(0,0,0,0.28)"
    else:  # "💡 Neon Glow"
        image_url = ""
        overlay = ""
        chip_bg = "rgba(0,0,0,0.28)"

    # Safe CSS: use image + gradient fallback. .stApp target works in Streamlit.
    if image_url:
        background_css = f"""
            background-image: {overlay}, url('{image_url}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        """
    else:
        # neon gradient fallback for no-image theme
        background_css = "background: radial-gradient(circle at 10% 20%, #ff2d95 0%, transparent 10%), radial-gradient(circle at 90% 80%, #04e5ff 0%, transparent 10%), #000000;"

    st.session_state["_chip_bg"] = chip_bg

    st.markdown(f"""
    <style>
    .stApp {{
        {background_css}
        color: #f0f0f0;
        transition: background 0.6s ease;
    }}
    /* subtle overlay to ensure cards/readability */
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.22);
        pointer-events: none;
    }}

    /* Entrance / micro animations */
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .card-animate {{
        animation: fadeUp 620ms cubic-bezier(.2,.9,.2,1) both;
    }}

    /* Pulse & bounce */
    @keyframes pulse {{
      0% {{ transform: scale(1); }} 50% {{ transform: scale(1.18); }} 100% {{ transform: scale(1); }}
    }}
    .pulse {{ animation: pulse 1.6s ease-in-out infinite; display:inline-block; }}

    @keyframes bounce {{
      0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-8px); }}
    }}
    .bounce {{ animation: bounce 1.1s ease-in-out infinite; display:inline-block; }}

    /* chip style override (uses session stored chip-bg) */
    .genre-chip-custom {{
      background: {chip_bg};
      color: #fff;
      padding: 6px 12px;
      border-radius: 16px;
      margin: 6px;
      display: inline-block;
      border: 1px solid rgba(255,255,255,0.08);
      font-weight: 700;
      backdrop-filter: blur(4px);
      -webkit-backdrop-filter: blur(4px);
    }}

    /* Make tabs pill-like */
    button[role="tab"] {{
      border-radius: 999px !important;
      padding: 8px 14px !important;
      font-weight: 700 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ===== Time-based greeting =====
def time_greeting(username):
    h = datetime.now().hour
    if h < 12:
        prefix = "Good Morning"
        banner = "linear-gradient(90deg,#ffd49a,#ffb347)"
    elif h < 18:
        prefix = "Good Afternoon"
        banner = "linear-gradient(90deg,#ff9a76,#ff5c4d)"
    else:
        prefix = "Good Evening"
        banner = "linear-gradient(90deg,#2b2d42,#4b6cb7)"
    return f"{prefix}, {username} 🍿 ready for a movie night?", banner

# ===== Reason formatter =====
def format_reason(reason: str) -> str:
    if not reason:
        return ""
    watched_movies, genres = [], []

    if "You watched" in reason:
        try:
            after = reason.split("You watched ", 1)[1]
            watched_text = after.split(" and ", 1)[0] if " and " in after else after
            watched_movies = [m.strip() for m in watched_text.split(",") if m.strip()]
        except Exception:
            pass

    if "You selected genre(s)" in reason:
        try:
            genre_text = reason.split("You selected genre(s) ", 1)[1]
            genres = [g.strip() for g in genre_text.split(",") if g.strip()]
        except Exception:
            pass

    if not watched_movies and not genres:
        return ""

    parts = ["<div style='margin-top:8px;color:#399ed7;font-size:0.9rem;'>💡 Why recommended:<br>"]
    if watched_movies:
        parts.append("🎬 You Watched :<br>")
        for m in watched_movies:
            parts.append(f"&nbsp;&nbsp;&nbsp;&nbsp;• {m}<br>")
    if genres:
        parts.append("🎯 Matches your genre(s):<br>")
        for g in genres:
            parts.append(f"&nbsp;&nbsp;&nbsp;&nbsp;• {g}<br>")
    parts.append("</div>")
    return "".join(parts)

# ===== Card Renderer with Details =====
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

    reason_html = format_reason(reason) if reason else ""

    html = textwrap.dedent(f"""<div class="movie-card card-animate" style="border:1.5px solid {border_color};
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
    <span class='pulse'>⭐</span> {row["IMDB_Rating"]:.1f}/10
  </div>
  {reason_html}
</div>

</div>""")
    st.markdown(html, unsafe_allow_html=True)

    _details = df.loc[df['Series_Title'] == row['Series_Title']]
    if not _details.empty:
        d = _details.iloc[0]
        overview = d['Overview'] if pd.notna(d.get('Overview')) else "Not available."
        runtime = str(d['Runtime']).strip() if pd.notna(d.get('Runtime')) else "N/A"
        if runtime != "N/A" and "min" not in runtime.lower():
            runtime = runtime + " min"
        stars = [d.get('Star1'), d.get('Star2'), d.get('Star3'), d.get('Star4')]
        stars = [s for s in stars if pd.notna(s) and str(s).strip()]
    else:
        overview, runtime, stars = "Not available.", "N/A", []

    with st.expander("🔎 View details", expanded=False):
        st.markdown(f"**Overview:** {overview}")
        st.markdown(f"**Runtime:** {runtime}")
        if stars:
            st.markdown("**Stars:**")
            for s in stars:
                st.markdown(f"- {s}")

    if show_button:
        key = f"watched_{section}_{row.name}"
        if row['Series_Title'] not in watched_list:
            if st.button("▶ Watch Now", key=key, type="primary", use_container_width=True):
                watched_list.append(row['Series_Title'])
                update_watched(username, watched_list)
                st.rerun()
        else:
            st.button("▶ Watch Now", key=key, disabled=True, use_container_width=True)

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
                if not st.session_state.genres:
                    st.session_state.temp_selected_genres = []
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

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("Next ➡️", type="primary", use_container_width=True):
            if st.session_state.temp_selected_genres:
                update_user_genres(st.session_state.username, st.session_state.temp_selected_genres)
                st.session_state.genres = st.session_state.temp_selected_genres.copy()
                st.session_state.scroll_to_top = True
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("Please select at least one genre to continue.")

# ===== Search helpers =====
def search_top_movies(searchterm: str):
    if not searchterm:
        return df.sort_values(by="IMDB_Rating", ascending=False)["Series_Title"].head(10).tolist()
    results = df[df["Series_Title"].str.lower().str.contains(str(searchterm).lower()) |
                 df["Genre"].str.lower().str.contains(str(searchterm).lower())]
    return results["Series_Title"].head(10).tolist()

def search_watched_movies(searchterm: str):
    watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
    if not searchterm:
        return watched_df["Series_Title"].tolist()
    results = watched_df[watched_df["Series_Title"].str.lower().str.contains(str(searchterm).lower()) |
                         watched_df["Genre"].str.lower().str.contains(str(searchterm).lower())]
    return results["Series_Title"].head(10).tolist()

def search_recommended_movies(searchterm: str):
    recs = recommend_for_user(st.session_state.genres, st.session_state.watched, 10)
    if not searchterm:
        return recs["Series_Title"].tolist()
    results = recs[recs["Series_Title"].str.lower().str.contains(str(searchterm).lower()) |
                   recs["Genre"].str.lower().str.contains(str(searchterm).lower())]
    return results["Series_Title"].head(10).tolist()

# ===== Dashboard Page =====
def dashboard_page():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if st.session_state.get("scroll_to_top", False):
        st.markdown("<script>window.scrollTo({top: 0, behavior: 'instant'});</script>", unsafe_allow_html=True)
        st.session_state.scroll_to_top = False

    # New premium theme picker (radio makes preview/apply immediate)
    theme_choice = st.sidebar.radio("🎨 Theme", ["🎭 Cinematic Curtain", "🌌 Galaxy Night", "💡 Neon Glow"])
    apply_premium_theme(theme_choice)

    if st.sidebar.button("🌙 Toggle Card Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
    if st.sidebar.button("🚪 Logout"):
        st.session_state.page = "login_signup"
        st.session_state.username = ""
        st.session_state.genres = []
        st.session_state.watched = []
        st.session_state.temp_selected_genres = []
        st.rerun()

    # Personalized header (time based) with subtle banner
    greeting, banner_bg = time_greeting(st.session_state.username or "Movie Buff")
    st.markdown(f"""
    <div style="background:{banner_bg};
                padding:22px;border-radius:14px;text-align:center;color:white;
                box-shadow:0 6px 18px rgba(0,0,0,0.25);">
        <h1 style="margin:0;font-weight:900;letter-spacing:.3px;">{greeting}</h1>
        <div style="margin-top:8px;font-size:20px;opacity:.95;">
            <span class="bounce">🍿</span> <span style="margin-left:10px;" class="pulse">⭐</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Favorite genres as chips (uses chip bg from theme function)
    if st.session_state.genres:
        chip_bg = st.session_state.get("_chip_bg", "rgba(255,255,255,0.12)")
        chips_html = " ".join([
            f"<span class='genre-chip-custom'>{g}</span>"
            for g in st.session_state.genres
        ])
        st.markdown(f"<div style='text-align:center;margin-top:12px;'>{chips_html}</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])

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
