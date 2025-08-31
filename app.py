import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json, os, textwrap
from math import ceil
from datetime import datetime
from streamlit_searchbox import st_searchbox

USER_DATA_FILE = "user_data.json"

# ===== Load FontAwesome (reliable icons instead of emoji boxes) =====
st.markdown(
    "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'>",
    unsafe_allow_html=True,
)

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

    if watched_idx:
        scores[watched_idx] = -99999

    rec_df = df.iloc[np.argsort(scores)[::-1]]
    rec_df = rec_df[~rec_df['Series_Title'].isin(watched_titles)]

    if preferred_genres:
        signup_df = rec_df[rec_df['Genre'].str.contains('|'.join(preferred_genres), case=False, na=False)]
    else:
        signup_df = rec_df.head(0)

    rec_df = pd.concat([signup_df.head(3), rec_df]).drop_duplicates()
    return rec_df.head(top_n)[['Series_Title','Genre','IMDB_Rating','Certificate','Released_Year']]

# ===== Emoji / Icon mapping =====
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

# ===== Base CSS (cards/buttons) =====
st.markdown("""
<style>
/* base card/button styles */
.movie-card {
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    cursor: pointer;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 14px;
    position: relative;
    margin-bottom: 20px;
    opacity: 1;
    border-radius: 12px;
    overflow: hidden;
}
.movie-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 18px 40px rgba(0,0,0,0.35);
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

/* small classes for animations */
.pulse { animation: pulse 1.6s ease-in-out infinite; display:inline-block; }
.bounce { animation: bounce 1.1s ease-in-out infinite; display:inline-block; }

@keyframes pulse { 0% { transform: scale(1);} 50% { transform: scale(1.16);} 100% { transform: scale(1);} }
@keyframes bounce { 0%,100% { transform: translateY(0);} 50% { transform: translateY(-8px);} }
</style>
""", unsafe_allow_html=True)

# ===== Theme definitions & apply function =====
THEMES = {
    "Cinema Classic 🎬": {
        "image": "https://images.unsplash.com/photo-1542204165-5ea6adf4c5e9?auto=format&fit=crop&w=2000&q=80",
        "overlay": "linear-gradient(180deg, rgba(0,0,0,0.55), rgba(0,0,0,0.65))",
        "chip": "rgba(0,0,0,0.35)",
        "card_border": "#ffd700",
        "card_bg": "rgba(255,255,255,0.02)",
        "text_color": "#fff",
        "header_bg": "linear-gradient(90deg,#7f0000,#ff4b4b)"
    },
    "Neon Night 🌌": {
        "image": "",
        "overlay": "linear-gradient(180deg, rgba(5,5,20,0.4), rgba(10,10,40,0.6))",
        "chip": "rgba(0,0,0,0.25)",
        "card_border": "#00f0ff",
        "card_bg": "rgba(6,6,20,0.7)",
        "text_color": "#e8f9ff",
        "header_bg": "linear-gradient(90deg,#7b2ff7,#2af598)"
    },
    "Popcorn Fun 🍿": {
        "image": "https://images.unsplash.com/photo-1544025162-d76694265947?auto=format&fit=crop&w=2000&q=80",
        "overlay": "linear-gradient(180deg, rgba(255,255,255,0.06), rgba(0,0,0,0.06))",
        "chip": "rgba(255,255,255,0.9)",
        "card_border": "#ffb347",
        "card_bg": "rgba(255,255,255,0.96)",
        "text_color": "#111",
        "header_bg": "linear-gradient(90deg,#ffd27a,#ffb347)"
    },
    "Minimal Dark 🕶️": {
        "image": "",
        "overlay": "linear-gradient(180deg, rgba(0,0,0,0.6), rgba(0,0,0,0.8))",
        "chip": "rgba(255,255,255,0.06)",
        "card_border": "#444444",
        "card_bg": "rgba(20,20,20,0.88)",
        "text_color": "#eaeaea",
        "header_bg": "linear-gradient(90deg,#0f0f0f,#222222)"
    }
}

def apply_theme(choice: str):
    theme = THEMES.get(choice, THEMES["Minimal Dark 🕶️"])
    st.session_state["_chip_bg"] = theme["chip"]
    st.session_state["_card_border"] = theme["card_border"]
    st.session_state["_card_bg"] = theme["card_bg"]
    st.session_state["_card_text"] = theme["text_color"]
    st.session_state["_header_bg"] = theme["header_bg"]

    if theme["image"]:
        bg_css = f"background-image: {theme['overlay']}, url('{theme['image']}'); background-size: cover; background-position: center; background-attachment: fixed;"
    else:
        bg_css = f"background: {theme['overlay']};"

    st.markdown(f"""
    <style>
    .stApp {{
        {bg_css}
        color: {theme['text_color']};
        transition: background 0.6s ease;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.12);
        pointer-events: none;
    }}

    .header-box {{
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        color: {theme['text_color']};
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}

    .genre-chip-custom {{
      background: {theme['chip']};
      color: { '#111' if choice == 'Popcorn Fun 🍿' else '#fff' };
      padding: 8px 14px;
      border-radius: 20px;
      margin: 6px;
      display: inline-block;
      font-weight: 700;
      border: 1px solid rgba(255,255,255,0.06);
      backdrop-filter: blur(4px);
    }}

    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .card-animate {{ animation: fadeUp 560ms cubic-bezier(.2,.9,.2,1) both; }}

    @keyframes pulse {{
      0% {{ transform: scale(1); }} 50% {{ transform: scale(1.16); }} 100% {{ transform: scale(1); }}
    }}
    .pulse {{ animation: pulse 1.6s ease-in-out infinite; display:inline-block; }}

    @keyframes bounce {{
      0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-8px); }}
    }}
    .bounce {{ animation: bounce 1.1s ease-in-out infinite; display:inline-block; }}

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
    elif h < 18:
        prefix = "Good Afternoon"
    else:
        prefix = "Good Evening"
    return f"{prefix}, {username} 🍿 ready for a movie night?"

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

# ===== Card Renderer with Details (fixed to avoid Markdown code-block) =====
def movie_card(row, watched_list, username, section, reason=None, show_button=True, signup_genres=None):
    # pick colors from session (applied theme)
    card_bg = st.session_state.get("_card_bg", "#fdfdfe")
    text_color = st.session_state.get("_card_text", "#222")
    border_color = st.session_state.get("_card_border", "#e2e3e6")
    genre_color = "#b2b2b2" if st.session_state.get("_card_text", "#222") == "#fff" else "#666"
    rating_color = "#fcb900"

    emoji, genre_text = get_dominant_genre_with_emoji(row["Genre"], signup_genres)

    cert_value = row["Certificate"] if pd.notna(row["Certificate"]) and str(row["Certificate"]).strip() else "UA"
    cert_value = cert_value.strip()
    cert_colors = {"U": "#27ae60", "UA": "#f39c12", "A": "#c0392b"}
    cert_color = cert_colors.get(cert_value.upper(), "#7f8c8d")

    reason_html = format_reason(reason) if reason else ""

    # build HTML and IMPORTANT: strip leading whitespace so Streamlit doesn't treat it as a code block
    html = textwrap.dedent(f"""\
    <div class="movie-card card-animate" style="border:1.8px solid {border_color};
    border-radius:12px;padding:14px;background:{card_bg};color:{text_color};
    box-shadow:0 6px 18px rgba(0,0,0,0.12);min-height:160px;height:auto;
    display:flex;flex-direction:column;justify-content:space-between;
    overflow-wrap:break-word;word-break:break-word;white-space:normal;">
      <div>
        <div style="font-weight:800;font-size:1.05rem;line-height:1.25;color:{text_color};">
          {row["Series_Title"]} <span style="color:rgba(255,255,255,0.5);font-weight:600;">({row["Released_Year"]})</span>
        </div>
        <div style="display:inline-block;background:{cert_color};color:#fff;padding:5px 10px;border-radius:8px;
                    font-size:0.85rem;font-weight:700;min-width:44px;text-align:center;margin-top:8px;">
          {cert_value}
        </div>
      </div>

      <div>
        <div style="color:{genre_color};margin-top:10px;">
          <!-- Using FontAwesome film icon + fallback emoji -->
          <i class="fa-solid fa-film" aria-hidden="true" style="margin-right:6px;"></i>
          <span style="font-style: italic;">{genre_text}</span>
        </div>
        <div style="color:{rating_color};margin-top:8px;font-weight:700;">
          <span class='pulse'><i class="fa-solid fa-star"></i></span> {row["IMDB_Rating"]:.1f}/10
        </div>
        {reason_html}
      </div>
    </div>
    """).strip()  # <-- strip leading/trailing whitespace to prevent Markdown turning it into a code block

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

    # Theme picker (4 polished themes)
    theme_choice = st.sidebar.radio("🎨 Theme", list(THEMES.keys()))
    apply_theme(theme_choice)
    st.session_state['theme_choice'] = theme_choice

    if st.sidebar.button("🌙 Toggle Card Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
    if st.sidebar.button("🚪 Logout"):
        st.session_state.page = "login_signup"
        st.session_state.username = ""
        st.session_state.genres = []
        st.session_state.watched = []
        st.session_state.temp_selected_genres = []
        st.rerun()

    # personalized header (glass + time-based greeting)
    greeting = time_greeting(st.session_state.username or "Movie Buff")
    header_bg = st.session_state.get("_header_bg", "linear-gradient(90deg,#2b2d42,#4b6cb7)")
    st.markdown(f"""
    <div class="header-box" style="background:{header_bg}; margin-bottom:14px;">
      <h1 style="margin:0;font-weight:900;letter-spacing:.3px;font-size:34px;">{greeting}</h1>
      <div style="margin-top:8px;font-size:20px;">
        <span class="bounce"><i class="fa-solid fa-popcorn"></i></span> &nbsp; <span class="pulse"><i class="fa-solid fa-star"></i></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Favorite genres as chips
    if st.session_state.genres:
        chips_html = " ".join([f"<span class='genre-chip-custom'>{g}</span>" for g in st.session_state.genres])
        st.markdown(f"<div style='text-align:center;margin-top:8px;margin-bottom:18px;'>{chips_html}</div>", unsafe_allow_html=True)

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
