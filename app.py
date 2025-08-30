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

def signup_user(username, profile):  # <-- changed: capture profile at signup
    data = load_user_data()
    if username in data: return False
    # store profile once, as requested
    data[username] = {"genres": [], "watched": [], "profile": profile}
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

# ===== Certificate utils (normalize + allowed-by-profile) =====
def normalize_certificate(cert):
    """Map various certificates to a common set."""
    if pd.isna(cert):
        return "UA 13+"
    c = str(cert).strip().upper()

    # Common Indian & MPAA / TV mappings
    if c in {"U"}: return "U"
    if c in {"UA", "U/A"}: return "UA 13+"
    if "UA" in c and "7" in c: return "UA 7+"
    if "UA" in c and "13" in c: return "UA 13+"
    if c in {"A", "ADULTS ONLY", "18", "18+"}: return "A"

    if c in {"G"}: return "G"
    if c in {"PG"}: return "PG"
    if c in {"PG-13", "PG13"}: return "PG-13"
    if c in {"R"}: return "R"
    if c in {"NC-17", "NC17"}: return "NC-17"

    # TV ratings → approximate
    if c in {"TV-Y", "TV-Y7"}: return "U"
    if c in {"TV-PG"}: return "PG"
    if c in {"TV-14"}: return "PG-13"
    if c in {"TV-MA"}: return "A"

    return "UA 13+"

def get_allowed_certificates(profile_type):
    """Profile → allowed certificates."""
    if profile_type == "Kids":
        return {"U", "UA 7+", "G"}
    elif profile_type == "Teen":
        return {"U", "UA 7+", "UA 13+", "PG-13", "PG", "G"}
    else:  # Adult
        return {"U", "UA 7+", "UA 13+", "PG-13", "PG", "G", "R", "A", "NC-17"}

def get_allowed_in_session():
    """Get allowed certs for the current logged-in user's profile."""
    profile = st.session_state.get("profile")
    if not profile and st.session_state.get("username"):
        user = load_user(st.session_state["username"])
        profile = (user or {}).get("profile", "Teen")
        st.session_state.profile = profile
    if not profile:
        profile = "Teen"  # safe default
    return get_allowed_certificates(profile)

# ===== Load Model/Data =====
@st.cache_resource
def load_model():
    df = joblib.load("movies_df.pkl")
    cosine_sim = joblib.load("cosine_similarity.pkl")
    indices = joblib.load("title_indices.pkl")
    # add normalized certificate column once
    if "CertNorm" not in df.columns:
        df["CertNorm"] = df["Certificate"].apply(normalize_certificate)
    return df, cosine_sim, indices

df, cosine_sim, indices = load_model()

# ===== Recommendation Logic =====
def recommend_for_user(preferred_genres, watched_titles, top_n=10, allowed_certs=None):
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

    # filter by profile certificates (if provided)
    if allowed_certs is not None:
        rec_df = rec_df[rec_df["CertNorm"].isin(allowed_certs)]

    if preferred_genres:
        signup_df = rec_df[rec_df['Genre'].str.contains('|'.join(preferred_genres), case=False, na=False)]
    else:
        signup_df = rec_df.head(0)

    rec_df = pd.concat([signup_df.head(3), rec_df]).drop_duplicates()
    return rec_df.head(top_n)[['Series_Title','Genre','IMDB_Rating','Certificate','Released_Year']]

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
/* Make primary buttons modern (used for ▶ Watch Now + Next) */
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
  {reason_html}
</div>

</div>""")
    st.markdown(html, unsafe_allow_html=True)

    # --- Expander: Show Overview, Runtime, Stars ---
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
        # Stylish ▶ Watch Now button (primary + full-width) that still marks as watched
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
        # profile selection ONLY at signup
        profile = st.selectbox("Choose Profile Type (saved with your account)", ["Kids", "Teen", "Adult"])
        if st.button("Signup", type="primary") and username and password:
            if len(password) < 5:
                st.error("Password must be at least 5 characters long.")
            else:
                if signup_user(username, profile):
                    st.session_state.username = username
                    st.session_state.profile = profile
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
                st.session_state.profile = user.get("profile", "Teen")
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

    # Centered & styled Next button
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
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
    allowed = get_allowed_in_session()
    df_allowed = df[df["CertNorm"].isin(allowed)]
    if not searchterm:
        return df_allowed.sort_values(by="IMDB_Rating", ascending=False)["Series_Title"].head(10).tolist()
    results = df_allowed[df_allowed["Series_Title"].str.lower().str.contains(str(searchterm).lower()) |
                         df_allowed["Genre"].str.lower().str.contains(str(searchterm).lower())]
    return results["Series_Title"].head(10).tolist()

def search_watched_movies(searchterm: str):
    watched_df = df[df['Series_Title'].isin(st.session_state.watched)]
    if not searchterm:
        return watched_df["Series_Title"].tolist()
    results = watched_df[watched_df["Series_Title"].str.lower().str.contains(str(searchterm).lower()) |
                         watched_df["Genre"].str.lower().str.contains(str(searchterm).lower())]
    return results["Series_Title"].head(10).tolist()

def search_recommended_movies(searchterm: str):
    allowed = get_allowed_in_session()
    recs = recommend_for_user(st.session_state.genres, st.session_state.watched, 10, allowed_certs=allowed)
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
    if st.sidebar.button("🌙 Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
    if st.sidebar.button("🚪 Logout"):
        st.session_state.page = "login_signup"
        st.session_state.username = ""
        st.session_state.genres = []
        st.session_state.watched = []
        st.session_state.temp_selected_genres = []
        st.session_state.profile = None
        st.rerun()
    st.markdown(f"""
    <div style="
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 36px;
        font-weight: 600;
        color: #e74c3c;
        margin-bottom: 25px;
        padding-top: 15px;
        letter-spacing: 1px;
    ">
        Welcome, {st.session_state.username}!
    </div>
    """, unsafe_allow_html=True)

    allowed = get_allowed_in_session()

    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎥 Your Watching", "🎯 Recommendations"])

    with tab1:
        # Top movies (filtered by profile)
        top_movies = df[df["CertNorm"].isin(allowed)].sort_values(by="IMDB_Rating", ascending=False)
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
        recs = recommend_for_user(st.session_state.genres, st.session_state.watched, 10, allowed_certs=allowed)
        reason_map = {}
        for idx, row in recs.iterrows():
            reasons = []
            watched_reasons = [
                w for w in st.session_state.watched
                if w in indices and cosine_sim[indices[w]][idx] > 0.1
            ]
            if watched_reasons:
                reasons.append("You watched " + ", ".join(watched_reasons[:3]))
            genre_matches = [g for g in st.session_state.genres if g.lower() in str(row["Genre"]).lower()][:3]
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
