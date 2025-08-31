import streamlit as st
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
