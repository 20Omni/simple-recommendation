import streamlit as st
import pandas as pd

# ---- Sample DF for testing search ---
df = pd.DataFrame({
    "Series_Title": [
        "The Matrix", "Inception", "Interstellar", "The Dark Knight",
        "Avengers: Endgame", "Gladiator", "The Prestige", "Titanic",
        "Jurassic Park", "Avatar"
    ],
    "Genre": [
        "Sci-Fi, Action", "Sci-Fi, Thriller", "Sci-Fi, Drama", "Action, Crime",
        "Action, Sci-Fi", "Action, Drama", "Drama, Mystery", "Drama, Romance",
        "Adventure, Sci-Fi", "Sci-Fi, Adventure"
    ]
})

# ---- Improved Search Test Function ----
def search_and_render(df_tab):
    search_query = st.text_input(
        "🔍 Search by movie title or genre",
        key="search_test",
        placeholder="Type to search..."
    ).strip().lower()

    filtered_df = df_tab
    if search_query:
        filtered_df = df_tab[
            df_tab["Series_Title"].str.lower().str.contains(search_query) |
            df_tab["Genre"].str.lower().str.contains(search_query)
        ].copy()

        # show suggestions
        suggestions = filtered_df["Series_Title"].head(5).tolist()
        if suggestions:
            st.caption("Suggestions:")
            cols = st.columns(len(suggestions))
            for i, title in enumerate(suggestions):
                if cols[i].button(title, key=f"sugg_test_{i}"):
                    st.session_state["search_test"] = title
                    st.experimental_rerun()
        else:
            st.warning("No results found")
            return

    if filtered_df.empty:
        st.warning("No results found")
    else:
        st.write(filtered_df)

# ---- Mock Dashboard Page for Scroll Test ----
def dashboard_page():
    # ✅ Scroll-to-top snippet
    st.markdown(
        "<script>window.scrollTo({top: 0, left: 0, behavior: 'auto'});</script>",
        unsafe_allow_html=True
    )

    st.title("Dashboard Scroll-and-Search Test")
    st.write("This page should always load scrolled to the very top.")
    st.write("Type below to test live search with clickable suggestions:")

    search_and_render(df)

# ---- Run Page ----
dashboard_page()
