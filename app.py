import streamlit as st
import pickle
import pandas as pd
import sqlite3
import os

# ===== DB Setup =====
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS watch_history (
            username TEXT,
            movie TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS genre_preferences (
            username TEXT,
            genre TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ===== Load model =====
with open("hybrid_recommender.pkl", "rb") as f:
    data = pickle.load(f)

recommend_for_user_func = data["recommend_for_user"]
movie_metadata = data["movie_metadata"]

# ===== Helpers =====
def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False

def check_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None

def mark_watched(username, movie):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO watch_history VALUES (?, ?)", (username, movie))
    conn.commit()
    conn.close()

def get_watch_history(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT movie FROM watch_history WHERE username=?", (username,))
    movies = [row[0] for row in c.fetchall()]
    conn.close()
    return movies

def save_genre_preference(username, genre):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO genre_preferences VALUES (?, ?)", (username, genre))
    conn.commit()
    conn.close()

def get_genre_preferences(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT genre FROM genre_preferences WHERE username=?", (username,))
    genres = [row[0] for row in c.fetchall()]
    conn.close()
    return genres

# ===== Reason Formatter =====
def format_reason(reason: str) -> str:
    if not reason:
        return ""

    watched_movies = []
    genres = []

    # Extract watched movies
    if "You watched" in reason:
        try:
            watched_text = reason.split("You watched ")[1].split(" and ")[0]
            watched_movies = [m.strip() for m in watched_text.split(",")]
        except:
            pass

    # Extract genres
    if "You selected genre(s)" in reason:
        try:
            genre_text = reason.split("You selected genre(s) ")[1]
            genres = [g.strip() for g in genre_text.split(",")]
        except:
            pass

    parts = ["<div style='margin-top:8px;color:#399ed7;font-size:0.9rem;'>💡 Why recommended:<br>"]

    if watched_movies:
        parts.append("🎬 Similar to:<br>")
        for movie in watched_movies:
            parts.append(f"&nbsp;&nbsp;&nbsp;&nbsp;• {movie}<br>")

    if genres:
        parts.append("🎯 Matches your genre(s):<br>")
        for g in genres:
            parts.append(f"&nbsp;&nbsp;&nbsp;&nbsp;• {g}<br>")

    parts.append("</div>")
    return "".join(parts)

# ===== Movie Card UI =====
def movie_card(title, genres, rating, reason, watched=False):
    reason_html = format_reason(reason)

    card_html = f"""
    <div style='
        border:1px solid #ddd;
        border-radius:10px;
        padding:15px;
        margin-bottom:15px;
        background-color:#fff;
        box-shadow:0 2px 5px rgba(0,0,0,0.1);
    '>
        <h4 style='margin:0;'>{title}</h4>
        <p style='margin:5px 0; color:gray; font-style:italic;'>{genres}</p>
        <p style='margin:5px 0; color:#f39c12;'>⭐ {rating}/10</p>
        {reason_html}
        {"<p style='color:green;'>✅ Watched</p>" if watched else ""}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# ===== App =====
st.title("🎬 Movie Recommender System")

if "username" not in st.session_state:
    st.session_state["username"] = None

menu = ["Login", "Signup", "Dashboard"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Signup":
    st.subheader("Create Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Signup"):
        if add_user(new_user, new_pass):
            st.success("Account created successfully!")
        else:
            st.error("Username already exists.")

elif choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_user(username, password):
            st.session_state["username"] = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid username or password.")

elif choice == "Dashboard":
    if st.session_state["username"]:
        st.subheader(f"Welcome, {st.session_state['username']} 👋")

        # Watch history
        st.markdown("### 🎞️ Your Watch History")
        history = get_watch_history(st.session_state['username'])
        if history:
            for movie in history:
                st.write(f"✅ {movie}")
        else:
            st.info("No movies watched yet.")

        # Genre preferences
        st.markdown("### 🎭 Select Your Favorite Genres")
        all_genres = sorted(set([g for sub in movie_metadata['genres_clean'].str.split(",") for g in sub]))
        selected_genre = st.selectbox("Pick a genre:", all_genres)
        if st.button("Add Genre Preference"):
            save_genre_preference(st.session_state['username'], selected_genre)
            st.success(f"Added {selected_genre} to preferences")

        # Recommendations
        st.markdown("### 🎯 Your Recommendations")
        recs = recommend_for_user_func(st.session_state['username'])
        for movie, reason in recs:
            movie_info = movie_metadata[movie_metadata['title'] == movie].iloc[0]
            watched = movie in history
            movie_card(
                movie_info['title'],
                movie_info['genres_clean'],
                movie_info['rating'],
                reason,
                watched
            )
            if not watched:
                if st.button(f"Mark as Watched: {movie}", key=f"watch_{movie}"):
                    mark_watched(st.session_state['username'], movie)
                    st.success(f"Marked '{movie}' as watched!")

    else:
        st.warning("Please log in first.")
