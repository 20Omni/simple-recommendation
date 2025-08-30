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
        json.dump(data, f, indent=4)

# ===== Age Certificate Filtering =====
def get_allowed_certificates(profile_type):
    if profile_type == "Kids":
        return ["U", "UA 7+", "G"]
    elif profile_type == "Teen":
        return ["U", "UA 7+", "UA 13+", "PG-13", "PG"]
    elif profile_type == "Adult":
        return ["U", "UA 7+", "UA 13+", "PG-13", "PG", "R", "A", "NC-17"]
    else:
        return ["U"]

# ===== Model & Data Load =====
@st.cache_resource
def load_model_and_data():
    model = joblib.load("model.pkl")
    movies = pd.read_csv("movies.csv")
    return model, movies

model, movies = load_model_and_data()
user_data = load_user_data()

# ===== Signup =====
def signup_user(username, password):
    if username in user_data:
        st.error("Username already exists")
        return False
    profile = st.selectbox("Choose Profile Type", ["Kids", "Teen", "Adult"])
    user_data[username] = {"password": password, "genres": [], "watched": [], "profile": profile}
    save_user_data(user_data)
    st.success("Signup successful! Please login.")
    return True

# ===== Login =====
def login_user(username, password):
    if username in user_data and user_data[username]["password"] == password:
        return True
    return False

# ===== Recommendation System =====
def recommend_for_user(username):
    genres = user_data[username]["genres"]
    profile = user_data[username]["profile"]
    allowed = get_allowed_certificates(profile)

    if not genres:
        return pd.DataFrame()

    filtered = movies[movies["genres"].apply(lambda g: any(genre in g for genre in genres))]
    filtered = filtered[filtered["certificate"].isin(allowed)]
    return filtered.sample(min(10, len(filtered)))

# ===== UI Pages =====
def genre_selection_page(username):
    st.title("Select Your Favorite Genres 🎬")
    genres = list(set(g for gs in movies["genres"] for g in gs.split(", ")))
    selected = st.multiselect("Choose genres you like:", genres)
    if st.markdown("""<style>.next-btn button{background:#FF4B4B;color:white;border-radius:12px;font-size:18px;padding:8px 20px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);} .next-btn button:hover{background:#E03E3E;}</style>""", unsafe_allow_html=True):
        pass
    if st.button("Next ➡️", key="next", help="Proceed", use_container_width=True):
        user_data[username]["genres"] = selected
        save_user_data(user_data)
        st.session_state.page = "dashboard"

# ===== Dashboard =====
def dashboard(username):
    st.title(f"Welcome {username}! 🎉")
    profile = user_data[username]["profile"]
    allowed = get_allowed_certificates(profile)

    tab1, tab2, tab3 = st.tabs(["Recommendations", "Top Rated", "Your Watching"])

    with tab1:
        recs = recommend_for_user(username)
        if recs.empty:
            st.info("Select genres first to get recommendations!")
        for _, row in recs.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(row["title"])
                st.write(f"{row['certificate']} | Rating: {row['rating']}")
            with col2:
                if st.button("▶ Watch Now", key=f"watch_{row['id']}"):
                    user_data[username]["watched"].append(row["id"])
                    save_user_data(user_data)
                    st.success(f"Marked {row['title']} as watched!")

    with tab2:
        top = movies[movies["certificate"].isin(allowed)].sort_values(by="rating", ascending=False).head(10)
        st.dataframe(top[["title", "certificate", "rating"]])

    with tab3:
        watched_ids = user_data[username]["watched"]
        watched_movies = movies[movies["id"].isin(watched_ids)]
        st.dataframe(watched_movies[["title", "certificate", "rating"]])

# ===== App Flow =====
st.sidebar.title("Movie Recommender")
choice = st.sidebar.radio("Menu", ["Login", "Signup"])

if choice == "Signup":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Signup"):
        signup_user(username, password)

elif choice == "Login":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.username = username
            if not user_data[username]["genres"]:
                st.session_state.page = "genre"
            else:
                st.session_state.page = "dashboard"
        else:
            st.error("Invalid credentials")

# Page Router
if "page" in st.session_state and "username" in st.session_state:
    if st.session_state.page == "genre":
        genre_selection_page(st.session_state.username)
    elif st.session_state.page == "dashboard":
        dashboard(st.session_state.username)
