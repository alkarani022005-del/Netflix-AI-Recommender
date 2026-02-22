import streamlit as st
import pandas as pd
import random
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG

# =========================
st.set_page_config(page_title="Netflix AI Recommender", layout="wide")

# 🔑 PASTE YOUR NEW TMDB API KEY HERE
API_KEY = "Ocb3cfcfee499772d8d1162d74eb97a6"
# =========================
# LOAD DATA
# =========================
movies = pd.read_csv("movies.csv")

movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['combined_features'] = movies['overview'] + " " + movies['genres']

# =========================
# ML MODEL
# =========================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies(movie_title, num_recommendations=5):
    idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended = [movies.iloc[i[0]].title for i in similarity_scores[1:num_recommendations+1]]
    return recommended

# =========================
# FETCH POSTER FUNCTION
# =========================
# TMDB POSTER FETCH
def fetch_poster(movie_id):
    api_key = "YOUR_API_KEY_HERE"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    
    response = requests.get(url)
    data = response.json()
    
    if 'poster_path' in data and data['poster_path'] is not None:
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    else:
        return "https://via.placeholder.com/500x750?text=No+Image"
# =========================
# NETFLIX STYLE CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #141414;
}

.main-title {
    font-size: 50px;
    font-weight: bold;
    color: #E50914;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 18px;
    margin-bottom: 30px;
}

.movie-title {
    text-align: center;
    color: white;
    font-size: 16px;
    margin-top: 10px;
}

img {
    border-radius: 8px;
    transition: transform 0.3s;
}

img:hover {
    transform: scale(1.08);
}

.stButton>button {
    background-color: #E50914;
    color: white;
    font-size: 18px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">NETFLIX MOVIE RECOMMENDER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Movie Recommendations 🎬</div>', unsafe_allow_html=True)

st.markdown("## 🔥 Trending Now")

trending_movies = movies.sort_values(by="vote_average", ascending=False).head(5)

cols = st.columns(5)

for i, movie in enumerate(trending_movies['title']):
    poster = fetch_poster(movie)
    with cols[i]:
        if poster:
            st.image(poster, use_container_width=True)
        st.markdown(f"<div class='movie-title'>{movie}</div>", unsafe_allow_html=True)
        
# =========================
# HERO FEATURED SECTION
# =========================
featured_movie = random.choice(movies['title'].values)
featured_poster = fetch_poster(featured_movie)

if featured_poster:
    st.markdown(
        f"""
        <div style="position: relative;">
            <img src="{featured_poster}" style="width:100%; border-radius:12px;">
            <div style="
                position:absolute;
                bottom:30px;
                left:40px;
                color:white;
                font-size:40px;
                font-weight:bold;">
                🔥 {featured_movie}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    f"<h2 style='color:white;'>🔥 Featured Movie: {featured_movie}</h2>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# SEARCH BAR
# =========================
search_query = st.text_input("🔍 Search for a movie")

if search_query:
    filtered_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)]
else:
    filtered_movies = movies

# =========================
# GENRE FILTER
# =========================
all_genres = set()
for genre_list in movies['genres']:
    for g in genre_list.split():
        all_genres.add(g)

selected_genre = st.selectbox("🎭 Filter by Genre", ["All"] + sorted(list(all_genres)))

if selected_genre != "All":
    filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(selected_genre)]

# =========================
# SELECT MOVIE
# =========================
selected_movie = st.selectbox("🎥 Select a Movie:", filtered_movies['title'].values)

# =========================
# RECOMMEND BUTTON
# =========================
if st.button("🔥 Recommend"):
    with st.spinner("Finding best movies for you..."):
        recommendations = recommend_movies(selected_movie)

    st.markdown("## 🎬 Top Recommendations For You")

    cols = st.columns(5)

    for i, movie in enumerate(recommendations):
        poster = fetch_poster(movie)

        with cols[i]:
            if poster:
                st.image(poster, use_container_width=True)
            st.markdown(f"<div class='movie-title'>{movie}</div>", unsafe_allow_html=True)
