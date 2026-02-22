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
    recommended = similarity_scores[1:num_recommendations+1]

    recommended_movies = []
    for i in recommended:
        recommended_movies.append({
            "title": movies.iloc[i[0]].title,
            "id": movies.iloc[i[0]].id
        })

    return recommended_movies

# =========================
# FETCH POSTER
# =========================
def fetch_poster(movie_id):
    api_key = st.secrets["TMDB_API_KEY"]

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    data = response.json()

    if 'poster_path' in data and data['poster_path']:
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
}
.subtitle {
    text-align: center;
    color: white;
}
.movie-title {
    text-align: center;
    color: white;
    font-size: 16px;
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

# =========================
# TRENDING
# =========================
st.markdown("## 🔥 Trending Now")

trending_movies = movies.sort_values(by="vote_average", ascending=False).head(5)
cols = st.columns(5)

for i, row in trending_movies.iterrows():
    with cols[list(trending_movies.index).index(i)]:
        poster = fetch_poster(row['id'])
        st.image(poster, use_container_width=True)
        st.markdown(f"<div class='movie-title'>{row['title']}</div>", unsafe_allow_html=True)

# =========================
# FEATURED MOVIE
# =========================
featured_row = movies.sample(1).iloc[0]
featured_poster = fetch_poster(featured_row['id'])

st.markdown("---")
st.image(featured_poster, use_container_width=True)
st.markdown(f"<h2 style='color:white;'>🔥 Featured Movie: {featured_row['title']}</h2>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# SEARCH
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
# RECOMMEND
# =========================
if st.button("🔥 Recommend"):
    recommendations = recommend_movies(selected_movie)

    st.markdown("## 🎬 Top Recommendations For You")
    cols = st.columns(5)

    for i, movie in enumerate(recommendations):
        with cols[i]:
            poster = fetch_poster(movie['id'])
            st.image(poster, use_container_width=True)
            st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
