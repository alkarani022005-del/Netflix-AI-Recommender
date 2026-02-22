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
# CUSTOM NETFLIX STYLE
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
    border-radius: 10px;
    transition: transform 0.3s;
}

img:hover {
    transform: scale(1.05);
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
# LOAD DATA
# =========================
movies = pd.read_csv("movies.csv")

movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['combined'] = movies['overview'] + " " + movies['genres']

# =========================
# ML MODEL
# =========================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# =========================
# FETCH POSTER FUNCTION
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
# RECOMMEND FUNCTION
# =========================
def recommend(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

    recommended_names = []
    recommended_posters = []

    for i in similarity_scores:
        movie_id = movies.iloc[i[0]].id
        recommended_names.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_names, recommended_posters

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">NETFLIX MOVIE RECOMMENDER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Movie Recommendations 🎬</div>', unsafe_allow_html=True)

# =========================
# TRENDING SECTION
# =========================
st.markdown("## 🔥 Trending Now")

trending_movies = movies.sort_values(by="vote_average", ascending=False).head(5)
trend_cols = st.columns(5)

for i, row in enumerate(trending_movies.itertuples()):
    with trend_cols[i]:
        poster = fetch_poster(row.id)
        st.image(poster, use_container_width=True)
        st.markdown(f"<div class='movie-title'>{row.title}</div>", unsafe_allow_html=True)

# =========================
# FEATURED SECTION
# =========================
featured = random.choice(movies.itertuples())
featured_poster = fetch_poster(featured.id)

st.markdown("---")
st.markdown("## ⭐ Featured Movie")

st.image(featured_poster, use_container_width=True)
st.markdown(f"<h2 style='color:white;'>{featured.title}</h2>", unsafe_allow_html=True)

st.markdown("---")

# =========================
# SEARCH & FILTER
# =========================
search_query = st.text_input("🔍 Search for a movie")

if search_query:
    filtered_movies = movies[movies['title'].str.contains(search_query, case=False)]
else:
    filtered_movies = movies

# Genre filter
all_genres = set()
for g in movies['genres']:
    for item in g.split():
        all_genres.add(item)

selected_genre = st.selectbox("🎭 Filter by Genre", ["All"] + sorted(list(all_genres)))

if selected_genre != "All":
    filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(selected_genre)]

selected_movie = st.selectbox("🎥 Select a Movie", filtered_movies['title'].values)

# =========================
# RECOMMEND BUTTON
# =========================
if st.button("🎬 Recommend"):
    names, posters = recommend(selected_movie)

    st.markdown("## 🎯 Recommended For You")
    rec_cols = st.columns(5)

    for i in range(5):
        with rec_cols[i]:
            st.image(posters[i], use_container_width=True)
            st.markdown(f"<div class='movie-title'>{names[i]}</div>", unsafe_allow_html=True)
