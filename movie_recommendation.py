import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Expanded dataset with poster URLs
data = {
    "movie": [
        "Inception", "Titanic", "The Dark Knight", "Interstellar", "Avatar",
        "The Matrix", "Gladiator", "Forrest Gump", "The Shawshank Redemption",
        "Jurassic Park", "The Lion King", "Frozen", "Avengers: Endgame",
        "Black Panther", "Toy Story"
    ],
    "genre": [
        "Sci-Fi", "Romance", "Action", "Sci-Fi", "Fantasy",
        "Sci-Fi", "Action", "Drama", "Drama",
        "Adventure", "Animation", "Animation", "Action",
        "Action", "Animation"
    ],
    "poster_url": [
        "https://static1.srcdn.com/wordpress/wp-content/uploads/Leonardo-DiCaprio-Inception-The-Extractor-Poster.jpg",
        "https://image.tmdb.org/t/p/original/8MFJ4aAr85B5lVCecxGSd9iX6FX.jpg",
        "https://image.tmdb.org/t/p/original/c94GEWkz12pYfg9fO1weiN1ibU4.jpg",
        "https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg",
        "https://image.tmdb.org/t/p/original/FpH1WMn80l2ZeDjPoOW5aWZfJT.jpg",
        "https://image.tmdb.org/t/p/w440_and_h660_face/cgwASCaaYxq31SU1xoMbUc4Re4m.jpg",
        "https://c8.alamy.com/comp/R59H7F/gladiator-original-movie-poster-R59H7F.jpg",
        "https://image.tmdb.org/t/p/original/saHP97rTPS5eLmrLQEcANmKrsFl.jpg",
        "https://image.tmdb.org/t/p/original/hxcHfW0o5QhY6slqGc7dkEkvo6U.jpg",
        "https://image.tmdb.org/t/p/original/7xpRzGR4jXLTXdjfFBtdE4OboJr.jpg",
        "https://image.tmdb.org/t/p/original/7bop06WN9YpgreK5Xs3qU29vgCa.jpg",
        "https://image.tmdb.org/t/p/original/h2uN3MVsyCZjBS1CSqUpOrQ1XWb.jpg",
        "https://upload.wikimedia.org/wikipedia/en/0/0d/Avengers_Endgame_poster.jpg",
        "https://image.tmdb.org/t/p/original/fj7sX7w0MfIxWylcizp5ArPIMFs.jpg",
        "https://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg"
    ]
}

movies = pd.DataFrame(data)

# Convert genres to vectors
cv = CountVectorizer()
matrix = cv.fit_transform(movies["genre"])

# Calculate similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie):
    index = movies[movies["movie"] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommended = []
    for i in movies_list:
        recommended.append(movies.iloc[i[0]].movie)
    return recommended

# Streamlit UI
st.title("🎬 Movie Recommender System")

# Sidebar layout (branding + controls)
st.sidebar.markdown("## 🎬 Movie Recommender")
st.sidebar.markdown("🍿 **Welcome to your personalized movie recommender!**")
st.sidebar.markdown("---")

st.sidebar.header("Choose Your Movie")
selected_movie = st.sidebar.selectbox(
    "Pick a movie",
    movies["movie"].values
)

recommend_button = st.sidebar.button("Recommend")

st.sidebar.header("Filters")
genre_filter = st.sidebar.multiselect(
    "Filter by genre",
    movies["genre"].unique()
)

# Main content
if recommend_button:
    # Show selected movie poster
    selected_poster = movies[movies["movie"] == selected_movie]["poster_url"].values[0]
    st.write(f"### Selected Movie: {selected_movie}")
    st.image(selected_poster, caption=selected_movie, width=240)

    # Show recommendations in rows of 3
    recommendations = recommend(selected_movie)

    # Apply genre filter if selected
    if genre_filter:
        recommendations = [
            m for m in recommendations
            if movies[movies["movie"] == m]["genre"].values[0] in genre_filter
        ]

    st.write("### Recommended Movies:")

    for i in range(0, len(recommendations), 3):
        cols = st.columns(3)
        for idx, movie in enumerate(recommendations[i:i+3]):
            poster_url = movies[movies["movie"] == movie]["poster_url"].values[0]
            with cols[idx]:
                st.image(poster_url, caption=movie, width=180)