# app.py
from flask import Flask, render_template_string, request
import pandas as pd
import pickle
import requests  # Import the requests library

app = Flask(__name__)

# --- TMDb API Configuration ---
# The API key you provided is included here.
API_KEY = "8386ab94d9aee13375ef76a056c7a8af"

# --- Load the preprocessed data ---
try:
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    print("Pickle files not found. Please run preprocess.py first.")
    exit()


# --- Helper function to fetch poster from API ---
def fetch_poster_url(movie_id):
    """Fetches the poster URL for a movie ID from the TMDb API."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
    # Return a placeholder if the poster is not found or if the API call fails
    return "https://via.placeholder.com/500x750.png?text=No+Image+Found"


# --- Recommendation Function ---
def recommend(movie):
    """Returns a list of 5 recommended movies, each with a title and movie_id."""
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies_data = []
        for i in movies_list:
            movie_id = movies.iloc[i[0]].movie_id
            title = movies.iloc[i[0]].title
            recommended_movies_data.append({"title": title, "movie_id": movie_id})
        return recommended_movies_data
    except (IndexError, KeyError):
        return []


# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommender</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: auto; text-align: center; }
        h1 { color: #bb86fc; margin-bottom: 20px; }
        form { margin-bottom: 40px; }
        input[type="text"] { width: 60%; padding: 12px; margin: 10px 0; border: 1px solid #333; border-radius: 6px; background-color: #2c2c2c; color: #e0e0e0; font-size: 1rem; }
        button { background-color: #bb86fc; color: #121212; border: none; padding: 12px 24px; font-size: 1rem; border-radius: 6px; cursor: pointer; transition: background-color 0.3s; font-weight: bold; }
        button:hover { background-color: #3700b3; color: white; }
        .results-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
        .movie-card { background: #1e1e1e; border-radius: 8px; overflow: hidden; width: 200px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); text-align: center; transition: transform 0.2s; }
        .movie-card:hover { transform: scale(1.05); }
        .movie-card img { width: 100%; height: 300px; object-fit: cover; }
        .movie-card h3 { font-size: 1rem; padding: 10px; margin: 0; color: #e0e0e0; }
        h2 { color: #03dac6; }
        p.error { color: #cf6679; font-size: 1.2rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Movie Recommender System 🎬</h1>
        <form action="/recommend" method="post">
            <input list="movie_titles" type="text" id="movie" name="movie" placeholder="Type a movie you like..." required>
            <datalist id="movie_titles">
                {% for title in movie_titles %}
                    <option value="{{ title }}">
                {% endfor %}
            </datalist>
            <br>
            <button type="submit">Get Recommendations</button>
        </form>

        {% if recommendations %}
            <h2>Recommended for you:</h2>
            <div class="results-container">
                {% for movie in recommendations %}
                    <div class="movie-card">
                        <img src="{{ movie.poster_url }}" alt="{{ movie.title }} Poster">
                        <h3>{{ movie.title }}</h3>
                    </div>
                {% endfor %}
            </div>
        {% endif %}

        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
"""


# --- Flask Routes ---
@app.route('/')
def home():
    movie_titles = sorted(movies['title'].values)
    return render_template_string(HTML_TEMPLATE, movie_titles=movie_titles)


@app.route('/recommend', methods=['POST'])
def recommend_movies():
    selected_movie = request.form.get('movie')
    movie_titles = sorted(movies['title'].values)

    if selected_movie not in movie_titles:
        error_message = f"Movie '{selected_movie}' not found. Please choose from the list."
        return render_template_string(HTML_TEMPLATE, movie_titles=movie_titles, error=error_message)

    # Get recommendations (list of dicts with 'title' and 'movie_id')
    recommended_movies = recommend(selected_movie)

    # Fetch poster for each recommended movie
    recommendations_with_posters = []
    for movie in recommended_movies:
        recommendations_with_posters.append({
            'title': movie['title'],
            'poster_url': fetch_poster_url(movie['movie_id'])
        })

    return render_template_string(HTML_TEMPLATE, movie_titles=movie_titles,
                                  recommendations=recommendations_with_posters)


# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    print("Open http://127.0.0.1:5000 in your web browser.")
    app.run(debug=True)
