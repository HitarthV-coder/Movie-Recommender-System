# preprocess.py (Corrected Original Version)
import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- Data Loading ---
print("Loading data...")
credits = pd.read_csv('tmdb_5000_credits.csv')
# Add this line to remove the duplicate title column from the credits dataframe
credits.drop('title', axis=1, inplace=True)
movies_df = pd.read_csv('tmdb_5000_movies.csv') # Renamed to avoid confusion

# --- Data Merging and Cleaning ---
# Merge using the movie ID, which is more reliable than title
movies = movies_df.rename(columns={'id': 'movie_id'}).merge(credits, on='movie_id')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# --- Helper Functions for Data Transformation ---
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# --- Applying Transformations ---
print("Transforming data...")
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
# The 'crew' column from the merge will be different, we need to adapt
movies['crew'] = movies['crew'].apply(fetch_director)
movies.dropna(inplace=True) # drop rows where director might be missing

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# --- Create 'tags' Column ---
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# --- Stemming ---
print("Stemming tags...")
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# --- Vectorization ---
print("Vectorizing text...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# --- Similarity Calculation ---
print("Calculating similarity...")
similarity = cosine_similarity(vectors)

# --- Saving Processed Data ---
print("Saving processed data files...")
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("\nPreprocessing complete! You can now run the updated app.py.")