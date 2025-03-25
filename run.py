import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load Processed Data
data = pd.read_csv('processed_data.csv')

# Load Trained Model and Vectorizer
model = joblib.load('final_optimized_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Input Movie Title for Recommendation
movie_title = input("Enter movie title: ")

# Find Matching Movie in Data
if movie_title in data['title'].values:
    movie_idx = data[data['title'] == movie_title].index[0]

    # Extract Movie Metadata and Vectorize
    movie_vector = vectorizer.transform([data.loc[movie_idx, 'metadata']])

    # Find Similarity Scores with All Movies
    similarity_scores = cosine_similarity(
        movie_vector, vectorizer.transform(data['metadata']))

    # Get Top 10 Recommendations
    top_indices = similarity_scores.argsort()[0][-11:-1][::-1]
    recommendations = data.loc[top_indices, ['title', 'genres', 'rating']]

    # Display Recommendations
    print("\nTop Recommended Movies:")
    print(recommendations)

else:
    print("Movie not found in the dataset. Please try another title.")
