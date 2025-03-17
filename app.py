import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load data
data = pd.read_csv('data.csv')

# Use HashingVectorizer
vectorizer = HashingVectorizer(
    n_features=2000, ngram_range=(1, 2), alternate_sign=False)
X = vectorizer.fit_transform(data['metadata'])

# KMeans Clustering
kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
data['cluster'] = kmeans.fit_predict(X)

# Save model
joblib.dump(kmeans, 'kmeans_model.pkl')

# UI Header
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title('🎬 Hybrid Movie Recommender System')
st.write('Get personalized movie recommendations based on your interests')

# Search Bar
query = st.text_input('🔎 Search for a movie', '')

# Function to recommend items
def recommend(query):
    query_vector = vectorizer.transform([query])
    cluster = kmeans.predict(query_vector)[0]
    cluster_items = data[data['cluster'] == cluster]
    similarity = cosine_similarity(
        query_vector, X[cluster_items.index]).flatten()
    cluster_items = cluster_items.assign(similarity=similarity)
    recommendations = cluster_items.sort_values(
        by='similarity', ascending=False).drop_duplicates('title').head(10)
    return recommendations[['title', 'genres', 'year', 'similarity']]


if query:
    st.subheader('📽️ Recommended Movies')
    recommendations = recommend(query)
    if not recommendations.empty:
        st.dataframe(recommendations)

# Filter Options
st.sidebar.subheader('🎯 Filter Recommendations')
selected_genre = st.sidebar.selectbox(
    'Select Genre', ['All'] + list(data['genres'].unique()))

# Fix range issue
min_year = int(data['year'].min()) if not data['year'].isnull().all() else 1900
max_year = int(data['year'].max()) if not data['year'].isnull().all() else 2025
if min_year < max_year:
    selected_year = st.sidebar.slider(
        'Select Year', min_year, max_year, (min_year, max_year))
else:
    st.sidebar.warning('Invalid year range.')

filtered_data = data
if selected_genre != 'All':
    filtered_data = filtered_data[filtered_data['genres'].str.contains(
        selected_genre)]
if min_year < max_year:
    filtered_data = filtered_data[(filtered_data['year'] >= selected_year[0]) & (
        filtered_data['year'] <= selected_year[1])]

st.subheader('🎬 Filtered Movies')
st.dataframe(filtered_data[['title', 'genres', 'year']
                           ].drop_duplicates().head(10))

# Cluster Distribution
st.subheader('📊 Cluster Distribution')
st.bar_chart(data['cluster'].value_counts())