## 📄 **Hybrid Movie Recommendation System**  
This is the **Hybrid Movie Recommendation System** project, covering the core concepts, steps, implementation details, and solutions to key problems such as **homonyms**, **cold start**, and **grey sheep**. The project aims to create an efficient recommendation system using clustering, similarity analysis, and metadata-based filtering. 🚀  

---

## 🏆 **Objective**  
The goal of this project is to build a **Hybrid Movie Recommendation System** that effectively handles common issues like:  
✅ Homonyms (similar or identical movie titles)  
✅ Cold Start Problem (new items without prior data)  
✅ Grey Sheep Problem (unique user behavior)  

---

## 📂 **Dataset Description**  
The dataset includes the following attributes:  

| Column Name | Description | Importance |  
|------------|-------------|------------|  
| `movieId` | Unique identifier for each movie | Identification |  
| `rating` | User rating for the movie | Used to measure relevance |  
| `title` | Title of the movie | Important for similarity matching |  
| `genres` | Genre of the movie | Used for clustering and recommendations |  
| `tag` | User-generated tags for the movie | Used for similarity calculation |  
| `metadata` | Combined features (genre, director, year, etc.) | For TF-IDF vectorization |  
| `year` | Release year of the movie | Used for time-based recommendations |  
| `popularity` | Popularity score of the movie | Used for ranking |  
| `runtime` | Duration of the movie | Used for similarity analysis |  
| `director` | Director of the movie | Used for similarity calculation |  
| `actors` | Leading actors in the movie | Used for similarity calculation |  
| `language` | Language of the movie | Used for filtering |  
| `country` | Country of origin | Used for filtering |  
| `budget` | Budget of the movie | Used for similarity analysis |  
| `revenue` | Revenue generated | Used for ranking |  

---

## 🔍 **Approach and Techniques**  
This project combines **content-based filtering** and **clustering** to build a hybrid recommendation system. The following methods were used:  

### ✅ **1. Data Preprocessing**  
- Removed missing/null values  
- Handled duplicates  
- Combined metadata to create a meaningful feature vector  
- Cleaned the dataset for consistency  

---

### ✅ **2. Vectorization using TF-IDF**  
Used **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text-based metadata into numerical vectors.  
**Why TF-IDF?**  
✔️ Converts text to numerical form  
✔️ Handles high-dimensional data  
✔️ Captures importance of terms  

```python
vectorizer = TfidfVectorizer(max_features=2005)
X = vectorizer.fit_transform(data['metadata'])
```

---

### ✅ **3. Clustering using MiniBatchKMeans**  
Used **MiniBatchKMeans** instead of KMeans or GMM to avoid memory issues and improve training time.  
**Why MiniBatchKMeans?**  
✔️ Memory-efficient  
✔️ Faster convergence  
✔️ Works well with large datasets  

```python
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=10, batch_size=100)
data['cluster'] = kmeans.fit_predict(X)
```

---

### ✅ **4. Homonyms Detection and Resolution**  
Homonyms create confusion when two or more movies have similar or identical names.  
**Solution:**  
- Used clustering and similarity matching to disambiguate them.  
- Combined metadata features like `year`, `genre`, `actors` to differentiate similar titles.  

```python
def resolve_homonym(query):
    matches = data[data['title'].str.contains(query, case=False)]
    if len(matches) > 1:
        matches = matches.sort_values(by='popularity', ascending=False)
    return matches.head(1)['title'].values[0]
```

---

### ✅ **5. Cold Start Problem**  
Cold Start Problem arises when the system encounters a new movie or user without enough data.  
**Solution:**  
- For new movies → Recommended based on genre and popularity  
- For new users → Recommended based on popular trends  

```python
if cluster not in data['cluster'].unique():
    top_items = data.groupby('cluster').apply(lambda x: x.nlargest(5, 'rating')).reset_index(drop=True)
```

---

### ✅ **6. Grey Sheep Problem**  
Grey Sheep Problem happens when a user's preferences do not align with any existing pattern.  
**Solution:**  
- Used clustering and similarity scores  
- For ambiguous cases → Recommended high-rated and popular movies  

```python
similarity = cosine_similarity(query_vector, X[cluster_items.index]).flatten()
recommendations = cluster_items.assign(similarity=similarity).sort_values(by='similarity', ascending=False)
```

---

## 📈 **Performance Evaluation**  
We evaluated the system using the following metrics:  

| Metric | Description | Purpose |  
|--------|-------------|---------|  
| **Silhouette Score** | Measures how similar an object is to its cluster | Cluster Quality |  
| **Davies-Bouldin Index** | Measures the average similarity ratio between clusters | Cluster Separation |  
| **F1 Score** | Harmonic mean of precision and recall | Model Quality |  
| **Recall** | Percentage of relevant items retrieved | Model Performance |  
| **Precision** | Percentage of relevant items among retrieved items | Model Performance |  

### ✅ **Performance Code:**
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, f1_score, recall_score, precision_score

silhouette = silhouette_score(X[valid_clusters.index], valid_clusters['cluster'])
db_score = davies_bouldin_score(X[valid_clusters.index].toarray(), valid_clusters['cluster'])

f1 = f1_score(valid_clusters['rating'] > 3.5, valid_clusters['cluster'], average='macro')
recall = recall_score(valid_clusters['rating'] > 3.5, valid_clusters['cluster'], average='macro')
precision = precision_score(valid_clusters['rating'] > 3.5, valid_clusters['cluster'], average='macro')
```

---

## 📊 **Visualization**  
**1. Cluster Size Distribution**  
Shows the number of items per cluster.  

```python
sns.countplot(data['cluster'])
plt.title('Cluster Size Distribution')
plt.show()
```

**2. Silhouette Score Distribution**  
Shows cluster formation quality.  

```python
sns.histplot(data['cluster'], kde=True)
plt.title('Silhouette Score Distribution')
plt.show()
```

**3. Top Genres Pie Chart**  
Shows genre popularity.  

```python
top_genres = data['genres'].value_counts().nlargest(10)
plt.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%')
plt.show()
```

---

## 💾 **Saving the Final Model**  
Saved the model and vectorizer for future use.  

```python
import joblib

joblib.dump(kmeans, 'final_optimized_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
```

---

## 🚀 **Why This Approach Is Better?**  
✅ **MiniBatchKMeans** → Memory-efficient and fast  
✅ **TF-IDF** → Efficient handling of high-dimensional data  
✅ **Hybrid Approach** → Combines clustering and similarity-based recommendations  
✅ **Cold Start + Grey Sheep Handling** → Makes recommendations even in tricky cases  

---

## 🏁 **How to Use This Project?**  
1. Preprocess the data  
2. Train the model using MiniBatchKMeans  
3. Handle homonyms and cold start issues  
4. Generate recommendations using cosine similarity  
5. Evaluate performance using various metrics  
6. Visualize performance using charts  

---

## 🚨 **Challenges and Solutions**  
| Problem | Solution |  
|---------|----------|  
| Homonyms | Handled using metadata and similarity matching |  
| Cold Start | Suggested popular movies and genre-based recommendations |  
| Grey Sheep | Suggested high-rated and similar movies based on clustering |  
| Memory Issues | Used MiniBatchKMeans to avoid memory overflow |  

---

## 🎉 **Project Status: COMPLETED ✅**  
👉 Efficient and scalable hybrid recommendation system is ready!  
👉 Handles large datasets without memory issues!  
👉 Effective handling of homonyms, cold start, and grey sheep problems!  

---
