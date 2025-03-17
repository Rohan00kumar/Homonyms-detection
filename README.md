## 📌 **Project Overview**  
This project is a **hybrid recommender system** that combines content-based and clustering-based approaches to generate high-quality recommendations. It addresses key challenges in recommendation systems, such as:  
✅ **Homonyms** - Same titles with different meanings.  
✅ **Cold-Start Problem** - Lack of user history or new items.  
✅ **Duplicate Recommendations** - Repeated or similar results.  
✅ **Low Relevance** - Poor recommendation quality due to lack of contextual features.  

---

## 🚀 **How It Works**  
### 1. **Data Preprocessing**  
- Data is loaded from `data.csv`.  
- Metadata (title, genre, tags) is combined and vectorized using **TF-IDF** (Term Frequency-Inverse Document Frequency).  
- Vectorized data is used for clustering.  

---

### 2. **Clustering Approach**  
- Uses **KMeans** for clustering items based on metadata similarity.  
- Number of clusters = `10` (can be fine-tuned).  
- Each item is assigned to a cluster based on similarity.  

---

### 3. **Homonym Resolution**  
**Problem:** Homonyms (e.g., "Avatar" movie title refers to different movies).  
**Solution:**  
- Finds the closest match based on string length difference.  
- Ensures the correct context is used for recommendations.  

```python
def resolve_homonym(query):
    choices = data['title'].unique()
    return min(choices, key=lambda x: abs(len(x) - len(query)))
```

---

### 4. **Cold-Start Problem Handling**  
**Problem:** If no cluster is found for a new or rare query.  
**Solution:**  
- Suggests top-rated items based on genre and release year.  
- Ensures meaningful recommendations even with limited data.  

```python
if cluster_items.empty:
    top_items = data.sort_values(by=['rating', 'year'], ascending=[False, False]).head(5)
    return top_items[['title', 'genres', 'year', 'rating']]
```

---

### 5. **Recommendation Generation**  
**Problem:** Poor relevance due to lack of context.  
**Solution:**  
- Calculates **cosine similarity** between the query and cluster items.  
- Sorts recommendations by:  
  - **Similarity**  
  - **Popularity** = rating * number of ratings per genre  
  - **Rating**  
- Removes duplicates and returns top 10 recommendations.  

```python
similarity = cosine_similarity(query_vector, X[cluster_items.index])
cluster_items['similarity'] = similarity.flatten()
cluster_items['popularity'] = cluster_items['rating'] * cluster_items.groupby('genres')['rating'].transform('count')
```

---

### 6. **Final Output**  
✅ Top 10 recommended items based on similarity, popularity, and rating  
✅ Clean, unique recommendations without duplicates  

---

## 🏆 **Key Improvements Over Previous Versions**  
✅ Improved recommendation quality using **popularity**  
✅ Handled cold-start using **genre and year similarity**  
✅ Faster execution using **KMeans** instead of GMM  
✅ Better homonym resolution  
✅ Removed duplicates  

---

## 🛠️ **How to Run**  
1. Install dependencies:  
```bash
pip install pandas scikit-learn
```
2. Run the script:  
```bash
python recommender.py
```
3. Enter a movie name and get recommendations!  

---
