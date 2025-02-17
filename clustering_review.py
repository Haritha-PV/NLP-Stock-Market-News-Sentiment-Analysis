
import pandas as pd
import numpy as np
import sqlite3
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import nltk
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


conn = sqlite3.connect('Database.db')

df = pd.read_sql_query('Select * from New_Delhi_Reviews' , conn)

df

reviews = df['review_full'].dropna()

# Text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

cleaned_reviews = reviews.apply(clean_text)

print(df.info())  # Check data types of each column
print(df.head())  # Display the first few rows of the dataset

# 3. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1, 3))  # Restrict features to improve speed
X = tfidf.fit_transform(cleaned_reviews)

# 4. PCA for Dimensionality Reduction
pca = PCA(n_components=10, random_state=42)  # Retain top 50 components
X_reduced = pca.fit_transform(X.toarray())

# 5. Mini-Batch KMeans Clustering
inertia = []
sil_scores = []

# Clustering and metric calculation
k_values = range(2, 11)  # Values of k from 2 to 10
for k in k_values:
    mb_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=500)
    mb_kmeans.fit(X_reduced)  # Replace X_reduced with your actual feature matrix
    labels = mb_kmeans.labels_
    inertia.append(mb_kmeans.inertia_)
    sil_scores.append(silhouette_score(X_reduced, labels))

# Verify lengths of k_values, inertia, and sil_scores
print(f"Length of k_values: {len(k_values)}")
print(f"Length of inertia: {len(inertia)}")
print(f"Length of sil_scores: {len(sil_scores)}")

# Find the best k based on the highest silhouette score
best_k = k_values[sil_scores.index(max(sil_scores))]
best_sil_score = max(sil_scores)

print(f"Best k based on Silhouette Score: {best_k}")
print(f"Best Silhouette Score: {best_sil_score}")

# Plot Elbow Method for Inertia
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o', label='Inertia')
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.legend()
plt.grid()
plt.show()

# Silhouette Score Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, sil_scores, marker='o', label='Silhouette Score', color='orange')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid()
plt.show()