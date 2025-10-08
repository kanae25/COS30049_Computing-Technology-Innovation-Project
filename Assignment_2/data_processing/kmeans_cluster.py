# kmeans_cluster.py — K-Means clustering on TF-IDF (+SVD) with external & internal metrics
# Uses: outputs/processed/emails_merged.processed.csv

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).parent.parent
DATA = ROOT / "outputs" / "processed" / "emails_merged.processed.csv"

df = pd.read_csv(DATA)
X_text = df["text"].astype(str).tolist()
y_true = df["label"].astype(int).values  # ground truth for external metrics

# TF-IDF -> SVD to reduce noise, speed KMeans
tfidf = TfidfVectorizer(min_df=5, ngram_range=(1,2))
X = tfidf.fit_transform(X_text)
svd = TruncatedSVD(n_components=100, random_state=42)
X_red = svd.fit_transform(X)

kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
clusters = kmeans.fit_predict(X_red)

# Internal metric
sil = silhouette_score(X_red, clusters)

# External metrics (use labels only for evaluation, not for training)
h, c, v = homogeneity_completeness_v_measure(y_true, clusters)
nmi = normalized_mutual_info_score(y_true, clusters)
ari = adjusted_rand_score(y_true, clusters)

# Best mapping of cluster ids to {0,1} via Hungarian algorithm for an accuracy-like score
cont = pd.crosstab(y_true, clusters).values
row_ind, col_ind = linear_sum_assignment(cont.max() - cont)  # maximize matches
best_match = cont[row_ind, col_ind].sum()
matched_acc = best_match / cont.sum()

print("=== K-Means (k=2) on TF-IDF + SVD ===")
print(f"Silhouette (internal): {sil:.4f}")
print(f"Homogeneity: {h:.4f} | Completeness: {c:.4f} | V-measure: {v:.4f}")
print(f"NMI: {nmi:.4f} | ARI: {ari:.4f}")
print(f"Best-matched accuracy (post-hoc label mapping): {matched_acc:.4f}")
print("\nContingency table (rows=true label 0/1, cols=cluster 0/1):")
print(pd.crosstab(y_true, clusters))
