import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import seaborn as sns
import matplotlib.pyplot as plt

# Function to cluster features using KMeans
def cluster_features(features, num_clusters=9):  # Fixed number of clusters (9)
    # Perform KMeans clustering with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels, kmeans.cluster_centers_

# Function to reduce dimensionality and plot clusters
def plot_clusters(features, cluster_labels, prompts, use_umap=True):
    if use_umap:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    features_reduced = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=cluster_labels, cmap='viridis')

    for i, _ in enumerate(prompts):
        plt.text(features_reduced[i, 0], features_reduced[i, 1], str(i + 1), fontsize=9, ha='center', va='center')

    plt.title("Feature Clusters with Line Numbers as Labels")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

# Correlation matrix plot
def plot_correlation_matrix(features):
    # Compute the correlation matrix
    correlation_matrix = np.corrcoef(features, rowvar=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Features")
    plt.show()

# Analyze the clusters and generate a report
def analyze_clusters(features, cluster_labels, prompts, top_k=5):
    cluster_info = {}

    for cluster in np.unique(cluster_labels):
        # Get the indices of the texts that belong to this cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_samples = [prompts[i] for i in cluster_indices]
        
        # Sort the features within the cluster and get the top k
        cluster_features = features[cluster_indices]
        cluster_mean = np.mean(cluster_features, axis=0)
        top_indices = np.argsort(cluster_mean)[::-1][:top_k]

        cluster_info[cluster] = {
            'cluster_samples': cluster_samples,
            'top_features': top_indices,
            'mean_feature': cluster_mean
        }

    return cluster_info

# Generate a textual report of the cluster analysis
def generate_cluster_report(cluster_info):
    report = []

    for cluster, info in cluster_info.items():
        report.append(f"\n=== Cluster {cluster} ===")
        report.append(f"Top Features: {info['top_features']}")
        report.append(f"Mean Feature Vector: {info['mean_feature'][:10]}...")  # Displaying top 10 features for brevity
        report.append(f"Text Samples:")
        
        for sample in info['cluster_samples'][:5]:  # Show top 5 text samples
            report.append(f"  - {sample}")

    return "\n".join(report)