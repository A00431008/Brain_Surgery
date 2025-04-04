from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def cluster_features(features, num_clusters=5):
    # Clusters features using KMeans and returns cluster labels and cluster centers.
    # Fit the KMeans model to cluster the features
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # Return the cluster labels for each feature and the cluster centers
    return cluster_labels, kmeans.cluster_centers_

def plot_clusters(features, cluster_labels):
    # Reduce the dimensionality of features for visualization and plot the clustered data.
    # Uses PCA or t-SNE for dimensionality reduction.
    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    features_reduced = pca.fit_transform(features)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=cluster_labels, cmap='viridis')
    
    plt.title("Feature Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # Show legend for clusters
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

def visualize_tsne(features, cluster_labels):
    # Use t-SNE to reduce the dimensionality of features and plot the result.
    tsne = TSNE(n_components=2, random_state=42)
    features_reduced = tsne.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=cluster_labels, cmap='viridis')
    
    plt.title("t-SNE Clustering of Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Show legend for clusters
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()