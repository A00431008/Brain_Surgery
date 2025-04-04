from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def cluster_features(features, num_clusters=8):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels, kmeans.cluster_centers_


def plot_clusters(features, cluster_labels, prompts):
    # Use t-SNE to reduce the dimensionality of features and plot the result.
    tsne = TSNE(n_components=2, random_state=42)
    features_reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))

    # Create scatter plot with points colored by cluster label
    scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=cluster_labels, cmap='viridis')

    # Add labels as the prompt number (line number)
    for i, _ in enumerate(prompts):
        plt.text(features_reduced[i, 0], features_reduced[i, 1], str(i+1), fontsize=9, ha='center', va='center')  # Label with line number

    plt.title("t-SNE Clustering of Features with Line Numbers as Labels")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

def plot_elbow_method(features, max_clusters=20):
    inertia = []
    
    # Loop through different cluster sizes
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
    
    # Plot the inertia
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters+1), inertia, marker='o')
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.grid(True)
    plt.show()