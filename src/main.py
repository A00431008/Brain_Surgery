from model_wrapper import GPT2WrapperCPU
from model_wrapper import GPT2WrapperGPU
from data_generator import DataGenerator
from autoencoder import SparseAutoencoder
from feature_interpretability import cluster_features, plot_clusters, plot_correlation_matrix, analyze_clusters, generate_cluster_report
from analysis import ActivationAnalyzer

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_prompts_from_file(file_path):
    """Loads prompts from a text file."""
    try:
        with open(file_path, "r") as file:
            prompts = file.readlines()
        prompts = [prompt.strip() for prompt in prompts]  # Remove any extra whitespace/newlines
        return prompts
    
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

# Check if GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"  # Define the device here

# Select the correct wrapper class based on the device
if device == "cuda":
    wrapper = GPT2WrapperGPU(device=device) 
else:
    wrapper = GPT2WrapperCPU(device=device)

# Load prompts from file
file_name = "prompts.txt"
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
prompts_file_path = os.path.join(data_dir, file_name)
prompts = load_prompts_from_file(prompts_file_path)

# Data Generator for generation of training data
data_generator = DataGenerator(model_wrapper=wrapper, prompts=prompts)
data_generator.generate_data()

data_generator.save_data(data_dir)
data_generator.save_prompts_and_responses(data_dir)

data = data_generator.get_data()

# Test generated Data
# print("Generated Data:", data)

# === Load Activation Data ===
data_path = os.path.join(data_dir, "activations.npz")

# Load activations and determine input dimension
data = np.load(data_path, allow_pickle=True)['activations']
X = np.array([np.concatenate([v.flatten() for v in act.values()]) for act in data])
input_dim = X.shape[1]  # Number of features

# === Initialize Autoencoder ===
autoencoder = SparseAutoencoder(input_dim=input_dim, hidden_dim_ratio=2, l1_lambda=1e-5)

# === Train Autoencoder ===
autoencoder.train(data_path=data_path, epochs=50, lr=0.001, patience=5)

# === Plot Training Loss ===
loss_history = np.load("loss_history.npy")

plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Sparse Autoencoder Training Loss")
plt.legend()
plt.grid()
plt.show()


# === Apply Clustering with the Chosen Number of Clusters ===
num_clusters = 9  # Fixed number of clusters
cluster_labels, cluster_centers = cluster_features(X, num_clusters=num_clusters)

# === Visualize Clusters using UMAP ===
plot_clusters(X, cluster_labels, prompts, use_umap=True)

# === Visualize Correlation Matrix of Features ===
plot_correlation_matrix(X)

# === Analyze and generate report for clusters ===
cluster_info = analyze_clusters(X, cluster_labels, prompts)

# === Generate and print the cluster report ===
report = generate_cluster_report(cluster_info)
print(report)


# Analyze the activations 
# ==============================================================================
analyzer = ActivationAnalyzer(top_k=5, batch_size=64)
model_path = "best_autoencoder.pth"
analyzer.load_data(data_path=data_path, model_path=model_path, device=device)

# Encode the data
analyzer.encode_data()

# Analyze latent features
analyzer.analyze_latent_features()