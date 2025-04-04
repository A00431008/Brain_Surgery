import torch
import numpy as np
from autoencoder import SparseAutoencoder
import os

# === Load Data ===
data = np.load("data/activations.npz", allow_pickle=True)
print(data.files)  # This will print the keys in the .npz file
activations = data["activations"]
texts = data["texts"]  # Corresponding generated text snippets

# === Prepare Input ===
X = np.array([np.concatenate([v.flatten() for v in act.values()]) for act in activations])
input_dim = X.shape[1]

# === Load Trained Autoencoder ===
autoencoder = SparseAutoencoder(input_dim=input_dim)
autoencoder.load_state_dict(torch.load("best_autoencoder.pth", map_location=torch.device("cpu")))
print(autoencoder)

autoencoder.eval()

# === Encode Data ===
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32)
    _, encoded = autoencoder(X_tensor)

# === Analyze Latent Features ===
encoded_np = encoded.numpy()
num_features = encoded_np.shape[1]

top_k = 5  # Show top 5 strongest activations for each feature

for feature_idx in range(min(num_features, 10)):  # Inspect first 10 dimensions
    print(f"\n=== Latent Feature {feature_idx} ===")
    
    # Get top-k samples for this dimension
    top_indices = np.argsort(encoded_np[:, feature_idx])[::-1][:top_k]
    
    for i, idx in enumerate(top_indices):
        score = encoded_np[idx, feature_idx]
        print(f"\nSample #{i + 1} (Activation: {score:.4f}):")
        print(texts[idx])