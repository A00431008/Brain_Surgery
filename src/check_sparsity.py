import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from autoencoder import SparseAutoencoder

# --- Load activations ---
data_path = os.path.join("data", "activations.npz")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find activation file at {data_path}")

data = np.load(data_path, allow_pickle=True)['activations']
X = np.array([np.concatenate([v.flatten() for v in act.values()]) for act in data])

# --- Set up model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X.shape[1]

model = SparseAutoencoder(input_dim=input_dim)
model.load_state_dict(torch.load("best_autoencoder.pth", map_location=device))
model.to(device)
model.eval()

# --- Convert X to tensor ---
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# --- Get encoded activations ---
with torch.no_grad():
    _, encoded = model(X_tensor)

encoded_np = encoded.cpu().numpy().flatten()

# --- Plot histogram ---
plt.figure(figsize=(10, 6))
plt.hist(encoded_np, bins=100, color='darkorange', alpha=0.85)
plt.title("Histogram of Encoder Activations")
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
