import torch
import numpy as np
from autoencoder import SparseAutoencoder

class ActivationAnalyzer:
    def __init__(self, top_k=5, batch_size=64):
        self.top_k = top_k
        self.batch_size = batch_size
        self.activations = []
        self.texts = []

    def load_data(self, data_path, model_path, device="cpu"):
        # Load data
        data = np.load(data_path, allow_pickle=True)
        self.activations = data["activations"]
        self.texts = data["texts"]
        
        # Prepare input
        self.X = torch.tensor(
            [np.concatenate([v.flatten() for v in act.values()]) for act in self.activations],
            dtype=torch.float32
        )
        self.input_dim = self.X.shape[1]
    
        # Load the trained autoencoder model
        self.autoencoder = SparseAutoencoder(input_dim=self.input_dim)
        self.autoencoder.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.autoencoder.eval()


    def encode_data(self):
        encoded = []
        with torch.no_grad():
            for start_idx in range(0, len(self.X), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.X))
                batch = self.X[start_idx:end_idx]
                _, batch_encoded = self.autoencoder(batch)
                encoded.append(batch_encoded)
        self.encoded_np = torch.cat(encoded, dim=0).numpy()

    def analyze_latent_features(self):
        # Analyze latent features
        num_features = self.encoded_np.shape[1]
        
        for feature_idx in range(min(num_features, 10)):  # Inspect the first 10 dimensions
            print(f"\n=== Latent Feature {feature_idx} ===")
            
            # Get top-k samples for this dimension
            top_indices = np.argsort(self.encoded_np[:, feature_idx])[::-1][:self.top_k]
            
            for i, idx in enumerate(top_indices):
                score = self.encoded_np[idx, feature_idx]
                print(f"\nSample #{i + 1} (Activation: {score:.4f}):")
                print(self.texts[idx])