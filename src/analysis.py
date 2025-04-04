import torch
import numpy as np
import os
from autoencoder import SparseAutoencoder

class AutoencoderAnalysis:
    def __init__(self, hidden_dim_ratio=2, top_k=10):
        self.hidden_dim_ratio = hidden_dim_ratio
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
        
        self.prompts, self.X_tensor = self._load_data()
        self.autoencoder = self._load_autoencoder()
    
    def _load_data(self):
        """Loads activation data and prompts."""
        prompts_file = os.path.join(self.data_dir, "prompts.txt")
        activations_file = os.path.join(self.data_dir, "activations.npz")

        with open(prompts_file, "r") as f:
            prompts = [line.strip() for line in f.readlines()]

        activations_data = np.load(activations_file, allow_pickle=True)['activations']
        X = np.array([np.concatenate([v.flatten() for v in act.values()]) for act in activations_data])
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        return prompts, X_tensor

    def _load_autoencoder(self):
        """Loads the trained Sparse Autoencoder model."""
        input_dim = self.X_tensor.shape[1]
        autoencoder = SparseAutoencoder(input_dim=input_dim, hidden_dim_ratio=self.hidden_dim_ratio)
        autoencoder.load_state_dict(torch.load(os.path.join(self.data_dir, "autoencoder.pth"), map_location=self.device))
        autoencoder.to(self.device)
        autoencoder.eval()
        return autoencoder

    def analyze_neurons(self, neuron_indices=[0, 7, 42, 133, 256]):
        """Analyzes top-k activating text snippets for given neurons."""
        os.makedirs("feature_analysis", exist_ok=True)

        with torch.no_grad():
            _, encoded = self.autoencoder(self.X_tensor)  # Encoded shape: (num_samples, hidden_dim)

        encoded_np = encoded.cpu().numpy()

        for neuron_idx in neuron_indices:
            activations_for_neuron = encoded_np[:, neuron_idx]
            top_indices = activations_for_neuron.argsort()[-self.top_k:][::-1]

            print(f"\n🔍 Top {self.top_k} prompts for neuron {neuron_idx}:\n")
            with open(f"feature_analysis/feature_{neuron_idx}.txt", "w") as f:
                f.write(f"Neuron {neuron_idx} — Top {self.top_k} prompts:\n\n")
                for rank, idx in enumerate(top_indices):
                    prompt = self.prompts[idx]
                    activation_val = activations_for_neuron[idx]
                    print(f"{rank+1}. ({activation_val:.4f}) {prompt}")
                    f.write(f"{rank+1}. ({activation_val:.4f}) {prompt}\n")
