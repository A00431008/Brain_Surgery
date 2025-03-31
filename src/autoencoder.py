import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, l1_lambda=1e-5):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def compute_loss(self, x, decoded, encoded):
        # Compute MSE Loss (Reconstruction Loss)
        reconstruction_loss = nn.MSELoss()(decoded, x)
        # Compute L1 Regularization Loss (Sparsity)
        l1_loss = self.l1_lambda * torch.norm(encoded, 1)
        # Total Loss = Reconstruction Loss + Sparsity Penalty
        return reconstruction_loss + l1_loss

    def train(self, data_path="activations.npz", epochs=10, lr=0.001):
        # Load activations data
        data = np.load(data_path, allow_pickle=True)['activations']
        X = np.array([np.concatenate([v.flatten() for v in act.values()]) for act in data])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X.shape[1]  # Feature size (activation dimension)
        
        # Move model to the appropriate device (CPU or GPU)
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Convert the activations data to a torch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        # Train the autoencoder
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Forward pass
            reconstructed, encoded = self(X_tensor)
            # Compute the loss
            loss = self.compute_loss(X_tensor, reconstructed, encoded)
            # Backpropagate the loss
            loss.backward()
            # Update model parameters
            optimizer.step()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # Save the trained model
        torch.save(self.state_dict(), "autoencoder.pth")
        print("Autoencoder model saved!")

