import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_ratio=2, l1_lambda=1e-5):
        super(SparseAutoencoder, self).__init__()

        hidden_dim = input_dim * hidden_dim_ratio  # Make encoder 2-4x larger
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)

        # Decoder with tied weights
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.decoder.weight = nn.Parameter(self.encoder.weight.T)  # Tied weights

        self.l1_lambda = l1_lambda

    def forward(self, x):
        # Forward pass through encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def compute_loss(self, x, decoded, encoded):
        # Reconstruction loss (Mean Squared Error)
        reconstruction_loss = nn.MSELoss()(decoded, x)
        
        # L1 Regularization to enforce sparsity
        l1_loss = self.l1_lambda * torch.norm(encoded, 1)
        
        # Total Loss: Reconstruction Loss + Sparsity Penalty (L1 Regularization)
        return reconstruction_loss + l1_loss

    def train(self, 
        #       data_path=os.path.join(os.path.abspath(__file__), "../data/activations.npz"), 
        #       epochs=10, lr=0.001, patience=5, max_norm=1.0):
        # """Train the Sparse Autoencoder with early stopping and gradient clipping"""
                data_path="data/activations.npz", 
          epochs=10, lr=0.001, patience=5, max_norm=1.0):
        """Train the Sparse Autoencoder with early stopping and gradient clipping"""
     
     # Check if data_path is valid
        if not isinstance(data_path, str) or not os.path.exists(data_path):
         print(f"Invalid data_path: {data_path}")
         return
        
        # Load activations data
        data = np.load(data_path, allow_pickle=True)['activations']
        X = np.array([np.concatenate([v.flatten() for v in act.values()]) for act in data])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X.shape[1]
        self.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Convert the activations data to a torch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device) 

        # Early Stopping setup
        best_loss = float('inf')
        epochs_no_improvement = 0
        loss_history = []

        best_model_path = "best_autoencoder.pth"

        # Train the autoencoder
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, encoded = self(X_tensor)
            
            # Compute the Loss
            loss = self.compute_loss(X_tensor, reconstructed, encoded)

            # Backpropagation
            loss.backward()

            # Update model parameters
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
            optimizer.step()

            
            loss_value = loss.item()
            loss_history.append(loss_value)

            # Early stopping check: if loss hasn't improved for 'patience' epochs, stop
            if loss_value < best_loss:
                best_loss = loss_value
                epochs_no_improvement = 0
                torch.save(self.state_dict(), best_model_path)  # Save best model
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best loss: {best_loss:.4f}")
                break

            # Print training progress
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_value:.4f}")

        # Save final model
        torch.save(self.state_dict(), "autoencoder.pth")
        print("Training complete. Autoencoder model saved.")

        # Save loss history for visualization
        np.save("loss_history.npy", np.array(loss_history))
        print("Loss history saved!")


# class SparseAutoencoderDup(nn.Module):
#     def __init__(self, input_dim, hidden_dim=512, l1_lambda=1e-5):
#         super(SparseAutoencoder, self).__init__()

#         # Encoder and Decoder with tied weights
#         self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)  # Encoder
#         self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)  # Decoder
#         self.l1_lambda = l1_lambda

#     def forward(self, x):
#         # Forward pass through encoder and decoder
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded, encoded

#     def compute_loss(self, x, decoded, encoded):
#         # Reconstruction loss (Mean Squared Error)
#         reconstruction_loss = nn.MSELoss()(decoded, x)
        
#         # L1 Regularization to enforce sparsity
#         l1_loss = self.l1_lambda * torch.norm(encoded, 1)
        
#         # Total Loss: Reconstruction Loss + Sparsity Penalty (L1 Regularization)
#         return reconstruction_loss + l1_loss

#     def train(self, 
#               data_path=os.path.join(os.path.abspath(__file__), "../data/activations.npz"), 
#               epochs=10, lr=0.001, patience=5):
#         """
#         Train the Sparse Autoencoder with early stopping
#         - `patience` defines how many epochs to wait for improvement before stopping
#         """
#         # Load activations data
#         data = np.load(data_path, allow_pickle=True)['activations']
#         X = np.array([np.concatenate([v.flatten() for v in act.values()]) for act in data])

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         input_dim = X.shape[1]  # Feature size (activation dimension)
        
#         # Move model to the appropriate device (CPU or GPU)
#         # self.to(device)
        
#         optimizer = optim.Adam(self.parameters(), lr=lr)
        
#         # Convert the activations data to a torch tensor
#         X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

#         # Early Stopping setup
#         best_loss = float('inf')
#         epochs_no_improvement = 0
        
#         # Train the autoencoder
#         for epoch in range(epochs):
#             optimizer.zero_grad()
#             # Forward pass
#             reconstructed, encoded = self(X_tensor)
            
#             # Compute the loss
#             loss = self.compute_loss(X_tensor, reconstructed, encoded)
            
#             # Backpropagation
#             loss.backward()
            
#             # Update model parameters
#             optimizer.step()

#             # Early stopping check: if loss hasn't improved for 'patience' epochs, stop
#             if loss.item() < best_loss:
#                 best_loss = loss.item()
#                 epochs_no_improvement = 0  # Reset counter if improvement
#             else:
#                 epochs_no_improvement += 1

#             if epochs_no_improvement >= patience:
#                 print(f"Early stopping triggered at epoch {epoch + 1}.")
#                 break

#             # Print training progress
#             print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        
#         # Save the trained model
#         torch.save(self.state_dict(), "autoencoder.pth")
#         print("Autoencoder model saved!")
