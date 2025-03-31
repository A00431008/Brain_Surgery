import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_lambda=1e-5):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.l1_lambda = l1_lambda
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def compute_loss(self, x, decoded, encoded):
        reconstruction_loss = nn.MSELoss()(decoded, x)
        l1_loss = self.l1_lambda * torch.norm(encoded, 1)
        return reconstruction_loss + l1_loss


