from model_wrapper import GPT2Wrapper
from autoencoder import SparseAutoencoder

import torch
from torch import optim

def train_autoencoder(data, input_dim, hidden_dim, epochs=50, lr=0.001, l1_lambda=1e-5):
    model = SparseAutoencoder(input_dim, hidden_dim, l1_lambda)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        decoded, encoded = model(data_tensor)
        loss = model.compute_loss(data_tensor, decoded, encoded)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
    
    return model

# Main Function
def main():
    wrapper = GPT2Wrapper(device="cpu")
    prompt = "Once upon a time"
    
    print("Generating text...")
    text, activations = wrapper.generate_text(prompt)

    print("Generated Text:", text)
    print("Captured Activations:", len(activations), "layers")
    
    # Check for activations in layer_0 to avoid errors
    if 'layer_0' in activations:
        print("Example Activation Shape:", activations['layer_0'].shape)
    else:
        print("No activations captured from layer_0.")

if __name__ == "__main__":
    main()