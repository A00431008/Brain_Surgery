from model_wrapper import GPT2WrapperCPU
from model_wrapper import GPT2WrapperGPU
# from model_wrapper import LlamaWrapperCPU
# from model_wrapper import LlamaWrapperGPU 
from data_generator import DataGenerator
from autoencoder import SparseAutoencoder
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
    #wrapper = LlamaWrapperGPU(device=device)
else:
    wrapper = GPT2WrapperCPU(device=device)
    #wrapper = LlamaWrapperCPU(device=device)

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
print("Generated Data:", data)

# === Load Activation Data ===
npz_file = os.path.join(data_dir, "activations.npz")

# Load activations and determine input dimension
data = np.load(npz_file, allow_pickle=True)['activations']
X = np.array([np.concatenate([v.flatten() for v in act.values()]) for act in data])
input_dim = X.shape[1]  # Number of features

# === Initialize Autoencoder ===
autoencoder = SparseAutoencoder(input_dim=input_dim, hidden_dim_ratio=2, l1_lambda=1e-5)

# === Train Autoencoder ===
autoencoder.train(data_path=npz_file, epochs=50, lr=0.001, patience=5)

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