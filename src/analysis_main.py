# SEPARATE ANALYSIS DRIVER TO TEST THE ANALYZER

from model_wrapper import GPT2WrapperCPU
from model_wrapper import GPT2WrapperGPU
# from model_wrapper import LlamaWrapperCPU
# from model_wrapper import LlamaWrapperGPU 
from data_generator import DataGenerator
from autoencoder import SparseAutoencoder
from feature_interpretability import cluster_features, plot_clusters, plot_elbow_method
from analysis import ActivationAnalyzer

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"  # Define the device here

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
data_path = os.path.join(data_dir, "activations.npz")


# Analyze the activations 
# ==============================================================================
analyzer = ActivationAnalyzer(top_k=5, batch_size=64)
model_path = "best_autoencoder.pth"
analyzer.load_data(data_path=data_path, model_path=model_path, device=device)

# Encode the data
analyzer.encode_data()

# Analyze latent features
analyzer.analyze_latent_features()