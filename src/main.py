from model_wrapper import GPT2Wrapper
from data_generator import DataGenerator
import os
import torch

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
device = "cuda" if torch.cuda.is_available() else "cpu"
wrapper = GPT2Wrapper(device=device)

# Load prompts from file
file_name = "prompts.txt"
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
prompts_file_path = os.path.join(data_dir, file_name)
prompts = load_prompts_from_file(prompts_file_path)

# Data Generator for generation of training data
data_generator = DataGenerator(model_wrapper=wrapper, prompts=prompts)
data_generator.generate_data()
data_generator.save_data(data_dir)
data = data_generator.get_data()

# Test generated Data
print("Generated Text:", data)

