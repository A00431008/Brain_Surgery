from model_wrapper import GPT2Wrapper
from data_generator import DataGenerator
import os
import torch

def load_prompts_from_file(filename="../data/prompts.txt"):
    """Loads prompts from a text file."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, filename)
        
        with open(file_path, "r") as file:
            prompts = file.readlines()
        prompts = [prompt.strip() for prompt in prompts]  # Remove any extra whitespace/newlines
        return prompts
    
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return []

# Check if GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
wrapper = GPT2Wrapper(device=device)

# Load prompts from file
prompts = load_prompts_from_file('../data/prompts.txt')

print("Generating text...")
text, activations = wrapper.generate_text(prompts)

print("Generated Text:", text)
print("Captured Activations:", len(activations), "layers")
    
# Check for activations in middle layer to avoid errors
if 'layer_6' in activations:
    print("Example Activation Shape:", activations['layer_6'].shape)
else:
    print("No activations captured from layer_6.")
