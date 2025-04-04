import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

class GPT2Wrapper:
    def __init__(self, model_name="gpt2", device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Fix pad token issue
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.activations = {}

    def _activation_hook(self, layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]  # Extract hidden state if tuple
            if isinstance(output, torch.Tensor):
                self.activations[layer_name] = output.detach().cpu()
            else:
                print(f"Warning: Output at {layer_name} is not a tensor. Type: {type(output)}")
        return hook
    
    def register_hooks(self):
        # Register hook on the middle layer
        middle_layer_idx = len(self.model.transformer.h) // 2
        block = self.model.transformer.h[middle_layer_idx]
        block.register_forward_hook(self._activation_hook(f"layer_{middle_layer_idx}"))
        print(f"Hook registered for middle layer_{middle_layer_idx}")  # Print confirmation for the middle layer hook

        # for i, block in enumerate(self.model.transformer.h):
        #     block.register_forward_hook(self._activation_hook(f"layer_{i}"))
        #     print(f"Hook registered for layer_{i}")  # Print confirmation for each hook registration

    def generate_text(self, prompt, max_length=50):
        self.activations = {}  # Reset activations
        self.register_hooks()

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model.generate(inputs.input_ids, max_length=max_length)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Print activations for the middle layer
        middle_layer_idx = len(self.model.transformer.h) // 2
        if f"layer_{middle_layer_idx}" in self.activations:
            print(f"Activations for layer_{middle_layer_idx} (Shape: {self.activations[f'layer_{middle_layer_idx}'].shape}):")
            print(self.activations[f'layer_{middle_layer_idx}'][0, :5, :5])  # Print first 5 tokens, first 5 features for the first token
        else:
            print(f"No activations captured for layer_{middle_layer_idx}")

        # # Print activations for every layer
        # for i in range(len(self.model.transformer.h)):
        #     if f"layer_{i}" in self.activations:
        #         print(f"Activations for layer_{i} (Shape: {self.activations[f'layer_{i}'].shape}):")
        #         print(self.activations[f'layer_{i}'][0, :5, :5])  # Print first 5 tokens, first 5 features for the first token
        #     else:
        #         print(f"No activations captured for layer_{i}")

        # Visualization: Plot activations for the middle layer
        # if f"layer_{middle_layer_idx}" in self.activations:
        #     activation_tensor = self.activations[f'layer_{middle_layer_idx}']
        #     activations_to_plot = activation_tensor[0, :5, :10].detach().cpu().numpy()  # First 5 tokens, first 10 features
        #     plt.imshow(activations_to_plot, cmap='viridis', aspect='auto')
        #     plt.colorbar()
        #     plt.title(f'Activations for Middle Layer {middle_layer_idx}')
        #     plt.show()

        # # Visualization: Plot activations for each layer
        # for i in range(len(self.model.transformer.h)):
        #     activation_tensor = self.activations[f'layer_{i}']
        #     activations_to_plot = activation_tensor[0, :5, :10].detach().cpu().numpy()  # First 5 tokens, first 10 features
        #     plt.imshow(activations_to_plot, cmap='viridis', aspect='auto')
        #     plt.colorbar()
        #     plt.title(f'Activations for Layer {i}')
        #     plt.show()

        return generated_text, self.activations







