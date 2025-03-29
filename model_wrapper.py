# This is the model wrapper class that handles text generation with activation collection
# Plan is to start with gpt-2 and then swap with larger model later on
# ======================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelWrapper:
    def __init__(self, model_name="gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.activations = []
        self.hook = None
    
    def hook_fn(self, module, input, output):
        # Method to Intercept forward pass and activations
        self.activations.append(output.detach().cpu().numpy())

    def add_hooks(self, layer_idx=6):
        # Attach hook to specific transformer layer
        self.hook = self.model.transformer.h[layer_idx].register_forward_hook(self.hook_fn)

    def remove_hooks(self):
        # Remote hook if it exists
        if self.hook:
            self.hook.remove()
    
    def generate_text(self, prompt, max_length=50):
        # Method to Generate Text and return it along with activations
        self.activations.clear()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(output[0]), self.activations









