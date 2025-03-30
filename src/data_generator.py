# Class to generate training data for the autoencoder

import numpy as np
import json

class DataGenerator:
    def __init__(self, model_wrapper, prompts):
        # Initializes data generator with model wrapper and list of prompts
        self.model_wrapper = model_wrapper
        self.prompts = prompts
        self.data = []

    def generate_data(self):
        # Method to iteratively work through a corpus of prompts, 
        # calling the model to generate text and activations for each prompt
        for prompt in self.prompts:
            generated_text, activations = self.model_wrapper.generate_text(prompt)
            self.data.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "activations": activations
            })
        
    def save_data(self, filename="activations.npz"):
        # Method to save the outputs in a format that can be input to the autoencoder for training
        prompts = [entry["prompt"] for entry in self.data]
        texts = [entry["generated_text"] for entry in self.data]
        activations = [entry["activations"] for entry in self.data]

        # use numpy's savez_compressed() to save the prompts, texts and activations as a .npz file
        np.savez_compressed(filename, prompts=prompts, texts=texts, activations=activations)

        # Same the data as json as well
        json_filename = filename.replace(".npz", ".json")
        with open(json_filename, "w") as f:
            json.dump(self.data, f, indent=4)

        print(f"Data saved to {filename} and {json_filename} successfully")