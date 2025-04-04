# Class to generate training data for the autoencoder

import numpy as np
import json
import csv
import os

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
        
    def save_data(self, filepath=os.path.join(os.path.abspath(__file__), "../data"), filename="activations.npz"):
        # Method to save the outputs in a format that can be input to the autoencoder for training
        prompts = [entry["prompt"] for entry in self.data]
        texts = [entry["generated_text"] for entry in self.data]
        activations = [entry["activations"] for entry in self.data]

        # use numpy's savez_compressed() to save the prompts, texts and activations as a .npz file
        np.savez_compressed(os.path.join(filepath, filename), prompts=prompts, texts=texts, activations=activations)

        # Same the data as json as well
        json_filename = filename.replace(".npz", ".json")
        
        # with open(os.path.join(filepath,json_filename), "w") as f:
        #     json.dump(self.data, f, indent=4)

        print(f"Data saved to {filename} and {json_filename} successfully")
    
    def get_data(self):
        return self.data
    

    def save_prompts_and_responses(self, filepath=os.path.join(os.path.abspath(__file__), "../data"), filename="prompts_and_responses.csv"):
        """Method to save prompts and generated text pairs to a CSV file."""
        prompts = [entry["prompt"] for entry in self.data]
        texts = [entry["generated_text"] for entry in self.data]

        # Save prompts and generated texts in a separate CSV file
        with open(os.path.join(filepath, filename), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Prompt", "Generated Text"])
            for prompt, generated_text in zip(prompts, texts):
                writer.writerow([prompt, generated_text])
        
        print(f"Prompts and generated responses saved to {filename} successfully")

    def get_data(self):
        return self.data