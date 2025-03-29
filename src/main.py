from model_wrapper import GPT2Wrapper

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