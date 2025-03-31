# =============================
# Feature Analysis
# =============================
def analyze_features(model_path="autoencoder.pth", data_path="activations.npz", num_features=5):
    data = np.load(data_path, allow_pickle=True)
    activations = data['activations']
    prompts = data['prompts']

    input_dim = sum(act.flatten().shape[0] for act in activations[0].values())
    model = SparseAutoencoder(input_dim, hidden_dim=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X = np.array([np.concatenate([v.flatten() for v in act.values()]) for act in activations])
    X_tensor = torch.tensor(X, dtype=torch.float32)

    _, encoded = model(X_tensor)
    encoded = encoded.detach().numpy()

    top_features = np.argsort(-np.abs(encoded), axis=0)[:num_features]

    for i in range(num_features):
        print(f"Feature {i}:")
        for j in top_features[:, i]:
            print(f" - {prompts[j]}")
        print("-" * 30)
