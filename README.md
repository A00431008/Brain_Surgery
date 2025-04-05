# Brain_Surgery
MCDA5511 Assignment #4: Brain Surgery

## Submitted By
- Louise Fear
- Sudeep Raj Badal
- Mohammed Abdul Thoufiq
- Sukanta Dey Amit

## 1. Finding out the layer for insertion of Hooks

For finding the best layer for insertion of hooks, we first ran the wrapper class by inserting hooks in every single layer and checking the results for it.
While looking for the activations while hooks are inserted on each of the layers one by one, it was found that the largest values for number of activations were
found at layer 6 followed by layer 7. This showed that placing hooks on layer 6 (which is the middle layer out of the 12) gives us better activations and it is capturing
more complex relationships giving us more interesting and abstract features.

This also aligns well when put together with the idea from the `Scaling Monosemanticity paper` that placing the hooks at the middle layer firstly makes the autoencoder training and inference cheaper since the residual stream is smaller than MLP layer and focusing on the residual stream prevents cross-layer superposition. 

Cross-layer superposition is the idea that gradient descent isn't affected by the specific layer that it is implemented in so the features would be spread across the layers. Even though there is not a concrete solution for it available yet, focusing on the residual stream - which is sum of outputs of all previous layers - will add the activations of features represented in cross-layer superposition. Thigh might not fully solve the issue but this might be able to bypass the issue a little bit

Source: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#discussion-limitations

## Training Data Generation for Sparse Autoencoder

In this section, we built a class that generates training data for the sparse autoencoder. The autoencoder was trained on activations collected from a Language Model using a corpus of prompts which is iterated through to generate corresponding text and activation. It is stored in suitable format for autoencoder training. The specifications for the corpus of prompts are shows below. 

## Prompts Generation

For the training data, we selected a diverse set of **100 prompts** covering five key themes
- **Fitness & Health**
- **Technology & AI**
- **Society & Culture**
- **Environment & Sustainability**
- **Media & Entertainment**

These themes were chosen to ensure the prompts span a wide range of domains to test the broad basis for the model's activation analysis. These themes are common in natural language processing because of the commonality of the themes in real world documents available and use of language based on which LLM's are trained upon. These themes are also extremely rich in semantic content.

## Summary Statistics

- **Number of Prompts**: 100
- **Average Number of Words per Prompt**: 22 words
- **Total Number of Words Across Prompts**: 2,200 words
- **Themes Covered**: 5 
- **Most Frequent Words**: "health", "technology", "environment", "fitness", "culture"

