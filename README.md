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
