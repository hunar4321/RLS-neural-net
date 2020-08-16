# RLS_Learning
Using Recursive Leasting Squares (RLS) with Neural Network for one-shot learning

**Advantages of using RLS for learning instead of gradient descent**
1. Fast learning, sample efficient can learn on one-shot
2. Online Learning (suitable for real time learning)
3. No worries about local minima

**Disadvantages:**
1. Computationally inefficient if the size of the input is big.
2. Sensitive of overflow and underflow of the estimated value and this can lead to unstability in some cases
3. The current implementation works with a single hidden unit neural network. It is not clear if adding more layers will be useful since learning only happens in the last layer 

Currently we are working to overcome the limitations above by using a novel algorthim we call it **Predictive Hebbian Unified Neurons** or PHUN which shows promssing results. For more info about PHUN visit us at: https://www.brainxyz.com/ 
