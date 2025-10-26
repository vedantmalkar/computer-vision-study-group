import numpy as np 

def sgd_step(W, dW, learning_rate):
    W -= learning_rate * dW      #updates Weights based on gradient
    return W

def create_batches(X, Y, batch_size):
    N = X.shape[0]
    
    indices = np.random.permutation(N)  #random shuffling
    
    for i in range(0, N, batch_size):           #creating batches randomly in sets of batch_size
        batch_indices = indices[i:i + batch_size]
        yield X[batch_indices], Y[batch_indices]        # allows for 1 batch to be stored at a time
