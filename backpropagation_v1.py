import numpy as np
from sklearn import datasets
import sklearn


def sigmoid(x, dot= False):
    if dot == True:
        return x*(1-x)
    return 1.0/(1.0 + np.exp(-x))

def tanh(x, dot = False):
    if dot == True:
        return 1.0 - x**2
    return np.tanh(x)

def relu(x, dot = False):
    if dot == True:
        x[x<0] = 0;
        x[x>=0] = 1;
        return x
    return np.maximum(x,  0)


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, activation):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = activation(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, activation, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)



# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, activation, num_passes=40000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}
    import random
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
        # p= [random.randint(0,(X.shape[0]-1)) for k in range(100)]
        x = X
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = activation(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        delta2 = delta3.dot(W2.T) * activation(a1, dot = True)
        # Weight two deltas
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        # Weight one deltas
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)


        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, calculate_loss(model, activation))

    return model

if __name__ == '__main__':
    from sklearn import datasets, linear_model


    # Generate a dataset and plot it
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)


    num_examples = len(X) # training set size
    nn_input_dim = 2 # input layer dimensionality
    nn_output_dim = 2 # output layer dimensionality
    act = sigmoid
    # Gradient descent parameters (I picked these by hand)
    epsilon = 1e-02 # learning rate for gradient descent
    reg_lambda = 0.01 # regularization strength


    # Build a model with a 3-dimensional hidden layer
    model = build_model(3, act, print_loss=True)


    y_hat = []
    for e in X:
        y_hat.append( predict(model, act,  e))

    from sklearn.metrics import accuracy_score
    print (accuracy_score(y, y_hat, normalize=True, sample_weight=None))
