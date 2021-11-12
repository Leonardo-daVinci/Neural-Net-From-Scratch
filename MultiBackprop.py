import numpy as np

# we consider single hidden layer with 3 neurons
# these are the gradient values received from next layer
d_values = np.array([[1.0, 1.0, 1.0]])
batch_dvalues = np.array([[1., 1., 1.],
                          [2., 2., 2.],
                          [3., 3., 3.]])

# we have 4 inputs
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# we have 3 sets of inputs
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# one bias per neuron
biases = np.array([2, 3, 0.5])

# let layer outputs be z
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])

if __name__ == "__main__":
    # sum the weights of the input and multiply by gradient values
    dx0 = sum(weights[0] * d_values[0])
    dx1 = sum(weights[1] * d_values[0])
    dx2 = sum(weights[2] * d_values[0])
    dx3 = sum(weights[3] * d_values[0])

    dinputs = np.array([dx0, dx1, dx2, dx3])
    print(dinputs)

    # we can also perform the same operation as follows
    dinputs = np.dot(d_values, weights.T)
    print(dinputs)

    # for a batch of inputs
    batch_dinputs = np.dot(batch_dvalues, weights.T)
    print(batch_dinputs)

    # similarly calculating dweights
    # here we basically sum the inputs for each weight then multiply by input gradient
    dweights = np.dot(inputs.T, batch_dvalues)
    print(dweights)

    # for biases, derivative is simple sum of each d_value
    dbiases = np.sum(batch_dvalues, axis=0, keepdims=True)
    print(dbiases)

    # for reLU function. the derivative is just same value if its positive otherwise 0
    # thus d relu will be dvalues with negative values as zero

    new_dvalues = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]])
    drelu = new_dvalues.copy()
    drelu[z <= 0] = 0
    print(drelu)
