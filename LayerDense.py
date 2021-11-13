import numpy as np

np.random.seed(0)

# inputs are generally referred to as X
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # weights are in the shape of features x number of neurons - this avoids taking transpose in forward pass
        # we get a gaussian distribution using randn but we want the range of weights to be -1 to 1, we multiply by 0.10
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        # biases are in the shape of 1 x number of neurons
        self.biases = np.zeros((1, n_neurons))

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs  # we need to remember inputs for calculating gradients
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ReLU:
    # forward pass
    def forward(self, inputs):
        self.inputs = inputs  # used for gradient calculation
        self.output = np.maximum(0, inputs)

    # backward pass
    def backward(self, dvalues):
        self.dinputs = self.inputs.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # backward pass
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)  # This will become the resulting gradient array

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        batch_loss = np.mean(sample_losses)
        return batch_loss


class CategoricalCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # handling type of class targets
        if len(y_true.shape) == 1:
            # scalar class values
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            # one-hot encoded class values
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # backward pass for categorical cross-entropy is just (- y_true) / y_pred
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # there are multiple labels in each sample. hence we are using first one to count
        labels = len(dvalues[0])

        # for sparse labels, we need to convert them into one-hot vectors
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            # note that if the index of row is i then the ith place has one in eye matrix

        # calculating the gradient
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        # normalization is required so that the gradients don't explode


# we create this class because calculations get simpler if we have the combo below
class ActivationSoftmaxLossCategoricalCrossEntropy:

    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropyLoss()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # changing one-hot encoded labels to discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


if __name__ == "__main__":
    X, y = spiral_data(100, 3)

    dense1 = LayerDense(2, 3)
    activation1 = ReLU()

    dense2 = LayerDense(3, 3)
    loss_activation = ActivationSoftmaxLossCategoricalCrossEntropy()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    print(loss_activation.output[:5])
    print(f"Loss is {loss}")

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)
    print(f"Accuracy is {accuracy}")

    # backward pass
    # here we use the output of final layer as dvalues
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
