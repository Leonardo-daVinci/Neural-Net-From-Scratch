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

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


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


# layer1 = LayerDense(4, 5)
# # here output of layer 1 will be (x,5), so we need to have layer 2 of the shape (5,y)
# layer2 = LayerDense(5, 2)
# layer1.forward(X)
# # print(layer1.output)
# layer2.forward(layer1.output)
# # print(layer2.output)
X, y = spiral_data(100, 3)
# layer1 = LayerDense(2, 5)
# activation1 = ReLU()
# layer1.forward(X)
# # print(layer1.output)
#
# activation1.forward(layer1.output)
# print(activation1.output)

dense1 = LayerDense(2, 3)
activation1 = ReLU()

dense2 = LayerDense(3, 3)
activation2 = Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = CategoricalCrossEntropyLoss()
loss = loss_function.calculate(activation2.output, y)
print("Loss", loss)
