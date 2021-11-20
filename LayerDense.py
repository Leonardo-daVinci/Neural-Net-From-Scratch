import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()


class LayerDense:
    def __init__(self, n_inputs, n_neurons, L1_weight=0., L2_weight=0., L1_bias=0., L2_bias=0.):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.L1_weight = L1_weight
        self.L2_weight = L2_weight
        self.L1_bias = L1_bias
        self.L2_bias = L2_bias

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs  # we need to remember inputs for calculating gradients
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass
    def backward(self, dvalues):
        # gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # using L1 and L2 regularization
        # derivative of L1 is just 1 if parameter is positive else -1
        # derivative of L2 is 2 * parameter
        if self.L1_weight > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.L1_weight * dL1

        if self.L1_bias > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.L1_bias * dL1

        if self.L2_weight > 0:
            self.dweights += 2 * self.L2_weight * self.weights

        if self.L2_bias > 0:
            self.dbiases += 2 * self.L2_bias * self.biases

        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class ReLU:
    # forward pass
    def forward(self, inputs):
        self.inputs = inputs  # used for gradient calculation
        self.output = np.maximum(0, inputs)

    # backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
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
        data_loss = np.mean(sample_losses)
        return data_loss


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


class SGD:

    # adding momentum to SGD
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # if we're using momentum
        if self.momentum:

            # if layer doesn't not have momentum arrays, create them as zero arrays
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # update to parameters is learning rate * param gradients
            # we use momentum which is a fraction of the update done to the parameter
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class AdaGrad:

    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # to keep the parameters from growing too fast, we do the following
        # cache += param_gradient ** 2
        # param_updates = learning rate * param_gradient / (sqrt(cache) + epsilon)
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        # this may make learning very slow hence it is seldom used

    def post_update_params(self):
        self.iterations += 1


class RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # just like AdaGrad, we have a following ways to control parameters
        # cache = rho * cache + (1- rho) * gradient ** 2
        # updating the params is same as AdaGrad
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# Adam incorporates RMSprop along with momentum in SGD
class Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, momentum=0.9, rho=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.momentum = momentum
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # If layer does not contain cache arrays and momentum arrays
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.momentum * layer.weight_momentums + (1 - self.momentum) * layer.dweights
        layer.bias_momentums = self.momentum * layer.bias_momentums + (1 - self.momentum) * layer.dbiases

        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.momentum ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.momentum ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.rho ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.rho ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += - self.current_learning_rate * weight_momentums_corrected / (
                    np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += - self.current_learning_rate * bias_momentums_corrected / (
                    np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


if __name__ == "__main__":
    X, y = spiral_data(samples=1000, classes=3)

    dense1 = LayerDense(2, 64, L2_weight=5e-4, L2_bias=5e-4)
    activation1 = ReLU()

    dense2 = LayerDense(64, 3)
    loss_activation = ActivationSoftmaxLossCategoricalCrossEntropy()

    # using RMSprop optimizer
    optimizer = Adam(learning_rate=0.05, decay=5e-7)

    for epoch in range(10001):

        dense1.forward(X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(
                f"Epoch is {epoch} Accuracy is {accuracy:.3f} and loss is {loss:.3f} with learning rate {optimizer.current_learning_rate}")

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # update the parameters
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    # validating the model
    X_test, y_test = spiral_data(samples=100, classes=3)

    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_test)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)

    print(f"Validation accuracy is : {accuracy:.3f} and loss is : {loss:.3f}")
