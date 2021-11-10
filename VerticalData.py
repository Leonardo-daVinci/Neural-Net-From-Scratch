import nnfs
import numpy as np
from nnfs.datasets import vertical_data

from LayerDense import LayerDense, ReLU, CategoricalCrossEntropyLoss

nnfs.init()

# create dataset
X, y = vertical_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()

# create model
dense1 = LayerDense(2, 3)
activation1 = ReLU()
dense2 = LayerDense(3, 3)
activation2 = ReLU()

# loss function
loss_function = CategoricalCrossEntropyLoss()

# helper variables
lowest_loss = np.inf
best_dense1_weights = dense1.weights.copy()
best_dense2_weights = dense2.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_biases = dense2.biases.copy()

# using random values and tweaking every iteration
for iteration in range(10000):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print("New parameters found at iteration: ", iteration)
        best_dense1_weights = dense1.weights.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

    else:
        dense1.weights = best_dense1_weights.copy()
        dense2.weights = best_dense2_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.biases = best_dense2_biases.copy()
