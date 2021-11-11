# let us try to use just a single neuron and try to minimize the output of ReLU
inputs = [1.0, -2.0, 3.0]
weights = [-3.0, -1.0, 2.0]
bias = 1.0

# output of neuron can be written as
# y = ReLU(sum(mul(xi, wi)))
# differentiation will be as follows
# dy/dx0 = dReLU/dSum * dSum/dMul * dMul/dx0

# derivative wrt weights and biases are to tune them, wrt inputs are to chain multiple layers

if __name__ == "__main__":
    # forward pass on this neuron
    z = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias

    # applying activation
    a = max(z, 0)

    # Backward pass

    # derivative from next layer (Suppose)
    d_value = 1.0

    # derivative of relu
    dReLU_dz = d_value * (1.0 if z > 0 else 0.0)
    print(dReLU_dz)

    # partial derivative of multiplication
    dSum_dxw0 = 1  # derivative of sum is always 1
    dSum_dxw1 = 1
    dSum_dxw2 = 1
    dSum_dBias = 1

    dReLU_dxw0 = dReLU_dz * dSum_dxw0
    dReLU_dxw1 = dReLU_dz * dSum_dxw1
    dReLU_dxw2 = dReLU_dz * dSum_dxw2
    dReLU_dBias = dReLU_dz * dSum_dBias

    print(dReLU_dxw0, dReLU_dxw1, dReLU_dxw2, dReLU_dBias)

    dMul_dx0 = weights[0]
    dMul_dx1 = weights[1]
    dMul_dx2 = weights[2]

    dRelu_dx0 = dReLU_dxw0 * dMul_dx0
    dRelu_dx1 = dReLU_dxw1 * dMul_dx1
    dRelu_dx2 = dReLU_dxw2 * dMul_dx2

    dMul_dw0 = inputs[0]
    dMul_dw1 = inputs[1]
    dMul_dw2 = inputs[2]

    dRelu_dw0 = dReLU_dxw0 * dMul_dw0
    dRelu_dw1 = dReLU_dxw1 * dMul_dw1
    dRelu_dw2 = dReLU_dxw2 * dMul_dw2

    print("Weights derivatives :", dRelu_dw0, dRelu_dw1, dRelu_dw2)
    print("input derivatives: ", dRelu_dx0, dRelu_dx1, dRelu_dx2)

    # in-short, we can do it as follows
    dRelu_dx0 = d_value * (1.0 if z > 0 else 0.0) * 1 * weights[0]
    # derivative wrt x0 = d_value * derivative of relu * derivative of sum * derivative of multiplication

    # this dx will be back-propagated to previous layers
    dx = [dRelu_dx0, dRelu_dx1, dRelu_dx2]

    # this will be used to tweak the weights and biases
    dw = [dRelu_dw0, dRelu_dw1, dRelu_dw2]
    db = [dReLU_dBias]
