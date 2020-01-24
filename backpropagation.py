# Backpropagation example
# A neural network of 1 layer with 1 node, without a bias.
# Goal is to train the network so that W is such that input of X results in output of Y
# Great explanation by Mikael Laine at https://www.youtube.com/watch?v=8d6jf7s6_Qs

import matplotlib.pyplot as plt

X = 1.50  # input
Y = 0.75  # desired output

W = -0.12  # random initial weight

learning_rate = 0.001
iterations = 1000
error = []

print("initial :", "W = {:0.3f}".format(W))

for _ in range(iterations):
    # forward pass
    a = W * X  # actual output

    # back propagation
    e = (a - Y) ** 2  # error measured via MSE
    error.append(e)

    da = 2 * (a - Y)  # da = d_e / d_a = derivative of e = how changes in a change e
    dw = X  # dw = d_a / d_W = X = derivative of a = how changes in W change a
    W -= learning_rate * da * dw  # adjust W to negative of gradient (+ chain rule)

print("result  : W = {:0.3f} (after {} iterations)".format(W, iterations))
print("expected: W = {:0.3f}".format(Y / X))

plt.plot(range(iterations), error)
plt.title("MSE (mean squared error)")
plt.xlabel("training iterations")
plt.ylabel("mse")
plt.show()
