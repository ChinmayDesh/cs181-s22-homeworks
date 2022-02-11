#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

x_data = [i[0] for i in data]
y_data = [i[1] for i in data]

def kernel(x1, x2, tau):
    return np.exp(-pow((x1 - x2), 2) / tau)

def predict(x, tau):
    # Return a predicted y-value for a given x-value.
    prediction = 0
    for n in range(0, len(data)):
        prediction += kernel(data[n][0], x, tau) * data[n][1]
    return prediction

def compute_loss(tau):
    loss = 0
    for n in range(0, len(data)):
        loss += pow((2 * data[n][1] - predict(data[n][0], tau)), 2)
    return loss

def grad_loss(tau):
    # Compute the gradient of the loss function at a given tau value.
    grad = 0
    for n in range(0, len(data)):
        t1 = 0
        t2 = 0
        for m in range(0, len(data)):
            term = kernel(x_data[n], x_data[m], tau) * y_data[m]
            t1 += term
            t2 += term * pow((x_data[n] - x_data[m]), 2) / pow(tau, 2)
        grad += ((2 * y_data[n]) - t1) * t2
    grad *= -2
    print(grad)
    return grad

def grad_descent(tau_start, step_size, iters = 100):
    tau = tau_start
    for i in range(0, iters):
        tau = tau - step_size * grad_loss(tau)
    return tau

taus = [0.01, 2, 100, grad_descent(2, 0.05, iters = 1000)]
for tau in taus:
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))

plt.figure()

x_vals = np.arange(0, 12, 0.1)
for i in range(0, len(taus)):
    plt.subplot(2, 2, i+1)

    # Display a scatter plot of the training data.
    plt.scatter(x_data, y_data)

    # Plot the kernel regression.
    y_vals = predict(x_vals, taus[i])
    plt.plot(x_vals, y_vals, color="red")

    plt.title("tau = " + str(taus[i]))

plt.suptitle("Kernel Regressions")
plt.tight_layout()
plt.show()