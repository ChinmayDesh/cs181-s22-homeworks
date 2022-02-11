#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)

print("y is:")
print(y_train)

def kernel(x1, x2, tau):
    return np.exp(-pow((x1 - x2), 2) / tau)

def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    # Compute the distance between each test point and training point. Each row is a test point.
    x_train_dists = kernel(np.tile(x_train, (x_test.size, 1)),
        np.tile(np.array([x_test]).transpose(), (1, x_train.size)), tau)

    # Get the K nearest neighbors for each test point (i.e. for each row).
    x_train_indices = np.fliplr(np.argsort(x_train_dists)) # Sort in descending order.
    neighbor_indices = np.delete(x_train_indices, np.s_[k:np.shape(x_train_indices)[1]], axis=1)
    y_train_neighbors = np.take(y_train, neighbor_indices)

    # Calculate predictions by averaging the neighboring y-values.
    predictions = np.mean(y_train_neighbors, axis = 1)

    return predictions


def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    y_test = predict_knn(k=k)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for k in (1, 3, len(x_train)-1):
    plot_knn_preds(k)