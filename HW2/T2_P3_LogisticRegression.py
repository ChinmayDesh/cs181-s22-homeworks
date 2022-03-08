import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None
    
    def __compute_loss(self, X, y):
        loss = 0
        n_classes = len(np.unique(y))
        for i, x in enumerate(X):
            y_vec = np.zeros(n_classes)
            y_vec[y[i]] = 1
            for j, y_val in enumerate(y_vec):
                loss -= y_val * np.log(softmax(np.dot(self.W, x))[j])
        return loss

    def fit(self, X, y):
        X = np.hstack((np.array([np.ones(X.shape[0])]).T, X))

        self.losses = []

        n_classes = len(np.unique(y))
        self.W = np.random.rand(n_classes, X.shape[1]) # Each row of W contains the weights for the row index's class.
        for _ in range(200000):
            grad = np.zeros(self.W.shape)
            for i, x in enumerate(X):
                true_y = y[i]
                y_vec = np.zeros(n_classes)
                y_vec[true_y] += 1
                pred = softmax(np.dot(self.W, x))
                for y_val in range(n_classes):
                    grad_row_inc = (pred[y_val] - y_vec[y_val]) * x + self.lam * self.W[y_val]
                    grad[y_val] += grad_row_inc
            grad /= X.shape[0]

            self.losses.append(self.__compute_loss(X, y))

            self.W = self.W - self.eta * grad

    def predict(self, X_pred):
        preds = []
        for x in np.hstack((np.array([np.ones(X_pred.shape[0])]).T, X_pred)):
            preds.append(np.argmax(softmax(np.dot(self.W, x))))
        return np.array(preds)

    def visualize_loss(self, output_file, show_charts=False):
        plt.figure()
        grid_x = np.linspace(0, len(self.losses), len(self.losses))
        plt.plot(grid_x, self.losses)
        plt.savefig(output_file)
        if show_charts:
            plt.show()