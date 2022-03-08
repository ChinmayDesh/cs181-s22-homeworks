import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        self.y_classes = np.sort(np.unique(y))

        n_classes = len(np.unique(y))
        self.n_classes = n_classes
        x_dim = X.shape[1]

        # Estimate mean vector.
        self.mean = np.zeros((n_classes, x_dim))
        class_counts = np.zeros(n_classes)
        for i, x in enumerate(X):
            self.mean[y[i]] += x
            class_counts [y[i]] += 1
        self.mean /= np.array([class_counts]).T

        self.cov = np.zeros((x_dim, x_dim))
        self.covs = np.zeros((n_classes, x_dim, x_dim))
        # If shared covariance matrix is asked for, estimate it.
        if self.is_shared_covariance:
            for i, x in enumerate(X):
                for k in self.y_classes:
                    if y[i] == k:
                        self.cov += np.array([(x - self.mean[y[i]])]).T @ np.array([(x - self.mean[y[i]])])
            self.cov /= X.shape[0]
        # If separate covariance matrices are asked for, estimate them.
        else:
            for i, x in enumerate(X):
                for k in self.y_classes:
                    if y[i] == k:
                        self.covs[y[i]] += np.array([(x - self.mean[y[i]])]).T @ np.array([(x - self.mean[y[i]])])
            for i in range(self.covs.shape[0]):
                self.covs[i] /= class_counts[i]
        self.class_counts = class_counts

    # TODO: Implement this method!
    def predict(self, X_pred):
        preds = []
        for x in X_pred:
            pred = self.y_classes[0]
            highest_likelihood = 0
            for y in self.y_classes:
                prob = 0
                if self.is_shared_covariance:
                    prob = mvn.pdf(x, self.mean[y], self.cov)
                else:
                    prob = mvn.pdf(x, self.mean[y], self.covs[y])
                if prob > highest_likelihood:
                    highest_likelihood = prob
                    pred = y
            preds.append(pred)
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        nll = 0
        if self.is_shared_covariance:
            for i, x in enumerate(X):
                for y_val in self.y_classes:
                    if y_val == y[i]:
                        nll += np.array([(x - self.mean[y_val])]) @ np.linalg.inv(self.cov) @ np.array([(x - self.mean[y_val])]).T
                        nll -= np.log((2 * np.pi)**self.n_classes)
                        nll -= np.log(np.linalg.det(self.cov))
                        nll += np.log(self.class_counts[y_val] / X.shape[0])
        else:
            for i, x in enumerate(X):
                for y_val in self.y_classes:
                    if y_val == y[i]:
                        nll += np.array([(x - self.mean[y_val])]) @ np.linalg.inv(self.covs[y_val]) @ np.array([(x - self.mean[y_val])]).T
                        nll -= np.log((2 * np.pi)**self.n_classes)
                        nll -= np.log(np.linalg.det(self.covs[y_val]))
                        nll += np.log(self.class_counts[y_val] / X.shape[0])
        nll /= -2
        return nll
        
