import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None
    
    def dist(self, a, b):
        return (((a[0] - b[0])/3)**2 + (a[1] - b[1])**2)

    # TODO: Implement this method!
    def predict(self, X_pred):        
        preds = []
        for x in X_pred:
            dist = []
            for i in range(len(self.y)):
                dist.append([self.dist(x, self.X[i]), self.y[i]])
            nearest = 0
            y_votes = []
            while nearest < self.K:
                min = dist[0]
                for element in dist:
                    if min[0] > element[0]:
                        min = element
                y_votes.append(min[1])
                dist.remove(min)
                nearest += 1
            if y_votes.count(0) > y_votes.count(1) and y_votes.count(0) > y_votes.count(2):
                preds.append(0)
            elif y_votes.count(1) > y_votes.count(0) and y_votes.count(1) > y_votes.count(2):
                preds.append(1)
            else:
                preds.append(2)

        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y