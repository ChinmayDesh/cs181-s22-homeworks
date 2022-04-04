# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
=======
from scipy.spatial.distance import cdist
>>>>>>> Stashed changes

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
<<<<<<< Updated upstream

    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        pass

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        # TODO: change this!
        return small_dataset[:self.K]
=======
        self.rss = np.array([])
    
    # Calculuate the residual sum of squares.
    def calc_rss(self, means, X, assignments):
        rss = 0
        for n, image in enumerate(X):
            mean = means[assignments[n]]
            rss += np.linalg.norm(image - mean) ** 2
        return rss

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        ITERS = 10
        # Initialize means randomly.
        self.means = np.random.uniform(size=(self.K, 784))
        assignments = np.empty(X.shape[0], dtype=np.int64)
        for _ in range(ITERS):
            # Assign each image to its closest prototype.
            for n, image in enumerate(X):
                # Calculate the distance between the image and each prototype.
                distances = np.empty(self.means.shape[0])
                for k, mean in enumerate(self.means):
                    distances[k] = np.linalg.norm(image - mean)
                # Assign the image to the index of the closest mean.
                assignments[n] = np.argmin(distances)

            # Set each mean to the mean of images assigned to it.
            new_means = np.zeros(self.means.shape)
            cluster_sizes = np.zeros(self.means.shape[0], dtype=np.int64)
            for n, image in enumerate(X):
                new_means[assignments[n]] += image
                cluster_sizes[assignments[n]] += 1
            self.cluster_sizes = cluster_sizes
            for i in range(cluster_sizes.shape[0]):
                if cluster_sizes[i] != 0:
                    new_means[i] /= cluster_sizes[i]
            self.means = new_means

            self.rss = np.append(self.rss, self.calc_rss(self.means, X, assignments))

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.means
    
    def get_cluster_sizes(self):
        return self.cluster_sizes
    
    def get_rss(self):
        return self.rss
>>>>>>> Stashed changes

class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
    
<<<<<<< Updated upstream
    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        pass

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        # TODO: Change this!
        return small_dataset[:n_clusters]
=======
    def fit(self, X):
        N_CLUSTERS = 10
        clustered_X = list(X[:,np.newaxis]) # Each row is a cluster, stored as an array of images.
        while len(clustered_X) > N_CLUSTERS:
            # Find the two closest clusters.
            closest = 0
            distances = 0
            if self.linkage == "min":
                # Flatten clustered images into an array of images.
                flattened_clusters = np.vstack(clustered_X)
                # Make array of cluster identities for each of the images.
                cluster_names = []
                for i in range(len(clustered_X)):
                    cluster_names.append(np.ones(clustered_X[i].shape[0], dtype=np.int64) * i)
                cluster_names = np.concatenate(cluster_names)
                # Get distances between all images.
                all_distances = cdist(flattened_clusters, flattened_clusters)

                # Get minimum distance between each cluster.
                distances = np.zeros((len(clustered_X), len(clustered_X)))
                for i in range(distances.shape[0]):
                    for j in range(distances.shape[1]):
                        if i != j:
                            keep_rows = cluster_names == i
                            keep_cols = cluster_names == j
                            distances[i][j] = np.min(all_distances[np.array(keep_rows)][0][np.array(keep_cols)])
                # Find the closest pair.
                closest = np.unravel_index(np.argmin(np.where(distances == 0, np.max(distances), distances)), (len(clustered_X), len(clustered_X)))
            elif self.linkage == "max":
                # Flatten clustered images into an array of images.
                flattened_clusters = np.vstack(clustered_X)
                # Make array of cluster identities for each of the images.
                cluster_names = []
                for i in range(len(clustered_X)):
                    cluster_names.append(np.ones(clustered_X[i].shape[0], dtype=np.int64) * i)
                cluster_names = np.concatenate(cluster_names)
                # Get distances between all images.
                all_distances = cdist(flattened_clusters, flattened_clusters)

                # Get maximum distance between each cluster.
                distances = np.zeros((len(clustered_X), len(clustered_X)))
                for i in range(distances.shape[0]):
                    for j in range(distances.shape[1]):
                        if i != j:
                            keep_rows = cluster_names == i
                            keep_cols = cluster_names == j
                            distances[i][j] = np.max(all_distances[np.array(keep_rows)][0][np.array(keep_cols)])
                # Find the closest pair.
                closest = np.unravel_index(np.argmin(np.where(distances == 0, np.max(distances), distances)), (len(clustered_X), len(clustered_X)))
            elif self.linkage == "centroid":
                # Flatten clusters into their centroids.
                cluster_centroids = []
                for cluster in clustered_X:
                    cluster_centroids.append(np.mean(cluster, axis=0))
                # Get distances between each centroid.
                distances = cdist(cluster_centroids, cluster_centroids)
                # Find the closest pair.
                closest = np.unravel_index(np.argmin(np.where(distances == 0, np.max(distances), distances)), (len(clustered_X), len(clustered_X)))
            else:
                return
            # Merge the two closest clusters.
            clustered_X[closest[0]] = np.append(clustered_X[closest[0]], clustered_X[closest[1]], axis=0)
            clustered_X.pop(closest[1])
            print(len(clustered_X))
        
        self.clusters = clustered_X

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        means = []
        for cluster in self.clusters:
            means.append(np.mean(cluster, axis=0))
        return np.vstack(means)
    
    # Returns the size of each cluster.
    def get_cluster_sizes(self):
        cluster_sizes = []
        for cluster in self.clusters:
            cluster_sizes.append(cluster.shape[0])
        return np.array(cluster_sizes)
>>>>>>> Stashed changes

# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    plt.show()

<<<<<<< Updated upstream
# ~~ Part 2 ~~
make_mean_image_plot(large_dataset, False)

# ~~ Part 3 ~~
# TODO: Change this line! standardize large_dataset and store the result in large_dataset_standardized
large_dataset_standardized = large_dataset
make_mean_image_plot(large_dataset_standardized, True)
=======
# # ~~ Part 1 ~~
# KMeansClassifier = KMeans(K=10)
# KMeansClassifier.fit(large_dataset)
# rss = KMeansClassifier.get_rss()
# fig = plt.figure()
# plt.suptitle("K-Means Residual Sum of Squares across Training")
# plt.ylabel("Residual Sum of Squares")
# plt.xlabel("Training Iterations")
# plt.plot(np.linspace(1, 10, 10), rss)
# plt.show()

# # ~~ Part 2 ~~
# make_mean_image_plot(large_dataset, False)

# # ~~ Part 3 ~~
# def standardize(dataset):
#     standardized = dataset.copy().astype(np.float64)
#     for i in range(standardized.shape[1]):
#         standardized[:,i] -= np.mean(standardized[:,i])
#         var = np.std(standardized[:,i])
#         if var != 0:
#             standardized[:,i] /= var
#     return standardized
# large_dataset_standardized = standardize(large_dataset)
# make_mean_image_plot(large_dataset_standardized, True)
>>>>>>> Stashed changes

# Plotting code for part 4
LINKAGES = [ 'max', 'min', 'centroid' ]
n_clusters = 10

fig = plt.figure(figsize=(10,10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    mean_images = hac.get_mean_images(n_clusters)
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if m_idx == 0: plt.title(l)
        if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
        plt.imshow(m.reshape(28,28), cmap='Greys_r')
plt.show()

# TODO: Write plotting code for part 5
<<<<<<< Updated upstream

# TODO: Write plotting code for part 6
=======
for l in LINKAGES:
    fig = plt.figure()
    plt.suptitle("Cluster sizes for HAC, " + l + " distance")
    hac = HAC(l)
    hac.fit(small_dataset)
    sizes = hac.get_cluster_sizes()
    plt.bar(np.linspace(0, 9, 10), sizes)
    plt.show()
fig = plt.figure()
plt.suptitle("Cluster sizes for K-Means")
KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(large_dataset)
sizes = KMeansClassifier.get_cluster_sizes()
plt.bar(np.linspace(0, 9, 10), sizes)
plt.show()

# TODO: Write plotting code for part 6
>>>>>>> Stashed changes
