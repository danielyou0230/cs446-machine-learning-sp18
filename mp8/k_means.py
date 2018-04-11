from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
file = "data/data/iris.data"
dataset = pd.read_csv(file)
data = dataset.as_matrix()[:, :-1]
# generate labels
types = dataset.as_matrix()[:, -1]
keys = np.unique(types)
dics = dict(zip(keys, list(range(len(keys)))))
# convert labels to integers
label = np.array([dics[itr] for itr in types])

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)

def assign_step(k_clus, C, data):
    """
    Assign step in K-means Clustering.
    Arguments:
        k_clus(int): Number of clusters
        C(ndarray): Current center of each clusters
        data(ndarray): Dataset

    Returns:
        cluster(ndarray): Each entry represents the cluster of the data.
    """
    def dist(center, X):
        """
        Measure the Euclidian distance between the centre and all points.
        """
        return np.sum((center - X) ** 2, axis=1)
    # assign data to clusters
    distances = np.array([dist(C[itr, :], data) for itr in range(k_clus)]).T
    cluster = distances.argmin(axis=1)
    return cluster

def centre_step(k_clus, data, cluster):
    """
    Centering step in K-means Clustering.
    Arguments:
        k_clus(int): Number of clusters
        data(ndarray): Dataset
        cluster(ndarray): The cluster which all the data belongs to,
                        returned by assign_step()
    Returns:
        tmp(ndarray): New center
    """
    tmp = [np.mean(data[cluster == itr], axis=0) for itr in range(k_clus)]
    tmp = np.array(tmp)
    return tmp

def k_means(C):
    # Write your code here!
    C = np.array(C)
    k_clus = C.shape[0]
    C_final = C
    # init assignment
    cluster = assign_step(k_clus, C, data)

    criterion = True
    step = 0
    while criterion:
        # Recenter
        tmp = centre_step(k_clus, data, cluster)
        # Reassign
        cluster = assign_step(k_clus, C_final, data)

        # Calculate the change of centers
        delta = (C_final - tmp).sum(axis=0)
        print("Step {0} | Delta: {1}".format(step, sum(delta)))

        # update the centers
        C_final = tmp
        # stopping condition
        step += 1
        if step > 20:
            criterion = False
    return C_final

k_means(C)







