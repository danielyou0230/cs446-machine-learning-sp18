"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import random
import scipy
from scipy.stats import multivariate_normal
from collections import Counter


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-1):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        # np.array of size (n_components, n_dims)
        self._mu = np.ones((self._n_components, self._n_dims))

        # Initialized with uniform distribution.
        # np.array of size (n_components, 1)
        self._pi = np.random.uniform(low=0.0, high=1.0, size=(n_components, 1))

        # Regularization on covariance matrix
        reg_I = self._reg_covar * np.identity(n_dims)
        self._reg = np.array([reg_I for _  in range(n_components)])
        # Initialized with identity.
        I = np.identity(n_dims) * 10000
        # np.array of size (n_components, n_dims, n_dims)
        self._sigma = np.array([I for _ in range(n_components)]) + reg_I


    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        # initialize mu
        index = np.random.random_integers(low=0, high=x.shape[0] - 1,
                                          size=(self._n_components))
        self._mu = x[index]
        # self._mu = np.array([x.mean(axis=0) for _ in range(self._n_components)])
        # print("mu shape: {0}".format(self._mu.shape))

        for step in range(self._max_iter):
            print("Step: {:5d} / {:5d}".format(step + 1, self._max_iter))
            # E-step
            z_ik = self._e_step(x)
            # M-step
            self._m_step(x, z_ik)
        pass

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        # z_ik = normalized \pi_j * p_theta_j(x_i)
        z_ik = self.get_posterior(x)
        return z_ik

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        N = x.shape[0]
        # Update the parameters.
        self._pi = (z_ik.sum(axis=0) / N)[:, np.newaxis]

        self._mu = z_ik.T.dot(x) / (N * self._pi)

        self._sigma = [(z_ik[:,itr][:, np.newaxis] * (x - self._mu[itr,:])).T
                       .dot(x - self._mu[itr,:]) / self._pi[itr][0]
                       for itr in range(self._n_components)]
        self._sigma = np.array(self._sigma) / N + self._reg

    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N, n_components).
        """
        ret = [self._multivariate_gaussian(x, self._mu[itr, :], self._sigma[itr,:,:])
               for itr in range(self._n_components)]

        return np.array(ret).T

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        N_k = self.get_conditional(x)
        marginal = (N_k * self._pi.T).sum(axis=1)
        return marginal

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        conditional = self.get_conditional(x)
        marginal = self.get_marginals(x)

        z_ik = conditional / marginal[:, np.newaxis]
        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,)
            sigma_k(numpy.ndarray): Array containing one single covariance matrix
                (ndims, ndims)
        """
        # print(mu_k)
        # print(sigma_k)
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        self.cluster_label_map = list()

        # Acquire the soft assignments for all instances
        cluster = self.get_posterior(x).argmax(axis=1)
        # Distinct labels
        distinct_labels = [int(itr) for itr in np.unique(y)]

        assignment = [list(y[cluster == itr]) for itr in range(self._n_components)]

        # Debugging
        # occurences = [Counter(itr) for itr in assignment]
        # print(occurences)

        for idx, itr in enumerate(assignment):
            # If the cluster contains no instances
            # random assignment the cluster to the label
            if len(Counter(itr).most_common(1)) == 0:
                print("Random assignment applied to cluster {:d}".format(idx))
                assign_to = random.choice(distinct_labels)
            # Assignment the cluster to the most common labels
            else:
                assign_to = int(Counter(itr).most_common(1)[0][0])

            # Assignments
            self.cluster_label_map.append(assign_to)

        print(self.cluster_label_map)

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """

        cluster = self.get_posterior(x).argmax(axis=1)
        y_hat = [self.cluster_label_map[itr] for itr in cluster]

        return np.array(y_hat)
