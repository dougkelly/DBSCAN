# Imports
import numpy as np
import random
from collections import defaultdict, Counter
from scipy.spatial import distance


class DBSCAN(object):

    def __init__(self, eps=0.5, MinPts=5, metric='euclidean'):
        """My implentation of DBSCAN unsupervised clustering algorithm.

        Referenced Sklearn source: \
          # pylint disable=E501
          https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/dbscan_.py
        Referenced: \
          https://github.com/alwintsui/dbscan/blob/master/dbscan/dbscan.py
        Args:
            eps: float, default 0.5. The maximum distance between 2 samples \
              for them to be considered as in the same neighborhood.
            MinPts: int, default 5
            metric: ('euclidean', 'cosine') distance metric used to compute  \
              pointwise distances.

        Returns:
            DBSCAN object.
            """
        # maximum distance threshold between 2 points.
        self.eps = eps
        # Minimum number of points to form a cluster, including point itself.
        self.MinPts = MinPts
        # List to keep track of indexes of clustered points.
        self.visited = None
        # Default dict stores [point index] = cluster_ID.
        # Per sklearn docs: noise stored with cluster_ID = -1.
        self.clusters = defaultdict(set)
        # Initialized to 0 for first class.
        self.cluster_ID = 0
        # Either euclidean or cosine distance metric.
        self.metric = metric

    def fit(self, X):
        """Function to compute clusters from points.

        Args:
          X: (array of samples with shape (N,1)).

        Returns:
          None (see predict function for accessing point clusters).
        """
        # store copy of data for use and public access within object
        self.samples = X
        # Optimization: initializing boolean array to index points visited
        self.visited = np.repeat(False, len(self.samples))
        # Note: point represents a sample index
        for point in range(len(self.samples)):
            if not self.visited[point]:
                self.visited[point] = True
                # Retrieving all point indices within eps range
                NeighborPts = self._regionQuery(point)
                # Handle 'noise' by assigning it to cluster -1
                if len(NeighborPts) < self.MinPts:
                    self.clusters[point] = -1
                else:
                    # Builds cluster
                    self._expandCluster(point, NeighborPts)
                    # Iterates to next cluster
                    self.cluster_ID += 1

    def _expandCluster(self, point, NeighborPts):
        """Private function that builds cluster out from point.

        Args:
            point: individual sample passed from fit function.
            NeighborPts: all samples within eps distance of point.

        Returns:
            None
        """
        # Add point to current cluster
        self.clusters[point] = self.cluster_ID
        # Note: neighbor represents a neighboring sample index
        for neighbor in NeighborPts:
            if not self.visited[neighbor]:
                self.visited[neighbor] = True
                hood = self._regionQuery(neighbor)
                if len(hood) >= self.MinPts:
                    NeighborPts.extend(hood)
                # critical lines below; identifies all still unvisited points \
                # within neighboring points and assigns to current cluster
                if self.visited[neighbor] is None:
                    self.clusters[neighbor] = self.cluster_ID

    def _regionQuery(self, point):
        """Private helper function to calculate pointwise distances.

        This function calculcates pointwise distancesusing a user specified \
          distance metric between single point and all other points in sample.

        Args:
            point: individual sample passed from fit function.

        Returns:
            List of all point indices within eps distance from point passed \
              from fit function.
        """
        # Actual sample retrieved for distance computations
        point = self.samples[point]
        # list to store all neighboring points found with eps distance.
        neighbor_points = []
        if self.metric == 'euclidean':
            for idx, pt in enumerate(self.samples):
                if distance.euclidean(point, pt) <= self.eps:
                    neighbor_points.append(idx)
        else:
            for idx, pt in enumerate(self.samples):
                if distance.cosine(point, pt) <= self.eps:
                    neighbor_points.append(idx)
        return neighbor_points

    def predict(self, X):
        """Returns array of (X, cluster_ID). If model not fit, calls fit().

        Args:
          X: Numpy array of data.
        Returns:
          Numpy array of cluster values.

        """
        if len(self.clusters) == 0:
            self.fit(X)
        else:
            return np.asarray(self.clusters.values())
