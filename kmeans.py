import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, n_clusters=10, distance_metric='euclidean'):
        assert isinstance(n_clusters, int) and n_clusters > 0, "n_clusters must be positive integer"
        self.k = n_clusters

        assert isinstance(distance_metric, str), "distance_metric must be a string"
        self.distance_metric = distance_metric

    def fit(self, points, predefined_centers=None, max_iters=50, threshold=1e-8):
        self.ndim = len(points[0])
        self.points = points
        self.centers = np.zeros(shape=(self.k, self.ndim))

        # start with the centers that are "fixed"
        n_predefined = 0
        if predefined_centers is not None:
            n_predefined = len(predefined_centers)
            self.centers[:n_predefined] = predefined_centers

        # then randomly initialize the other ones
        random_points = points.copy()
        np.random.shuffle(random_points)
        self.centers[n_predefined:] = random_points[:self.k-n_predefined]

        last_sse = 0
        for _ in range(max_iters):
            # find which center is closest to which point
            self.assignments = np.argmin(cdist(self.centers, self.points, metric=self.distance_metric), axis=0)

            # recompute each center as the average of its constituent points. if the "fixed" centers are horrible and are closest to no points, then die!
            print(self.assignments)
            for i in range(n_predefined, len(self.centers)):
                assert i in self.assignments, f"cluster {i} has no members!"
                self.centers[i] = np.average(self.points[self.assignments==i], axis=0)

            # check sum-squared error to see if we've converged
            sse = np.sum(np.linalg.norm(self.points - self.centers[self.assignments], axis=0)**2 ) / len(self.points)
            if last_sse > 0 and (last_sse - sse < threshold * sse):
                break

            last_sse = sse

        return self

