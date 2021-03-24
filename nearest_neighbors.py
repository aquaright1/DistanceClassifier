import numpy as np
from sklearn.neighbors import KDTree, BallTree


class LinearNNSearch:
    def __init__(self, data: np.ndarray, fit: bool = False):
        self.data = data
        self.fit = fit

    def nn_distance(self, point: np.ndarray) -> float:
        '''
        point: the point to find nearest neighbor distance to, as a numpy array of features (coordinates in feature space)

        returns: the distance from point to its nearest neighbor in the data, as a float
        '''

        square_diffs = (data - point)**2
        distances = np.sqrt(square_diffs.sum(axis=1))

        if self.fit:
            distances = distances[distances != 0]

        return np.partition(distances, 0)[0]


class TreeNNSearch:
    def __init__(self, data: np.ndarray, tree: 'Tree', fit: bool):
        # TODO: could also configure the tree?
        self.tree = tree(data)
        self.fit = fit

    def nn_distance(self, point: np.ndarray) -> float:
        k = 2 if self.fit else 1
        dist, _ = self.tree.query(np.array([point]), k=k)

        if k == 1:
            return dist[0][0]
        else:
            return dist[0][1]

    def kd_tree(data: np.ndarray, fit: bool = False) -> 'TreeNNSearch':
        return TreeNNSearch(data, KDTree, fit)

    def ball_tree(data: np.ndarray, fit: bool = False) -> 'TreeNNSearch':
        return TreeNNSearch(data, BallTree, fit)


class NearestNeighborMethod:
    LINEAR = LinearNNSearch
    KD_TREE = TreeNNSearch.kd_tree
    BALL_TREE = TreeNNSearch.ball_tree


if __name__ == '__main__':
    # sanity check
    data = np.array([
        [1, 4, 3],
        [5, 3, 4],
        [2, 2, 6],
        [1, 3, 3],
        [6, 4, 1]
    ])

    point_fit = data[0, :]
    point = np.array([2, 5, 1])

    linear_fit = LinearNNSearch(data, fit=True)
    linear = LinearNNSearch(data)

    print(linear_fit.nn_distance(point_fit))
    print(linear.nn_distance(point))
    print()

    kd_tree_fit = TreeNNSearch.kd_tree(data, fit=True)
    kd_tree = TreeNNSearch.kd_tree(data)

    print(kd_tree_fit.nn_distance(point_fit))
    print(kd_tree.nn_distance(point))
    print()

    ball_tree_fit = TreeNNSearch.ball_tree(data, fit=True)
    ball_tree = TreeNNSearch.ball_tree(data)

    print(ball_tree_fit.nn_distance(point_fit))
    print(ball_tree.nn_distance(point))
