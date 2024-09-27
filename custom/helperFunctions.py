import math
from typing import Set
from typing import Tuple


import numpy as np

def _adjacent(coord, moore_neighborhood=False):
    # 定义在每个坐标轴（x，y，z）上可能的改变量，即每个坐标值可以增大1，保持不变，或者减少1
    d = [-1, 0, 1]

    def condition(manhattan_dist):
        if moore_neighborhood:
            # The Moore neighbourhood of a cell is the cell itself and the cells at a Chebyshev distance（两个点在各维度上坐标差值的绝对值的最大值） of 1. By https://en.wikipedia.org/wiki/Moore_neighborhood
            # 包括除了完全重合（距离为0）的所有向量
            return manhattan_dist != 0
        else:
            # The von Neumann neighbourhood of a cell is the cell itself and the cells at a Manhattan distance of 1. By https://en.wikipedia.org/wiki/Von_Neumann_neighborhood
            # 只包括曼哈顿距离为1的向量
            return manhattan_dist == 1
    # sum(abs([x, y, x])) 计算向量在各个轴上的绝对值，然后计算这些绝对值得和，得到 曼哈顿距离
    directions = [np.array([x, y, z]) for x in d for y in d for z in d if condition(np.sum(np.abs([x, y, z])))]
    return np.array([coord + direction for direction in directions])


# Pre-compute for faster operations
adjacent_6 = _adjacent(np.array([0, 0, 0]), False)
'''
von Neumann neighbourhood
与中心点相邻的6个点(3*3的kernel，不包括对角线？)
array([[-1,  0,  0],
       [ 0, -1,  0],
       [ 0,  0, -1],
       [ 0,  0,  1],
       [ 0,  1,  0],
       [ 1,  0,  0]])'''
adjacent_26 = _adjacent(np.array([0, 0, 0]), True)
'''
adjacent_26 是通过计算 "Moore 邻域" 的相对坐标生成的向量数组。

具体来说，adjacent_26 表示从三维空间中的一个中心点出发，所有与中心点共享至少一个顶点的26个相邻点的坐标增量列表。这种邻域包括:
(此时的中心点应该考虑为一个体素？)
与中心点共享一个面（如在 x、y 或 z 轴方向相邻）的点。
与中心点共享一条边（对角线相邻）的点。
以及与中心点仅共享一个顶点的点。
因此，adjacent_26 包含以下特性：

包括除了中心点本身（坐标增量为 [0, 0, 0]）之外的所有可能的坐标增量组合。
这些增量由三个维度的每个可能变化组合(-1, 0, 1)组成，但排除 [0, 0, 0]（即曼哈顿距离不为零的情况）。
这种邻域类型考虑到所有可能的相邻关系，因此相对于 Von Neumann 邻域，Moore 邻域更加全面，涵盖了26个可能的相邻位置。
'''


def adjacent(coord, moore_neighborhood=False):
    """Returns a numpy array of adjacent coordinates to the given coordinate

    By default von Neumann neighborhood which returns only coordinates sharing a face (6 coordinates)
    Moore neighborhood returns all adjacent coordinates sharing a at least a vertex (26 coordinates)
    """
    return coord + (adjacent_26 if moore_neighborhood else adjacent_6)


def get_numpy_sphere(radius, hollow=False):
    """Returns a numpy 3D bool array with True where the sphere lies and False elsewhere as well as the centre

    If hollow==True then only the outer shell of the sphere will be True

    Suggested use:
        use this to create a sphere, then np.nonzero() to get the coordinates of the points
        which are of interest. Then add the coordinates of the point where circle lies like this:
            sphere_around_point = tuple(map(sum, zip(np.nonzero(sphere), at_point)))
        now you can iterate over this to color in the points or something like that
    """

    shape = ((math.ceil(radius) * 2) + 1,) * 3
    centre = np.array([round(radius)] * 3)
    dist_mat = np.full(shape, 0.0)
    for x in range(len(dist_mat)):
        for y in range(len(dist_mat[x])):
            for z in range(len(dist_mat[x][y])):
                dist_mat[x][y][z] = np.linalg.norm(centre - np.array([x, y, z]))
    sphere = dist_mat <= radius
    if hollow:
        sphere &= radius - 1 < dist_mat
    return sphere, centre


def get_coords_in_sphere_at_point(radius, point, hollow=False):
    sphere, centre = get_numpy_sphere(radius, hollow=hollow)
    coords = np.nonzero(sphere)
    sphere_around_point = tuple(map(sum, zip(coords, point, -centre)))
    return sphere_around_point


def find_radius_via_sphere(at_point: Tuple[int, int, int], allowed_types: Set[int], model: np.ndarray):
    """Returns the maximum radius of a sphere which fits into the model at the given point

    This only considers voxels in the model which have a value in allowed_types (e.g. 1)
    and views everything else as empty.
    """
    max_radius = 50
    for radius in range(1, max_radius):
        sphere_around_point = get_coords_in_sphere_at_point(radius + 0.5, at_point, hollow=True)
        for x, y, z in zip(*sphere_around_point):
            try:
                if model[round(x), round(y), round(z)] not in allowed_types:
                    return radius
            except IndexError:
                pass
    return max_radius
    # TODO
    # raise Exception(f"ERROR: Within radius of {max_radius} no valid voxels found!")


def adjacent_euclidean(coord, dist=2):
    """Returns a numpy array of adjacent coordinates to the given coordinate"""
    d = list(range(-math.floor(dist), math.ceil(dist) + 1))

    def condition(euclidean_dist):
        return euclidean_dist <= dist

    directions = [
        np.array([x, y, z]) for x in d for y in d for z in d if condition(np.sqrt(np.sum(np.power([x, y, z], 2))))
    ]
    return [coord + direction for direction in directions]
