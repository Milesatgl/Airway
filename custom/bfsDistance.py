import queue
from typing import Tuple, Dict, List
from pathlib import Path
import numpy as np
from skimage.morphology import skeletonize

from helperFunctions import adjacent
from utils import getOutputPath

Coordinate = Tuple[int, int, int]

def getSkeleton(model: np.ndarray, objPath: str | None = None):
    model[model != 1] = 0
    skeleton = skeletonize(model)
    skeleton[skeleton != 0 ] = 1
    print(f"Model loaded with skeleton shape {skeleton.shape}")

    if objPath is not None:
        from trimesh import voxel
        voxleModel = voxel.VoxelGrid(skeleton)
        mesh = voxleModel.as_boxes()
        mesh.export(objPath)

        oriModel = voxel.VoxelGrid(model)
        oriMesh = oriModel.as_boxes()
        oriMesh.export(Path(objPath).parent / 'S1_modelForSkeVoxel.obj')

    return skeleton


def find_first_voxel(target: np.ndarray) -> List[int]:
    """Find first (highest) voxel in the lung"""
    for layer in range(len(target)):
        possible_coords = []
        found = False
        # for every layer，遍历每个体素
        # 有可能起始位置这一层对应的模型不是一个点，而是一个面或者截面
        for line in range(len(target[layer])):
            for voxel in range(len(target[layer][line])):
                if target[layer][line][voxel] == 1:
                    # element shape: (3,)
                    possible_coords.append(np.array([layer, line, voxel]))
                    found = True
        if found:
            possible_coords = np.array(possible_coords)
            avg = np.sum(possible_coords, axis=0) / len(possible_coords)
            print("Average starting coordinate:", avg)
            best = possible_coords[0]
            # 选择距离均值最近的点？
            for pc in possible_coords:
                if np.sum(np.abs(avg - pc)) < np.sum(np.abs(avg - best)):
                    best = pc
            print("Closest starting coordinate to average:", list(best))
            return list(best)
    raise ValueError('Unable to find first voxel')
        

def traverSkeletion(skeleton: np.ndarray, outputDataPath: Path):

    distance_to_coords_file = outputDataPath / "S1_map_distance_to_coords"
    coord_to_distance_file = outputDataPath / "S1_map_coord_to_distance.txt"
    coord_to_previous_file = outputDataPath / "S1_map_coord_to_previous.txt"
    coord_to_next_count_file = outputDataPath / "S1_map_coord_to_next_count.txt"

    firstVoxel = find_first_voxel(target=skeleton)
    bfs_queue = queue.Queue()
    # 起始位置，fisrtVoxel，距离起始位置的曼哈顿距离为0
    # np.array(firstVoxel) shape is (1, 3)
    bfs_queue.put((np.array(firstVoxel), 0))
    visited: Dict[Coordinate, int] = {tuple(firstVoxel): 0} # type: ignore # {（体素坐标），该体素到firstVoxel的曼哈顿距离}

    distance_to_coords: List[List[np.ndarray]] = []
    coord_to_previous: Dict[Tuple[int, int, int], np.ndarray] = {}
    coord_to_next_count: Dict[Tuple[np.ndarray], int] = {}

    vis_count = 0

    # Traverse entire tree to mark each pixels manhattan distance to the first pixel
    while not bfs_queue.empty():
        # curr: array([ 10,  31, 125])
        curr, dist = bfs_queue.get()
        if len(distance_to_coords) <= dist:
            # 从主干开始到第一个分支，都会走这里，一个距离对应一个体素
            distance_to_coords.append([curr]) # [[firstVoxel]]
        else:
            # 用 体素到firstvoxel的距离 作为索引
            # distance_to_coords[dist]即索引dist处的元素，是一个列表，代表了距离firstVoxel为dist的所有voxel
            # 当体素个数大于 dist 时，一个距离对应了多个体素
            # 所以在当前距离的对应的体素列表后 append
            distance_to_coords[dist].append(curr)
        vis_count += 1

        # Print progress
        if vis_count % 10000 == 0:
            print(vis_count)
        next_count = 0
        # 遍历 curr 周围的26个邻域位置，这些邻域的 dist 都是 +1
        # 没有分叉点的部位是一条线，nextCount是1，queue里面放的也是一个tuple（邻域体素坐标，距离）
        # 有分叉点的部位，一个体素的邻域可能有几个，nextCount>1,
        for adj in adjacent(curr, moore_neighborhood=True):
            x, y, z = adj

            # Iterate over bronchus
            if skeleton[x][y][z] == 1:
                # 在什么情况下，curr的邻域体素会在visited里面？
                # 1、处于中间位置的体素，这时它上面那一个就是visited
                if tuple(adj) not in visited:
                    bfs_queue.put((adj, dist + 1))
                    # adj的父是curr
                    coord_to_previous[tuple(adj)] = curr
                    visited[tuple(adj)] = dist + 1
                    next_count += 1
        coord_to_next_count[tuple(curr)] = next_count # curr voxel有几个子voxel
    
    np_dist_to_coords = np.array(distance_to_coords, dtype=object) # 得到的是一个array[list[array]]
    # print(np_dist_to_coords)
    
    np.savez_compressed(distance_to_coords_file, np_dist_to_coords)
    print(f"Writing distance to coords with shape: {np_dist_to_coords.shape}")

    for dictionary, filename in [
        (visited, coord_to_distance_file),
        (coord_to_previous, coord_to_previous_file),
        (coord_to_next_count, coord_to_next_count_file),
    ]:
        with open(filename, "w") as curr_file:
            for coord, dist in dictionary.items():
                x, y, z = coord
                curr_file.write(f"{x}, {y}, {z}: {dist}\n")

    return {
        'visited': visited,
        'coord_to_previous': coord_to_previous,
        'coord_to_next_count': coord_to_next_count,
        'distance_to_coords': np_dist_to_coords
    }
    

def distance(c1: Coordinate, c2: Coordinate):
    # return math.sqrt(sum(map(lambda a, b: (a-b)*(a-b), zip(c1, c2))))
    return np.linalg.norm(np.array(c1) - np.array(c2))

    
def get_distance_in_model_from_skeleton(model: np.ndarray, visited: Dict[Coordinate, int], outputDataPath: Path):

    distance_mask_path = outputDataPath / "S1_distance_mask"
    distance_mask: np.ndarray = np.zeros(model.shape)
    origin: Dict[Coordinate, Coordinate] = {}
    bfs_queue = queue.Queue()
    # 初始化
    for coord, dist in visited.items():
        bfs_queue.put(coord)
        distance_mask[coord] = dist
        origin[coord] = coord # 意思是骨架作为origin吗？这个骨架还在不断扩展，直到充满模型？
    while not bfs_queue.empty():
        curr = bfs_queue.get()
        # 遍历骨架上某个体素的每一个邻域体素
        for adj in map(tuple, adjacent(curr)):
            # 要求这个邻域体素在model里面
            if model[adj] == 1:
                # 如果这个邻域体素在骨架上
                if adj in origin:
                    # origin[curr]是curr对应的起始点体素
                    # origin[adj] 是adj 对应的起始点体素
                    # adj是cur的邻域
                    # 如果 curr的起始点与 adj 的距离 >= adj的起始点 与 adj 的距离
                    if distance(origin[curr], adj) >= distance(origin[adj], adj):
                        # 保证origin向外围扩展？
                        continue
                # else add point 
                # 这个邻域体素不在泛骨架（泛化的骨架，从骨架向周围慢慢扩散）上，或者在泛骨架上，但是 curr的起始点与 adj 的距离 < adj的起始点 与 adj 的距离,adj在curr的正上或者正下方？
                bfs_queue.put(adj) # 不在骨架上，但在模型里面
                distance_mask[adj] = distance_mask[curr] # 此时该邻域体素的dist等同于中心点curr的dist
                origin[adj] = origin[curr] # 将curr的邻域adj也加入到origin中，且adj的origin是curr？泛化骨架？
    np.savez_compressed(distance_mask_path, distance_mask)
    print(*map(str, zip(*np.unique(distance_mask, return_counts=True))))
    return distance_mask


def bfsDistanceMethod(model: np.ndarray, outputDataPath: Path):
    skeleton = getSkeleton(model=model, objPath=str(outputDataPath.joinpath('S1_skeletonVoxel.obj')))
    skeletonInfo = traverSkeletion(skeleton=skeleton, outputDataPath=outputDataPath)
    _ = get_distance_in_model_from_skeleton(model=model, visited=skeletonInfo['visited'], outputDataPath=outputDataPath)
    return skeletonInfo


if __name__ == "__main__":
    import sys
    # reducedModelPath = Path(r'E:\conda_projects\Airway\data\CY_14210425\S0_Airway_preprocessed.npz')

    projDir = sys.argv[1]
    reducedModelPath = Path(projDir).joinpath('S0_Airway_preprocessed.npz')

    reducedModel = np.load(str(reducedModelPath))['arr_0']
    _ = bfsDistanceMethod(model=reducedModel, outputDataPath=reducedModelPath.parent)


# "D:\mambaforge\envs\skeletonA\python.exe" E:\conda_projects\Airway\custom\bfsDistance.py E:\conda_projects\Airway\data\UN_10122643T