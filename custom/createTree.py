import queue
import math
from typing import Tuple, Set, Dict, List
from pathlib import Path
import numpy as np

from helperFunctions import adjacent, find_radius_via_sphere

def distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

def calc_diameter(area):
    return math.sqrt(4 * area / math.pi)

def parse_coord(coord, split_on):
    text = coord.replace("[", "").replace("]", "").strip().split(split_on)
    if len(text) == 1:
        return int(text[0])
    else:
        return tuple([int(a) for a in text if a != ""])

def groupVoxel(
        distance_to_coords: List[List[np.ndarray]],
        coord_to_previous: Dict[Tuple[np.ndarray], np.ndarray]
):
    # Maps group id (1, 0) to group_id (0, 0) to show the predecessor
    prev_group = {}

    # A list of all groups, where each entry corresponds to a list 
    # with all the groups in the distance
    all_groups = []

    # Maps group to average coordinate of the group
    group_to_avg_coord = {}

    group_diameter = {}
    group_area = {}

    # Each iteration corresponds to 1 depth level from the start point
    # 从顶到底，每循环一层，距离顶部更深一级（距离firstVoxel的曼哈顿距离+1）
    # distance_to_coords[dist]即索引dist处的元素，是一个列表，代表了距离 firstVoxel 为 dist 的所有 voxel
    for curr_dist, coords in enumerate(distance_to_coords):
        coords_set = {tuple(coord) for coord in coords} # why use set ???
        print("Current manhattan distance: {}".format(curr_dist), end=" -> ")

        # Groups is a dictionary where each coordinate maps to an interger which stands
        # for it's group number. A group in this project is regarded as a set of coordinates
        # which have the same manhattan distance from the start point(fisrt voxel) and are
        # more connected. 也就是说 同一级 且 是adjacent？
        # 寻找每一级 有多少个 group？每一个dist 都是从0开始计数group的
        # 这一级有多少个 coord，就有多少个 group，共同构成groups
        # 对于非分叉点的部位，每一个体素就是一个单独的group
        # 对于每个group，如果curr的邻域adj也在骨架上，那么adj也是属于这个group
        # 在骨架上，邻域，dist相同的是什么点？
        groups = {}
        group_index = 0

        # Iterate over each coord in the current depth level. Later on this coordinate will
        # be added to a bfs queue and each adjacent coordinate will be marked as belonging
        # to this group. The loop will not iterate over visited coords, therefore this loop
        # will only visit as many coords as there are groups
        for coord in coords:

            # Convert coords to tuple as arrays cannot be hashed in dictionaries. Tuple[int, int, int]
            coord = tuple(coord)
            
            # Make sure the coordinate has not been visited
            if coord in groups:
                continue

            groups[coord] = group_index
            group_coords_sum = np.array(coord)
            group_size = 1 # 同一个group里面体素的个数
            bfs_queue = queue.Queue()
            bfs_queue.put(coord)

            # Count any adjacent coords to the current group
            while not bfs_queue.empty():
                curr = bfs_queue.get()

                # Iterate over adjacent coords
                for adj in adjacent(curr, moore_neighborhood=True):
                    adj = tuple(adj)

                    # If the adjacent is an actual coordinate (not empty space) 
                    # and has not been visited
                    # Then mark it as belonging to this group
                    # ??? 意思是这个 adjacent 需要在 skeleton 上，而且与 curr 是处于同一级（curr_dist）的(在同一个coords_set里面)
                    # fistVoxel 是 [10, 31, 125]
                    # 比如 (88, 98, 59) 和 (89, 98, 58)
                    # 比如 (149, 74, 183)和 (150, 73, 184)
                    # (178, 52, 198),(179, 51, 197), (179, 52, 196)
                    # firstVoxel is (9, 30, 124)
                    # (173, 87, 76), (174, 88, 77), (175, 88, 76)
                    if adj in coords_set and adj not in groups.keys():
                        bfs_queue.put(adj)
                        groups[adj] = group_index
                        group_coords_sum += np.array(adj)
                        group_size += 1

            # Group_id is the unique identifier for each group; this one will be used in
            # dicts to access them
            group_id = (curr_dist, group_index)

            # Remember the previous group for each group. Used to build the tree
            if curr_dist != 0:
                # all_groups 的索引等于distance_to_coords的索引
                # all_groups 的元素是 索引所在层级（曼哈顿距离） 所有体素的 groups
                # prev_group 字典长度 是 dist 数量减一；键是curr dist下某个group，值是 上一个 dist下 curr对应coord 的上一个体素所属的 group index
                # 起点是 (1, 0), 对于终点，prev_group反映的是终点的上一个group
                prev_group[group_id] = all_groups[(curr_dist - 1)][tuple(coord_to_previous[coord])]
            
            # Add the information about the group for saving as attribute
            group_area[group_id] = group_size
            group_diameter[group_id] = calc_diameter(group_size)

            # Count the average coordinate for each group, this will be the split location
            group_to_avg_coord[group_id] = group_coords_sum / group_size

            group_index += 1

        all_groups.append(groups)
        print(f"{group_index} group count")

    return {
        'prev_group': prev_group,
        'all_groups': all_groups,
        'group_to_avg_coord': group_to_avg_coord,
        'group_diameter': group_diameter,
        'group_area': group_area
    }


def getSuccessor(prev_group: Dict[Tuple[int, int], int]):
    # Create successor count for each node
    # Will be used to detemine groups which only connect 2 other groups if there are only
    successor_count = {(0, 0): 0} # 接替的
    # prev_group 是从 curr_dist 为 1 开始的
    # 对于prev_group的最后一个元素，successor_count的key是上一个元素
    # successor_count是从0到倒数第二个group，不包括终点？
    for group, prev_group_index in prev_group.items():
        curr_dist, group_index = group # 第一个：(1, 0): 0
        key = (curr_dist - 1, prev_group_index) # 这其实是 group 的上一级, 第一个是（0，0）？
        if key not in successor_count:
            # 什么情况下会进入这一级？
            successor_count[key] = 0
        successor_count[key] += 1 # 字典键是key，值代表这个key有几个children ？
    return successor_count


def groupAnalysis(
        prev_group: Dict[Tuple[int, int], int], group_area: Dict[Tuple[int, int], int],
        group_to_avg_coord: Dict[Tuple[int, int], np.ndarray],
        model: np.ndarray,
        group_diameter: Dict[Tuple[int, int], float],
        output_data_path: Path
):
    
    FINAL_COORDS_FILE = output_data_path / "S2_final_coords"
    FINAL_EDGES_FILE = output_data_path / "S2_final_edges"
    EDGE_ATTRIBUTES_FILE = output_data_path / "S2_edge_attributes"
    COORD_ATTRIBUTES_FILE = output_data_path / "S2_coord_attributes"

    # Create successor count for each node
    successor_count = getSuccessor(prev_group=prev_group)

    # Build minimal tree
    minimal_tree = {(0, 0): (0, 0)}
    edge_area_per_group_id = {(0, 0): [1]}

    not_skip_groups = {(0, 0)}

    # Propagates prev until node with succesor_count of not 1 appears,
    # i.e. either 0 (end node), or >1 which is a split,好像没有0吧
    for group, prev_group_index in prev_group.items():
        curr_dist, group_index = group
        prev = (curr_dist - 1, prev_group_index)
        print(f"Prev: {prev}, curr: {group}")
        if prev not in successor_count:
            continue

        if prev not in edge_area_per_group_id:
            edge_area_per_group_id[prev] = []
        if successor_count[prev] == 1:
            # 具有 1 个 succesor 的group 在 tree 中 属于同一个 节点？
            minimal_tree[group] = minimal_tree[prev] # Tree[current group] = Tree[prev group]
            if group not in edge_area_per_group_id:
                edge_area_per_group_id[group] = edge_area_per_group_id[prev].copy()
            # Use this if not skeletonize
            edge_area_per_group_id[group].append(group_area[group])

        else:
            minimal_tree[group] = prev
            not_skip_groups.add(prev)
            edge_area_per_group_id[group] = [group_area[group]]

    # Remove nodes which add no information
    for group, successors in successor_count.items():
        # Filter nodes, make sure not to filter start node
        # 去除中间体素，只保留 起点 和 分叉点
        if group not in not_skip_groups:
            minimal_tree.pop(group, None)


    # Save nodes
    xs = []
    ys = []
    zs = []
    group_attr = []

    # Calculate final coordinates and group coordinates
    for group_id in minimal_tree:
        c = group_to_avg_coord[group_id]
        xs.append(c[0])
        ys.append(c[1])
        zs.append(c[2])
        # 这是对节点的 area 重新计算了？
        group_area[group_id] = find_radius_via_sphere(c.tolist(), {1}, model) * 2 # 直径
        group_diameter[group_id] = (group_area[group_id] / 2) ** 2 * math.pi # math.pi * r **2 面积
        group_attr.append(np.array([
            group_diameter[group_id], group_area[group_id], group_id[0]
        ], dtype=object))

    final_coords = np.array([xs, ys, zs]) # (3, 319)

    xs = []
    ys = []
    zs = []
    edge_attr = []

    # Calculate edge and coord attributes
    for group_id in minimal_tree:
        # Skip first node
        if group_id != (0, 0):
            # Get coordinates for previous nodes
            c = group_to_avg_coord[group_id]
            prev_group_id = minimal_tree[group_id]
            prev_c = group_to_avg_coord[prev_group_id]
            xs.append([c[0], prev_c[0]])
            ys.append([c[1], prev_c[1]])
            zs.append([c[2], prev_c[2]])

            # Add edge attributes
            area1 = group_diameter[group_id]
            area2 = group_diameter[prev_group_id]
            curr_edge_areas = [round((area1 + area2) / 2)] * len(edge_area_per_group_id[group_id]) # 值代表结尾和起始的平均面积，list长度代表本段的长度？
            print(group_id, curr_edge_areas)
            # avg_area = sum(curr_edge_areas) / len(curr_edge_areas)
            # avg_diameter = sum([calc_diameter(a) for a in curr_edge_areas]) / len(curr_edge_areas)
            # edge_attr.append(np.array([avg_diameter, avg_area]))
            edge_attr.append(np.array(curr_edge_areas))


    final_edges = np.array([xs, ys, zs]) # (3, 318, 2)

    np.savez_compressed(FINAL_COORDS_FILE, np.array(final_coords))
    np.savez_compressed(FINAL_EDGES_FILE, np.array(final_edges))
    np.savez_compressed(COORD_ATTRIBUTES_FILE, np.array(group_attr))
    np.savez_compressed(EDGE_ATTRIBUTES_FILE, np.array(edge_attr, dtype=object))

    return {
        'final_coords': np.array(final_coords),
        'final_edges': np.array(final_edges),
        'coord_attr': np.array(group_attr),
        'edge_attr': np.array(edge_attr, dtype=object)
    }


def createTree(
        reducedModel: np.ndarray,
        distance_to_coords: List[List[np.ndarray]],
        coord_to_previous: Dict[Tuple[np.ndarray], np.ndarray],
        outputDataPath: Path
):
    groupInfo  = groupVoxel(distance_to_coords, coord_to_previous)
    roughTreeInfo = groupAnalysis(
        prev_group=groupInfo['prev_group'],
        group_area=groupInfo['group_area'],
        group_to_avg_coord=groupInfo['group_to_avg_coord'],
        model=reducedModel,
        group_diameter=groupInfo['group_diameter'],
        output_data_path=outputDataPath
    )
    return roughTreeInfo


if __name__ == '__main__':

    # reducedModelPath = Path(r'E:\conda_projects\Airway\data\CY_14210425\S0_Airway_preprocessed.npz')
    import sys
    projDir = sys.argv[1]
    reducedModelPath = Path(projDir).joinpath('S0_Airway_preprocessed.npz')
    txtDir = reducedModelPath.parent

    coord_to_previous = {}
    with open(txtDir.joinpath("S1_map_coord_to_previous.txt"), "r") as dist_file:
        for line in dist_file.read().split("\n"):
            if line != "":
                [first_half, second_half] = line.split(":")
                coord = parse_coord(first_half, ",")
                prev = parse_coord(second_half, " ")
                coord_to_previous[coord] = prev

    distance_to_coords_file = reducedModelPath.parent / 'S1_map_distance_to_coords.npz'
    distance_to_coords = np.load(distance_to_coords_file, allow_pickle=True)["arr_0"]

    _ = createTree(
        reducedModel=np.load(reducedModelPath)['arr_0'],
        distance_to_coords=distance_to_coords,
        coord_to_previous=coord_to_previous,
        outputDataPath=txtDir
    )

# "D:\mambaforge\envs\skeletonA\python.exe" E:\conda_projects\Airway\custom\createTree.py E:\conda_projects\Airway\data\UN_10122643T

# coord_to_previous = {}
# coord_to_next_count = {}
# for dictionary, filename in [
#     (coord_to_previous, txtDir.joinpath("map_coord_to_previous.txt")),
#     (coord_to_next_count, txtDir.joinpath("map_coord_to_next_count.txt")),
# ]:  
#     assert filename.is_file(), f"{filename} is not a valid file, exiting..."
#     with open(filename, "r") as dist_file:
#         for line in dist_file.read().split("\n"):
#             if line != "":
#                 [first_half, second_half] = line.split(":")
#                 coord = parse_coord(first_half, ",")
#                 prev = parse_coord(second_half, " ")
#                 dictionary[coord] = prev