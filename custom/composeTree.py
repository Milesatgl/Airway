import sys
import math

import numpy as np
import networkx as nx
from pathlib import Path
from typing import Tuple, List, Dict

def get_value_from_model(coord: List[int], reduced_model: np.ndarray) -> int:
    return reduced_model[coord[0], coord[1], coord[2]]

def get_weight(coord1, coord2):
    '''欧式距离'''
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)

def check_axis(axis_id: int, coord_list: List[int], direction: str, reduced_model: np.ndarray) -> Tuple[int, int]:
    '''returns the lobe bumber and the path length to this coordinate'''
    maximum = np.shape(reduced_model)[axis_id] # 某个轴向 model 的最大 长度？
    coord = coord_list[axis_id]
    path_len = 0
    # print(  "check: " + direction + " of " + str(axisID) + " coord: " + str(coord)
    #        + " maximum " + str(maximum))
    curr_coord_list = coord_list.copy() # 此处浅拷贝的作用是什么呢？List[int]的浅拷贝就是深拷贝？
    while coord in range(0, maximum - 1):
        path_len += 1
        coord += 1 if direction == 'positive' else -1 # 1 if direction == 'positive' else -1
        curr_coord_list[axis_id] = coord
        lobe = get_value_from_model(curr_coord_list, reduced_model)
        if lobe > 1:
            break
    return lobe, path_len


def get_lobe(coords: Tuple[float, float, float], reduced_model: np.ndarray) -> int:
    orig_coord_list = [int(round(i)) for i in coords]
    lobe_paths: Dict[int, int] = {}

    for i in range(0, 7):
        lobe_paths.update({i: 8192})
    
    # 总共三个轴 0，1，2
    for axis_id in range(0, 3):
        for direction in ['positive', 'negtive']:
            lobe_path_len = check_axis(axis_id, orig_coord_list, direction, reduced_model)
            # if lobe_paths.get(lobe_path_len[0]) > lobe_path_len[1]:
            #     lobe_paths.update({lobe_path_len[0]: lobe_path_len[1]})
            if (defaultPathLen := lobe_paths.get(lobe_path_len[0])) and defaultPathLen > lobe_path_len[1]:
                # 只有长度小于8192的被更新了？
                lobe_paths.update({lobe_path_len[0]: lobe_path_len[1]})

    # if more than maximum_path_length pixel between split and lobe, set lobe number to 0
    maximum_path_length = 24
    if lobe_paths.get(min(lobe_paths, key=lobe_paths.get)) > maximum_path_length: # type: ignore
        return 0
    else:
        return min(lobe_paths, key=lobe_paths.get) # type: ignore


def create_nodes(graph: nx.Graph, np_coord: np.ndarray, np_coord_attributes: list, reduced_model: np.ndarray):
    '''returns a dict with association coordinates -> node Number'''
    # get node coordinates
    max_coords = np.shape(np_coord)[1] # 节点数量
    dic_coords_to_nodes: Dict[Tuple[float, float, float], int] = {}
    i = 0
    while i < max_coords:
        curr_coord = (np_coord[0][i], np_coord[1][i], np_coord[2][i])
        dic_coords_to_nodes.update({curr_coord: i})
        # level counts from root, where root = 0
        level_val = 8192 # 默认值的作用是什么？
        if i == 0:
            level_val = 0

        # first split never belongs to a lobe(肺叶？)
        if i == 1:
            lobe_val = 0
        else:
            # lobe_val = get_lobe(curr_coord, reduced_model)
            lobe_val = 0 # 这个现在用不到

        group_size = np_coord_attributes[i][1] # 这个其实是直径，不过也是反映了面积
        group = np_coord_attributes[i][2] # 这个是 curr dist

        graph.add_node(
            i,
            x=curr_coord[0],
            y=curr_coord[1],
            z=curr_coord[2],
            lobe=lobe_val,
            level=level_val,
            group_size=group_size,
            group=group
        )
        i += 1

    return dic_coords_to_nodes
            
            
def create_edges(
        graph: nx.Graph, np_edges: np.ndarray,
        dic_coords: Dict[Tuple[float, float, float], int],
        edge_attributes: list
):
    '''returns a dict with association edge -> weight'''
    dic_edges_to_weight = {}
    max_edges = np.shape(np_edges)[1]
    i = 0
    while i < max_edges:
        coord1 = (np_edges[0][i][0], np_edges[1][i][0], np_edges[2][i][0])
        coord2 = (np_edges[0][i][1], np_edges[1][i][1], np_edges[2][i][1])
        curr_weight = get_weight(coord1, coord2)
        edge = (dic_coords[coord1], dic_coords[coord2]) # （当前node，前一个node）
        dic_edges_to_weight.update({edge: curr_weight})
        group_sizes = " ".join([str(attr) for attr in edge_attributes[i]])
        graph.add_edge(edge[0], edge[1], weight=curr_weight, group_sizes=group_sizes)
        i += 1
    return dic_edges_to_weight


def set_level(input_graph: nx.Graph):
    """Takes a graph and set the level attribute for every node.
    定义从根节点(节点0)到指定节点的距离(层级深度)。
    1、寻找级别最低的邻居节点(有效父节点)；
    2、更新当前节点的级别为这个父节点的级别+1. 
    就是说寻找当前节点的父节点，然后当前节点的level是父节点level+1 ？
    实现层次结构上的级别推理。"""
    graph = input_graph.copy()
    for node in nx.nodes(graph):
        neighbors = nx.neighbors(graph, node)
        # identify parent 跳过第一个
        if node != 0:
            parent = node
            # possible parrent
            for poss_parent in neighbors:
                if graph.nodes[parent]["level"] > graph.nodes[poss_parent]["level"]:
                    parent = poss_parent
            # print("parent {} -- level -> {}"
            #        .format(parent,graph.nodes[parent]['level']))
            graph.nodes[node]['level'] = graph.nodes[parent]['level'] + 1
    return graph


def set_attribute_to_node(
        graph: nx.Graph,
        filter_by_value_attribute: Tuple[str, int],
        target: Tuple[str, int]
):
    '''
    set_attribute_to_node(graph, filter, target) set or update existing attributes to
    a node filtered by filter.
    
    graph -> a graph
    filter = (filterAttrib, filterVal) -> Nodes whom filterAttrib has filterVal
    target = (targetAttrib, targetValue) -> Set new targetValue to a nodes targetAttrib'''
    graph = graph.copy()

    def filter_for_attrib(node_id: int):
        return graph.nodes[node_id][filter_by_value_attribute[0]] == filter_by_value_attribute[1]
    
    view = nx.subgraph_view(graph, filter_node=filter_for_attrib)

    for node in nx.nodes(view):
        graph.nodes[node][target[0]] = target[1]

    return graph


def show_stats(graph: nx.Graph, pat_id: str):
    print(f"\nstatistics for the graph of patient: {pat_id}")
    print(f"nodes: {graph.number_of_nodes()}")
    print(f"edges: {graph.number_of_edges()}\n")


def composeTree(
            reduced_model: np.ndarray, final_coords: np.ndarray,
            final_edges: np.ndarray, coord_attributes: list, edge_attributes: list,
            patient_id: str, output_data_path: Path
):
        # print(np.unique(reduced_model))
        # Remove all voxels 7, 8 and 9 since these are veins/arteries and not useful in classification
        reduced_model[reduced_model >= 7] = 0
        # Create empty graph
        graph = nx.Graph(patient=patient_id)
        # Compose graph
        dic_coords = create_nodes(graph, final_coords, coord_attributes, reduced_model)
        dic_edges = create_edges(graph, final_edges, dic_coords, edge_attributes)

        # set levels to the graph
        graph = set_level(graph)
        # level 2 does not belong to a lobe
        graph = set_attribute_to_node(graph, ("level", 2), ('lobe', 0))

        show_stats(graph, patient_id)

        nx.write_graphml(graph, output_data_path / "S3_composeTree.graphml")


if __name__ == '__main__':
    # reducedModelPath = Path(r'E:\conda_projects\Airway\data\CY_14210425\S0_Airway_preprocessed.npz')
    import sys
    projDir = sys.argv[1]
    reducedModelPath = Path(projDir) / 'S0_Airway_preprocessed.npz'

    coord_file_path = reducedModelPath.parent / "S2_final_coords.npz"
    edges_file_path = reducedModelPath.parent / "S2_final_edges.npz"
    coord_attributes_file_path = reducedModelPath.parent / "S2_coord_attributes.npz"
    edge_attributes_file_path = reducedModelPath.parent / "S2_edge_attributes.npz"

    reduced_model = np.load(reducedModelPath)["arr_0"]

    final_coords = np.load(coord_file_path)["arr_0"]
    final_edges = np.load(edges_file_path)["arr_0"]
    coord_attributes = np.load(coord_attributes_file_path, allow_pickle=True)["arr_0"]
    edge_attributes = np.load(edge_attributes_file_path, allow_pickle=True)["arr_0"]

    composeTree(
        reduced_model=reduced_model,
        final_coords=final_coords,
        final_edges=final_edges,
        coord_attributes=coord_attributes,
        edge_attributes=edge_attributes,
        patient_id=reducedModelPath.parent.name,
        output_data_path=reducedModelPath.parent
    )

# "D:\mambaforge\envs\skeletonA\python.exe" E:\conda_projects\Airway\custom\composeTree.py E:\conda_projects\Airway\data\UN_10122643T