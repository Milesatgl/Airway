"""Creates a color mask for lingula and the other

"""
import random
import re
from queue import Queue, PriorityQueue
from typing import Tuple, Callable, Dict

import numpy as np
import networkx as nx
from pathlib import Path

from helperFunctions import adjacent, get_coords_in_sphere_at_point

import yaml

CLASSIFICATION_CONFIG_PATH = Path(r'E:\conda_projects\Airway\airway\configs\classification.yaml')

def get_dict_from_yaml(curr_config_path: Path, ignore_if_does_not_exist=False) -> Dict:
    if ignore_if_does_not_exist and not curr_config_path.exists():
        return {}
    assert curr_config_path.exists(), f"Config {curr_config_path} does not exist!"
    with curr_config_path.open("r") as config_file:
        return yaml.load(config_file.read(), yaml.FullLoader)
    

classification_config = get_dict_from_yaml(CLASSIFICATION_CONFIG_PATH)


def get_first_matching_ids(tree: nx.Graph, condition: Callable[[nx.Graph, int], bool]):
    allowed = {"0"}
    ids = []
    bfs_successors = dict(nx.bfs_successors(tree, "0"))
    for node_id in tree.nodes():
        if node_id in allowed:
            if condition(tree, node_id):
                ids.append(node_id)
            else:
                allowed.update(bfs_successors.get(node_id, []))
    return ids

def is_lobe(tree: nx.Graph, node_id: int) -> bool:
    return re.fullmatch(r"[RL](Lower|Middle|Upper)Lobe", tree.nodes[node_id]["split_classification"]) is not None

def is_segment(tree: nx.Graph, node_id: int) -> bool:
    return classification_config.get(tree.nodes[node_id]["split_classification"], {}).get("clustering_endnode", False)



def fill_color_with_priority_queue(node_properties, model, distance_mask, color_mask):
    queue = PriorityQueue()
    for node, point, curr_color, radius, parent_dist in node_properties:
        queue.put((-distance_mask[point], parent_dist, point))
        color_mask[point] = curr_color
    while not queue.empty():
        _, min_dist, curr_point = queue.get()
        for adj in map(tuple, adjacent(curr_point)):
            if color_mask[adj] == 0 and model[adj] == 1 and min_dist < distance_mask[adj]:
                queue.put((-distance_mask[adj], min_dist, adj))
                color_mask[adj] = color_mask[curr_point]


def find_legal_point(node, distances):
    p = get_point(node)
    queue = Queue()
    queue.put(p)
    visited = {p}
    while not queue.empty():
        for adj in map(tuple, adjacent(queue.get())):
            if adj not in visited:
                if distances[adj] != 0:
                    return adj
                visited.add(adj)
                queue.put(adj)


def fill_sphere_around_point(
    radius: int,
    point: Tuple[int, int, int],
    color_mask: np.ndarray,
    curr_color: int,
):
    sphere_around_point = get_coords_in_sphere_at_point(radius * 2.5, point)
    color_mask[sphere_around_point] = curr_color


def color_hex_to_floats(h: str):
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def get_color_variation(color, variance=0.1):
    def var(h):
        return max(0.0, min(h * (1 + random.uniform(-1, 1) * variance), 1.0))

    return tuple(map(var, color))


def get_point(node):
    return round(node["x"]), round(node["y"]), round(node["z"])

def get_nodes_visit_order(tree: nx.Graph, distance_mask: np.ndarray, should_color_node: Callable):
    # The 0th color is unassigned, the 1st color is just bronchus.
    # The distinction is important because the 0th color can be changed.
    color_hex_codes = [color_hex_to_floats("ffffff")] * 2
    first_node = list(tree.nodes)[0]
    map_node_id_to_color_id: Dict[str, int] = {"0": 1}
    map_node_id_to_color_id_if_colored: Dict[str, int] = {"0": 1}
    nodes_visit_order = []
    for (parent_index, successors) in nx.bfs_successors(tree, first_node):
        parent_node = tree.nodes[parent_index]
        parent_dist = distance_mask[find_legal_point(parent_node, distance_mask)] + parent_node["group_size"]
        for s in successors:
            succ_node = tree.nodes[s]
            point = find_legal_point(succ_node, distance_mask)
            succ_radius = succ_node["group_size"] / 2
            color_id = len(color_hex_codes) if should_color_node(s) else map_node_id_to_color_id[parent_index]
            map_node_id_to_color_id[s] = color_id
            nodes_visit_order.append((succ_node, point, color_id, succ_radius, parent_dist))
            if should_color_node(s):
                map_node_id_to_color_id_if_colored[s] = color_id
                if "color" in succ_node:
                    color_hex_codes.append(color_hex_to_floats(succ_node["color"]))
                elif parent_index in map_node_id_to_color_id:
                    parent_color = color_hex_codes[map_node_id_to_color_id[parent_index]]
                    color_hex_codes.append(get_color_variation(parent_color))
                else:
                    color_hex_codes.append(color_hex_to_floats("ffffff"))
    return nodes_visit_order, color_hex_codes, map_node_id_to_color_id_if_colored


def createColorMask(model: np.ndarray, distanceMask: np.ndarray, classifiedTree: nx.Graph,  output_data_path: Path):

    def should_color_func(condition: Callable[[nx.Graph, int], bool]) -> Callable:
        return lambda s: s in get_first_matching_ids(classifiedTree, condition)
    
    print(get_first_matching_ids(classifiedTree, is_lobe))
    print(get_first_matching_ids(classifiedTree, is_segment))
    for func, filename in [
        (lambda _: True, f"bronchus_color_mask.npz"),
        (should_color_func(is_segment), f"segments.npz"),
        (should_color_func(is_lobe), f"lobes.npz"),
    ]:
        color_mask = np.full(model.shape, 0)
        nodes_visit_order, color_hex_codes, map_node_id_to_color_id = get_nodes_visit_order(
            classifiedTree, distanceMask, func
        )
        map_color_id_to_node_id = {value: key for key, value in map_node_id_to_color_id.items()}
        map_color_id_to_classification = {}
        for c in map_color_id_to_node_id:
            node = classifiedTree.nodes[map_color_id_to_node_id[c]]
            classification = node.get("split_classification_gt", "")
            if classification == "":
                classification = node.get("split_classification", "")
            map_color_id_to_classification[c] = classification
        print(color_hex_codes)

        fill_color_with_priority_queue(nodes_visit_order, model, distanceMask, color_mask)

        print("Colors:")
        for color, occ in zip(*np.unique(color_mask, return_counts=True)):
            print(f"Color {color} appears {occ:,} times in color mask")
        np.savez_compressed(
            output_data_path / filename,
            color_mask=color_mask,
            color_codes=np.array(color_hex_codes),
            color_id_to_node_id=map_color_id_to_node_id, # type: ignore
            color_id_to_classification=map_color_id_to_classification, # type: ignore
        )


if __name__ == "__main__":
    reducedModelPath = Path(r'E:\conda_projects\Airway\data\customTest\airTree_preprocessed.npz')
    distanceMaskPath = reducedModelPath.parent / 'distance_mask.npz'
    classifiedTreePath = reducedModelPath.parent / 'postProcessingTree_classified.graphml'
    createColorMask(
        model=np.load(reducedModelPath)['arr_0'],
        distanceMask=np.load(distanceMaskPath)['arr_0'],
        classifiedTree=nx.read_graphml(classifiedTreePath),
        output_data_path=reducedModelPath.parent
    )