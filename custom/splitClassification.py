""" Classify splits in graphml tree
"""
import copy
import itertools
import math
import sys
import yaml
from queue import PriorityQueue
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime
from utils import getOutputPath

trees_thrown_out = 0
global_angles = []

CLASSIFICATION_CONFIG_PATH = Path(r'E:\conda_projects\Airway\airway\configs\classification.yaml')
ARRAY_ENCODING_PATH = Path(r'E:\conda_projects\Airway\airway\configs\array_encoding.yaml')



def get_dict_from_yaml(curr_config_path: Path, ignore_if_does_not_exist=False) -> Dict:
    if ignore_if_does_not_exist and not curr_config_path.exists():
        return {}
    assert curr_config_path.exists(), f"Config {curr_config_path} does not exist!"
    with curr_config_path.open("r") as config_file:
        return yaml.load(config_file.read(), yaml.FullLoader)


def get_point(node: Dict):
    return np.array([node["x"], node["y"], node["z"]])


def cost_exponential_diff_function(curr_vec: np.ndarray, target_vec: np.ndarray, exp=2, div=math.pi / 3):
    '''评估两个向量之间的方向相似性'''
    # 向量内积（点积） / （curr向量模长*target向量模长）--> cos(theta)(两向量单位化后的点积)
    angle_pre_arccos = (curr_vec @ target_vec) / (np.linalg.norm(curr_vec) * np.linalg.norm(target_vec))
    # 计算向量夹角(theta)，clip限制点积结果，避免超出arccos输入超出定义域
    angle_radians = np.arccos(np.clip(angle_pre_arccos, -1, 1))
    global_angles.append(angle_radians)
    # 使用 div 标准化 角度
    return (angle_radians / div) ** exp

def show_classification_vectors(tree: nx.Graph, successors: Dict):
    for node_id, children_ids in successors.items():
        node = tree.nodes[node_id]
        node_point = get_point(node)
        curr_classification = node["split_classification"]
        print(node_id, children_ids, curr_classification)
        for child_id in children_ids:
            child_node = tree.nodes[child_id]
            child_point = get_point(child_node)
            vec = child_point - node_point
            print(f"\tVector {node_id}->{child_id}: {list(vec)} ({tree.nodes[child_id]['split_classification']})")
        print()
    return tree

def add_defaults_to_classification_config(classification_config):

    defaults = {"children": [], "deep_descendants": [], "descendants": [], "take_best": False}

    for cid in classification_config:
        for key, val in defaults.items():
            classification_config[cid][key] = classification_config[cid].get(key, copy.deepcopy(val))
            # key 存在则值不变，key不存在则补充 键为key，值为val
    for cid in classification_config:
        if "vector" in classification_config[cid]:
            # 转换列表为数组
            classification_config[cid]["vector"] = np.array(classification_config[cid]["vector"])


def add_default_split_classification_id_to_tree(tree: nx.Graph):
    for node in tree.nodes:
        tree.nodes[node]["split_classification_gt"] = ""
        tree.nodes[node]["split_classification"] = f"c{node}"
    tree.nodes["0"]["split_classification"] = "Trachea"


def add_deep_descendants_to_classification_config(classification_config):
    def recursive_get(classification):
        if classification not in classification_config:
            return []
        cc = classification_config[classification]
        dd = cc["deep_descendants"]
        dd += cc.get("descendants", [])
        for child in cc["children"]:
            dd += recursive_get(child)
        cc["deep_descendants"] = list(set(dd))
        return dd

    recursive_get("Trachea")


def add_cost_by_level_in_tree(tree: nx.Graph, successors: Dict[str, List[str]]):
    def recursive_add_cost(curr_id="0", cost=1000000.0):
        tree.nodes[curr_id]["cost"] = float(0.0) # 全部初始化为 0 了，那这个函数还有什么作用
        for child_id in successors.get(curr_id, []):
            recursive_add_cost(child_id, cost / 2)

    recursive_add_cost()
    tree.nodes["0"]["cost"] = float(0.0)


def get_total_cost_in_tree(tree, successors):
    def rec_total(curr_id="0"):
        return tree.nodes[curr_id]["cost"] + sum(rec_total(child_id) for child_id in successors.get(curr_id, []))

    return rec_total()


def get_all_classifications_in_tree(tree, successors):
    def rec(curr_id):
        return [tree.nodes[curr_id]["split_classification"]] + sum([rec(i) for i in successors.get(curr_id, [])], [])

    return rec("0")


def add_colors_in_tree(tree, classification_config):
    for node_id in tree.nodes:
        node = tree.nodes[node_id]
        try:
            node["color"] = classification_config[node["split_classification"]]["color"]
        except KeyError:
            pass


def is_valid_tree(
    tree: nx.Graph,
    classification_config: Dict[str, Dict[str, Any]],
    successors: Dict[str, List[str]],
    start_node_id: str = "0",
):
    required_descendants = set()
    have_appeared = set()

    def recursive_is_valid_tree(current_id):
        nonlocal required_descendants, have_appeared, tree
        classification = tree.nodes[current_id]["split_classification"]

        # Make sure each classification appears only once
        if classification in have_appeared:
            print(f"Classification {classification} appears twice!")
            # 第一级不合格：已经出现过的分类第二次出现
            return False
        have_appeared.add(classification) # 第一次是 Trachea

        # Remember required descendants for subtree
        required_descendants.discard(classification) # 统计父节点（classification）的 所有 子节点分类信息（required_descendants），首先扔掉父节点分类信息
        curr_descendants = set()
        if classification in classification_config:
            # 下一级子节点分类信息
            curr_descendants = set(classification_config[classification].get("descendants", []))
        required_descendants |= curr_descendants # 第一次是 空，初始状态下，所有 descendants 都是 空

        # Recursively iterate over each node and require each node to be valid
        # 第二级不合格："当前节点下的所有节点的分类信息都是唯一的（都满足第一级限制）" 这个条件不满足
        for child_id in successors.get(current_id, []):
            if not recursive_is_valid_tree(child_id):
                return False

        # Tree is valid only if all descendants have been removed in the recursive steps above
        # isdisjoint() 方法用于判断两个集合是否包含相同的元素，如果没有返回 True，否则返回 False, 都为空时，返回True
        # 第三级不合格：required_descendants 和 curr_descendants 不包括相同的元素，也就是没有交集？
        if not required_descendants.isdisjoint(curr_descendants):
            # 如果两个集合不包括相同的元素
            print(
                f"Invalid because {required_descendants} is required as descendant, but is not available."
                f" Descendants: {curr_descendants} for node {classification}"
            )
            return False
        return True

    return recursive_is_valid_tree(start_node_id)



def classify_tree(
    starting_tree: nx.Graph,
    successors: Dict[str, List[str]],
    classification_config: Dict[str, Dict[str, Any]],
    starting_node="0",
    starting_cost=0,
):
    """
    Creates every valid classification for a tree based on the rules in classification.yaml

    Terminology:
        starting_* - function was called with these parameters
        curr_* - node which is temporarily considered root node in while loop
        child_* - nodes and their attributes which are children of curr
    """
    global trees_thrown_out

    # queue contains the tree currently being worked on, and the current steps to work on
    print(f'Classify tree for node {starting_node}, start cost is {starting_cost}')
    tree_variations_queue = PriorityQueue()
    tree_variations_queue.put((starting_cost, starting_tree, [starting_node]))

    print(starting_cost, starting_tree, starting_node, starting_tree.nodes[starting_node]["split_classification"])
    cost_hack = 0

    # While there are any tree variations in queue iterate over them
    print(f'Entering while loop')
    while not tree_variations_queue.empty():
        curr_cost, curr_tree, next_node_id_list = tree_variations_queue.get()
        print(f'Processing tree {id(curr_tree)}, node to be processed are: {next_node_id_list}, current cost: {curr_cost}')
        # Save which classifications have already been used so no invalid trees are created unnecessarily
        # 初始的 curr_classifications_used is {'Trachea', 'c1', 'c10', 'c100', 'c101', 'c102', 'c103', 'c104'... }
        curr_classifications_used = {curr_tree.nodes[node]["split_classification"] for node in curr_tree.nodes}

        # If there is a tree variation which has no next nodes in list, then return it if it is a valid tree.
        # Sine tree variations is a priority queue this must be the best possible (lowest cost) tree
        if len(next_node_id_list) == 0:
            # next_node_id_list 为 空 时，才会进入这个函数
            print(f'Node to be processed is empty, validing tree {id(curr_tree)} for starting node: {starting_node}')
            if is_valid_tree(curr_tree, classification_config, successors, starting_node):
                print(f"Valid tree {id(curr_tree)}, current cost is {curr_cost}, returnning")
                return [(curr_cost, curr_tree)]
            else:
                print(f'Invalid tee {id(curr_tree)}, moving on to next tree in queue')
                trees_thrown_out += 1
                continue

        # Divide next node list into curr node id, and rest which still need to be checked
        # next_node_id_list 是待分类的节点，取出第一个当前操作用，其余的放到rest_node_ids
        (curr_node_id, *rest_node_ids) = next_node_id_list # 只有一个元素时，rest_node_ids是一个空列表
        curr_node = curr_tree.nodes[curr_node_id]
        curr_classification = curr_node["split_classification"]
        curr_node_point = get_point(curr_node)
        print(f"Processing node {curr_node_id}, nodes to be processed: {rest_node_ids}")

        # Only handle if current classification (i.e. Bronchus/RB3, etc) is actually in classification config
        if curr_classification not in classification_config:
            print(f'Unbale to find {curr_classification} in config file, moving on to next tree in queue')
            continue

        # If there are more children than in the config then extend list to account for all of them
        children_in_rules: List[Optional[str]] = [
            child
            for child in classification_config[curr_classification]["children"]
            if child not in curr_classifications_used
        ] # 这个是 分类 的 名字
        # The ids as strings of nodes which succeed current node
        successor_ids: List[str] = successors.get(curr_node_id, []) # 后继的所有 node id
        adjust_for_unaccounted_children: int = len(successor_ids) - len(children_in_rules)
        children_in_rules.extend([None] * adjust_for_unaccounted_children) # 对于 多的 后继节点， 补充 None 为分类名

        # Defines list of all permutations(排列) of children including their cost
        # e.g. [(34.3, [('3', 'Bronchus')]) cost and the permutation where the node id specifies which
        # classification should be used
        cost_with_perm: List[Tuple[int, List[Tuple[str, str | None]]]] = []
        for perm in set(itertools.permutations(children_in_rules, r=len(successor_ids))):
            # successors_with_permutations example: [('2', 'RBronchus'), ('7', 'LBronchus')]
            successors_with_permutations: List[Tuple[str, str | None]] = list(zip(successor_ids, perm))
            print(f'Cost calculation for successors-permutations: {successors_with_permutations}')

            # Create a list of all descendants for each children, this then can be used to check whether any
            # of them share descendants when this list has non unique members

            # sum(iterable: List[List[str], List[str], ...], start: EmptyList)
            # Return the sum of a 'start' value (default: 0) plus an iterable of numbers
            # When the iterable is empty, return the start value.
            # This function is intended specifically for use with numeric values and may
            # reject non-numeric types.
            descendant_list = sum(
                [
                    list(classification_config.get(p, {}).get("deep_descendants", set()) if p is not None else set())
                    + ([] if p is None else [p])
                    for _, p in successors_with_permutations
                ],
                [],
            ) # 没看懂这个sum的作用
            
            permutation_shares_descendants = len(descendant_list) != len(set(descendant_list))
            if permutation_shares_descendants:
                # 保证 两个分支 各自的 所有子分支 没有交集
                print(f'Permutation shares descendants: {[d for d in set(descendant_list) if descendant_list.count(d) > 1]}')
                continue

            # Then check whether all children config rules have vectors defined, if not just take the best
            print(f'Initialize permutation cost as curr_cost: {curr_cost}')
            perm_cost = curr_cost # 每一个 子节点+children命名 组合 的 损失，是在当前节点的损失基础上累加的
            do_all_classifications_have_vectors = any(
                classification in classification_config
                and "vector" in classification_config.get(classification, {})
                for _, classification in successors_with_permutations
            )

            # Calculate cost of current permutation
            if do_all_classifications_have_vectors:
                for child_id, classification in successors_with_permutations:
                    print(f'Calculating cost for child node {child_id} with calssification {classification}')
                    child_node = curr_tree.nodes[child_id]
                    child_point = get_point(child_node)
                    vec = child_point - curr_node_point
                    if classification in classification_config:
                        target_vec = classification_config[classification]["vector"]
                        child_node["cost"] = float(cost_exponential_diff_function(vec, target_vec, 1, 1))
                        perm_cost += child_node["cost"]
                        print(f'Update permutaion cost to {perm_cost}')
            # cost_with_perm = [
            #  (cost0, [('2', 'RBronchus'), ('7', 'LBronchus')]),
            #  (cost1, [('7', 'RBronchus'), ('2', 'LBronchus')]),
            #  ...
            #]
            cost_with_perm.append((perm_cost, successors_with_permutations))

            # Only add first permutation if not all children have vectors
            if not do_all_classifications_have_vectors:
                print("Break cost calculation since not all classifications have vectors")
                break

        # Sort by cost, so we evaluate low cost first
        cost_with_perm.sort(key=lambda k: k[0])
        print(f'Finshed cost calculation, best guess is {cost_with_perm[0][1]} with cost {cost_with_perm[0][0]}')

        # If cost_with_perm is not empty
        if cost_with_perm:
            for perm_cost, successors_with_permutations in cost_with_perm:
                # (cost0, [('2', 'RBronchus'), ('7', 'LBronchus')])
                # print("successors with permutations:", successors_with_permutations)
                perm_tree = curr_tree.copy()
                treeId = id(perm_tree)
                print(f'Initilizing new tree {treeId} for each permutaion, current perm is {successors_with_permutations}')
                # [('2', 'RBronchus'), ('7', 'LBronchus')]
                for child_id, classification in successors_with_permutations:
                    if classification is not None:
                        perm_tree.nodes[child_id]["split_classification"] = classification
                # successors_with_permutations 是 当前节点的 子节点，也同样放到 rest_node_ids 里面
                next_nodes = rest_node_ids.copy() + [
                    child_id
                    for child_id, classification in successors_with_permutations
                    if classification in classification_config
                ]
                print(f'Nodes to processed for this new tree {treeId} are {next_nodes}')
                take_best = classification_config[curr_node["split_classification"]]["take_best"]
                print(f'Take_best status: {take_best}')
                if take_best:
                    # 开始 take_best 之后才会进入 递归
                    for child_node_id in successors[curr_node_id]:
                        print(f'Starting recur classify_tree for tree {treeId} and child {child_node_id}')
                        perm_cost, perm_tree = classify_tree(
                            perm_tree, successors, classification_config, child_node_id, perm_cost + cost_hack # type: ignore
                        )[0]
                        print(f'Finished recur classify_tree for tree {treeId} and child {child_node_id}, current cost: {perm_cost}, current tree: {id(perm_tree)}, removing this child from Nodes to processed')

                        next_nodes.remove(child_node_id)
                cost_hack += 0.000001
                print(f"Putting new tree variation")
                tree_variations_queue.put((perm_cost + cost_hack, perm_tree, next_nodes))
                if take_best:
                    # 开启take_best之后，操作完第一个 备选 tree 就结束函数
                    print(f'Take best enabled, skiping other permutaions')
                    break
                # print("Breaking for node", node['split_classification'], "since it is specified as take_best")
        else:
            print("WEIRD ELSE? cost wiht perm is empty...")
            tree_variations_queue.put((curr_cost, curr_tree, []))
    # return [(curr_cost, curr_tree)]
    raise Exception("Sacrebleu! Only invalid trees are possible!")


def splitClassification(tree: nx.Graph):
    classification_config = get_dict_from_yaml(CLASSIFICATION_CONFIG_PATH)
    successors: Dict[str, List[str]] = dict(nx.bfs_successors(tree, "0"))

    add_defaults_to_classification_config(classification_config)
    add_default_split_classification_id_to_tree(tree) # 初始化 classification
    add_deep_descendants_to_classification_config(classification_config)
    add_cost_by_level_in_tree(tree, successors)
    # print("\n".join(map(str, classification_config.items())))

    all_trees = classify_tree(tree, successors, classification_config)
    all_trees.sort(key=lambda x: x[0])
    
    print(f"Invalid trees thrown out: {trees_thrown_out}")
    print(f"All trees: {len(all_trees)}")
    return all_trees

    # validated_trees = []
    # for cost, curr_tree in all_trees:
    #     if is_valid_tree(curr_tree, classification_config, successors):
    #         validated_trees.append((cost, curr_tree))
    # print(f"Valid trees: {len(validated_trees)}")

    # for curr_cost, curr_tree in validated_trees:
    #     all_classifications = get_all_classifications_in_tree(curr_tree, successors)
    #     print(f"Cost={curr_cost:.2f}, {'B1+2 is in tree' if 'LB1+2' in all_classifications else ''}")


    # try:
    #     classified_tree = validated_trees[0][1]
    # except IndexError:
    #     print("ERROR: Could not create valid tree! Using invalid tree instead.", file=sys.stderr)
    #     classified_tree = all_trees[0][1]
    # add_colors_in_tree(classified_tree, classification_config)
    # show_classification_vectors(classified_tree, successors)
    # nx.write_graphml(classified_tree, output_path)


if __name__ == "__main__":
    import sys
    start = datetime.now()
    # treePath = Path(r"E:\conda_projects\Airway\data\CY_14210425\S4_postProcessingTree.graphml")
    projDir = sys.argv[1]
    treePath = Path(projDir) / 'S4_postProcessingTree.graphml'
    graph = nx.read_graphml(str(treePath))
    allTrees = splitClassification(graph)
    nx.write_graphml(allTrees[0][1], getOutputPath(treePath, '_classified.graphml', 'S5_'))
    # treePath.parent.joinpath(treePath.stem+'_classified.graphml')
    print(f'Elapsed time: {datetime.now() - start}')


# "D:\mambaforge\envs\skeletonA\python.exe" E:\conda_projects\Airway\custom\postProcessingTree.py E:\conda_projects\Airway\data\UN_10122643T