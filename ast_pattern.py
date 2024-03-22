import pandas as pd
import torch
from tqdm import *
import sys
sys.setrecursionlimit(10000)
from prepare_data import get_blocks

def is_descendant(i, j): #判断i是否为j的后代节点
    if i == j:
        return True
    
    if not isinstance(j, str):
        for _, child in j.children():
            if is_descendant(i, child):
                return True

    return False

def build_descendant(blocks):
    n = len(blocks)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        if blocks[i].node in ['Compound', 'End']:
            continue
        for j in range(n):
            if j != i and is_descendant(blocks[i].node, blocks[j].node):
                matrix[i][j] = 1
    return matrix

def find_root(node_list, descendant_matrix):
    root_list = []
    for i in node_list:
        is_root = True
        for j in node_list:
            if j != i and descendant_matrix[i][j]:
                is_root = False
        if is_root:
            root_list.append(i)

    return root_list

def build_tree(node_list, descendant_matrix):
    if len(node_list) == 0:
        return {}
    
    tree = {}
    root_list = find_root(node_list, descendant_matrix)
    setA = set(node_list)
    setB = set(root_list)
    child_list = list(setA - setB)
    child_root = find_root(child_list, descendant_matrix)
    for root in root_list:
        tree[root] = []
        for child in child_root:
            if descendant_matrix[child][root]:
                tree[root].append(child)
    subtree = build_tree(child_list, descendant_matrix)
    tree.update(subtree)
    return tree

def adjacency_matrix(tree, n):
    adj_matrix = [[1] * n for _ in range(n)]
    for i in range(n):
        for j in tree.get(i, []):
            adj_matrix[i][j] = 0
            adj_matrix[j][i] = 0
    row_list = []
    for i in range(n):
        tensor = torch.tensor(adj_matrix[i])
        row_list.append(tensor)
    tensor_matrix = torch.stack(row_list)
    return tensor_matrix.bool()

def trans2matrix(r):
    blocks = []
    get_blocks(r, blocks)
    descendant_matrix = build_descendant(blocks)
    node_list = list(range(len(blocks)))
    tree = build_tree(node_list, descendant_matrix)
    matrix = adjacency_matrix(tree, len(blocks))
    return matrix

if __name__ == '__main__':
    path_train = 'data/matrix_train.pkl'
    path_dev = 'data/matrix_dev.pkl'
    path_test = 'data/matrix_test.pkl'

    train = pd.read_pickle(path_train)
    dev = pd.read_pickle(path_dev)
    test = pd.read_pickle(path_test)

    print(train)

