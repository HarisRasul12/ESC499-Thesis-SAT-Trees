import numpy as np
import pytest
from satree.treemodder.builder import build_complete_tree, create_literals
from satree.treemodder.tree_utils import get_ancestors


def test_build_complete_tree():
    depth = 2
    tree_structure, TB, TL = build_complete_tree(depth)
    # A complete binary tree of depth 2 has (2^(2+1)-1)=7 nodes.
    assert len(tree_structure) == 7
    # Branching nodes: indices 0 to (2^2 -1)-1 = indices 0,1,2.
    assert TB == [0, 1, 2]
    # Leaves are nodes 3,4,5,6.
    assert TL == [3, 4, 5, 6]


def test_create_literals():
    branch_nodes = [0, 1]
    leaf_nodes = [2, 3]
    feature_indices = np.array(['0', '1'])
    class_labels = [0, 1]
    dataset_size = 3
    literals, current_index = create_literals(branch_nodes, leaf_nodes, feature_indices, class_labels, dataset_size,
                                              fixed_tree=True)
    assert 'a_0_0' in literals
    assert 's_0_0' in literals
    assert 'z_0_2' in literals
    assert 'g_2_0' in literals
    assert 'p_0' in literals
    assert current_index > len(literals)


def test_get_ancestors():
    # For node index 5 in a complete binary tree: its parent is (5-1)//2 = 2, and parent's parent is 0.
    ancestors_left = get_ancestors(5, 'left')
    ancestors_right = get_ancestors(5, 'right')
    # For node 5 (which is odd), it is a left child, so 'left' ancestors should include its parent.
    assert 2 in ancestors_left
    # For side 'right', we expect an empty list.
    assert ancestors_right == [0]
