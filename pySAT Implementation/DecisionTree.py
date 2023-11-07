from graphviz import Digraph

class TreeNode:
    def __init__(self, node_id, is_leaf=False):
        self.node_id = node_id
        self.is_leaf = is_leaf
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.depth = 0
        self.feature = None  # β(t): feature selection function
        self.threshold = None  # α(t): threshold selection function
        self.label = None  # θ(t): leaf labeling function

    def set_parent(self, parent_node):
        self.parent = parent_node
        self.depth = parent_node.depth + 1

    def set_children(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child


class DecisionTree:
    def __init__(self, root):
        self.root = root

    def predict(self, x):
        """
        Predict the label of a given data point x by traversing the tree.
        """
        def Θ(t, xi):
            if t.is_leaf:
                return t.label
            elif xi[t.feature] <= t.threshold:
                return Θ(t.left_child, xi)
            else:
                return Θ(t.right_child, xi)

        return Θ(self.root, x)

    def calculate_accuracy(self, X, gamma):
        """
        Calculate the accuracy of the decision tree on a dataset X with labels gamma.
        """
        correct_predictions = sum(1 for i, xi in enumerate(X, start=1) if self.predict(xi) == gamma[i])  # start=1 to match the gamma indices
        return correct_predictions / len(X)

    
    def visualize(self, path='tree_visualization'):
        dot = Digraph(comment='Decision Tree')

        def add_nodes_edges(t):
            if t.is_leaf:
                label = f"Leaf: {t.label}"
            else:
                label = f"{t.feature} <= {t.threshold}"
            dot.node(str(t.node_id), label)

            if t.left_child:
                dot.edge(str(t.node_id), str(t.left_child.node_id), label='yes')
                add_nodes_edges(t.left_child)

            if t.right_child:
                dot.edge(str(t.node_id), str(t.right_child.node_id), label='no')
                add_nodes_edges(t.right_child)

        add_nodes_edges(self.root)
        dot.render(path, view=True)

#Create node
def create_tree_node(node_id, is_leaf=False):
    return TreeNode(node_id, is_leaf)
