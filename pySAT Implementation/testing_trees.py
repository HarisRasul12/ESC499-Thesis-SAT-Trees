from graphviz import Digraph
#testing trees
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
        correct_predictions = sum(1 for i, xi in enumerate(X) if self.predict(xi) == gamma[i])
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

# Helper function to create a tree node
def create_tree_node(node_id, is_leaf=False):
    return TreeNode(node_id, is_leaf)


if __name__ == "__main__":
    # Define a simple tree structure
    # Let's assume we have a binary classification problem with features F1 and F2
    # and labels 0 and 1.

    # Create leaf nodes with labels
    leaf1 = create_tree_node(4, is_leaf=True)
    leaf1.label = 0  # Label for leaf node 1

    leaf2 = create_tree_node(5, is_leaf=True)
    leaf2.label = 1  # Label for leaf node 2

    # Create branching nodes with feature selections and thresholds
    branch1 = create_tree_node(2)
    branch1.feature = 'F1'  # Feature F1
    branch1.threshold = 5  # Threshold for F1

    branch2 = create_tree_node(3)
    branch2.feature = 'F2'  # Feature F2
    branch2.threshold = 3  # Threshold for F2

    # Create the root node with feature selections and thresholds
    root = create_tree_node(1)
    root.feature = 'F1'  # Feature F1
    root.threshold = 10  # Threshold for F1

    # Link the nodes to form the tree
    root.set_children(branch1, branch2)
    branch1.set_children(leaf1, leaf2)
    branch2.set_children(leaf1, leaf2)

    # Create the decision tree
    tree = DecisionTree(root)

    # Define a small dataset with known labels
    X = [
    {'F1': 5, 'F2': 3},  # Data point 1
    {'F1': 7, 'F2': 4},  # Data point 2
    {'F1': 2, 'F2': 6},  # Data point 3
    {'F1': 3, 'F2': 5},  # Data point 4
    ]

    # Labels for the dataset
    gamma = [
        0,  # Correct label for first data point
        1,  # Correct label for second data point
        0,  # Correct label for third data point
        1   # Correct label for fourth data point
    ]

    # Test predictions
    for i, x in enumerate(X):
        prediction = tree.predict(x)
        print(f"Data point {i+1}: {x}, Predicted label: {prediction}, Actual label: {gamma[i]}")

    # Calculate and print the accuracy
    accuracy = tree.calculate_accuracy(X, gamma)
    print(f"Accuracy of the decision tree: {accuracy * 100:.2f}%")

    # Now you can visualize the tree
    tree.visualize()