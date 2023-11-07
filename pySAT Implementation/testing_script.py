from decision_tree_module import Feature, DecisionTree, DecisionTreeNode

# 1. Setting Up Features
age_feature = Feature(name="Age", possible_values=list(range(101))) # Age values from 0 to 100
gender_feature = Feature(name="Gender", possible_values=["Male", "Female"])

# 2. Building the Decision Tree
tree = DecisionTree()

# Root node: Decision based on age
tree.root.feature = age_feature
tree.root.threshold = 18

# Left child of root: Represents Age <= 18
young_node = DecisionTreeNode(label="Young", parent=tree.root)
tree.root.left = young_node

# Right child of root: Represents Age > 18, further decision based on Gender
gender_decision_node = DecisionTreeNode(feature=gender_feature, parent=tree.root)
tree.root.right = gender_decision_node

# Children for gender_decision_node
male_node = DecisionTreeNode(label="Adult Male", parent=gender_decision_node)
female_node = DecisionTreeNode(label="Adult Female", parent=gender_decision_node)

gender_decision_node.left = male_node
gender_decision_node.right = female_node

# 1. Making Predictions:
data_example1 = {"Age": 15, "Gender": "Male"}
prediction1 = tree.predict(data_example1)
print(f"For data {data_example1}, prediction is {prediction1}.")  # Output: For data {'Age': 15, 'Gender': 'Male'}, prediction is Young.

data_example2 = {"Age": 25, "Gender": "Female"}
prediction2 = tree.predict(data_example2)
print(f"For data {data_example2}, prediction is {prediction2}.")  # Output: For data {'Age': 25, 'Gender': 'Female'}, prediction is Adult Female.

# 2. Visualize Tree:
tree.visualize()


data_points = [
    {"Age": 15, "Gender": "Male"},
    {"Age": 25, "Gender": "Female"},
    {"Age": 40, "Gender": "Male"},
    {"Age": 30, "Gender": "Female"}
]

true_labels = ["Young", "Adult Female", "Adult Male", "Adult Female"]

tree_accuracy = tree.accuracy(data_points, true_labels)
print(f"Decision Tree Accuracy: {tree_accuracy:.2f}%")