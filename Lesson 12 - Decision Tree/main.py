# ------------------- Import Necessary Libraries -------------------
# Import packages for data handling, model training, evaluation, and visualization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris                   # To load a sample dataset
from sklearn.model_selection import train_test_split       # To split data into training and testing sets
from sklearn.tree import DecisionTreeClassifier, plot_tree   # Decision Tree algorithm and plot function
from sklearn.metrics import accuracy_score                 # For model evaluation

# ------------------- Definition -------------------
# Decision Tree:
# A Decision Tree is a flowchart-like structure used for classification or regression.
# - Each internal node represents a "test" on a feature.
# - Each branch represents an outcome of the test.
# - Each leaf node represents a class label or a decision outcome.
#
# The model recursively splits the dataset into subsets based on feature values to best separate the classes.
# They are intuitive to interpret and work well on both classification and regression tasks.

# ------------------- Load the Dataset -------------------
iris = load_iris()        # Load the iris dataset
X = iris.data             # Feature matrix (sepal and petal measurements)
y = iris.target           # Target vector (iris species)

# ------------------- Split Data for Training and Testing -------------------
# Split the data into training (70%) and testing (30%) sets to evaluate generalization performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------- Train the Decision Tree Classifier -------------------
# Create the Decision Tree model. 'random_state' ensures reproducibility.
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data.
decision_tree_model.fit(X_train, y_train)

# ------------------- Make Predictions -------------------
# Predict the species for the test set.
y_pred = decision_tree_model.predict(X_test)

# ------------------- Evaluate the Model -------------------
# Calculate the accuracy of the model's predictions.
accuracy = accuracy_score(y_test, y_pred)
print("📊 Decision Tree Accuracy:", accuracy)

# ------------------- Visualize the Decision Tree -------------------
# Plot the trained decision tree for better understanding of the model decision process.
plt.figure(figsize=(12, 8))
plot_tree(decision_tree_model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,            # Colors nodes based on class
          rounded=True,           # Rounded corners for better aesthetics
          fontsize=10)
plt.title("Decision Tree - Iris Dataset")
plt.show()
