# Import necessary libraries
import matplotlib.pyplot as plt

# ------------------- Definition -------------------

# Scatter Plot:
# A scatter plot is a type of data visualization that uses dots to represent the values
# of two different variables. It is useful for identifying correlations, trends, and outliers.

# ------------------- Sample Data -------------------

# Let's assume we're plotting study hours vs exam scores
study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
exam_scores = [35, 40, 50, 52, 55, 65, 70, 75, 78, 85]

# ------------------- Create Scatter Plot -------------------

plt.figure(figsize=(10, 6))  # Set figure size
plt.scatter(study_hours, exam_scores, color='green', marker='o', s=80, edgecolor='black')

# ------------------- Add Titles and Labels -------------------

plt.title("📊 Scatter Plot: Study Hours vs Exam Scores", fontsize=14)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score (%)")
plt.grid(True, linestyle='--', alpha=0.6)

# ------------------- Show Plot -------------------

plt.show()
